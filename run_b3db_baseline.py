import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, matthews_corrcoef, brier_score_loss
from xgboost import XGBClassifier

# 1. Load & Curate B3DB_classification.tsv
TSV_PATH = r"e:/BBBP minor + portfolio project/B3DB_classification.tsv"# Change path if inside data/raw/
df = pd.read_csv(TSV_PATH, sep="\t")

smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
label_col = "BBB+/BBB-" if "BBB+/BBB-" in df.columns else "label"

records = []
for _, row in df.iterrows():
    smi = str(row[smiles_col]).strip()
    lbl_raw = str(row[label_col]).strip()
    if lbl_raw not in ["BBB+", "BBB-"]:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    inchikey = Chem.MolToInchiKey(mol)
    y = 1 if lbl_raw == "BBB+" else 0
    records.append({"SMILES": can_smi, "InChIKey": inchikey, "label": y, "mol": mol})

clean_df = pd.DataFrame(records)

# Remove conflicting InChIKeys and deduplicate
label_counts = clean_df.groupby("InChIKey")["label"].nunique()
valid_keys = label_counts[label_counts == 1].index
clean_df = clean_df[clean_df["InChIKey"].isin(valid_keys)].drop_duplicates(subset=["InChIKey"]).reset_index(drop=True)

print(f"\n[DATA SUMMARY] B3DB Compounds (Post-Dedup): {len(clean_df)}")

# 2. Bemis-Murcko Scaffold Split (80% Train / 20% Test)
scaffold_to_indices = defaultdict(list)
for idx, mol in enumerate(clean_df["mol"]):
    scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    scaffold_to_indices[scaff].append(idx)

# Sort scaffold buckets largest to smallest for deterministic 80/20 allocation
sorted_scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)
train_idx, test_idx = [], []
train_cutoff = int(0.8 * len(clean_df))

for bucket in sorted_scaffolds:
    if len(train_idx) + len(bucket) <= train_cutoff:
        train_idx.extend(bucket)
    else:
        test_idx.extend(bucket)

print(f"[SPLIT SUMMARY] Train: {len(train_idx)} | Internal Scaffold Test: {len(test_idx)}")

# 3. Featurization: 15 Continuous 2D CNS Descriptors + 2048-bit Morgan FP
DESC_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds,
    Descriptors.RingCount, Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    Descriptors.MolMR, Descriptors.NHOHCount, Descriptors.NOCount,
    Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings, Descriptors.LabuteASA
]
N_DESC = len(DESC_FUNCS)

def featurize_molecules(mols):
    X_list, fp_objs = [], []
    for mol in mols:
        desc_vals = [fn(mol) for fn in DESC_FUNCS]
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_objs.append(fp)
        fp_arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)
        X_list.append(np.concatenate([desc_vals, fp_arr]))
    return np.array(X_list, dtype=np.float32), fp_objs

print("Featurizing molecules (RDKit 2D + Morgan ECFP4)...")
X_all, fps_all = featurize_molecules(clean_df["mol"])
y_all = clean_df["label"].values

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
fps_train = [fps_all[i] for i in train_idx]
fps_test = [fps_all[i] for i in test_idx]

# 4. Applicability Domain (5-NN Tanimoto Similarity Threshold)
k = 5
train_knn_sims = []
for i, fp in enumerate(fps_train):
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
    sims[i] = -1.0  # exclude self
    top_k = np.partition(sims, -k)[-k:]
    train_knn_sims.append(np.mean(top_k))

ad_threshold = np.mean(train_knn_sims) - 1.5 * np.std(train_knn_sims)

test_in_ad = []
for fp in fps_test:
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
    top_k = np.partition(sims, -k)[-k:]
    test_in_ad.append(np.mean(top_k) >= ad_threshold)

ad_coverage = np.mean(test_in_ad) * 100.0

# 5. ColumnTransformer (Scale ONLY continuous descriptors, passthrough binary FPs)
desc_Indices = list(range(N_DESC))
fp_Indices = list(range(N_DESC, N_DESC + 2048))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), desc_Indices),
        ("fp", "passthrough", fp_Indices),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss", n_jobs=-1, random_state=42),
}

# 6. Train, Calibrate & Print Markdown Table
os.makedirs("models", exist_ok=True)
print("\n### Copy-Paste Results Table for README.md:\n")
print("| Model | Internal ROC-AUC | Internal MCC | AD Coverage (%) | Brier Score |")
print("|---|---|---|---|---|")

for name, clf in models.items():
    base_pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    cal_model = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)
    cal_model.fit(X_train, y_train)
    
    y_prob = cal_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    print(f"| **{name}** | {auc:.3f} | {mcc:.3f} | {ad_coverage:.1f}% | {brier:.3f} |")
    
    slug = name.lower().replace(" ", "_")
    joblib.dump(cal_model, f"models/{slug}_calibrated.pkl")