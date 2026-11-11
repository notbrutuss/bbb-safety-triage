# BBB Permeability Prediction: A QSAR/ML Pipeline with Applicability Domain and Calibrated Probabilities

## Overview

A reproducible pipeline for predicting blood-brain barrier (BBB) permeability from molecular structure. Combines RDKit-based featurization, ensemble machine learning (Logistic Regression, Random Forest, XGBoost), probability calibration, and a Tanimoto applicability domain check.

**Status:** Baseline complete (internal scaffold split). External validation and Streamlit app in progress.

## Motivation

BBB permeability is a critical ADMET endpoint in CNS drug discovery. Most published QSAR models report high AUC but fail in practice because they lack applicability domain checks and produce uncalibrated probabilities. This project addresses both.

## Data

| Dataset | Role | Compounds (post-dedup) |
|---|---|---|
| B3DB | Training + internal scaffold test | 7805 |
| TITAN-BBB | External test (locked) | not yet integrated |
| QSAR-BBB | External test (locked) | not yet integrated |

- **Deduplication:** InChIKey-based, conflicting labels dropped.
- **Split:** Bemis-Murcko scaffold split, 80/20 train/internal-test. Train: 6244, Internal test: 1561.
- **External test:** frozen separately, not touched during training. Will be evaluated in Week 5.

## Pipeline

```
SMILES → RDKit standardization → 15 CNS-relevant descriptors + 2048-bit Morgan FP
→ ColumnTransformer (StandardScaler on descriptors only, FPs passthrough)
→ CalibratedClassifierCV(sigmoid, cv=3)
→ 5-NN Tanimoto Applicability Domain check

```


## Results — Internal Scaffold Split

| Model | ROC-AUC | MCC | Test AD Coverage (%)* | Brier Score |
|---|---|---|---|---|
| Logistic Regression | 0.798 | 0.326 | 20.6% | 0.217 |
| Random Forest | 0.819 | 0.384 | 20.6% | 0.214 |
| XGBoost | 0.822 | 0.386 | 20.6% | 0.215 |

*AD coverage computed on the scaffold-split test set using a 5-NN Tanimoto threshold (mean − 1.5·std of training-set similarities). Identical across models because AD is a property of the feature space, not the classifier. Low coverage (20.6%) reflects deliberate chemical novelty in the scaffold split; in-domain subset metrics are planned for Week 5.*

**Notes:**
- RF and XGB perform comparably, suggesting the descriptor + fingerprint features drive performance more than model class.
- MCC at 0.5 threshold is modest relative to AUC; optimal threshold tuning is planned.

## Repository Structure

```
├── data/
│ ├── raw/ # Original B3DB TSV
│ ├── processed/ # Deduplicated, standardized
│ └── external_test/ # Locked TITAN-BBB / QSAR-BBB (not yet populated)
├── models/ # Saved calibrated .pkl models
├── src/ # (refactor in progress — currently in run_b3db_baseline.py)
├── run_b3db_baseline.py # End-to-end baseline script
├── README.md
└── requirements.txt

```


## Setup

```
git clone https://github.com/notbrutuss/bbb-qsar.git
cd bbb-qsar
pip install -r requirements.txt
```
## Usage
```
python run_b3db_baseline.py
```

Produces:

Post-dedup compound count and split sizes

Internal scaffold-split metrics for LR, RF, XGB

Saved calibrated models in models/

## Roadmap
- [x] Week 1: Data curation, dedup, scaffold split
- [x] Week 2: Featurization, baseline models, StandardScaler pipeline
- [x] Week 3: Applicability domain, calibration
- [ ] Week 4–5: External validation on TITAN-BBB / QSAR-BBB, Streamlit app
- [ ] Week 6: Transporter-mechanism error analysis (P-gp, BCRP, LAT1)
- [ ] Week 7–8: Manuscript draft
- [ ] Week 9: ChemRxiv/bioRxiv preprint
## Limitations
- External test set not yet evaluated; only internal scaffold-split results reported.

- AD threshold is a heuristic 5-NN Tanimoto cutoff; conformal prediction planned.

- Single random seed for calibration folds; stability across seeds not yet tested.

- Transporter mechanism analysis (P-gp efflux, LAT1 influx) not yet implemented.

## Planned Extensions
The pipeline is endpoint-agnostic. A second endpoint (neurotoxicity or developmental toxicity) is planned as a post-baseline extension; dataset curation and label harmonization will determine which is feasible.

## References
- OECD QSAR Validation Principles (2004)

- B3DB: [citation to fill]

- TITAN-BBB: [citation to fill]

- QSAR-BBB: [citation to fill]

## Contact
- Ujjawal Awasthi  — MSc Toxicology, 
- National Forensic Sciences University , Gandhinagar .
- ujjawalawasthibio77@gmail.com | github.com/notbrutuss