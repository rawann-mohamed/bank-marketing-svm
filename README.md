# Bank Marketing Prediction with SVM

Predicting whether a bank client will subscribe to a term deposit after a marketing phone call, using Support Vector Machines. Built as a Data Mining course project.

## Dataset
[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`): 41,188 records, 20 attributes, binary target `y` (about 11% "yes").
The CSV is not included in this repo. Download it from the link above and put `bank-additional-full.csv` in the same folder as the notebook.

## Pipeline
1. **Exploration:** data types, target distribution, hidden missing values (`"unknown"`, `pdays = 999`), duplicates.
2. **Cleaning:** removed 12 duplicate rows; replaced `pdays` with a `prev_contacted` flag; ordinal-encoded `education`.
3. **Train/test split (80/20, stratified)** before any learned transformation.
4. **Fitted on the training set only:** median imputation, IQR capping (`duration`, `campaign`), one-hot encoding, StandardScaler / RobustScaler. This avoids data leakage.
5. **Feature selection:** variance threshold, correlation, Chi-square, ANOVA F-test, Mutual Information. 10 of 55 features dropped with almost no change in F1.
6. **Modeling:** linear, RBF and polynomial SVMs tuned by hand on a validation set; `class_weight="balanced"` for the 89/11 imbalance.
7. **Final evaluation:** the test set was used once, at the end.

`duration` is excluded from the model: it is only known after the call, so it would leak the outcome. It is used only as a benchmark.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline: always "no" | 0.887 | 0.00 | 0.00 | 0.00 |
| Final SVM (RBF, C=1, balanced) | 0.850 | 0.40 | 0.64 | 0.49 |

Adding `duration` back (validation benchmark) raises recall to 0.91 and F1 to 0.58, but that model could not be used in practice.

## Limitations
- Precision is low: many clients predicted "yes" will say no.
- Only one validation split was used (no cross-validation).
- The data is from 2008-2010 and the split is random, so it does not test future campaigns.

## Run
```bash
pip install -r requirements.txt
jupyter notebook bank_marketing_svm.ipynb
```

**Tech:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
