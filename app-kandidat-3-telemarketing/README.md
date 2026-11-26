# Bank Telemarketing Predictive Lead Scoring (Decision Tree vs Logistic Regression)

Interactive **Decision Support System (DSS)** for pre-call telemarketing lead prioritization comparing **Decision Tree (CART)** and **Logistic Regression** on the UCI Bank Marketing benchmark (`bank-full.csv` and `bank-additional-full.csv`).

## Key Methodological Safeguards

1. **Strict Anti-Leakage Guard (`duration` Exclusion):**
   Post-call call length (`duration`) is never available prior to dialing a prospect. Both the training pipeline and the batch CSV uploader automatically detect and remove `duration` so evaluation metrics (`ROC-AUC`, `PR-AUC`, `Lift Curve`, `ALIFT`) reflect genuine pre-call predictive performance.
2. **Sentinel Engineering (`pdays = -1` / `999`):**
   Creates the explicit binary indicator `never_contacted_before` and normalizes sentinel `pdays` values to prevent linear distortion in Logistic Regression while preserving `"unknown"` as an informative category.
3. **Algorithm-Specific Preprocessing Pipelines:**
   - **Logistic Regression Branch:** `OneHotEncoder(drop="first")` + `RobustScaler()` (mitigates extreme `balance` outliers and prevents dummy-variable collinearity).
   - **Decision Tree Branch:** `OneHotEncoder` + numeric `passthrough` so decision tree node rules retain original units (Euro, years, days).
4. **Train-Only Imbalance Handling:**
   Supports `Class Weighting (Balanced)`, `SMOTE (Training Fold Only)`, and `None (Baseline)` inside **5-Fold Stratified K-Fold Cross-Validation** and **Stratified Train-Test Split** (80/20), keeping the holdout test set strictly untouched.

## Quickstart (Python 3.13)

Run the Streamlit dashboard:

```powershell
cd "d:\05-Dev\Colab\Data Mining-Project\app-kandidat-3-telemarketing"
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

Run the automated test suite (`pytest`):

```powershell
cd "d:\05-Dev\Colab\Data Mining-Project"
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m pytest app-kandidat-3-telemarketing/tests/test_kandidat_3.py -v
```

## Dashboard Modules

- **Modul 1: Single Lead Scoring & Call Decision Simulator** — 3-group prospect profiler, side-by-side Decision Tree & Logistic Regression conversion probability gauges, Call Priority Badges, LR local log-odds driver breakdown, and DT root-to-leaf decision path trace.
- **Modul 2: Batch Lead Prioritization & Campaign Planner** — CSV batch upload or holdout batch simulation with Anti-Leakage Guard Banner, decile ranking (`Top 10%` to `Top 100%`), Call Capacity Quota slider, real-time Lift/Net Profit KPIs, Cumulative Gain chart, and CSV export.
- **Modul 3: Model Comparison & Evaluation Benchmark** — Side-by-side holdout benchmark table (`ROC-AUC`, `PR-AUC`, `F1-Score`, `ALIFT`), interactive `Probability Threshold` playground (`0.05` to `0.95`) updating dual Confusion Matrix heatmaps, combined ROC / Precision-Recall / Decile Lift curves, DT Feature Importance, and LR Odds Ratios ($e^{\beta_j}$).
