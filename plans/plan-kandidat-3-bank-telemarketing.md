# Execution Plan: Kandidat 3 - Bank Telemarketing Predictive Lead Scoring (Decision Tree vs Logistic Regression)

## 1. Proposal Review and Strategic Alignment

### 1.1 Source Proposal Summary
This execution plan operationalizes the architectural and methodological specification defined in [kandidat-3-bank-telemarketing-classification.md](../project/kandidat-3-bank-telemarketing-classification.md) for **Kandidat 3: Perbandingan Algoritma Decision Tree dan Logistic Regression untuk Prediksi Keberhasilan Kampanye Telemarketing Bank**.

The proposal addresses three intertwined retail banking challenges in outbound term-deposit telemarketing:
1. **Severe Class Imbalance & Low Conversion Rate:** On the UCI Bank Marketing benchmark, positive subscriptions (`y = yes`) represent only **11.70%** (5,289 / 45,211 rows) in `bank-full.csv` and **11.27%** (4,640 / 41,188 rows) in `bank-additional-full.csv`. Blanket calling wastes nearly 9 out of 10 call attempts.
2. **Operational Call Capacity Constraints & Cost Asymmetry:** Outbound agent time carries direct operational cost ($C_{\text{call}}$), while missed conversions incur opportunity cost ($V_{\text{profit}}$). Call centers operate under fixed call quotas and need ranked lead prioritization rather than naive `0.50` classification cutoffs.
3. **Customer Fatigue & Brand Erosion:** Repeated irrelevant calls (`campaign` frequency) reduce customer satisfaction and long-term retention.

### 1.2 Literature Benchmark Alignment
The implementation directly synthesizes the four verified studies reviewed in the proposal:
- **Moro, Cortez, & Rita (2014), *Decision Support Systems*:** Establishes strict **pre-call prediction** by excluding post-call attributes (`duration`), evaluating models using **ROC-AUC** and **ALIFT** (*Area of the Cumulative Lift Curve*), and demonstrating business lift (reaching ~79% of subscribers within the top 50% ranked prospects).
- **Ananthakrishnan, Sreya, & Anand (2026), *IJRPR*:** Empirically quantifies the `duration` leakage illusion (`ROC-AUC > 0.92` with `duration` vs realistic `0.75-0.81` without `duration`), contrasts Decision Tree vs Logistic Regression F1-score behavior, and highlights why raw **Accuracy** (~88.3% for an always-no classifier) is misleading.
- **Dong et al. (2026), *Entropy (MDPI)*:** Reinforces risk-sensitive / cost-sensitive classification, **SMOTE** within **5-Fold Stratified Cross-Validation**, and local/global explainability on `bank-additional-full.csv`.
- **Sutarman et al. (2024), *JTIIK*:** Demonstrates that Logistic Regression performance under imbalance and high one-hot dimensionality improves substantially when paired with proper **SMOTE** resampling (training split only) and **$L_2$ (Ridge) regularization**.

### 1.3 Strengths & Gaps Identified During Proposal Review
| Review Aspect | Assessment in Proposal | Execution Plan Enhancement |
| :--- | :--- | :--- |
| **Dual Dataset Schema Support** | Focuses on `bank-full.csv` (17 attrs) while noting `bank-additional-full.csv` (21 attrs) compatibility | Implement an explicit **Dataset Variant Switcher** in both data loader and UI sidebar so both `bank-full` and `bank-additional-full` train, score, and adapt the Macroeconomic panel dynamically. |
| **Sentinel Handling for `pdays`** | Identifies `pdays = -1` in `bank-full.csv` | Extend sentinel handling to cover both **`pdays = -1`** (`bank-full.csv`) and **`pdays = 999`** (`bank-additional-full.csv`), engineering `never_contacted_before` and neutral non-negative numeric encoding. |
| **Probability Calibration Shift** | Notes upward probability shift under `SMOTE` and `Class Weighting` | Surface clear UI calibration guidance and support cost-benefit optimal threshold $\tau^* = C_{\text{call}} / V_{\text{profit}}$ alongside capacity-constrained decile cutoffs. |
| **Offline vs Online Consistency** | Proposes serialized pipelines + scoring engine | Encapsulate dual model-specific `ColumnTransformer` branches (`lr_preprocessor` with `RobustScaler` + `drop="first"`, `dt_preprocessor` with `passthrough` numerics) cached cleanly via `@st.cache_resource` / `joblib`. |
| **Offline Fallback Reliability** | Assumes UCI dataset availability | Provide automatic UCI ZIP download + `.parquet` caching in `data/` **plus** a deterministic synthetic benchmark generator (`generate_bank_sample_data`) matching UCI schema, ~11.7% conversion rate, sentinels, `"unknown"` tokens, and realistic `duration` column for leakage guard demonstration. |

---

## 2. Technical Risk Mitigations

### Risk 1: Post-Call Data Leakage via `duration`
- **Risk:** Including `duration` (call length in seconds) inflates `ROC-AUC` above `0.90` because zero-second or short calls almost deterministically map to `y = no`. In pre-call operational scoring, future call duration does not exist.
- **Mitigation:**
  - `audit_and_remove_leakage(df, variant)` explicitly detects and drops `duration` immediately during data preparation, returning an audit record (`dropped_duration: bool`, `dropped_columns: list[str]`).
  - Batch CSV upload in Modul 2 runs the exact same guard and displays an **Anti-Leakage Guard Banner** confirming `duration` was stripped if present in uploaded historical CSVs.
  - Single Lead Scoring (Modul 1) never exposes a `duration` input field.

### Risk 2: Sentinel Value Distortion (`pdays = -1` and `pdays = 999`)
- **Risk:** `pdays` uses `-1` (`bank-full.csv`) or `999` (`bank-additional-full.csv`) to encode "never contacted in a prior campaign." Feeding raw `-1` or `999` into **Logistic Regression** creates severe linear distortion relative to valid contact intervals (1 to ~871 days).
- **Mitigation:**
  - Engineer binary indicator `never_contacted_before` (`1` if `pdays in {-1, 999}`, else `0`).
  - Replace sentinel `-1` or `999` in `pdays` with neutral `0` (or median valid non-sentinel / zeroed recency paired with the explicit `never_contacted_before` flag) so `RobustScaler` and linear weights separate prior-contact recency cleanly from never-contacted status without outlier distortion.

### Risk 3: Preprocessing, Scaling, and `SMOTE` Leakage Across Splits
- **Risk:** Applying `SMOTE`, `OneHotEncoder`, or `RobustScaler` before `Stratified Train-Test Split` or outside `Stratified K-Fold Cross-Validation` leaks holdout/validation information into training estimates and creates synthetic samples interpolated from test rows.
- **Mitigation:**
  - Execute `Stratified Train-Test Split` (80% train / 20% holdout test, `stratify=y`) **first** before fitting any encoder, scaler, or resampler.
  - Wrap `ColumnTransformer -> SMOTE -> Classifier` inside `imblearn.pipeline.Pipeline` during both `5-Fold Stratified Cross-Validation` and final model training on the 80% train split. The 20% holdout test set remains untouched and unresampled.

### Risk 4: Algorithm-Specific Preprocessing Divergence & Dummy Variable Trap
- **Risk:** Logistic Regression requires `Feature Scaling` (due to skewed `balance` ranging from `-8,019` to `102,127` Euro vs `campaign` in single digits) and reference-category dropping (`drop="first"`) to prevent multicollinearity (`dummy variable trap`), whereas Decision Tree (`CART`) benefits from unscaled units so node thresholds remain human-readable (e.g., `balance <= 1250.0`) and does not suffer from dummy collinearity.
- **Mitigation:** `build_preprocessor_pipelines(X_df)` constructs two tailored `ColumnTransformer` pipelines:
  - `lr_preprocessor`: `OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)` for categoricals + `RobustScaler()` for numerics (robust to extreme `balance` outliers).
  - `dt_preprocessor`: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categoricals + `"passthrough"` for numerics so decision rules display original Euro/day/age units.

### Risk 5: Informative `"unknown"` Categories
- **Risk:** Dropping rows containing `"unknown"` in `poutcome` (~81.7%) or `contact` (~28.8%) would discard the majority of the dataset and destroy population representativeness.
- **Mitigation:** Explicitly preserve `"unknown"` as a valid categorical level in both imputation/cleaning and `OneHotEncoder`.

### Risk 6: Probability Calibration Shift Under `SMOTE` and `Class Weighting`
- **Risk:** Setting `class_weight="balanced"` or resampling via `SMOTE` balances training class priors from ~11.7% toward ~50%, shifting raw predicted probabilities upward. Treating shifted probabilities blindly at `0.50` without understanding calibration skews expected conversion volume estimates.
- **Mitigation:**
  - Provide `None (Baseline)`, `Class Weighting (Balanced)`, and `SMOTE (Training Fold Only)` options in the Sidebar so analysts can observe probability shift directly.
  - Display **ROC-AUC**, **PR-AUC (Average Precision)**, **Cumulative Gain**, **Lift Curve**, and **ALIFT** (which are rank-order invariant under monotonic score shifts), alongside the interactive **Probability Threshold** tuning playground (`0.05` to `0.95`) and theoretical cost-benefit threshold $\tau^* = C_{\text{call}} / V_{\text{profit}}$.
  - Document calibration behavior clearly in Modul 3 and Modul 1 tooltips/alerts.

### Risk 7: Scikit-Learn 1.9 API Compatibility & Zero-Emoji UI Rule
- **Risk:** Newer scikit-learn releases (`1.8/1.9`) deprecate or remove `penalty` on `LogisticRegression` in favor of `l1_ratio` (`0.0` for L2, `1.0` for L1) and require `sparse_output=False` instead of `sparse=False` on `OneHotEncoder`. Additionally, emojis must never appear in Python UI or chart text.
- **Mitigation:** Inspect installed scikit-learn signature explicitly during implementation, use version-safe parameter mapping for regularization (`L2` Ridge / `L1` Lasso), enforce `sparse_output=False`, preserve exact standard English technical terms, and enforce automated regex verification for zero emojis across all `.py` files.

---

## 3. File Architecture

```text
d:\05-Dev\Colab\Data Mining-Project\
├── plans\
│   └── plan-kandidat-3-bank-telemarketing.md      # This execution plan
└── app-kandidat-3-telemarketing\
    ├── requirements.txt                           # Python 3.13 pinned/compatible dependencies
    ├── README.md                                  # Setup, run instructions, methodology summary
    ├── app.py                                     # 3-Module Streamlit + Plotly interactive dashboard
    ├── data\                                      # Auto-cached .parquet files & downloaded UCI zips (gitignored/runtime)
    ├── src\
    │   ├── __init__.py                            # Package exports
    │   ├── data_loader.py                         # UCI download + Parquet cache + realistic sample generator
    │   ├── preprocessing.py                       # Anti-leakage guard, sentinel engineering, dual ColumnTransformers
    │   └── modeling_engine.py                     # Stratified split, 5-fold CV, DT vs LR training, metrics, explainability
    └── tests\
        └── test_kandidat_3.py                     # Unit, leakage-guard, modeling, explainability & AppTest smoke tests
```

### Module Contracts
1. **`src/data_loader.py`**
   - `load_or_download_bank_data(data_dir, variant="bank-full", use_sample=False) -> pd.DataFrame`
   - `generate_bank_sample_data(variant="bank-full", n_rows=6000, random_state=42) -> pd.DataFrame`
   - Supports both `"bank-full"` (45,211 rows, 17 columns) and `"bank-additional-full"` (41,188 rows, 21 columns). Downloads official UCI Bank Marketing archive, caches parsed DataFrame to `data/<variant>.parquet`, and falls back gracefully on network/timeout error to the built-in realistic sample generator.
2. **`src/preprocessing.py`**
   - `audit_and_remove_leakage(df, variant="bank-full") -> tuple[pd.DataFrame, dict]`
   - `build_preprocessor_pipelines(X_df) -> tuple[ColumnTransformer, ColumnTransformer]`
   - Handles `duration` stripping, `pdays` sentinel (`-1` / `999`) -> `never_contacted_before` + neutral non-negative `pdays`, `"unknown"` preservation, `y` binary mapping (`yes -> 1`, `no -> 0`), and clean LR vs DT `ColumnTransformer` construction.
3. **`src/modeling_engine.py`**
   - `train_and_evaluate_models(df, variant="bank-full", imbalance_strategy="class_weight", test_size=0.20, dt_max_depth=6, dt_criterion="gini", lr_c=1.0, lr_penalty="l2", random_state=42) -> dict`
   - `compute_threshold_metrics(y_true, y_prob, threshold=0.50, call_cost=2.5, conversion_profit=80.0) -> dict`
   - `score_single_lead(model_bundle, lead_dict, threshold) -> dict`
   - `score_batch_leads(model_bundle, batch_df, threshold, model_choice="Logistic Regression") -> tuple[pd.DataFrame, dict]` (batch ranking + deciles + leakage guard)
   - `compute_decile_table_and_curves(y_true, y_prob)` (ROC, Precision-Recall, Cumulative Gain, Lift Curve by decile, ALIFT).
4. **`app.py`**
   - Uses `@st.cache_data` for data loading and `@st.cache_resource` / `@st.cache_data` for model training bundle keyed by dataset source, variant, imbalance strategy, and hyperparameters.
   - Implements Sidebar Control Center + 3 operational Tabs (Modul 1, Modul 2, Modul 3) with Plotly visualizations and strict adherence to English technical terminology and zero emojis.

---

## 4. CRISP-DM to KDD Mapping

| CRISP-DM Phase | KDD Stage | Implementation in `app-kandidat-3-telemarketing` |
| :--- | :--- | :--- |
| **1. Business Understanding** | Domain Goal Setting & Pre-KDD | Frame 11.7% conversion baseline, cost asymmetry ($C_{\text{call}}$ vs $V_{\text{profit}}$), customer fatigue prevention, and pre-call lead ranking priorities (`Recall`, `F1-Score`, `PR-AUC`, `Lift Curve`, `ALIFT`). |
| **2. Data Understanding** | **Data Selection** & EDA | Load `bank-full.csv` (17 attrs) or `bank-additional-full.csv` (21 attrs), audit class imbalance, inspect `"unknown"` frequencies, identify `pdays` sentinels (`-1` / `999`), and flag `duration` for mandatory removal. |
| **3. Data Preparation** | **Preprocessing** & **Transformation** | `audit_and_remove_leakage` drops `duration`, creates `never_contacted_before`, normalizes sentinel `pdays`, preserves `"unknown"`, splits via `Stratified Train-Test Split` (80/20), and applies dual `ColumnTransformer` pipelines (`OneHotEncoder` + `RobustScaler` for LR; `OneHotEncoder` + `passthrough` for DT) + train-only `SMOTE` or `Class Weighting`. |
| **4. Modeling** | **Data Mining** | Train **Decision Tree** (CART with `Gini Impurity`/`Entropy` and `max_depth` pruning) and **Logistic Regression** (`Sigmoid`, `Log-Loss`, $L_1/L_2$ regularization via `C`) inside **5-Fold Stratified K-Fold Cross-Validation** (`ROC-AUC`, `F1-Score`). |
| **5. Evaluation** | **Interpretation / Evaluation** | Evaluate holdout test set with `Confusion Matrix`, `Precision`, `Recall`, `Specificity`, `F1-Score`, $F_2$-Score, `ROC-AUC`, `PR-AUC` (`Average Precision`), `Cumulative Gain`, `Lift Curve`, and `ALIFT`; inspect DT feature importance + root-to-leaf rules vs LR `Odds Ratio` ($e^{\beta_j}$) + local log-odds contributions. |
| **6. Deployment** | Operational Knowledge Utilization | Expose pre-call inference engine in the 3-module Streamlit + Plotly dashboard for Single Lead Scoring (Modul 1), Batch Lead Prioritization & Call Capacity Planning with CSV export (Modul 2), and Interactive Threshold & Model Benchmark Tuning (Modul 3). |

---

## 5. Three-Week Execution Roadmap

| Week | Milestone | Key Deliverables & Activities |
| :--- | :--- | :--- |
| **Week 1: Data Foundation & Anti-Leakage Pipeline** | Data Acquisition, EDA, & Leakage-Free Preprocessing | Implement `src/data_loader.py` (UCI fetcher, `.parquet` cache, realistic sample generator for both `bank-full` and `bank-additional-full`) and `src/preprocessing.py` (anti-leakage `duration` removal, `pdays` sentinel engineering, dual `ColumnTransformer` pipelines). Validate zero data leakage on splits. |
| **Week 2: Modeling, Cross-Validation & Explainability** | Dual Model Training, Imbalance Handling & Scoring Engine | Implement `src/modeling_engine.py`: 80/20 `Stratified Train-Test Split`, 5-fold `Stratified K-Fold Cross-Validation`, train-only `SMOTE` & `Class Weighting`, Decision Tree (`CART`) vs Logistic Regression training, decile Lift/Gain/ALIFT computation, `Odds Ratio` & local log-odds breakdown, and DT root-to-leaf rule tracer. |
| **Week 3: Dashboard UI, Testing & Release Readiness** | 3-Module Interactive Streamlit Dashboard & QA | Build `app.py` (Sidebar Control Center, Modul 1 Single Lead Simulator, Modul 2 Batch Planner & CSV Exporter, Modul 3 Model Benchmark & Threshold Playground), write `README.md` & `requirements.txt`, and run automated unit + integration + `AppTest` + zero-emoji verification suite in `tests/test_kandidat_3.py`. |

---

## 6. Verification Strategy

1. **Automated Unit & Integration Tests (`tests/test_kandidat_3.py` via Python 3.13 `pytest`):**
   - **Data Loader Tests:** Validate schema, row counts, positive rate (~11.7%), sentinel presence (`-1` for `bank-full`, `999` for `bank-additional-full`), `"unknown"` values, and presence of `duration` in raw output for both dataset variants.
   - **Anti-Leakage & Preprocessing Tests:** Confirm `audit_and_remove_leakage` drops `duration`, logs removal in audit metadata, creates `never_contacted_before`, removes sentinel `-1`/`999` distortion from `pdays`, preserves `"unknown"`, and produces leak-free `lr_preprocessor` (`RobustScaler` + `drop="first"`) vs `dt_preprocessor` (`passthrough` numerics).
   - **Modeling & Imbalance Isolation Tests:** Confirm `train_and_evaluate_models` works for both `"bank-full"` and `"bank-additional-full"`, across `"class_weight"`, `"smote"`, and `"none"` strategies, computes finite `ROC-AUC`, `PR-AUC`, `ALIFT`, `CV ROC-AUC`, `Odds Ratio`, and DT feature importances, and keeps the holdout test set strictly stratified and un-resampled.
   - **Threshold & Single Lead Scoring Tests:** Verify Confusion Matrix counts (`tp + fp + tn + fn == n_test`), `precision`, `recall`, `specificity`, `f1`, `expected_net_profit`, optimal $\tau^* = C_{\text{call}} / V_{\text{profit}}$, Priority Badge thresholds, LR local log-odds top drivers, and non-empty DT root-to-leaf rule traces.
   - **Streamlit Smoke Test (`AppTest`):** Execute `streamlit.testing.v1.AppTest.from_file("app.py").run()` to verify the full 3-tab dashboard renders without uncaught exceptions.
2. **Compliance & Hygiene Audits:**
   - **Zero-Emoji Regex Check:** Scan every `.py` file under `app-kandidat-3-telemarketing/` using `[\U0001F300-\U0001FAFF\u2600-\u27BF]` and verify zero matches.
   - **Terminology Integrity Check:** Confirm all standard English ML/analytics terms remain intact across labels, headers, and docs.
