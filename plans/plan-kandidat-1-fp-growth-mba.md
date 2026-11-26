# Execution Plan: Kandidat 1 - Market Basket Analysis with FP-Growth (UCI Online Retail)

## 1. Proposal Review and Critical Gap Analysis

### 1.1 Review of `project/kandidat-1-fp-growth-mba.md`
The research and architecture proposal in `project/kandidat-1-fp-growth-mba.md` establishes a solid foundation for implementing **Market Basket Analysis (MBA)** with **Frequent Pattern Growth (FP-Growth)** on the **UCI Online Retail** dataset (`541,909` raw rows, `8` attributes: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`, spanning 01/12/2010 to 09/12/2011).

Key strengths verified in the proposal:
1. **Accurate Literature Grounding:** Properly distinguishes Chen, Sain, & Guo (2012) as an RFM + K-Means + Decision Tree study that defines the UCI dataset anomalies, while grounding FP-Growth benchmarks in Karimah & Udayanti (2026), Fairuzindah et al. (2025), Abadi et al. (2026), Nursasongka et al. (2024), and Noviana et al. (2023).
2. **Domain Anomaly Awareness:** Explicitly accounts for `'C'` cancellation prefixes (~8,905 rows), missing `Description` (1,454 rows), missing `CustomerID` (135,080 rows / ~24.93%), non-positive `Quantity`/`UnitPrice`, and non-product administrative `StockCode` values (`POST`, `DOT`, `M`, `CRUK`, `BANK CHARGES`, `D`, `C2`, `S`, `AMAZONFEE`, `PADS`, `B`).
3. **Complete Metric Coverage:** Incorporates all five core statistical and business evaluation metrics: **Support**, **Confidence**, **Lift**, **Conviction**, and **Leverage**.
4. **Actionable 3-Module Dashboard Architecture:** Separates exploratory data auditing (Module 1), dynamic parameter tuning and rule inspection (Module 2), and practical business decision simulation via bundling uplift & directed network graphs (Module 3).

### 1.2 Implementation Gaps Identified & Engineering Decisions
To move from conceptual specification to a production-grade, responsive Streamlit application without runtime bottlenecks or logical ambiguities, the following concrete engineering decisions are enforced:
- **Gap 1 — Memory & Latency Risk on High-Dimensional One-Hot Encoding:** Converting ~18,500–22,000 invoices by ~3,800–4,000 SKUs into a dense boolean DataFrame consumes excessive memory and degrades interactive slider responsiveness. We resolve this using CSR sparse matrices (`scipy.sparse.csr_matrix` -> `pd.DataFrame.sparse.from_spmatrix` with `Sparse[bool, False]`) combined with pre-pruning rare items below a `0.003` frequency floor.
- **Gap 2 — Description Fragmentation per `StockCode`:** Real UCI Online Retail transactions contain trailing/leading spaces, irregular internal spacing, casing differences, and minor typos for the same `StockCode`. Without normalization, one physical SKU splits across multiple columns and undercounts support. We enforce whitespace collapsing, uppercase normalization, and deterministic mode-based `Description` canonicalization per `StockCode`.
- **Gap 3 — `CustomerID` Null Policy Consistency:** Rather than hardcoding a single stance on the ~24.93% guest-checkout / null `CustomerID` rows, the preprocessing pipeline exposes `drop_missing_customer_id` (default `True` to align with Abadi et al., 2026 ~392k–397k clean rows benchmark, toggleable to `False` for full invoice-level basket coverage).
- **Gap 4 — Support Denominator Preservation After Rare-Item Pre-Pruning:** Dropping items with frequency below `0.003` from columns must NOT delete invoices that contained only those rare items from the row index; otherwise total invoice count $N$ shrinks and inflates `Support` for remaining items. All valid clean `InvoiceNo` rows remain in the sparse matrix index.
- **Gap 5 — Network / Offline Resilience:** Full UCI dataset downloads (~22 MB Excel/ZIP) can fail or time out in constrained environments. `data_loader.py` implements local Parquet caching (`data/online_retail.parquet`) with automatic timeout handling and seamless fallback to a schema-faithful synthetic UCI benchmark generator (`generate_retail_sample_data`).
- **Gap 6 — Strict UI Standards:** Zero emojis in any `.py` source file, page configuration, KPI card, badge, button, notification, or Plotly chart label; standard English technical terms preserved throughout.

---

## 2. Technical Risk Mitigations

| Risk ID | Technical / Modeling Risk | Impact | Concrete Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **R1** | **Sparse Matrix Memory Explosion & Slider Lag** | High memory footprint and slow `fpgrowth` re-computation when tuning UI sliders. | Construct binary basket matrix via vectorized `scipy.sparse.csr_matrix` (`bool`) wrapped in `pd.DataFrame.sparse.from_spmatrix` (`Sparse[bool, False]`), pre-prune items with `support < 0.003` floor before FP-Tree build, and isolate preprocessing in `@st.cache_data` separate from mining. |
| **R2** | **Inconsistent `Description` Variants per `StockCode`** | Split item columns, understated `Support` / `Confidence`, duplicate graph nodes. | Normalize `Description` (`strip`, collapse multi-spaces, `upper`) and map every `StockCode` deterministically to its mode `Description` (ties broken alphabetically) during `clean_retail_transactions`. |
| **R3** | **Trivial Rules from Administrative Fees** | `POST`, `DOT`, `M`, `CRUK`, `BANK CHARGES` dominate consequents and distort bundling advice. | Apply deterministic exclusion against `ADMIN_STOCK_CODES = {"POST", "DOT", "M", "CRUK", "BANK CHARGES", "D", "C2", "S", "AMAZONFEE", "PADS", "B"}` before basket aggregation and log exact removed count in `cleaning_audit`. |
| **R4** | **Redundant & Uninformative Association Rules** | Hundreds of overlapping superset rules clutter Module 2 and Module 3 UI. | Enforce `min_lift >= 1.0` (default `1.2`), cap `max_len` (default `3`), and implement subsumption-based `filter_redundant` pruning: remove rule $X_2 \Rightarrow Y$ whenever a strict subset rule $X_1 \subsetneq X_2$ for the same consequent $Y$ achieves $\text{Confidence}(X_1 \Rightarrow Y) \ge \text{Confidence}(X_2 \Rightarrow Y)$. |
| **R5** | **Zero-Itemset / Zero-Rule Crashes** | Strict slider combinations (`min_support=0.05`, `min_confidence=0.95`) trigger empty DataFrame KeyErrors in `association_rules`. | Explicitly guard empty `basket_df`, 0 frequent itemsets, <2-length-only itemsets, and 0 rules inside `mine_fpgrowth_rules`, returning well-typed empty DataFrames plus actionable threshold-adjustment guidance in the UI. |
| **R6** | **Parquet Serialization Type Errors & Offline Failure** | Mixed alphanumeric `StockCode` / `InvoiceNo` columns fail `pyarrow` write; UCI server timeout blocks startup. | Coerce string columns explicitly prior to Parquet export; wrap UCI HTTP download in `timeout=15` with local Parquet caching and automatic fallback to `generate_retail_sample_data()`. |
| **R7** | **Confidence = 1.0 Infinite Conviction Rendering** | `np.inf` conviction causes JSON/chart scaling warnings or table formatting glitches. | Preserve numeric `np.inf` in `rules_df` for exactness while displaying `"inf"` cleanly in formatted tables and clamping finite chart hover displays safely. |

---

## 3. File Architecture & Module Responsibilities

```text
app-kandidat-1-mba/
|-- app.py                    # 3-Module Streamlit + Plotly Dashboard with @st.cache_data
|-- requirements.txt          # Minimal dependency manifest (Python 3.13 environment)
|-- README.md                 # Setup, Python 3.13 run commands, dataset & architecture guide
|-- data/                     # Runtime cache directory for online_retail.parquet (auto-created)
|-- src/
|   |-- __init__.py           # Package exports
|   |-- data_loader.py        # UCI download + Parquet cache + schema-faithful sample generator
|   |-- preprocessing.py      # Deterministic cleaning audit, Description canonicalization, sparse basket matrix
|   `-- fpgrowth_engine.py    # FP-Growth mining, 5-metric evaluation, redundancy pruning, bundling simulator
`-- tests/
    `-- test_kandidat_1.py    # Unit tests, mathematical metric verification, no-emoji check, AppTest smoke test
```

### Module Contracts
1. `src/data_loader.py`
   - `generate_retail_sample_data(n_invoices: int = 2500, random_state: int = 42) -> pd.DataFrame`: Produces exact 8-column UCI Online Retail schema including strong co-occurrence series (`REGENCY TEA PLATE PINK/GREEN/ROSES`, `JUMBO BAG RED RETROSPOT/PINK POLKADOT`, `WHITE/RED HANGING HEART T-LIGHT HOLDER`, `ALARM CLOCK BAKELIKE RED/GREEN`), `'C'` cancellations, negative quantities, ~24.9% null `CustomerID`, null `Description`, whitespace/case variants, duplicates, multi-country distribution, and admin codes.
   - `load_or_download_retail_data(data_dir: str | Path, use_sample: bool = False) -> pd.DataFrame`: Loads cached `.parquet`, downloads UCI Online Retail ZIP/XLSX into `data_dir` and caches as `.parquet`, or falls back gracefully to `generate_retail_sample_data()` on network/parse error.
   - `load_uploaded_retail_file(uploaded_file) -> pd.DataFrame`: Parses user-uploaded `.csv` or `.xlsx` files and validates required columns.
2. `src/preprocessing.py`
   - `ADMIN_STOCK_CODES = {"POST", "DOT", "M", "CRUK", "BANK CHARGES", "D", "C2", "S", "AMAZONFEE", "PADS", "B"}`
   - `clean_retail_transactions(df: pd.DataFrame, drop_missing_customer_id: bool = True, country_filter: str | None = "United Kingdom") -> tuple[pd.DataFrame, dict]`: Sequential auditable cleaning preserving conservation identity (`raw_rows - sum(removed) == final_clean_rows`), canonicalized uppercase `Description` per `StockCode`, and complete `cleaning_audit` metrics.
   - `build_sparse_basket_matrix(clean_df: pd.DataFrame, item_col: str = "Description", min_item_freq: float = 0.003) -> tuple[pd.DataFrame, pd.DataFrame]`: Vectorized CSR sparse boolean DataFrame (`Sparse[bool, False]`) pre-pruned at `min_item_freq` alongside `item_metadata_df` (`item`, `support`, `transaction_count`, `total_quantity`, `avg_unit_price`, `total_revenue`).
3. `src/fpgrowth_engine.py`
   - `mine_fpgrowth_rules(basket_df: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.50, min_lift: float = 1.2, max_len: int = 3, filter_redundant: bool = True) -> dict`: Executes `mlxtend.frequent_patterns.fpgrowth` + `association_rules`, returns timing, formatted readable rule strings, 5 core metrics (`support`, `confidence`, `lift`, `conviction`, `leverage`), and redundancy pruning statistics.
   - `simulate_bundling_recommendations(rules_df: pd.DataFrame, item_metadata_df: pd.DataFrame, anchor_items: list[str], discount_pct: float = 10.0, top_k: int = 10) -> pd.DataFrame`: Computes `conditional_prob`, `baseline_prob`, `lift`, `unit_price`, `discounted_unit_price`, `estimated_incremental_basket_value`, and non-emoji `affinity_badge` (`Strong Bundle Candidate`, `High-Confidence Cross-Sell`, `Niche Affinity`).

---

## 4. 3-Week Development Roadmap

| Timeline | Phase Focus | Key Deliverables & Engineering Activities | Exit Criteria |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Data Pipeline & FP-Growth Engine** | Build `data_loader.py` (UCI downloader + Parquet cache + sample generator); implement `preprocessing.py` with `ADMIN_STOCK_CODES`, canonicalization, and `cleaning_audit`; build sparse CSR matrix builder (`min_item_freq=0.003`); implement `fpgrowth_engine.py` with 5 metrics and subsumption pruning. | Unit tests verify audit conservation, sparse dtype, exact metric math, and sub-second mining on benchmark data. |
| **Week 2** | **3-Module Streamlit & Plotly Dashboard** | Implement Sidebar Control Center; build Module 1 (5 KPI cards, Cleaning Audit waterfall/bar, Top-N bar, Basket Size histogram); build Module 2 (execution KPIs, Support-Confidence-Lift-Leverage scatter plot, keyword search, CSV export); build Module 3 (anchor selector, Product B cards, uplift calculator, NetworkX directed graph); configure `@st.cache_data`. | Interactive UI runs smoothly with zero re-cleaning on slider changes and zero emojis across all components. |
| **Week 3** | **Optimization, Validation & Packaging** | Stress-test zero-rule edge cases, multi-item basket simulation, and offline fallback; validate rule outputs against UCI benchmarks (Abadi et al., 2026 & Fairuzindah et al., 2025); write `AppTest` integration suite, `README.md`, and final verification report. | Full `pytest` suite passes (unit + `AppTest` + static no-emoji check) without warnings or failures. |

---

## 5. CRISP-DM and KDD Methodology Mapping

| CRISP-DM Phase | KDD Stage | Implemented Module / Function | Artifacts & Outputs |
| :--- | :--- | :--- | :--- |
| **1. Business Understanding** | Pre-KDD Scoping | `README.md`, Sidebar KPI framing, Module 3 Bundling Simulator | Cross-selling problem definition, target uplift KPIs (`Confidence`, `Lift`, `Incremental Basket Value`). |
| **2. Data Understanding** | **1. Data Selection** | `data_loader.py`, Module 1 KPIs & Audit Panel | Raw 8-column UCI dataset ingestion, schema validation, profiling of `'C'` cancellations and ~24.9% null `CustomerID`. |
| **3. Data Preparation** | **2. Preprocessing & 3. Transformation** | `preprocessing.py` (`clean_retail_transactions`, `build_sparse_basket_matrix`) | Cleaned commercial transactions, `cleaning_audit` log, canonicalized `Description` mapping, `Sparse[bool, False]` basket matrix. |
| **4. Modeling** | **4. Data Mining** | `fpgrowth_engine.py` (`mine_fpgrowth_rules`) | FP-Tree mining (`mlxtend.fpgrowth`), frequent itemsets up to `max_len`, candidate association rule extraction. |
| **5. Evaluation** | **5. Interpretation / Evaluation** | `fpgrowth_engine.py` (5 metrics + redundancy pruning), Module 2 Scatter & Table | Validated rules scored by `Support`, `Confidence`, `Lift`, `Conviction`, `Leverage` with trivial/redundant rules filtered. |
| **6. Deployment** | Post-KDD Operationalization | `app.py` (Modules 1, 2, 3), CSV export, Bundling & Network Graph Simulator | Production-ready Streamlit interactive dashboard for category managers and data analysts. |

---

## 6. Verification Strategy

1. **Automated Unit Verification (`pytest`)**:
   - Validate schema, anomaly injection ratios, and fallback resilience in `data_loader.py`.
   - Assert exact conservation equation in `cleaning_audit` (`raw_rows - sum(removed) == final_clean_rows`), `StockCode -> Description` canonicalization, `ADMIN_STOCK_CODES` removal, and `CustomerID` policy toggle.
   - Verify `build_sparse_basket_matrix` produces `Sparse[bool, False]` columns, enforces `min_item_freq`, and preserves full clean `unique_invoices` row denominator.
   - Verify hand-calculated `Support`, `Confidence`, `Lift`, `Conviction`, `Leverage` values on deterministic synthetic transactions, plus strict redundancy pruning behavior and 0-rule graceful handling.
2. **Headless Streamlit UI Integration Test (`streamlit.testing.v1.AppTest`)**:
   - Boot `app.py` with Python 3.13, verify zero uncaught exceptions on initial load, and exercise interactive sidebar controls (`min_support`, `drop_missing_customer_id`, `filter_redundant`).
3. **Static Compliance & Visual Hygiene Audit**:
   - Programmatically scan every `.py` file under `app-kandidat-1-mba/` for Unicode emoji codepoints (assert zero matches).
   - Verify all domain terms (`Support`, `Confidence`, `Lift`, `Conviction`, `Leverage`, `Frequent Itemsets`, `FP-Tree`, `One-Hot Encoding`, `Data Cleaning`, `Basket Size`, `Cross-Selling`, `Product Bundling`, `Network Graph`) remain intact.
