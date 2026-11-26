# Market Basket Analysis with FP-Growth (UCI Online Retail)

Standalone interactive **Streamlit + Plotly** web application implementing **Market Basket Analysis (MBA)** using the **Frequent Pattern Growth (FP-Growth)** algorithm on the **UCI Online Retail** transactional dataset.

## Features

1. **Modul 1: Dataset Explorer & Transaction Overview**
   - 5 executive KPI cards (Raw vs Clean Rows, Unique Transactions, Commercial SKUs, Average & Median Basket Size, Multi-Item Basket Ratio).
   - Deterministic Data Cleaning Audit chart tracking cancelled `'C'` invoices, missing `Description`, non-positive `Quantity`/`UnitPrice`, administrative `StockCode` removals (`POST`, `DOT`, `M`, `CRUK`, `BANK CHARGES`, etc.), missing `CustomerID` policy, duplicates, and country filters.
   - Interactive Top-N Best-Selling Products horizontal bar chart and Basket Size Distribution histogram.
2. **Modul 2: Dynamic Rule Explorer & Parameter Tuning**
   - Real-time FP-Growth status cards (Frequent Itemsets count, Valid Association Rules count, Computation Time in seconds).
   - Interactive Plotly Scatter Plot (`Support` vs `Confidence`, colored by `Lift`, sized by `Leverage`).
   - Keyword search, multi-column sortable rules table (`Support`, `Confidence`, `Lift`, `Conviction`, `Leverage`), redundant rule pruning, and CSV export.
3. **Modul 3: Product Bundling & Recommendation Simulator**
   - Multi-select Anchor Product A basket simulator with Top-K Recommended Product B cards and affinity badges (`Strong Bundle Candidate`, `High-Confidence Cross-Sell`, `Niche Affinity`).
   - Cross-Selling Impact Calculator comparing Baseline Purchase Probability (`Support(B)`) vs Conditional Purchase Probability (`Confidence(A -> B)`), bundle discount slider, and Estimated Incremental Basket Value (GBP).
   - Interactive Plotly + NetworkX Directed Network Graph (Ego-Network and Top-N Macro Rule Network).

## Quickstart (Python 3.13)

Ensure you are inside `d:\05-Dev\Colab\Data Mining-Project\app-kandidat-1-mba` and run with the shared Python 3.13 interpreter:

```powershell
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

## Running Tests

From `d:\05-Dev\Colab\Data Mining-Project`:

```powershell
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m pytest app-kandidat-1-mba/tests/test_kandidat_1.py -v
```

## Dataset Sources

- **Built-in UCI Benchmark Sample (Fast Default):** Generates realistic UCI Online Retail schema transactions (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`) with authentic retail anomalies (`'C'` cancellations, ~24.9% null `CustomerID`, administrative codes) and co-occurrence bundles.
- **UCI Full Dataset (Download & Parquet Cache):** Downloads the official UCI Online Retail dataset (`541,909` rows) into `data/online_retail.parquet` for fast cached reloads, with automatic graceful fallback to the benchmark sample if offline.
- **Custom CSV / Excel Upload:** Supports external transactional files matching the standard UCI column schema.
