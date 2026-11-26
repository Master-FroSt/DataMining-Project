# Kandidat 2: Customer Segmentation Using RFM & K-Means Clustering

Interactive Streamlit + Plotly Data Mining dashboard implementing end-to-end **RFM (Recency, Frequency, Monetary)** feature engineering, distribution preprocessing (**Winsorizing**, **Log Transformation**, **Box-Cox**, **StandardScaler**, **MinMaxScaler**), **K-Means (`k-means++`)** multi-metric optimization (**Elbow Method / WCSS**, **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index**), deterministic **RFM Persona** profiling, and **CRM Customer Lookup** export.

## Architecture Overview

- `src/data_loader.py` — Downloads and caches the UCI Online Retail dataset as `.parquet` in `data/` (with graceful offline fallback) and generates a realistic 12-month multi-segment benchmark sample (`Dec 2010 - Dec 2011`) complete with UCI transaction anomalies (`'C'` cancellations, null `CustomerID`, negative quantities, and `POST`/`M` administrative codes).
- `src/rfm_pipeline.py` — Cleans raw transactions with full audit statistics, computes customer-level RFM tables (`T_ref = max(InvoiceDate) + 1 day`), and runs configurable outlier handling, skewness transformation, and feature scaling while maintaining strict index alignment.
- `src/kmeans_engine.py` — Evaluates candidate cluster counts across `k = 2..10`, detects a consensus optimal `k`, projects customers and centroids into 2D `PCA` space, aggregates true unscaled business centroids directly from raw RFM values, maps deterministic business personas via $S_c = z(F_c) + z(M_c) - z(R_c)$, and builds the CRM Marketing Action Matrix.
- `app.py` — 3-module Streamlit dashboard:
  - **Modul 1: RFM Distribution & Preprocessing Monitor**
  - **Modul 2: Cluster Optimization & Evaluation Lab**
  - **Modul 3: Segment Profiling & Marketing Action Center**
- `tests/test_kandidat_2.py` — Unit, integration, edge-case, and `streamlit.testing.v1.AppTest` smoke tests.

## Running with Python 3.13

Navigate to the workspace root or app directory and run with Python 3.13:

```powershell
cd "d:\05-Dev\Colab\Data Mining-Project\app-kandidat-2-rfm"
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

## Running Automated Tests

From `d:\05-Dev\Colab\Data Mining-Project`:

```powershell
& "C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m pytest app-kandidat-2-rfm/tests/test_kandidat_2.py -v
```
