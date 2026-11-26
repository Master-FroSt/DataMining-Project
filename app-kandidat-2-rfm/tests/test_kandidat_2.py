"""Comprehensive unit, integration, edge-case, and UI smoke tests for Kandidat 2."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

APP_ROOT = Path(__file__).resolve().parents[1]


def _isolate_app_sys_path() -> None:
    """Keep APP_ROOT first on sys.path and remove sibling app-kandidat-* roots."""
    app_root_str = str(APP_ROOT)
    cleaned: list[str] = [app_root_str]
    for entry in sys.path:
        if entry == app_root_str:
            continue
        if "app-kandidat-" in entry and Path(entry).resolve() != APP_ROOT.resolve():
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def _clear_src_modules() -> None:
    """Remove any cached src / src.* modules from sys.modules."""
    for mod_name in list(sys.modules):
        if mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]


_isolate_app_sys_path()
_clear_src_modules()

from src.data_loader import generate_rfm_retail_sample_data, load_or_download_retail_data
from src.kmeans_engine import evaluate_k_range, fit_kmeans_and_profile
from src.rfm_pipeline import (
    ADMIN_STOCK_CODES,
    clean_transactions_for_rfm,
    compute_rfm_table,
    transform_and_scale_rfm,
)

_OWN_SRC_MODULES = {
    k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")
}


@pytest.fixture(autouse=True)
def _restore_kandidat2_src_isolation():
    """Ensure sys.path and sys.modules['src*'] point to Kandidat 2 during combined runs."""
    _isolate_app_sys_path()
    _clear_src_modules()
    sys.modules.update(_OWN_SRC_MODULES)
    yield


EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def test_zero_emojis_in_all_python_files() -> None:
    """Ensure strict compliance: zero emojis in any .py file under app-kandidat-2-rfm."""
    py_files = list(APP_ROOT.rglob("*.py"))
    assert len(py_files) >= 5
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        matches = EMOJI_PATTERN.findall(content)
        assert not matches, f"Emoji character(s) {matches} found in {py_file}"


def test_sample_data_schema_and_anomalies() -> None:
    """Sample generator should span Dec 2010 - Dec 2011 and contain realistic UCI anomalies."""
    df = generate_rfm_retail_sample_data(n_customers=200, random_state=42)
    expected_cols = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["InvoiceDate"].min() >= pd.Timestamp("2010-12-01")
    assert df["InvoiceDate"].max() <= pd.Timestamp("2011-12-10")
    assert df["CustomerID"].isna().sum() > 0
    assert (df["InvoiceNo"].astype(str).str.startswith("C")).sum() > 0
    assert (df["Quantity"] <= 0).sum() > 0
    assert (df["UnitPrice"] <= 0).sum() > 0
    assert df["StockCode"].isin(["POST", "M"]).sum() > 0


def test_load_or_download_sample_and_parquet_cache_and_fallback(tmp_path: Path) -> None:
    """Verify sample shortcut, Parquet caching round-trip, and graceful offline fallback."""
    sample_df = load_or_download_retail_data(tmp_path, use_sample=True)
    assert not sample_df.empty
    assert sample_df.attrs.get("data_source") == "built_in_sample"

    # Pre-seed parquet cache and verify it loads directly without network
    parquet_path = tmp_path / "online_retail.parquet"
    sample_df.head(100).to_parquet(parquet_path, index=False)
    cached_df = load_or_download_retail_data(tmp_path, use_sample=False)
    assert len(cached_df) == 100
    assert cached_df.attrs.get("data_source") == "parquet_cache"

    # Verify graceful fallback when no cache exists and network download raises
    empty_dir = tmp_path / "empty_cache"
    with patch("requests.get", side_effect=RuntimeError("Offline simulated")):
        fb_df = load_or_download_retail_data(empty_dir, use_sample=False)
    assert not fb_df.empty
    assert fb_df.attrs.get("data_source") == "fallback_sample"


def test_clean_transactions_audit_and_country_filter() -> None:
    """Verify cleaning removes cancellations, nulls, non-positive values, admin codes, duplicates."""
    raw = pd.DataFrame(
        {
            "InvoiceNo": ["500001", "500001", "C500002", "500003", "500004", "500005", "500006", "500007"],
            "StockCode": ["85123A", "85123A", "85123A", "85123A", "85123A", "POST", "M", "22423"],
            "Description": ["ITEM A", "ITEM A", "ITEM A", "ITEM A", "ITEM A", "POSTAGE", "Manual", "ITEM B"],
            "Quantity": [2, 2, -2, 0, 3, 1, 1, 5],
            "InvoiceDate": ["2011-12-01 10:00"] * 7 + ["2011-12-05 12:00"],
            "UnitPrice": [10.0, 10.0, 10.0, 10.0, 0.0, 15.0, 20.0, 5.0],
            "CustomerID": [10001.0, 10001.0, 10001.0, 10001.0, 10001.0, 10001.0, np.nan, 10002.0],
            "Country": ["United Kingdom"] * 7 + ["Germany"],
        }
    )

    clean_all, audit_all = clean_transactions_for_rfm(raw, country_filter="All Countries")
    assert audit_all["raw_rows"] == 8
    assert audit_all["removed_missing_customer"] == 1
    assert audit_all["removed_cancellations"] == 1
    assert audit_all["removed_non_positive_qty"] == 1
    assert audit_all["removed_non_positive_price"] == 1
    assert audit_all["removed_admin_codes"] == 1
    assert audit_all["removed_duplicates"] == 1
    assert audit_all["clean_rows"] == 2
    assert set(clean_all["CustomerID"]) == {"10001", "10002"}
    assert not clean_all["StockCode"].isin(ADMIN_STOCK_CODES).any()

    clean_de, audit_de = clean_transactions_for_rfm(raw, country_filter="Germany")
    assert audit_de["clean_rows"] == 1
    assert audit_de["removed_country_filter"] == 1
    assert clean_de["CustomerID"].tolist() == ["10002"]


def test_compute_rfm_table_math() -> None:
    """Verify T_ref = max + 1 day, Recency >= 1, unique invoice Frequency, exact Monetary."""
    clean_df = pd.DataFrame(
        {
            "InvoiceNo": ["1001", "1001", "1002", "1003"],
            "StockCode": ["A", "B", "A", "C"],
            "Description": ["D1", "D2", "D1", "D3"],
            "Quantity": [2, 3, 1, 4],
            "InvoiceDate": pd.to_datetime(
                ["2011-12-01 10:00", "2011-12-01 10:00", "2011-12-08 09:00", "2011-12-09 12:00"]
            ),
            "UnitPrice": [10.0, 5.0, 20.0, 25.0],
            "CustomerID": ["C1", "C1", "C1", "C2"],
            "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France"],
            "TotalPrice": [20.0, 15.0, 20.0, 100.0],
        }
    )

    rfm, t_ref = compute_rfm_table(clean_df)
    assert t_ref == pd.Timestamp("2011-12-10 12:00:00")
    c1 = rfm.loc[rfm["CustomerID"] == "C1"].iloc[0]
    c2 = rfm.loc[rfm["CustomerID"] == "C2"].iloc[0]

    assert c1["Recency"] == 2
    assert c1["Frequency"] == 2
    assert pytest.approx(c1["Monetary"]) == 55.0
    assert c1["Country"] == "United Kingdom"

    assert c2["Recency"] == 1
    assert c2["Frequency"] == 1
    assert pytest.approx(c2["Monetary"]) == 100.0
    assert c2["Country"] == "France"


@pytest.mark.parametrize("outlier_method", ["none", "iqr_1.5", "winsorize_3iqr", "winsorize_p99"])
@pytest.mark.parametrize("skew_method", ["none", "log1p", "boxcox"])
@pytest.mark.parametrize("scaler_method", ["standard", "minmax"])
def test_transform_and_scale_all_combinations(
    outlier_method: str,
    skew_method: str,
    scaler_method: str,
) -> None:
    """Every preprocessing combination maintains 1-to-1 row alignment and finite outputs."""
    raw = generate_rfm_retail_sample_data(n_customers=150, random_state=7)
    clean_df, _ = clean_transactions_for_rfm(raw)
    rfm_df, _ = compute_rfm_table(clean_df)

    X_scaled, rfm_filtered, diag = transform_and_scale_rfm(
        rfm_df,
        outlier_method=outlier_method,
        skew_method=skew_method,
        scaler_method=scaler_method,
    )
    assert X_scaled.shape == (len(rfm_filtered), 3)
    assert np.isfinite(X_scaled).all()
    assert rfm_filtered.index.tolist() == list(range(len(rfm_filtered)))
    if outlier_method != "iqr_1.5":
        assert len(rfm_filtered) == len(rfm_df)
        assert diag["rows_dropped"] == 0
    if scaler_method == "minmax":
        assert X_scaled.min() >= -1e-9
        assert X_scaled.max() <= 1.0 + 1e-9


def test_evaluate_k_range_and_fit_kmeans_across_k_values() -> None:
    """Verify evaluate_k_range metrics, consensus k, true centroids, and deterministic personas for k=2..10."""
    raw = generate_rfm_retail_sample_data(n_customers=300, random_state=42)
    clean_df, _ = clean_transactions_for_rfm(raw)
    rfm_df, _ = compute_rfm_table(clean_df)
    X_scaled, rfm_filtered, _ = transform_and_scale_rfm(rfm_df)

    eval_df = evaluate_k_range(X_scaled, k_min=2, k_max=10, random_state=42)
    expected_cols = {
        "k",
        "WCSS",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Consensus Score",
        "Recommended",
    }
    assert expected_cols.issubset(set(eval_df.columns))
    assert len(eval_df) == 9
    assert eval_df["WCSS"].is_monotonic_decreasing
    assert int(eval_df["Recommended"].sum()) == 1
    assert 2 <= eval_df.attrs["recommended_k"] <= 10

    for k in range(2, 11):
        res = fit_kmeans_and_profile(rfm_filtered, X_scaled, k=k, random_state=42)
        profile = res["profile_df"]
        cust = res["customer_df"]
        assert len(profile) == k
        assert profile["Persona"].nunique() == k
        assert int(profile["Customer Count"].sum()) == len(rfm_filtered)
        assert pytest.approx(float(profile["Revenue Share (%)"].sum()), abs=0.2) == 100.0
        assert len(res["marketing_matrix_df"]) == k
        assert len(res["pca_centroids_df"]) == k

        # Verify true unscaled business centroids match direct groupby aggregation
        direct_means = cust.groupby("Cluster")["Monetary"].mean().to_dict()
        for _, p_row in profile.iterrows():
            cid = int(p_row["Cluster"])
            assert pytest.approx(float(p_row["Monetary Mean (GBP)"]), abs=0.05) == direct_means[cid]

        # Highest health score cluster must hold Champions persona; lowest holds Hibernating/Lost
        assert "Champions" in profile.iloc[0]["Persona"]
        assert (
            "Hibernating" in profile.iloc[-1]["Persona"]
            or "Lost" in profile.iloc[-1]["Persona"]
        )


def test_streamlit_app_smoke() -> None:
    """Verify Streamlit app boots, renders all 3 modules, and handles interactive widget updates."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit AppTest raised exceptions: {at.exception}"
    assert len(at.tabs) == 3
    assert len(at.dataframe) >= 2

    # Interactive test: switch preprocessing selectors, country filter, and cluster k slider
    at.sidebar.selectbox[0].set_value("United Kingdom").run()
    assert not at.exception
    at.sidebar.selectbox[1].set_value("Winsorizing (1st - 99th Percentile)").run()
    assert not at.exception
    at.sidebar.selectbox[2].set_value("Box-Cox Transformation").run()
    assert not at.exception
    at.sidebar.selectbox[3].set_value("MinMaxScaler (0-1)").run()
    assert not at.exception
    at.sidebar.slider[0].set_value(5).run()
    assert not at.exception
    assert len(at.tabs) == 3
