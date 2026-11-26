"""Comprehensive unit, mathematical invariant, compliance, and AppTest suite for Kandidat 1."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from streamlit.testing.v1 import AppTest

APP_ROOT = Path(__file__).resolve().parents[1]


def _activate_own_app_path() -> None:
    """Put APP_ROOT first on sys.path and remove sibling app-kandidat-* roots."""
    cleaned_path = [
        p
        for p in sys.path
        if p != str(APP_ROOT)
        and not any(part.startswith("app-kandidat-") for part in Path(p).parts)
    ]
    sys.path[:] = [str(APP_ROOT), *cleaned_path]


_activate_own_app_path()
for _mod_name in list(sys.modules):
    if _mod_name == "src" or _mod_name.startswith("src."):
        del sys.modules[_mod_name]

from src.data_loader import (
    REQUIRED_COLUMNS,
    generate_retail_sample_data,
    load_or_download_retail_data,
)
from src.fpgrowth_engine import (
    mine_fpgrowth_rules,
    simulate_bundling_recommendations,
)
from src.preprocessing import (
    ADMIN_STOCK_CODES,
    build_sparse_basket_matrix,
    clean_retail_transactions,
)

_OWN_SRC_MODULES = {
    k: v
    for k, v in sys.modules.items()
    if k == "src" or k.startswith("src.")
}


@pytest.fixture(autouse=True)
def _isolate_own_src_modules() -> None:
    """Restore APP_ROOT sys.path and own src.* modules before each test."""
    _activate_own_app_path()
    for mod_name in list(sys.modules):
        if mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]
    sys.modules.update(_OWN_SRC_MODULES)

# Unicode ranges covering emojis, pictographs, dingbats, and variation selectors
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def test_no_emojis_in_any_python_file() -> None:
    """Strictly verify zero emojis exist in any .py file under app-kandidat-1-mba."""
    py_files = [
        p
        for p in APP_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert py_files, "Expected .py source files to exist."
    violations: list[str] = []
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        matches = EMOJI_PATTERN.findall(content)
        if matches:
            violations.append(f"{py_file}: {matches}")
    assert not violations, f"Emoji characters found in Python source files: {violations}"


def test_generate_retail_sample_data_schema_and_anomalies() -> None:
    """Verify generated sample matches UCI 8-column schema and includes realistic anomalies."""
    df = generate_retail_sample_data(n_invoices=600, random_state=42)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) > 600

    # Check cancellations ('C'), negative quantities, missing CustomerID, missing Description, admin codes
    assert df["InvoiceNo"].astype(str).str.startswith("C").any()
    assert (df["Quantity"] < 0).any()
    assert df["Description"].isna().any()

    null_cust_ratio = float(df["CustomerID"].isna().mean())
    assert 0.15 <= null_cust_ratio <= 0.35

    present_admin = set(df["StockCode"].dropna().astype(str).str.upper().unique()) & ADMIN_STOCK_CODES
    assert {"POST", "DOT", "M", "BANK CHARGES", "CRUK"} & present_admin
    assert "United Kingdom" in df["Country"].unique()
    assert df["Country"].nunique() >= 3


def test_load_or_download_retail_data_cache_and_fallback(tmp_path: Path) -> None:
    """Verify sample load, Parquet cache load, and graceful network failure fallback."""
    sample_df = load_or_download_retail_data(tmp_path, use_sample=True, n_sample_invoices=200)
    assert list(sample_df.columns) == REQUIRED_COLUMNS
    assert sample_df.attrs.get("data_source") == "sample"

    # Graceful fallback when UCI download raises RuntimeError
    with patch("src.data_loader._download_uci_dataframe", side_effect=RuntimeError("offline")):
        fallback_df = load_or_download_retail_data(tmp_path, use_sample=False, n_sample_invoices=200)
        assert list(fallback_df.columns) == REQUIRED_COLUMNS
        assert fallback_df.attrs.get("data_source") == "sample_fallback"

    # Parquet cache hit when online_retail.parquet exists
    parquet_path = tmp_path / "online_retail.parquet"
    sample_df.to_parquet(parquet_path, index=False)
    cached_df = load_or_download_retail_data(tmp_path, use_sample=False)
    assert len(cached_df) == len(sample_df)
    assert cached_df.attrs.get("data_source") == "uci_parquet_cache"


def test_clean_retail_transactions_audit_and_canonicalization() -> None:
    """Verify cleaning removes invalid rows, canonicalizes StockCode descriptions, and conserves counts."""
    raw = pd.DataFrame(
        [
            # Valid canonical items + noisy whitespace/case duplicate of StockCode 22661
            {"InvoiceNo": "536365", "StockCode": "22661", "Description": "REGENCY TEA PLATE PINK", "Quantity": 2, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            {"InvoiceNo": "536366", "StockCode": "22661", "Description": "REGENCY TEA PLATE PINK", "Quantity": 3, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            {"InvoiceNo": "536367", "StockCode": "22661", "Description": "  regency   tea plate pink ", "Quantity": 1, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17851.0, "Country": "United Kingdom"},
            # Exact duplicate of row 0
            {"InvoiceNo": "536365", "StockCode": "22661", "Description": "REGENCY TEA PLATE PINK", "Quantity": 2, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            # Cancelled invoice 'C'
            {"InvoiceNo": "C536368", "StockCode": "22662", "Description": "REGENCY TEA PLATE GREEN", "Quantity": -2, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            # Missing Description
            {"InvoiceNo": "536369", "StockCode": "22662", "Description": None, "Quantity": 1, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            # Non-positive Qty and Price
            {"InvoiceNo": "536370", "StockCode": "22662", "Description": "REGENCY TEA PLATE GREEN", "Quantity": 0, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 17850.0, "Country": "United Kingdom"},
            {"InvoiceNo": "536371", "StockCode": "22662", "Description": "REGENCY TEA PLATE GREEN", "Quantity": 1, "InvoiceDate": "2011-01-01", "UnitPrice": 0.0, "CustomerID": 17850.0, "Country": "United Kingdom"},
            # Admin StockCodes
            {"InvoiceNo": "536365", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "InvoiceDate": "2011-01-01", "UnitPrice": 18.0, "CustomerID": 17850.0, "Country": "United Kingdom"},
            {"InvoiceNo": "536365", "StockCode": "DOT", "Description": "DOTCOM POSTAGE", "Quantity": 1, "InvoiceDate": "2011-01-01", "UnitPrice": 12.0, "CustomerID": 17850.0, "Country": "United Kingdom"},
            # Missing CustomerID
            {"InvoiceNo": "536372", "StockCode": "22662", "Description": "REGENCY TEA PLATE GREEN", "Quantity": 4, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": np.nan, "Country": "United Kingdom"},
            # Non-UK country
            {"InvoiceNo": "536373", "StockCode": "22662", "Description": "REGENCY TEA PLATE GREEN", "Quantity": 4, "InvoiceDate": "2011-01-01", "UnitPrice": 1.65, "CustomerID": 12500.0, "Country": "France"},
        ]
    )

    clean_uk, audit_uk = clean_retail_transactions(raw, drop_missing_customer_id=True, country_filter="United Kingdom")
    assert audit_uk["raw_rows"] == 12
    assert audit_uk["cancelled_removed"] == 1
    assert audit_uk["missing_description_removed"] == 1
    assert audit_uk["non_positive_removed"] == 2
    assert audit_uk["admin_stock_codes_removed"] == 2
    assert audit_uk["missing_customer_removed"] == 1
    assert audit_uk["duplicates_removed"] == 1
    assert audit_uk["country_filtered_removed"] == 1
    assert audit_uk["final_clean_rows"] == 3

    # Exact row conservation invariant
    total_removed = (
        audit_uk["cancelled_removed"]
        + audit_uk["missing_description_removed"]
        + audit_uk["non_positive_removed"]
        + audit_uk["admin_stock_codes_removed"]
        + audit_uk["missing_customer_removed"]
        + audit_uk["duplicates_removed"]
        + audit_uk["country_filtered_removed"]
    )
    assert audit_uk["raw_rows"] - total_removed == audit_uk["final_clean_rows"]
    assert clean_uk["Description"].nunique() == 1
    assert clean_uk["Description"].iloc[0] == "REGENCY TEA PLATE PINK"

    # When keeping missing CustomerID and All Countries
    clean_all, audit_all = clean_retail_transactions(raw, drop_missing_customer_id=False, country_filter="All Countries")
    assert audit_all["missing_customer_removed"] == 0
    assert audit_all["country_filtered_removed"] == 0
    assert audit_all["final_clean_rows"] == 5
    assert len(clean_all) == 5


def test_sparse_basket_matrix_pre_pruning_and_denominator_preservation() -> None:
    """Verify sparse matrix dtype, min_item_freq pruning, and preserved invoice count denominator."""
    rows = []
    # 10 invoices: 8 contain COMMON_A & COMMON_B, 1 has ONLY RARE_ITEM (freq 0.10 < 0.25), 1 has COMMON_A
    for i in range(8):
        rows.append({"InvoiceNo": f"INV{i}", "StockCode": "A1", "Description": "COMMON_A", "Quantity": 2, "UnitPrice": 2.0, "CustomerID": 100.0, "Country": "United Kingdom"})
        rows.append({"InvoiceNo": f"INV{i}", "StockCode": "B1", "Description": "COMMON_B", "Quantity": 1, "UnitPrice": 4.0, "CustomerID": 100.0, "Country": "United Kingdom"})
    rows.append({"InvoiceNo": "INV8", "StockCode": "A1", "Description": "COMMON_A", "Quantity": 1, "UnitPrice": 2.0, "CustomerID": 100.0, "Country": "United Kingdom"})
    rows.append({"InvoiceNo": "INV9", "StockCode": "R1", "Description": "RARE_ITEM", "Quantity": 1, "UnitPrice": 9.0, "CustomerID": 100.0, "Country": "United Kingdom"})
    clean_df = pd.DataFrame(rows)

    basket_df, meta_df = build_sparse_basket_matrix(clean_df, min_item_freq=0.25)
    # All 10 invoices retained as rows so Support denominator remains N=10
    assert basket_df.shape == (10, 2)
    assert "RARE_ITEM" not in basket_df.columns
    assert all(isinstance(dt, pd.SparseDtype) for dt in basket_df.dtypes)

    # Verify item_metadata_df metrics
    meta_a = meta_df.loc[meta_df["item"] == "COMMON_A"].iloc[0]
    assert meta_a["transaction_count"] == 9
    assert math.isclose(meta_a["support"], 0.9, rel_tol=1e-6)
    assert meta_a["total_quantity"] == 17
    assert math.isclose(meta_a["avg_unit_price"], 2.0, rel_tol=1e-6)


def test_fpgrowth_metrics_and_redundant_rule_pruning() -> None:
    """Verify exact Support/Confidence/Lift/Conviction/Leverage math and redundancy pruning."""
    # Construct 10 deterministic baskets:
    # A & B & C appear together in 4 baskets (0..3)
    # A & C appear without B in 1 basket (4)
    # A & B appear without C in 1 basket (5)
    # B & C appear without A in 0 baskets
    # C alone in 2 baskets (6..7)
    # D alone in 2 baskets (8..9)
    # Look at rule A -> C vs {A, B} -> C:
    #   Support(A) = 6/10 = 0.6, Support(C) = 7/10 = 0.7
    #   Support(A, C) = 5/10 = 0.5 -> Conf(A -> C) = 5/6 = 0.833333
    #   Support(A, B) = 5/10 = 0.5, Support(A, B, C) = 4/10 = 0.4 -> Conf({A, B} -> C) = 4/5 = 0.80 <= 0.8333!
    # Therefore {A, B} -> C MUST be pruned when filter_redundant=True!
    baskets = [
        ["ITEM_A", "ITEM_B", "ITEM_C"],
        ["ITEM_A", "ITEM_B", "ITEM_C"],
        ["ITEM_A", "ITEM_B", "ITEM_C"],
        ["ITEM_A", "ITEM_B", "ITEM_C"],
        ["ITEM_A", "ITEM_C"],
        ["ITEM_A", "ITEM_B"],
        ["ITEM_C"],
        ["ITEM_C"],
        ["ITEM_D"],
        ["ITEM_D"],
    ]
    items = ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D"]
    rows, cols = [], []
    for r, b in enumerate(baskets):
        for it in b:
            rows.append(r)
            cols.append(items.index(it))
    mat = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(10, 4), dtype=bool)
    basket_df = pd.DataFrame.sparse.from_spmatrix(mat, columns=items)

    res_unpruned = mine_fpgrowth_rules(
        basket_df,
        min_support=0.35,
        min_confidence=0.70,
        min_lift=1.0,
        max_len=3,
        filter_redundant=False,
    )
    assert res_unpruned["status"] == "ok"
    rules_unpruned = res_unpruned["rules"]
    assert "ITEM_A, ITEM_B -> ITEM_C" in rules_unpruned["rule_label"].values
    assert "ITEM_A -> ITEM_C" in rules_unpruned["rule_label"].values

    # Check exact hand-calculated metrics for ITEM_A -> ITEM_C
    r_ac = rules_unpruned.loc[rules_unpruned["rule_label"] == "ITEM_A -> ITEM_C"].iloc[0]
    expected_sup = 0.5
    expected_conf = 5.0 / 6.0
    expected_lift = expected_conf / 0.7
    expected_lev = 0.5 - (0.6 * 0.7)
    expected_conv = (1.0 - 0.7) / (1.0 - expected_conf)
    assert math.isclose(r_ac["support"], expected_sup, rel_tol=1e-6)
    assert math.isclose(r_ac["confidence"], expected_conf, rel_tol=1e-6)
    assert math.isclose(r_ac["lift"], expected_lift, rel_tol=1e-6)
    assert math.isclose(r_ac["leverage"], expected_lev, rel_tol=1e-6)
    assert math.isclose(r_ac["conviction"], expected_conv, rel_tol=1e-6)

    # With filter_redundant=True, {ITEM_A, ITEM_B} -> ITEM_C (conf 0.80 <= 0.833) is pruned!
    res_pruned = mine_fpgrowth_rules(
        basket_df,
        min_support=0.35,
        min_confidence=0.70,
        min_lift=1.0,
        max_len=3,
        filter_redundant=True,
    )
    assert res_pruned["n_redundant_removed"] >= 1
    assert "ITEM_A, ITEM_B -> ITEM_C" not in res_pruned["rules"]["rule_label"].values
    assert "ITEM_A -> ITEM_C" in res_pruned["rules"]["rule_label"].values


def test_fpgrowth_empty_and_zero_rule_edge_cases() -> None:
    """Verify zero-itemset and zero-rule edge cases never crash."""
    empty_res = mine_fpgrowth_rules(pd.DataFrame())
    assert empty_res["n_rules"] == 0
    assert empty_res["status"] == "empty_basket"

    df_sample = generate_retail_sample_data(n_invoices=300, random_state=7)
    clean_df, _ = clean_retail_transactions(df_sample)
    basket_df, meta_df = build_sparse_basket_matrix(clean_df, min_item_freq=0.003)

    impossible_sup = mine_fpgrowth_rules(basket_df, min_support=0.99, min_confidence=0.50, min_lift=1.0)
    assert impossible_sup["n_itemsets"] == 0
    assert impossible_sup["n_rules"] == 0
    assert impossible_sup["status"] == "no_itemsets"

    impossible_lift = mine_fpgrowth_rules(basket_df, min_support=0.02, min_confidence=0.99, min_lift=99.0)
    assert impossible_lift["n_rules"] == 0
    assert impossible_lift["rules"].empty

    # Empty rules in bundling simulator
    sim_empty = simulate_bundling_recommendations(
        rules_df=impossible_lift["rules"],
        item_metadata_df=meta_df,
        anchor_items=["REGENCY TEA PLATE PINK"],
    )
    assert sim_empty.empty


def test_bundling_simulator_on_sample_bundles() -> None:
    """Verify bundling simulator produces ranked recommendations and accurate incremental value."""
    df_sample = generate_retail_sample_data(n_invoices=1500, random_state=42)
    clean_df, audit = clean_retail_transactions(
        df_sample,
        drop_missing_customer_id=True,
        country_filter="United Kingdom",
    )
    assert audit["canonicalized_descriptions"] > 0
    basket_df, meta_df = build_sparse_basket_matrix(clean_df, min_item_freq=0.003)
    mining = mine_fpgrowth_rules(
        basket_df,
        min_support=0.01,
        min_confidence=0.50,
        min_lift=1.2,
        max_len=3,
        filter_redundant=True,
    )
    assert mining["n_rules"] > 0

    rules_df = mining["rules"]
    single_rules = rules_df.loc[rules_df["antecedent_len"] == 1]
    assert not single_rules.empty
    anchor = sorted(single_rules.iloc[0]["antecedents_frozenset"])[0]
    sim_df = simulate_bundling_recommendations(
        rules_df=rules_df,
        item_metadata_df=meta_df,
        anchor_items=[anchor],
        discount_pct=10.0,
        top_k=5,
    )
    assert not sim_df.empty
    assert anchor not in sim_df["recommended_item"].values
    valid_badges = {"Strong Bundle Candidate", "High-Confidence Cross-Sell", "Niche Affinity"}
    assert set(sim_df["affinity_badge"]).issubset(valid_badges)

    top_row = sim_df.iloc[0]
    expected_net = round(top_row["conditional_prob"] * top_row["unit_price"] * 0.90, 2)
    assert math.isclose(top_row["estimated_incremental_basket_value"], expected_net, abs_tol=0.02)


def test_streamlit_app_smoke_and_interactions() -> None:
    """Headless AppTest smoke test validating all 3 modules render without exceptions."""
    app_path = APP_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()
    assert not at.exception, f"App raised exception on initial load: {at.exception}"
    assert len(at.tabs) == 3

    # Verify toggling sidebar controls works cleanly without exceptions
    if at.sidebar.toggle:
        at.sidebar.toggle[0].set_value(False)
    if at.sidebar.slider:
        at.sidebar.slider[0].set_value(0.015)
    at.run()
    assert not at.exception, f"App raised exception after sidebar update: {at.exception}"
