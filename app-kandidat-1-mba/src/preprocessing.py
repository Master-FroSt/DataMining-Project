"""Deterministic transaction cleaning, audit logging, and sparse basket matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ADMIN_STOCK_CODES: set[str] = {
    "POST",
    "DOT",
    "M",
    "CRUK",
    "BANK CHARGES",
    "D",
    "C2",
    "S",
    "AMAZONFEE",
    "PADS",
    "B",
}


def _canonicalize_descriptions_by_stock_code(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map each StockCode to its deterministic modal trimmed uppercase Description."""
    if df.empty:
        return df, 0

    work = df.copy()
    normalized_desc = (
        work["Description"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )
    work["Description"] = normalized_desc

    counts = (
        work.groupby(["StockCode", "Description"], sort=False)
        .size()
        .reset_index(name="freq")
        .sort_values(by=["StockCode", "freq", "Description"], ascending=[True, False, True])
    )
    canonical_map = counts.drop_duplicates(subset=["StockCode"], keep="first").set_index("StockCode")[
        "Description"
    ]
    mapped_desc = work["StockCode"].map(canonical_map).fillna(work["Description"])
    remapped_count = int((mapped_desc != normalized_desc).sum())
    work["Description"] = mapped_desc
    return work, remapped_count


def clean_retail_transactions(
    df: pd.DataFrame,
    drop_missing_customer_id: bool = True,
    country_filter: str | None = "United Kingdom",
) -> tuple[pd.DataFrame, dict]:
    """Clean UCI Online Retail line-item data and return deterministic cleaning audit metrics.

    Every sequential step removes disjoint rows so the conservation identity holds:
    ``raw_rows - sum(removed_counts) == final_clean_rows``.
    """
    work = df.copy()
    raw_rows = int(len(work))

    if raw_rows == 0:
        empty_audit = {
            "raw_rows": 0,
            "cancelled_removed": 0,
            "missing_description_removed": 0,
            "non_positive_removed": 0,
            "non_positive_qty_removed": 0,
            "non_positive_price_removed": 0,
            "admin_stock_codes_removed": 0,
            "missing_customer_removed": 0,
            "duplicates_removed": 0,
            "country_filtered_removed": 0,
            "country_filter": country_filter or "All Countries",
            "drop_missing_customer_id": drop_missing_customer_id,
            "final_clean_rows": 0,
            "retained_pct": 0.0,
            "unique_invoices": 0,
            "unique_skus": 0,
            "unique_stock_codes": 0,
            "avg_basket_size": 0.0,
            "median_basket_size": 0.0,
            "single_item_baskets": 0,
            "multi_item_baskets": 0,
            "multi_item_basket_ratio": 0.0,
            "canonicalized_descriptions": 0,
        }
        return work, empty_audit

    # Normalize key identifier and numeric fields safely
    work["InvoiceNo"] = work["InvoiceNo"].where(work["InvoiceNo"].notna(), None)
    work["InvoiceNo"] = work["InvoiceNo"].map(lambda v: "" if pd.isna(v) else str(v).strip())
    work["StockCode"] = work["StockCode"].where(work["StockCode"].notna(), None)
    work["StockCode"] = work["StockCode"].map(lambda v: "" if pd.isna(v) else str(v).strip().upper())
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
    work["CustomerID"] = pd.to_numeric(work["CustomerID"], errors="coerce")
    work["Country"] = work["Country"].fillna("Unspecified").astype(str).str.strip()

    # Step 1: Remove cancelled invoices ('C' prefix) or blank InvoiceNo
    is_cancelled = work["InvoiceNo"].str.upper().str.startswith("C") | (work["InvoiceNo"] == "")
    cancelled_removed = int(is_cancelled.sum())
    work = work.loc[~is_cancelled].copy()

    # Step 2: Remove missing or blank Description
    raw_desc_na = work["Description"].isna()
    clean_desc_str = work["Description"].fillna("").astype(str).str.strip()
    is_missing_desc = raw_desc_na | (clean_desc_str == "") | (clean_desc_str.str.lower().isin(["nan", "none"]))
    missing_description_removed = int(is_missing_desc.sum())
    work = work.loc[~is_missing_desc].copy()

    # Step 3: Remove non-positive Quantity or UnitPrice
    bad_qty = work["Quantity"].isna() | (work["Quantity"] <= 0)
    bad_price = work["UnitPrice"].isna() | (work["UnitPrice"] <= 0)
    is_non_positive = bad_qty | bad_price
    non_positive_qty_removed = int(bad_qty.sum())
    non_positive_price_removed = int(bad_price.sum())
    non_positive_removed = int(is_non_positive.sum())
    work = work.loc[~is_non_positive].copy()

    # Step 4: Remove administrative / non-product StockCodes
    is_admin = work["StockCode"].isin(ADMIN_STOCK_CODES) | (work["StockCode"] == "")
    admin_stock_codes_removed = int(is_admin.sum())
    work = work.loc[~is_admin].copy()

    # Step 5: Optional missing CustomerID removal (~24.9% in UCI Online Retail)
    if drop_missing_customer_id:
        is_missing_cust = work["CustomerID"].isna()
        missing_customer_removed = int(is_missing_cust.sum())
        work = work.loc[~is_missing_cust].copy()
    else:
        missing_customer_removed = 0

    # Canonicalize trimmed uppercase Description per StockCode before duplicate elimination
    work, canonicalized_descriptions = _canonicalize_descriptions_by_stock_code(work)

    # Step 6: Remove exact duplicate line items
    before_dedup = len(work)
    work = work.drop_duplicates().copy()
    duplicates_removed = int(before_dedup - len(work))

    # Step 7: Optional country segmentation filter
    normalized_country = (country_filter or "").strip()
    if normalized_country and normalized_country.lower() not in {"all", "all countries"}:
        before_country = len(work)
        work = work.loc[work["Country"] == normalized_country].copy()
        country_filtered_removed = int(before_country - len(work))
    else:
        normalized_country = "All Countries"
        country_filtered_removed = 0

    work = work.reset_index(drop=True)
    final_clean_rows = int(len(work))

    if final_clean_rows > 0:
        basket_sizes = work.groupby("InvoiceNo", sort=False)["Description"].nunique()
        unique_invoices = int(len(basket_sizes))
        unique_skus = int(work["Description"].nunique())
        unique_stock_codes = int(work["StockCode"].nunique())
        avg_basket_size = float(basket_sizes.mean())
        median_basket_size = float(basket_sizes.median())
        multi_item_baskets = int((basket_sizes >= 2).sum())
        single_item_baskets = int((basket_sizes == 1).sum())
        multi_item_basket_ratio = float(multi_item_baskets / unique_invoices) if unique_invoices > 0 else 0.0
    else:
        unique_invoices = 0
        unique_skus = 0
        unique_stock_codes = 0
        avg_basket_size = 0.0
        median_basket_size = 0.0
        multi_item_baskets = 0
        single_item_baskets = 0
        multi_item_basket_ratio = 0.0

    cleaning_audit = {
        "raw_rows": raw_rows,
        "cancelled_removed": cancelled_removed,
        "missing_description_removed": missing_description_removed,
        "non_positive_removed": non_positive_removed,
        "non_positive_qty_removed": non_positive_qty_removed,
        "non_positive_price_removed": non_positive_price_removed,
        "admin_stock_codes_removed": admin_stock_codes_removed,
        "missing_customer_removed": missing_customer_removed,
        "duplicates_removed": duplicates_removed,
        "country_filtered_removed": country_filtered_removed,
        "country_filter": normalized_country,
        "drop_missing_customer_id": drop_missing_customer_id,
        "final_clean_rows": final_clean_rows,
        "retained_pct": round((final_clean_rows / raw_rows) * 100.0, 2) if raw_rows > 0 else 0.0,
        "unique_invoices": unique_invoices,
        "unique_skus": unique_skus,
        "unique_stock_codes": unique_stock_codes,
        "avg_basket_size": round(avg_basket_size, 3),
        "median_basket_size": round(median_basket_size, 2),
        "single_item_baskets": single_item_baskets,
        "multi_item_baskets": multi_item_baskets,
        "multi_item_basket_ratio": round(multi_item_basket_ratio, 4),
        "canonicalized_descriptions": canonicalized_descriptions,
    }
    return work, cleaning_audit


def build_sparse_basket_matrix(
    clean_df: pd.DataFrame,
    item_col: str = "Description",
    min_item_freq: float = 0.003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform cleaned line-items into a memory-efficient boolean sparse basket matrix.

    Pre-prunes columns whose transaction frequency falls below ``min_item_freq`` while
    retaining every clean invoice row in the matrix index so true database support
    denominators remain exact.

    Returns:
        Tuple of (sparse boolean DataFrame indexed by ``InvoiceNo``, item metadata DataFrame).
    """
    meta_cols = [
        "item",
        "stock_code",
        "transaction_count",
        "support",
        "total_quantity",
        "avg_unit_price",
        "total_revenue",
    ]
    if clean_df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=meta_cols)

    if item_col not in clean_df.columns:
        raise ValueError(f"Column '{item_col}' not found in cleaned DataFrame.")

    work = clean_df.copy()
    work["line_revenue"] = work["Quantity"].astype(float) * work["UnitPrice"].astype(float)

    all_invoices = pd.Index(sorted(work["InvoiceNo"].unique()), name="InvoiceNo")
    n_invoices = len(all_invoices)
    if n_invoices == 0:
        return pd.DataFrame(), pd.DataFrame(columns=meta_cols)

    # Deduplicate (InvoiceNo, item) pairs for binary co-occurrence aggregation
    pairs = work[["InvoiceNo", item_col]].drop_duplicates()

    # Build item-level statistics over full cleaned transactions
    tx_counts = pairs.groupby(item_col, sort=False)["InvoiceNo"].count()
    agg_stats = work.groupby(item_col, sort=False).agg(
        stock_code=("StockCode", "first"),
        total_quantity=("Quantity", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        total_revenue=("line_revenue", "sum"),
    )
    item_metadata_df = agg_stats.join(tx_counts.rename("transaction_count")).reset_index()
    item_metadata_df = item_metadata_df.rename(columns={item_col: "item"})
    item_metadata_df["support"] = item_metadata_df["transaction_count"] / float(n_invoices)
    item_metadata_df = item_metadata_df[meta_cols].sort_values(
        by=["transaction_count", "total_quantity", "item"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Pre-prune rare items below min_item_freq floor before sparse matrix construction
    eligible_items = item_metadata_df.loc[item_metadata_df["support"] >= float(min_item_freq), "item"]
    eligible_items = sorted(eligible_items.tolist())
    if not eligible_items:
        empty_matrix = pd.DataFrame(index=all_invoices)
        return empty_matrix, item_metadata_df

    filtered_pairs = pairs.loc[pairs[item_col].isin(eligible_items)].copy()
    item_index = pd.Index(eligible_items, name="item")

    row_codes = all_invoices.get_indexer(filtered_pairs["InvoiceNo"])
    col_codes = item_index.get_indexer(filtered_pairs[item_col])
    valid_mask = (row_codes >= 0) & (col_codes >= 0)
    row_codes = row_codes[valid_mask]
    col_codes = col_codes[valid_mask]

    sparse_mat = csr_matrix(
        (np.ones(len(row_codes), dtype=bool), (row_codes, col_codes)),
        shape=(n_invoices, len(item_index)),
        dtype=bool,
    )
    basket_df = pd.DataFrame.sparse.from_spmatrix(
        sparse_mat,
        index=all_invoices,
        columns=item_index,
    )
    return basket_df, item_metadata_df
