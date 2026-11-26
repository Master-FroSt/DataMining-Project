"""RFM transaction cleaning, customer aggregation, and feature transformation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    "ADJUST",
    "ADJUST2",
}

RFM_COLUMNS: list[str] = ["Recency", "Frequency", "Monetary"]
VALID_OUTLIER_METHODS = {"none", "iqr_1.5", "winsorize_3iqr", "winsorize_p99"}
VALID_SKEW_METHODS = {"none", "log1p", "boxcox"}
VALID_SCALER_METHODS = {"standard", "minmax"}


def _format_customer_id(series: pd.Series) -> pd.Series:
    """Format CustomerID cleanly without trailing '.0' on integer-valued floats."""
    numeric = pd.to_numeric(series, errors="coerce")
    is_int_like = numeric.notna() & ((numeric % 1) == 0)
    formatted = series.astype(str).str.strip()
    if is_int_like.any():
        formatted = formatted.where(
            ~is_int_like,
            numeric.where(is_int_like, 0).astype("int64").astype(str),
        )
    return formatted


def _safe_skew(values: np.ndarray | pd.Series) -> float:
    """Compute skewness safely, returning 0.0 for constant or tiny arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or np.ptp(arr) < 1e-12:
        return 0.0
    val = float(pd.Series(arr).skew())
    return 0.0 if np.isnan(val) or np.isinf(val) else val


def clean_transactions_for_rfm(
    df: pd.DataFrame,
    country_filter: str = "All Countries",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw Online Retail transactions and return audit statistics.

    Removes missing CustomerID, 'C' cancellation invoices, Quantity <= 0,
    UnitPrice <= 0, invalid dates, non-product administrative StockCodes,
    exact duplicate rows, and applies optional Country filtering.
    """
    work = df.copy()
    work.columns = [str(col).strip() for col in work.columns]
    alias_map = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }
    rename_dict = {
        old: new for old, new in alias_map.items() if old in work.columns and new not in work.columns
    }
    if rename_dict:
        work = work.rename(columns=rename_dict)

    mandatory = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing_cols = [c for c in mandatory if c not in work.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required RFM columns: {missing_cols}")

    if "StockCode" not in work.columns:
        work["StockCode"] = "SKU"
    if "Description" not in work.columns:
        work["Description"] = "Retail Item"
    if "Country" not in work.columns:
        work["Country"] = "Unknown"

    raw_rows = int(len(work))

    work["InvoiceNo"] = work["InvoiceNo"].astype(str).str.strip()
    work["StockCode"] = work["StockCode"].astype(str).str.strip()
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")
    work["Country"] = work["Country"].fillna("Unknown").astype(str).str.strip()

    # Step 1: Remove missing CustomerID
    cid_str = work["CustomerID"].astype(str).str.strip()
    valid_cid_mask = work["CustomerID"].notna() & (~cid_str.isin(["", "nan", "None", "NaN", "<NA>"]))
    removed_missing_customer = int((~valid_cid_mask).sum())
    work = work.loc[valid_cid_mask].copy()

    # Step 2: Remove 'C' cancellation invoices
    cancel_mask = work["InvoiceNo"].str.upper().str.startswith("C")
    removed_cancellations = int(cancel_mask.sum())
    work = work.loc[~cancel_mask].copy()

    # Step 3: Remove Quantity <= 0 or NaN
    valid_qty_mask = work["Quantity"].notna() & (work["Quantity"] > 0)
    removed_non_positive_qty = int((~valid_qty_mask).sum())
    work = work.loc[valid_qty_mask].copy()

    # Step 4: Remove UnitPrice <= 0 or NaN
    valid_price_mask = work["UnitPrice"].notna() & (work["UnitPrice"] > 0)
    removed_non_positive_price = int((~valid_price_mask).sum())
    work = work.loc[valid_price_mask].copy()

    # Step 5: Remove invalid InvoiceDate
    valid_date_mask = work["InvoiceDate"].notna()
    removed_invalid_date = int((~valid_date_mask).sum())
    work = work.loc[valid_date_mask].copy()

    # Step 6: Remove non-product administrative StockCodes
    admin_mask = work["StockCode"].str.upper().isin(ADMIN_STOCK_CODES)
    removed_admin_codes = int(admin_mask.sum())
    work = work.loc[~admin_mask].copy()

    # Step 7: Remove exact duplicate rows
    before_dup = int(len(work))
    work = work.drop_duplicates().copy()
    removed_duplicates = before_dup - int(len(work))

    # Format CustomerID cleanly before country filtering
    if not work.empty:
        work["CustomerID"] = _format_customer_id(work["CustomerID"])
    else:
        work["CustomerID"] = pd.Series(dtype="object")

    # Step 8: Optional Country filter
    removed_country_filter = 0
    if country_filter and country_filter != "All Countries":
        before_country = int(len(work))
        work = work.loc[work["Country"] == country_filter].copy()
        removed_country_filter = before_country - int(len(work))

    work["TotalPrice"] = work["Quantity"].astype(float) * work["UnitPrice"].astype(float)
    work = work.reset_index(drop=True)

    clean_rows = int(len(work))
    retention_rate = round((100.0 * clean_rows / raw_rows), 2) if raw_rows > 0 else 0.0

    audit: dict[str, Any] = {
        "raw_rows": raw_rows,
        "removed_missing_customer": removed_missing_customer,
        "removed_cancellations": removed_cancellations,
        "removed_non_positive_qty": removed_non_positive_qty,
        "removed_non_positive_price": removed_non_positive_price,
        "removed_invalid_date": removed_invalid_date,
        "removed_admin_codes": removed_admin_codes,
        "removed_duplicates": removed_duplicates,
        "removed_country_filter": removed_country_filter,
        "country_filter": country_filter or "All Countries",
        "clean_rows": clean_rows,
        "retention_rate_pct": retention_rate,
        "unique_customers": int(work["CustomerID"].nunique()) if clean_rows > 0 else 0,
        "unique_invoices": int(work["InvoiceNo"].nunique()) if clean_rows > 0 else 0,
        "unique_countries": int(work["Country"].nunique()) if clean_rows > 0 else 0,
        "date_min": work["InvoiceDate"].min() if clean_rows > 0 else None,
        "date_max": work["InvoiceDate"].max() if clean_rows > 0 else None,
        "total_revenue": float(work["TotalPrice"].sum()) if clean_rows > 0 else 0.0,
    }
    return work, audit


def compute_rfm_table(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Aggregate clean transactions to customer-level Recency, Frequency, Monetary table.

    Sets reference snapshot T_ref = max(InvoiceDate) + 1 day so Recency >= 1 day.
    """
    if clean_df.empty:
        raise ValueError("Cannot compute RFM table from an empty cleaned transaction DataFrame.")

    work = clean_df.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")
    work = work.dropna(subset=["InvoiceDate", "CustomerID"])
    if "TotalPrice" not in work.columns:
        work["TotalPrice"] = work["Quantity"].astype(float) * work["UnitPrice"].astype(float)
    if "Country" not in work.columns:
        work["Country"] = "Unknown"

    t_ref = pd.Timestamp(work["InvoiceDate"].max()) + pd.Timedelta(days=1)

    rfm = (
        work.groupby("CustomerID", as_index=False)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    )
    rfm["Recency"] = (t_ref - rfm["last_purchase"]).dt.days.astype("int64")
    rfm["Frequency"] = rfm["Frequency"].astype("int64")
    rfm["Monetary"] = rfm["Monetary"].astype("float64")

    # Determine each customer's primary Country by highest cumulative spend
    country_pref = (
        work.groupby(["CustomerID", "Country"], as_index=False)["TotalPrice"]
        .sum()
        .sort_values(["CustomerID", "TotalPrice", "Country"], ascending=[True, False, True])
        .drop_duplicates(subset=["CustomerID"], keep="first")
        .loc[:, ["CustomerID", "Country"]]
    )

    rfm = rfm.merge(country_pref, on="CustomerID", how="left")
    rfm["Country"] = rfm["Country"].fillna("Unknown")
    rfm = (
        rfm.loc[:, ["CustomerID", "Country", "Recency", "Frequency", "Monetary"]]
        .sort_values("CustomerID")
        .reset_index(drop=True)
    )
    return rfm, t_ref


def transform_and_scale_rfm(
    rfm_df: pd.DataFrame,
    outlier_method: str = "winsorize_3iqr",
    skew_method: str = "log1p",
    scaler_method: str = "standard",
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Apply outlier handling, skewness transformation, and feature scaling to RFM.

    Returns:
        X_scaled: Standardized/scaled numpy array of shape (n_filtered, 3).
        rfm_filtered_df: Filtered RFM DataFrame with reset index aligned 1-to-1 with X_scaled.
        diagnostics: Before/after skewness metrics, dropped/capped counts, and intermediate tables.
    """
    if outlier_method not in VALID_OUTLIER_METHODS:
        raise ValueError(
            f"Unsupported outlier_method '{outlier_method}'. Expected one of {sorted(VALID_OUTLIER_METHODS)}."
        )
    if skew_method not in VALID_SKEW_METHODS:
        raise ValueError(
            f"Unsupported skew_method '{skew_method}'. Expected one of {sorted(VALID_SKEW_METHODS)}."
        )
    if scaler_method not in VALID_SCALER_METHODS:
        raise ValueError(
            f"Unsupported scaler_method '{scaler_method}'. Expected one of {sorted(VALID_SCALER_METHODS)}."
        )
    if rfm_df.empty:
        raise ValueError("Cannot transform an empty RFM DataFrame.")

    for col in RFM_COLUMNS:
        if col not in rfm_df.columns:
            raise ValueError(f"RFM DataFrame is missing column '{col}'.")

    raw_work = rfm_df.copy().reset_index(drop=True)
    rows_before = int(len(raw_work))
    skewness_before = {col: round(_safe_skew(raw_work[col]), 4) for col in RFM_COLUMNS}

    outliers_capped = {col: 0 for col in RFM_COLUMNS}
    iqr_bounds: dict[str, dict[str, float]] = {}

    # Step 1: Outlier Handling
    if outlier_method == "iqr_1.5":
        keep_mask = np.ones(len(raw_work), dtype=bool)
        for col in RFM_COLUMNS:
            vals = raw_work[col].astype(float)
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_bounds[col] = {"lower": lower, "upper": upper}
                keep_mask &= (vals >= lower) & (vals <= upper)
        # Guard against filtering out almost the entire dataset on narrow subsets
        if int(keep_mask.sum()) >= 3:
            rfm_filtered_df = raw_work.loc[keep_mask].copy().reset_index(drop=True)
        else:
            rfm_filtered_df = raw_work.copy().reset_index(drop=True)
        transformed = rfm_filtered_df[RFM_COLUMNS].astype(float).copy()
    elif outlier_method == "winsorize_3iqr":
        rfm_filtered_df = raw_work.copy().reset_index(drop=True)
        transformed = rfm_filtered_df[RFM_COLUMNS].astype(float).copy()
        for col in RFM_COLUMNS:
            vals = transformed[col]
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                capped_mask = (vals < lower) | (vals > upper)
                outliers_capped[col] = int(capped_mask.sum())
                transformed[col] = vals.clip(lower=lower, upper=upper)
                iqr_bounds[col] = {"lower": lower, "upper": upper}
    elif outlier_method == "winsorize_p99":
        rfm_filtered_df = raw_work.copy().reset_index(drop=True)
        transformed = rfm_filtered_df[RFM_COLUMNS].astype(float).copy()
        for col in RFM_COLUMNS:
            vals = transformed[col]
            lower = float(vals.quantile(0.01))
            upper = float(vals.quantile(0.99))
            if upper > lower:
                capped_mask = (vals < lower) | (vals > upper)
                outliers_capped[col] = int(capped_mask.sum())
                transformed[col] = vals.clip(lower=lower, upper=upper)
                iqr_bounds[col] = {"lower": lower, "upper": upper}
    else:
        rfm_filtered_df = raw_work.copy().reset_index(drop=True)
        transformed = rfm_filtered_df[RFM_COLUMNS].astype(float).copy()

    rows_after = int(len(rfm_filtered_df))
    rows_dropped = rows_before - rows_after

    # Step 2: Skewness Transformation
    boxcox_lambdas: dict[str, float | None] = {col: None for col in RFM_COLUMNS}
    if skew_method == "log1p":
        for col in RFM_COLUMNS:
            transformed[col] = np.log1p(np.maximum(transformed[col].to_numpy(dtype=float), 0.0))
    elif skew_method == "boxcox":
        for col in RFM_COLUMNS:
            arr = transformed[col].to_numpy(dtype=float)
            min_val = float(np.min(arr))
            pos_arr = arr if min_val > 0 else (arr - min_val + 1e-6)
            if np.ptp(pos_arr) < 1e-12 or len(pos_arr) < 3:
                boxcox_lambdas[col] = None
                continue
            try:
                bc_arr, lam = stats.boxcox(pos_arr)
                transformed[col] = bc_arr
                boxcox_lambdas[col] = round(float(lam), 4)
            except Exception:
                transformed[col] = np.log1p(np.maximum(arr, 0.0))
                boxcox_lambdas[col] = None

    # Step 3: Feature Scaling
    scaler = StandardScaler() if scaler_method == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(transformed[RFM_COLUMNS].to_numpy(dtype=float))
    X_scaled = np.asarray(X_scaled, dtype=float)

    rfm_scaled_df = pd.DataFrame(X_scaled, columns=RFM_COLUMNS)
    skewness_after = {
        col: round(_safe_skew(rfm_scaled_df[col]), 4) for col in RFM_COLUMNS
    }
    skewness_delta = {
        col: round(abs(skewness_after[col]) - abs(skewness_before[col]), 4)
        for col in RFM_COLUMNS
    }

    diagnostics: dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_dropped,
        "outlier_method": outlier_method,
        "skew_method": skew_method,
        "scaler_method": scaler_method,
        "skewness_before": skewness_before,
        "skewness_after": skewness_after,
        "skewness_delta": skewness_delta,
        "outliers_capped": outliers_capped,
        "iqr_bounds": iqr_bounds,
        "boxcox_lambdas": boxcox_lambdas,
        "rfm_transformed_df": transformed.reset_index(drop=True),
        "rfm_scaled_df": rfm_scaled_df.reset_index(drop=True),
        "scaler": scaler,
    }
    return X_scaled, rfm_filtered_df, diagnostics
