"""Data loading, Parquet caching, and realistic RFM sample generation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UCI_ONLINE_RETAIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)
PARQUET_FILENAME = "online_retail.parquet"

_CATALOG_ITEMS: list[tuple[str, str, float]] = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("47566", "PARTY BUNTING", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22197", "POPCORN HOLDER", 0.85),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
    ("23084", "RABBIT NIGHT LIGHT", 2.08),
    ("22633", "HAND WARMER UNION JACK", 2.10),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("22469", "HEART OF WICKER SMALL", 1.65),
    ("21733", "RED HANGING HEART T-LIGHT HOLDER", 2.95),
    ("22697", "GREEN REGENCY TEACUP AND SAUCER", 2.95),
    ("22699", "ROSES REGENCY TEACUP AND SAUCER", 2.95),
    ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
    ("23298", "SPOTTY BUNTING", 4.95),
    ("22178", "VICTORIAN GLASS HANGING T-LIGHT", 1.25),
    ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
]

_COUNTRIES: list[str] = [
    "United Kingdom",
    "Germany",
    "France",
    "EIRE",
    "Spain",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Portugal",
]
_COUNTRY_PROBS: list[float] = [0.82, 0.04, 0.035, 0.025, 0.02, 0.02, 0.015, 0.015, 0.01]


def _normalize_retail_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema and data types for Online Retail / Online Retail II logs."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    alias_map = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }
    rename_dict = {
        old: new for old, new in alias_map.items() if old in out.columns and new not in out.columns
    }
    if rename_dict:
        out = out.rename(columns=rename_dict)

    if "InvoiceNo" in out.columns:
        out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    if "StockCode" in out.columns:
        out["StockCode"] = out["StockCode"].astype(str).str.strip()
    if "Description" in out.columns:
        desc_series = out["Description"]
        out["Description"] = desc_series.where(desc_series.isna(), desc_series.astype(str).str.strip())
    if "Quantity" in out.columns:
        out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    if "InvoiceDate" in out.columns:
        out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    if "UnitPrice" in out.columns:
        out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")
    if "CustomerID" in out.columns:
        numeric_cid = pd.to_numeric(out["CustomerID"], errors="coerce")
        if numeric_cid.notna().sum() >= out["CustomerID"].notna().sum() * 0.8:
            out["CustomerID"] = numeric_cid
        else:
            cid_s = out["CustomerID"]
            out["CustomerID"] = cid_s.where(cid_s.isna(), cid_s.astype(str).str.strip())
    if "Country" in out.columns:
        country_s = out["Country"]
        out["Country"] = country_s.where(country_s.isna(), country_s.astype(str).str.strip())
    return out


def generate_rfm_retail_sample_data(
    n_customers: int = 800,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic 12-month (Dec 2010 - Dec 2011) Online Retail transaction log.

    Includes five behavioral archetypes (Champions/Wholesalers, Loyal, Regular,
    At-Risk, and Hibernating) plus UCI-authentic data quality anomalies:
    'C' cancellations, null CustomerID guest checkouts, non-positive quantities/prices,
    administrative StockCodes ('POST', 'M', 'DOT'), and exact duplicate rows.
    """
    if n_customers < 5:
        raise ValueError("n_customers must be at least 5 to represent multi-segment retail behavior.")

    rng = np.random.default_rng(random_state)
    start_ts = pd.Timestamp("2010-12-01 08:30:00")
    end_ts = pd.Timestamp("2011-12-09 12:50:00")
    total_seconds = int((end_ts - start_ts).total_seconds())

    # Segment shares summing cleanly to n_customers
    segment_specs = [
        # (name, share, r_days_min, r_days_max, f_min, f_max, lines_min, lines_max, qty_min, qty_max)
        ("Champions", 0.12, 1, 22, 14, 34, 3, 6, 12, 45),
        ("Loyal", 0.20, 10, 48, 6, 13, 2, 5, 6, 18),
        ("Regular", 0.28, 25, 105, 3, 6, 2, 4, 3, 10),
        ("At-Risk", 0.20, 115, 225, 4, 9, 2, 4, 5, 15),
        ("Hibernating", 0.20, 220, 368, 1, 2, 1, 3, 1, 5),
    ]

    counts = [max(1, int(round(n_customers * spec[1]))) for spec in segment_specs]
    diff = n_customers - sum(counts)
    counts[2] = max(1, counts[2] + diff)

    rows: list[dict[str, Any]] = []
    invoice_counter = 536365
    customer_id_start = 12347

    cust_idx = 0
    for spec, seg_count in zip(segment_specs, counts):
        _, _, r_min, r_max, f_min, f_max, l_min, l_max, q_min, q_max = spec
        for _ in range(seg_count):
            cid = float(customer_id_start + cust_idx * 7)
            cust_idx += 1
            country = str(rng.choice(_COUNTRIES, p=_COUNTRY_PROBS))
            if spec[0] == "Champions" and rng.random() < 0.25:
                country = "United Kingdom"

            n_invoices = int(rng.integers(f_min, f_max + 1))
            recency_days = int(rng.integers(r_min, r_max + 1))
            last_purchase_ts = end_ts - pd.Timedelta(
                days=recency_days,
                seconds=int(rng.integers(0, 28800)),
            )
            if last_purchase_ts < start_ts:
                last_purchase_ts = start_ts + pd.Timedelta(hours=1)

            # Build historical purchase dates spanning up to last_purchase_ts
            if n_invoices == 1:
                inv_timestamps = [last_purchase_ts]
            else:
                earliest_possible = start_ts
                if spec[0] == "At-Risk":
                    earliest_possible = start_ts
                span_sec = max(3600, int((last_purchase_ts - earliest_possible).total_seconds()))
                offsets = np.sort(rng.integers(0, span_sec, size=n_invoices - 1))
                inv_timestamps = [
                    earliest_possible + pd.Timedelta(seconds=int(off)) for off in offsets
                ] + [last_purchase_ts]

            for inv_ts in inv_timestamps:
                inv_no = str(invoice_counter)
                invoice_counter += 1
                n_lines = int(rng.integers(l_min, l_max + 1))
                item_indices = rng.choice(
                    len(_CATALOG_ITEMS),
                    size=min(n_lines, len(_CATALOG_ITEMS)),
                    replace=False,
                )
                for idx in item_indices:
                    stock_code, desc, base_price = _CATALOG_ITEMS[int(idx)]
                    qty = int(rng.integers(q_min, q_max + 1))
                    unit_price = round(float(base_price * rng.uniform(0.95, 1.15)), 2)
                    rows.append(
                        {
                            "InvoiceNo": inv_no,
                            "StockCode": stock_code,
                            "Description": desc,
                            "Quantity": qty,
                            "InvoiceDate": inv_ts,
                            "UnitPrice": unit_price,
                            "CustomerID": cid,
                            "Country": country,
                        }
                    )

                # Inject occasional 'POST' / 'M' admin line items on valid invoices
                if rng.random() < 0.03:
                    admin_code = "POST" if rng.random() < 0.7 else "M"
                    admin_desc = "POSTAGE" if admin_code == "POST" else "Manual"
                    rows.append(
                        {
                            "InvoiceNo": inv_no,
                            "StockCode": admin_code,
                            "Description": admin_desc,
                            "Quantity": 1,
                            "InvoiceDate": inv_ts,
                            "UnitPrice": 15.0 if admin_code == "POST" else 25.0,
                            "CustomerID": cid,
                            "Country": country,
                        }
                    )

    # Inject realistic UCI anomalies
    n_valid = len(rows)

    # 1. Cancellation ('C') rows with negative Quantity
    n_cancel = max(15, int(n_valid * 0.02))
    cancel_indices = rng.integers(0, n_valid, size=n_cancel)
    for idx in cancel_indices:
        src_row = rows[int(idx)]
        rows.append(
            {
                "InvoiceNo": f"C{invoice_counter}",
                "StockCode": src_row["StockCode"],
                "Description": src_row["Description"],
                "Quantity": -abs(int(src_row["Quantity"])),
                "InvoiceDate": src_row["InvoiceDate"] + pd.Timedelta(hours=2),
                "UnitPrice": src_row["UnitPrice"],
                "CustomerID": src_row["CustomerID"],
                "Country": src_row["Country"],
            }
        )
        invoice_counter += 1

    # 2. Null CustomerID rows (guest / internal transactions)
    n_null_cid = max(30, int(n_valid * 0.08))
    for _ in range(n_null_cid):
        stock_code, desc, base_price = _CATALOG_ITEMS[int(rng.integers(0, len(_CATALOG_ITEMS)))]
        inv_ts = start_ts + pd.Timedelta(seconds=int(rng.integers(0, total_seconds)))
        rows.append(
            {
                "InvoiceNo": str(invoice_counter),
                "StockCode": stock_code,
                "Description": desc,
                "Quantity": int(rng.integers(1, 10)),
                "InvoiceDate": inv_ts,
                "UnitPrice": base_price,
                "CustomerID": np.nan,
                "Country": "United Kingdom",
            }
        )
        invoice_counter += 1

    # 3. Non-positive Quantity (without 'C') and Zero/Negative UnitPrice anomalies
    n_anomalies = max(8, int(n_valid * 0.006))
    for i in range(n_anomalies):
        stock_code, desc, base_price = _CATALOG_ITEMS[int(rng.integers(0, len(_CATALOG_ITEMS)))]
        inv_ts = start_ts + pd.Timedelta(seconds=int(rng.integers(0, total_seconds)))
        cid = float(customer_id_start + int(rng.integers(0, n_customers)) * 7)
        if i % 2 == 0:
            rows.append(
                {
                    "InvoiceNo": str(invoice_counter),
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": -int(rng.integers(1, 6)),
                    "InvoiceDate": inv_ts,
                    "UnitPrice": base_price,
                    "CustomerID": cid,
                    "Country": "United Kingdom",
                }
            )
        else:
            rows.append(
                {
                    "InvoiceNo": str(invoice_counter),
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": int(rng.integers(1, 5)),
                    "InvoiceDate": inv_ts,
                    "UnitPrice": 0.0,
                    "CustomerID": cid,
                    "Country": "United Kingdom",
                }
            )
        invoice_counter += 1

    # 4. Exact duplicate rows
    n_dups = max(10, int(n_valid * 0.01))
    dup_indices = rng.integers(0, n_valid, size=n_dups)
    for idx in dup_indices:
        rows.append(dict(rows[int(idx)]))

    df = pd.DataFrame(rows)
    df = _normalize_retail_dataframe(df)
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    df.attrs["data_source"] = "built_in_sample"
    return df


def load_or_download_retail_data(
    data_dir: str | Path,
    use_sample: bool = False,
) -> pd.DataFrame:
    """Load UCI Online Retail from local Parquet cache, download it, or fall back to sample.

    Args:
        data_dir: Directory path used to store and retrieve cached `.parquet` files.
        use_sample: When True, bypasses network/cache and returns the built-in multi-segment sample.

    Returns:
        Normalized transaction DataFrame with standard 8-column Online Retail schema.
    """
    if use_sample:
        return generate_rfm_retail_sample_data()

    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = target_dir / PARQUET_FILENAME

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            df = _normalize_retail_dataframe(df)
            df.attrs["data_source"] = "parquet_cache"
            df.attrs["cache_path"] = str(parquet_path)
            return df
        except Exception:
            pass

    # Check if user placed local Excel or CSV manually inside data_dir
    for candidate_name in ("Online Retail.xlsx", "online_retail.xlsx", "online_retail.csv"):
        local_path = target_dir / candidate_name
        if local_path.exists():
            try:
                if local_path.suffix.lower() == ".csv":
                    raw_df = pd.read_csv(local_path, encoding="latin1")
                else:
                    raw_df = pd.read_excel(local_path, engine="openpyxl")
                raw_df = _normalize_retail_dataframe(raw_df)
                raw_df.to_parquet(parquet_path, index=False)
                raw_df.attrs["data_source"] = "local_file_cached"
                raw_df.attrs["cache_path"] = str(parquet_path)
                return raw_df
            except Exception:
                pass

    # Attempt downloading full UCI Online Retail dataset
    try:
        import requests

        resp = requests.get(UCI_ONLINE_RETAIL_URL, timeout=30)
        resp.raise_for_status()
        excel_bytes = io.BytesIO(resp.content)
        raw_df = pd.read_excel(excel_bytes, engine="openpyxl")
        raw_df = _normalize_retail_dataframe(raw_df)
        raw_df.to_parquet(parquet_path, index=False)
        raw_df.attrs["data_source"] = "uci_download"
        raw_df.attrs["cache_path"] = str(parquet_path)
        return raw_df
    except Exception as exc:
        fallback_df = generate_rfm_retail_sample_data()
        fallback_df.attrs["data_source"] = "fallback_sample"
        fallback_df.attrs["fallback_reason"] = str(exc)
        return fallback_df


def load_uploaded_retail_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse user-uploaded CSV, Excel, or Parquet transaction data into normalized schema."""
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)
    if suffix == ".csv":
        try:
            df = pd.read_csv(buffer, encoding="utf-8")
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin1")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(buffer, engine="openpyxl")
    elif suffix == ".parquet":
        df = pd.read_parquet(buffer)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Please upload a .csv, .xlsx, or .parquet file."
        )
    df = _normalize_retail_dataframe(df)
    df.attrs["data_source"] = "custom_upload"
    df.attrs["filename"] = filename
    return df
