"""Data ingestion, UCI caching, and realistic sample generation for UCI Online Retail."""

from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

UCI_DATASET_URLS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
)

# Realistic UCI Online Retail product catalog: (StockCode, Description, UnitPrice)
_SAMPLE_CATALOG: list[tuple[str, str, float]] = [
    ("22661", "REGENCY TEA PLATE PINK", 1.65),
    ("22662", "REGENCY TEA PLATE GREEN", 1.65),
    ("22659", "REGENCY TEA PLATE ROSES", 1.65),
    ("22699", "ROSES REGENCY TEACUP AND SAUCER", 2.95),
    ("22697", "GREEN REGENCY TEACUP AND SAUCER", 2.95),
    ("22698", "PINK REGENCY TEACUP AND SAUCER", 2.95),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("85099B", "JUMBO BAG RED RETROSPOT", 2.08),
    ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("22411", "JUMBO SHOPPER VINTAGE RED PAISLEY", 2.08),
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("21733", "RED HANGING HEART T-LIGHT HOLDER", 2.95),
    ("22727", "ALARM CLOCK BAKELIKE RED", 3.75),
    ("22726", "ALARM CLOCK BAKELIKE GREEN", 3.75),
    ("22728", "ALARM CLOCK BAKELIKE PINK", 3.75),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("20727", "LUNCH BAG BLACK SKULL.", 1.65),
    ("20728", "LUNCH BAG CARS BLUE", 1.65),
    ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("47566", "PARTY BUNTING", 4.95),
    ("23298", "SPOTTY BUNTING", 4.95),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("84991", "60 TEATIME FAIRY CAKE CASES", 0.55),
    ("21977", "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("22961", "JAM MAKING SET PRINTED", 1.45),
    ("22111", "SCOTTIE DOG HOT WATER BOTTLE", 4.95),
    ("22112", "CHOCOLATE HOT WATER BOTTLE", 4.95),
    ("21485", "RETROSPOT HEART HOT WATER BOTTLE", 4.95),
    ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
    ("22469", "HEART OF WICKER SMALL", 1.65),
    ("22470", "HEART OF WICKER LARGE", 2.95),
    ("82482", "WOODEN PICTURE FRAME WHITE FINISH", 2.55),
    ("82494L", "WOODEN FRAME ANTIQUE WHITE", 2.95),
    ("21790", "VINTAGE SNAP CARDS", 0.85),
    ("22178", "VICTORIAN GLASS HANGING T-LIGHT", 1.25),
    ("23298", "SPOTTY BUNTING", 4.95),
    ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
    ("20726", "LUNCH BAG WOODLAND", 1.65),
    ("22629", "SPACEBOY LUNCH BOX", 1.95),
    ("22630", "DOLLY GIRL LUNCH BOX", 1.95),
    ("22554", "PLASTERS IN TIN WOODLAND ANIMALS", 1.65),
    ("22551", "PLASTERS IN TIN SPACEBOY", 1.65),
    ("22556", "PLASTERS IN TIN CIRCUS PARADE", 1.65),
    ("21928", "JUMBO BAG SCANDINAVIAN BLUE PAISLEY", 2.08),
    ("21929", "JUMBO BAG PINK VINTAGE PAISLEY", 2.08),
    ("22385", "JUMBO BAG SPACEBOY DESIGN", 2.08),
    ("85099C", "JUMBO BAG BAROQUE BLACK WHITE", 2.08),
    ("85099F", "JUMBO BAG STRAWBERRY", 2.08),
    ("22467", "GUMBALL COAT RACK", 2.55),
    ("22468", "FAIRY SOAP ROLL", 0.85),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
    ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
    ("22750", "FELTCRAFT PRINCESS LOLA DOLL", 3.75),
    ("22751", "FELTCRAFT PRINCESS OLIVIA DOLL", 3.75),
    ("22197", "POPCORN HOLDER", 0.85),
    ("22198", "LARGE POPCORN HOLDER", 1.65),
    ("84946", "ANTIQUE SILVER T-LIGHT GLASS", 1.25),
    ("21137", "BLACK RECORD COVER FRAME", 3.75),
    ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95),
    ("22139", "RETROSPOT TEA SET CERAMIC 11 PC", 4.95),
    ("22149", "FELTCRAFT 6 FLOWER FRIENDS", 2.10),
    ("22150", "3 STRIPEY MICE FELTCRAFT", 1.95),
    ("22616", "PACK OF 12 LONDON TISSUES", 0.39),
    ("22617", "BAKING SET SPACEBOY DESIGN", 4.95),
    ("22624", "IVORY KITCHEN SCALES", 8.50),
    ("22625", "RED KITCHEN SCALES", 8.50),
    ("22627", "MINT KITCHEN SCALES", 8.50),
    ("21080", "SET/20 RED RETROSPOT PAPER NAPKINS", 0.85),
    ("21086", "SET/6 RED SPOTTY PAPER CUPS", 0.65),
    ("21094", "SET/6 RED SPOTTY PAPER PLATES", 0.85),
    ("22355", "CHARLOTTE BAG SUKI DESIGN", 0.85),
    ("22356", "CHARLOTTE BAG PINK POLKADOT", 0.85),
    ("20719", "WOODLAND CHARLOTTE BAG", 0.85),
    ("20724", "RED RETROSPOT CHARLOTTE BAG", 0.85),
    ("22910", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95),
    ("22722", "SET OF 6 SPICE TINS PANTRY DESIGN", 3.95),
    ("22725", "ALARM CLOCK BAKELIKE CHOCOLATE", 3.75),
    ("22729", "ALARM CLOCK BAKELIKE ORANGE", 3.75),
    ("22730", "ALARM CLOCK BAKELIKE IVORY", 3.75),
    ("21175", "GIN + TONIC DIET METAL SIGN", 2.55),
    ("21181", "PLEASE ONE PERSON METAL SIGN", 2.10),
    ("82583", "HOT BATHS METAL SIGN", 2.10),
    ("22413", "METAL SIGN TAKE IT OR LEAVE IT", 2.95),
    ("22077", "6 RIBBONS RUSTIC CHARM", 1.65),
    ("22078", "RIBBON REEL LACE DESIGN", 2.10),
    ("22079", "RIBBON REEL HEARTS DESIGN", 1.65),
    ("22666", "RECIPE BOX PANTRY YELLOW DESIGN", 2.95),
    ("22667", "RECIPE BOX RETROSPOT", 2.95),
    ("22668", "PINK BABY BUNTING", 2.95),
    ("22909", "SET OF 20 VINTAGE CHRISTMAS NAPKINS", 0.85),
    ("22632", "HAND WARMER RED RETROSPOT", 2.10),
    ("22633", "HAND WARMER UNION JACK", 2.10),
    ("22865", "HAND WARMER OWL DESIGN", 2.10),
    ("22866", "HAND WARMER SCOTTY DOG DESIGN", 2.10),
    ("22867", "HAND WARMER BIRD DESIGN", 2.10),
    ("22577", "WOODEN HEART CHRISTMAS SCANDINAVIAN", 0.85),
    ("22578", "WOODEN STAR CHRISTMAS SCANDINAVIAN", 0.85),
    ("22579", "WOODEN TREE CHRISTMAS SCANDINAVIAN", 0.85),
    ("21914", "BLUE HARMONICA IN BOX", 1.25),
    ("21915", "RED HARMONICA IN BOX", 1.25),
    ("22489", "PACK OF 12 TRADITIONAL CRAYONS", 0.42),
    ("22492", "MINI PAINT SET VINTAGE", 0.65),
    ("22558", "CLOTHES PEGS RETROSPOT PACK 24", 1.65),
    ("22560", "TRADITIONAL MODELLING CLAY", 1.25),
    ("21166", "COOK WITH WINE METAL SIGN", 2.08),
    ("22090", "PAPER BUNTING RETROSPOT", 2.95),
    ("22114", "HOT WATER BOTTLE TEA AND SYMPATHY", 4.25),
    ("22835", "HOT WATER BOTTLE I AM SO POORLY", 4.95),
    ("22993", "SET OF 4 PANTRY JELLY MOULDS", 1.25),
    ("23201", "JUMBO BAG ALPHABET", 2.08),
    ("23202", "JUMBO BAG VINTAGE LEAF", 2.08),
    ("23203", "JUMBO BAG VINTAGE DOILY", 2.08),
    ("23206", "LUNCH BAG APPLE DESIGN", 1.65),
    ("23209", "LUNCH BAG VINTAGE DOILY", 1.65),
    ("23240", "SET OF 4 KNICKERBOCKER GLORY TINS", 4.15),
    ("23245", "SET OF 3 REGENCY CAKE TINS", 4.95),
    ("23284", "DOORMAT KEEP CALM AND COME IN", 8.25),
    ("22690", "DOORMAT HOME SWEET HOME BLUE", 7.95),
    ("21523", "DOORMAT FANCY FONT HOME SWEET HOME", 7.95),
    ("21524", "DOORMAT SPOTTY HOME SWEET HOME", 7.95),
    ("22417", "PACK OF 60 SPACEBOY CAKE CASES", 0.55),
    ("22951", "60 CAKE CASES DOLLY GIRL DESIGN", 0.55),
    ("21980", "PACK OF 12 RED RETROSPOT TISSUES", 0.39),
    ("21981", "PACK OF 12 WOODLAND TISSUES", 0.39),
    ("22502", "PICNIC BASKET WICKER SMALL", 5.95),
    ("22501", "PICNIC BASKET WICKER LARGE", 9.95),
    ("22988", "SOLDIERS EGG CUP", 1.25),
    ("22989", "SET 2 PANTRY DESIGN TEA TOWELS", 3.25),
    ("22990", "COTTON APRON PANTRY DESIGN", 4.95),
]

_ADMIN_SAMPLE_ROWS: list[tuple[str, str, float]] = [
    ("POST", "POSTAGE", 18.00),
    ("DOT", "DOTCOM POSTAGE", 42.50),
    ("M", "Manual", 15.00),
    ("BANK CHARGES", "Bank Charges", 15.00),
    ("CRUK", "CRUK Commission", 2.50),
]

# Affinity clusters with anchor probabilities and tight conditional co-occurrence
_AFFINITY_CLUSTERS: list[tuple[list[str], float, float]] = [
    # (cluster descriptions, cluster trigger weight, conditional follow-up prob)
    (["REGENCY TEA PLATE PINK", "REGENCY TEA PLATE GREEN", "REGENCY TEA PLATE ROSES"], 0.14, 0.84),
    (["PINK REGENCY TEACUP AND SAUCER", "GREEN REGENCY TEACUP AND SAUCER", "ROSES REGENCY TEACUP AND SAUCER"], 0.14, 0.82),
    (["JUMBO BAG RED RETROSPOT", "JUMBO BAG PINK POLKADOT", "JUMBO STORAGE BAG SUKI", "JUMBO SHOPPER VINTAGE RED PAISLEY"], 0.16, 0.72),
    (["WHITE HANGING HEART T-LIGHT HOLDER", "RED HANGING HEART T-LIGHT HOLDER"], 0.18, 0.65),
    (["ALARM CLOCK BAKELIKE RED", "ALARM CLOCK BAKELIKE GREEN", "ALARM CLOCK BAKELIKE PINK"], 0.12, 0.78),
    (["LUNCH BAG RED RETROSPOT", "LUNCH BAG BLACK SKULL.", "LUNCH BAG CARS BLUE", "LUNCH BAG SPACEBOY DESIGN"], 0.14, 0.66),
    (["HEART OF WICKER SMALL", "HEART OF WICKER LARGE"], 0.10, 0.74),
    (["JAM MAKING SET PRINTED", "JAM MAKING SET WITH JARS"], 0.09, 0.72),
    (["PACK OF 72 RETROSPOT CAKE CASES", "60 TEATIME FAIRY CAKE CASES", "PACK OF 60 PINK PAISLEY CAKE CASES"], 0.11, 0.68),
    (["WOODEN PICTURE FRAME WHITE FINISH", "WOODEN FRAME ANTIQUE WHITE"], 0.09, 0.70),
    (["SCOTTIE DOG HOT WATER BOTTLE", "CHOCOLATE HOT WATER BOTTLE", "RETROSPOT HEART HOT WATER BOTTLE"], 0.08, 0.68),
]


def _normalize_retail_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw UCI Online Retail columns to Parquet-safe canonical types."""
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    out = out[REQUIRED_COLUMNS].copy()
    out["InvoiceNo"] = out["InvoiceNo"].where(out["InvoiceNo"].notna(), None)
    out["InvoiceNo"] = out["InvoiceNo"].map(lambda x: None if pd.isna(x) else str(x).strip())
    out["StockCode"] = out["StockCode"].where(out["StockCode"].notna(), None)
    out["StockCode"] = out["StockCode"].map(lambda x: None if pd.isna(x) else str(x).strip())
    out["Description"] = out["Description"].where(out["Description"].notna(), None)
    out["Description"] = out["Description"].map(lambda x: None if pd.isna(x) else str(x))
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0).astype("int64")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce").astype("float64")
    out["CustomerID"] = pd.to_numeric(out["CustomerID"], errors="coerce").astype("float64")
    out["Country"] = out["Country"].fillna("Unspecified").astype(str).str.strip()
    return out


def generate_retail_sample_data(
    n_invoices: int = 2500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic UCI Online Retail benchmark dataset with anomalies and strong bundles.

    Args:
        n_invoices: Number of synthetic invoices to generate (including cancellations).
        random_state: Deterministic NumPy seed.

    Returns:
        DataFrame matching the 8-column UCI Online Retail schema.
    """
    rng = np.random.default_rng(random_state)
    catalog_by_desc = {desc: (code, price) for code, desc, price in _SAMPLE_CATALOG}
    all_descs = [desc for _, desc, _ in _SAMPLE_CATALOG]

    countries = [
        "United Kingdom",
        "Germany",
        "France",
        "EIRE",
        "Spain",
        "Netherlands",
        "Switzerland",
        "Australia",
    ]
    country_probs = [0.87, 0.03, 0.03, 0.02, 0.015, 0.015, 0.01, 0.01]

    start_ts = pd.Timestamp("2010-12-01 08:26:00")
    span_seconds = int((pd.Timestamp("2011-12-09 12:50:00") - start_ts).total_seconds())

    customer_pool = np.arange(12346, 18287, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    base_invoice_num = 536365

    for idx in range(n_invoices):
        inv_num = base_invoice_num + idx
        is_cancelled = rng.random() < 0.038
        invoice_no = f"C{inv_num}" if is_cancelled else str(inv_num)

        inv_date = start_ts + pd.Timedelta(seconds=int(rng.integers(0, span_seconds)))
        country = str(rng.choice(countries, p=country_probs))

        # Approximate ~24.9% missing CustomerID at invoice level (matching UCI Online Retail)
        has_customer = rng.random() >= 0.249
        customer_id = float(rng.choice(customer_pool)) if has_customer else np.nan

        # Determine basket profile: single-item, standard multi-item, or wholesale large basket
        basket_profile = rng.random()
        if basket_profile < 0.12:
            target_size = 1
        elif basket_profile < 0.58:
            target_size = int(rng.integers(2, 6))
        elif basket_profile < 0.90:
            target_size = int(rng.integers(6, 15))
        else:
            target_size = int(rng.integers(16, 25))

        selected_items: list[str] = []
        # Inject structured co-occurrence bundles for multi-item baskets
        if target_size >= 2:
            for cluster_items, trigger_p, followup_p in _AFFINITY_CLUSTERS:
                if rng.random() < trigger_p:
                    anchor = str(rng.choice(cluster_items))
                    if anchor not in selected_items:
                        selected_items.append(anchor)
                    for partner in cluster_items:
                        if partner != anchor and len(selected_items) < target_size and rng.random() < followup_p:
                            selected_items.append(partner)
                    # Cross-cluster affinity: REGENCY TEA PLATE + REGENCY CAKESTAND 3 TIER
                    if (
                        "REGENCY TEA PLATE" in anchor
                        and "REGENCY CAKESTAND 3 TIER" not in selected_items
                        and len(selected_items) < target_size
                        and rng.random() < 0.55
                    ):
                        selected_items.append("REGENCY CAKESTAND 3 TIER")
                    break

        while len(selected_items) < target_size:
            candidate = str(rng.choice(all_descs))
            if candidate not in selected_items:
                selected_items.append(candidate)

        for item_desc in selected_items:
            stock_code, unit_price = catalog_by_desc[item_desc]

            # Realistic wholesale vs retail quantity
            if basket_profile >= 0.90:
                qty = int(rng.choice([6, 12, 24, 36, 48]))
            else:
                qty = int(rng.choice([1, 2, 3, 4, 6, 8, 12], p=[0.28, 0.25, 0.14, 0.12, 0.10, 0.06, 0.05]))

            if is_cancelled:
                qty = -abs(qty)

            # Inject ~0.6% missing Description, ~0.8% non-positive Qty/Price, ~2.0% casing/label drift
            desc_val: Any = item_desc
            price_val = unit_price

            anomaly_roll = rng.random()
            if not is_cancelled and anomaly_roll < 0.006:
                desc_val = None
            elif not is_cancelled and anomaly_roll < 0.011:
                qty = 0
            elif not is_cancelled and anomaly_roll < 0.016:
                price_val = 0.0
            elif anomaly_roll < 0.024:
                # Simulate slight StockCode description drift (e.g. typo/suffix) for canonicalization testing
                desc_val = f"{item_desc} (VARIANT)"
            elif anomaly_roll < 0.036:
                # Simulate noisy whitespace/lowercase variants per StockCode
                desc_val = f"  {item_desc.lower()}  "

            row = {
                "InvoiceNo": invoice_no,
                "StockCode": stock_code,
                "Description": desc_val,
                "Quantity": qty,
                "InvoiceDate": inv_date,
                "UnitPrice": price_val,
                "CustomerID": customer_id,
                "Country": country,
            }
            rows.append(row)

            # Inject ~0.7% exact duplicate line-items
            if rng.random() < 0.007:
                rows.append(dict(row))

        # Inject administrative non-product lines (POST, DOT, M, BANK CHARGES, CRUK) on ~4% of invoices
        if rng.random() < 0.04:
            admin_code, admin_desc, admin_price = _ADMIN_SAMPLE_ROWS[int(rng.integers(0, len(_ADMIN_SAMPLE_ROWS)))]
            rows.append(
                {
                    "InvoiceNo": invoice_no,
                    "StockCode": admin_code,
                    "Description": admin_desc,
                    "Quantity": -1 if is_cancelled else 1,
                    "InvoiceDate": inv_date,
                    "UnitPrice": admin_price,
                    "CustomerID": customer_id,
                    "Country": country,
                }
            )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df = _normalize_retail_schema(df)
    df.attrs["data_source"] = "sample"
    return df


def _download_uci_dataframe(timeout_sec: int = 20) -> pd.DataFrame:
    """Download the official UCI Online Retail dataset from repository mirrors."""
    last_error: Exception | None = None
    for url in UCI_DATASET_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MBA-FPGrowth/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                payload = resp.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    xlsx_names = [name for name in zf.namelist() if name.lower().endswith(".xlsx")]
                    csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                    if xlsx_names:
                        with zf.open(xlsx_names[0]) as fh:
                            raw_df = pd.read_excel(fh, engine="openpyxl")
                    elif csv_names:
                        with zf.open(csv_names[0]) as fh:
                            raw_df = pd.read_csv(fh, encoding="latin1")
                    else:
                        raise ValueError("Archive did not contain an .xlsx or .csv dataset file.")
            else:
                raw_df = pd.read_excel(io.BytesIO(payload), engine="openpyxl")

            return _normalize_retail_schema(raw_df)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, zipfile.BadZipFile) as exc:
            last_error = exc
            logger.warning("Failed downloading UCI dataset from %s: %s", url, exc)

    raise RuntimeError(f"Unable to download UCI Online Retail dataset: {last_error}")


def load_or_download_retail_data(
    data_dir: str | Path,
    use_sample: bool = False,
    n_sample_invoices: int = 2500,
) -> pd.DataFrame:
    """Load UCI Online Retail dataset from Parquet cache, download it, or fall back to sample data.

    Args:
        data_dir: Directory where Parquet caches and local files are stored.
        use_sample: If True, immediately return the built-in UCI benchmark sample dataset.
        n_sample_invoices: Invoice count used when generating the built-in benchmark sample.

    Returns:
        Standardized 8-column DataFrame with provenance metadata stored in ``df.attrs``.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if use_sample:
        sample_df = generate_retail_sample_data(n_invoices=n_sample_invoices)
        sample_df.attrs["data_source"] = "sample"
        return sample_df

    parquet_cache = data_path / "online_retail.parquet"
    if parquet_cache.exists():
        cached_df = pd.read_parquet(parquet_cache)
        cached_df = _normalize_retail_schema(cached_df)
        cached_df.attrs["data_source"] = "uci_parquet_cache"
        return cached_df

    # Check if user placed local Excel/CSV inside data_dir
    for local_name in ("Online Retail.xlsx", "online_retail.xlsx", "online_retail.csv"):
        local_file = data_path / local_name
        if local_file.exists():
            if local_file.suffix.lower() == ".xlsx":
                local_df = pd.read_excel(local_file, engine="openpyxl")
            else:
                local_df = pd.read_csv(local_file, encoding="latin1")
            local_df = _normalize_retail_schema(local_df)
            local_df.to_parquet(parquet_cache, index=False)
            local_df.attrs["data_source"] = "uci_local_cached"
            return local_df

    try:
        downloaded_df = _download_uci_dataframe()
        downloaded_df.to_parquet(parquet_cache, index=False)
        downloaded_df.attrs["data_source"] = "uci_downloaded"
        return downloaded_df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to built-in UCI benchmark sample: %s", exc)
        fallback_df = generate_retail_sample_data(n_invoices=n_sample_invoices)
        fallback_df.attrs["data_source"] = "sample_fallback"
        fallback_df.attrs["fallback_reason"] = str(exc)
        return fallback_df


def load_uploaded_retail_file(uploaded_file: Any) -> pd.DataFrame:
    """Read and validate a user-uploaded CSV or Excel retail transactions file."""
    filename = str(getattr(uploaded_file, "name", "")).lower()
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raw_df = pd.read_csv(uploaded_file, encoding_errors="replace")

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    # Provide safe defaults only for optional enrichment columns if missing
    if "InvoiceDate" not in raw_df.columns:
        raw_df["InvoiceDate"] = pd.Timestamp("2011-06-01")
    if "CustomerID" not in raw_df.columns:
        raw_df["CustomerID"] = np.nan
    if "Country" not in raw_df.columns:
        raw_df["Country"] = "United Kingdom"

    df = _normalize_retail_schema(raw_df)
    df.attrs["data_source"] = "uploaded_file"
    return df
