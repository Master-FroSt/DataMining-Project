"""Dataset acquisition, Parquet caching, and realistic synthetic fallback for UCI Bank Marketing."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_VARIANTS = ("bank-full", "bank-additional-full")

UCI_ARCHIVE_URLS = (
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
)

BANK_FULL_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]

BANK_ADDITIONAL_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]


def normalize_variant(variant: str) -> str:
    """Normalize and validate dataset variant identifier."""
    cleaned = str(variant).strip().lower().removesuffix(".csv")
    if cleaned not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported dataset variant '{variant}'. Expected one of {SUPPORTED_VARIANTS}."
        )
    return cleaned


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _calibrate_intercept(logit_base: np.ndarray, target_rate: float) -> float:
    """Find scalar intercept shifting mean sigmoid probability to target_rate."""
    low, high = -12.0, 12.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        mean_p = float(np.mean(_sigmoid(logit_base + mid)))
        if mean_p < target_rate:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def generate_bank_sample_data(
    variant: str = "bank-full",
    n_rows: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic UCI-schema Bank Marketing benchmark sample.

    Preserves ~11.7% positive rate (11.3% for bank-additional-full),
    pdays sentinels (-1 or 999), 'unknown' categories, and post-call
    'duration' so the anti-leakage guard can be verified cleanly.
    """
    resolved = normalize_variant(variant)
    rng = np.random.default_rng(random_state)

    jobs = np.array(
        [
            "management",
            "blue-collar",
            "technician",
            "admin.",
            "services",
            "retired",
            "self-employed",
            "entrepreneur",
            "unemployed",
            "housemaid",
            "student",
            "unknown",
        ]
    )
    job_probs = np.array([0.21, 0.21, 0.17, 0.11, 0.09, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01])
    job = rng.choice(jobs, size=n_rows, p=job_probs)

    age = np.clip(rng.normal(loc=40.5, scale=10.5, size=n_rows), 18, 92).astype(int)
    age = np.where(job == "retired", np.clip(rng.normal(66, 6, n_rows), 55, 92).astype(int), age)
    age = np.where(job == "student", np.clip(rng.normal(23, 3, n_rows), 18, 32).astype(int), age)

    marital = rng.choice(["married", "single", "divorced"], size=n_rows, p=[0.60, 0.28, 0.12])
    default = rng.choice(["no", "yes"], size=n_rows, p=[0.982, 0.018])
    housing = rng.choice(["yes", "no"], size=n_rows, p=[0.555, 0.445])
    loan = rng.choice(["no", "yes"], size=n_rows, p=[0.84, 0.16])
    campaign = np.clip(rng.geometric(p=0.45, size=n_rows), 1, 25).astype(int)

    if resolved == "bank-full":
        education = rng.choice(
            ["secondary", "tertiary", "primary", "unknown"],
            size=n_rows,
            p=[0.51, 0.30, 0.15, 0.04],
        )
        # Skewed balance distribution with negative overdrafts and high-balance tails
        raw_exp = rng.exponential(scale=1250.0, size=n_rows)
        overdraft = rng.binomial(1, 0.08, size=n_rows) * rng.uniform(50.0, 2200.0, size=n_rows)
        balance = np.round(raw_exp - overdraft, 0).astype(int)

        contact = rng.choice(["cellular", "unknown", "telephone"], size=n_rows, p=[0.648, 0.288, 0.064])
        day = rng.integers(1, 32, size=n_rows)
        month = rng.choice(
            ["may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "oct", "sep", "mar", "dec"],
            size=n_rows,
            p=[0.30, 0.15, 0.14, 0.12, 0.09, 0.07, 0.06, 0.03, 0.015, 0.012, 0.008, 0.005],
        )

        # Prior campaign history (~81.7% never contacted before -> pdays = -1)
        never_before = rng.random(size=n_rows) < 0.817
        pdays = np.where(never_before, -1, rng.integers(5, 450, size=n_rows)).astype(int)
        previous = np.where(never_before, 0, np.clip(rng.geometric(p=0.52, size=n_rows), 1, 12)).astype(int)
        poutcome = np.where(
            never_before,
            "unknown",
            rng.choice(["failure", "other", "success"], size=n_rows, p=[0.59, 0.22, 0.19]),
        )

        # Pre-call predictive latent logit (no duration leakage)
        logit = (
            0.012 * (age - 40.0)
            + 0.00018 * np.clip(balance, -2000, 12000)
            - 0.55 * (housing == "yes").astype(float)
            - 0.42 * (loan == "yes").astype(float)
            - 0.65 * (default == "yes").astype(float)
            - 0.72 * (contact == "unknown").astype(float)
            + 0.22 * (contact == "cellular").astype(float)
            - 0.14 * np.minimum(campaign - 1, 8)
            + 0.62 * np.isin(job, ["retired", "student"]).astype(float)
            + 0.22 * (job == "management").astype(float)
            - 0.28 * (job == "blue-collar").astype(float)
            + 1.25 * np.isin(month, ["mar", "sep", "oct", "dec"]).astype(float)
            + 0.45 * (month == "apr").astype(float)
            - 0.48 * (month == "may").astype(float)
            + 2.15 * (poutcome == "success").astype(float)
            + 0.35 * (poutcome == "failure").astype(float)
            + 0.45 * ((age >= 58) & (balance > 1200)).astype(float)
            + 0.50 * ((~never_before) & (pdays <= 180) & (campaign <= 2)).astype(float)
        )
        intercept = _calibrate_intercept(logit, target_rate=0.117)
        probs = _sigmoid(logit + intercept)
        y_bin = rng.binomial(1, probs)
        y = np.where(y_bin == 1, "yes", "no")

        # Post-call duration strongly correlated with y (for anti-leakage guard validation)
        duration = np.where(
            y_bin == 1,
            np.clip(rng.gamma(shape=3.2, scale=170.0, size=n_rows), 90, 3200),
            np.clip(rng.gamma(shape=1.6, scale=110.0, size=n_rows), 10, 1400),
        ).astype(int)

        df = pd.DataFrame(
            {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
                "y": y,
            }
        )
        df.attrs["data_source"] = "Built-in UCI Benchmark Sample (bank-full schema)"
        return df

    # bank-additional-full schema (21 attributes, pdays=999 sentinel + macro indicators)
    education = rng.choice(
        [
            "university.degree",
            "high.school",
            "basic.9y",
            "professional.course",
            "basic.4y",
            "basic.6y",
            "unknown",
            "illiterate",
        ],
        size=n_rows,
        p=[0.30, 0.23, 0.15, 0.13, 0.10, 0.05, 0.038, 0.002],
    )
    contact = rng.choice(["cellular", "telephone"], size=n_rows, p=[0.635, 0.365])
    month = rng.choice(
        ["may", "jul", "aug", "jun", "nov", "apr", "oct", "sep", "mar", "dec"],
        size=n_rows,
        p=[0.33, 0.17, 0.15, 0.13, 0.10, 0.06, 0.018, 0.015, 0.014, 0.013],
    )
    day_of_week = rng.choice(["mon", "tue", "wed", "thu", "fri"], size=n_rows)

    # Macroeconomic regimes mimicking 2008-2010 European telemarketing cycle
    regime_idx = rng.choice([0, 1, 2, 3], size=n_rows, p=[0.42, 0.30, 0.18, 0.10])
    regime_emp = np.array([1.2, 0.2, -1.8, -3.0])[regime_idx] + rng.normal(0, 0.2, n_rows)
    regime_cpi = np.array([93.95, 93.44, 92.89, 92.43])[regime_idx] + rng.normal(0, 0.15, n_rows)
    regime_cci = np.array([-39.5, -41.8, -46.2, -31.4])[regime_idx] + rng.normal(0, 1.8, n_rows)
    regime_euribor = np.array([4.86, 4.12, 1.31, 0.78])[regime_idx] + rng.normal(0, 0.18, n_rows)
    regime_employed = np.array([5208.0, 5191.0, 5099.1, 5017.5])[regime_idx] + rng.normal(0, 12.0, n_rows)

    emp_var_rate = np.round(np.clip(regime_emp, -3.4, 1.4), 1)
    cons_price_idx = np.round(np.clip(regime_cpi, 92.201, 94.767), 3)
    cons_conf_idx = np.round(np.clip(regime_cci, -50.8, -26.9), 1)
    euribor3m = np.round(np.clip(regime_euribor, 0.634, 5.045), 3)
    nr_employed = np.round(np.clip(regime_employed, 4963.6, 5228.1), 1)

    never_before = rng.random(size=n_rows) < 0.963
    pdays = np.where(never_before, 999, rng.integers(1, 28, size=n_rows)).astype(int)
    has_previous_only = (~never_before) | (rng.random(size=n_rows) < 0.10)
    previous = np.where(has_previous_only, np.clip(rng.geometric(p=0.6, size=n_rows), 1, 7), 0).astype(int)
    poutcome = np.where(
        previous == 0,
        "nonexistent",
        np.where(~never_before, "success", rng.choice(["failure", "success"], size=n_rows, p=[0.85, 0.15])),
    )

    logit = (
        -0.42 * emp_var_rate
        - 0.35 * (euribor3m - 2.5)
        + 0.04 * (cons_conf_idx + 40.0)
        + 0.38 * (contact == "cellular").astype(float)
        - 0.15 * np.minimum(campaign - 1, 8)
        + 0.55 * np.isin(job, ["retired", "student"]).astype(float)
        + 0.95 * np.isin(month, ["mar", "sep", "oct", "dec"]).astype(float)
        + 1.95 * (poutcome == "success").astype(float)
        + 0.25 * (poutcome == "failure").astype(float)
        - 0.22 * (housing == "yes").astype(float)
        - 0.18 * (loan == "yes").astype(float)
        + 0.45 * ((euribor3m < 1.5) & (age >= 50)).astype(float)
    )
    intercept = _calibrate_intercept(logit, target_rate=0.1127)
    probs = _sigmoid(logit + intercept)
    y_bin = rng.binomial(1, probs)
    y = np.where(y_bin == 1, "yes", "no")

    duration = np.where(
        y_bin == 1,
        np.clip(rng.gamma(shape=3.1, scale=165.0, size=n_rows), 85, 3200),
        np.clip(rng.gamma(shape=1.6, scale=105.0, size=n_rows), 8, 1300),
    ).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
            "y": y,
        }
    )
    df.attrs["data_source"] = "Built-in UCI Benchmark Sample (bank-additional-full schema)"
    return df


def _extract_target_csv_from_zip_bytes(raw_bytes: bytes, target_filename: str) -> pd.DataFrame | None:
    """Recursively locate and read semicolon-delimited target CSV inside UCI zip archives."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        for member in zf.namelist():
            member_lower = member.lower()
            if member_lower.endswith(f"/{target_filename}") or member_lower == target_filename:
                with zf.open(member) as f:
                    return pd.read_csv(f, sep=";")
        for member in zf.namelist():
            if member.lower().endswith(".zip"):
                nested_bytes = zf.read(member)
                nested_df = _extract_target_csv_from_zip_bytes(nested_bytes, target_filename)
                if nested_df is not None:
                    return nested_df
    return None


def load_or_download_bank_data(
    data_dir: str | Path,
    variant: str = "bank-full",
    use_sample: bool = False,
) -> pd.DataFrame:
    """Load Bank Marketing dataset from Parquet cache or UCI repository, falling back to sample data.

    Parameters
    ----------
    data_dir : str | Path
        Local directory for `.parquet` cache files.
    variant : str
        Either `"bank-full"` (45,211 rows, 17 attrs) or `"bank-additional-full"` (41,188 rows, 21 attrs).
    use_sample : bool
        When True, directly return the built-in benchmark sample dataset without network I/O.
    """
    resolved = normalize_variant(variant)
    if use_sample:
        return generate_bank_sample_data(variant=resolved)

    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{resolved}.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.attrs["data_source"] = f"Local Parquet Cache ({parquet_path.name})"
        return df

    target_csv = f"{resolved}.csv"
    for url in UCI_ARCHIVE_URLS:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (BankTelemarketingDSS/1.0)"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                archive_bytes = response.read()
            df = _extract_target_csv_from_zip_bytes(archive_bytes, target_csv)
            if df is not None and not df.empty:
                df.to_parquet(parquet_path, index=False)
                df.attrs["data_source"] = f"UCI Bank Marketing Official Download ({target_csv})"
                return df
        except Exception:
            continue

    fallback_df = generate_bank_sample_data(variant=resolved)
    fallback_df.attrs["data_source"] = f"Built-in Benchmark Sample Fallback ({resolved} unavailable offline)"
    return fallback_df
