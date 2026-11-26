"""Pre-call anti-leakage preprocessing, sentinel engineering, and model-specific pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

LEAKAGE_COLUMNS = ("duration",)
BINARY_ENGINEERED_COLUMNS = ("never_contacted_before",)


def audit_and_remove_leakage(
    df: pd.DataFrame,
    variant: str = "bank-full",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enforce pre-call data hygiene and engineer sentinel features.

    Steps performed:
    1. Drop post-call leakage columns (specifically `duration` if present).
    2. Engineer `never_contacted_before` (1 when `pdays` is -1, 999, or missing; else 0).
    3. Replace sentinel `pdays` values (-1 / 999) with neutral non-negative 0.0 so linear
       coefficients and scalers are not distorted.
    4. Preserve `"unknown"` / `"nonexistent"` as valid categorical levels.
    5. Encode target column `y` (`yes -> 1`, `no -> 0`) when present.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    clean_df = df.copy()
    raw_rows = int(len(clean_df))
    raw_columns = int(clean_df.shape[1])

    # 1. Strict anti-leakage guard: drop duration (case-insensitive safety check)
    dropped_columns: list[str] = []
    for col in list(clean_df.columns):
        if col.strip().lower() in LEAKAGE_COLUMNS:
            clean_df = clean_df.drop(columns=[col])
            dropped_columns.append(col)
    dropped_duration = len(dropped_columns) > 0

    # 2. Sentinel handling for pdays (-1 in bank-full, 999 in bank-additional-full)
    sentinel_pdays_count = 0
    if "pdays" in clean_df.columns:
        pdays_numeric = pd.to_numeric(clean_df["pdays"], errors="coerce")
        if "never_contacted_before" in clean_df.columns:
            existing_flag = pd.to_numeric(clean_df["never_contacted_before"], errors="coerce").fillna(0) > 0
            is_sentinel = existing_flag | pdays_numeric.isna() | (pdays_numeric < 0) | (pdays_numeric >= 999)
        else:
            is_sentinel = pdays_numeric.isna() | (pdays_numeric < 0) | (pdays_numeric >= 999)
        sentinel_pdays_count = int(is_sentinel.sum())
        clean_df["never_contacted_before"] = is_sentinel.astype(int)
        clean_df["pdays"] = pdays_numeric.where(~is_sentinel, 0.0).fillna(0.0).astype(float)
    elif "never_contacted_before" in clean_df.columns:
        clean_df["never_contacted_before"] = (
            pd.to_numeric(clean_df["never_contacted_before"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # 3. Clean categorical columns while explicitly preserving "unknown"
    unknown_counts: dict[str, int] = {}
    for col in clean_df.columns:
        if col == "y":
            continue
        if pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(clean_df[col]) or isinstance(
            clean_df[col].dtype, pd.CategoricalDtype
        ):
            series = clean_df[col].astype("object").where(clean_df[col].notna(), "unknown")
            series = series.astype(str).str.strip()
            series = series.replace({"": "unknown", "nan": "unknown", "None": "unknown", "NA": "unknown"})
            clean_df[col] = series
            unk_n = int((series == "unknown").sum())
            if unk_n > 0:
                unknown_counts[col] = unk_n

    # 4. Target encoding if present
    has_target = "y" in clean_df.columns
    positive_count: int | None = None
    negative_count: int | None = None
    positive_rate: float | None = None

    if has_target:
        raw_y = clean_df["y"]
        if pd.api.types.is_numeric_dtype(raw_y) and not pd.api.types.is_bool_dtype(raw_y):
            y_series = pd.to_numeric(raw_y, errors="raise").astype(int)
        else:
            y_str = raw_y.astype(str).str.strip().str.lower()
            mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
            mapped = y_str.map(mapping)
            if mapped.isna().any():
                invalid_vals = sorted(y_str[mapped.isna()].unique().tolist())
                raise ValueError(f"Unsupported target values in column 'y': {invalid_vals}")
            y_series = mapped.astype(int)
        clean_df["y"] = y_series
        positive_count = int((y_series == 1).sum())
        negative_count = int((y_series == 0).sum())
        positive_rate = float(positive_count / len(y_series))

    remaining_features = [c for c in clean_df.columns if c != "y"]
    never_rate = (
        float(clean_df["never_contacted_before"].mean())
        if "never_contacted_before" in clean_df.columns
        else 0.0
    )

    audit = {
        "variant": variant,
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "clean_rows": int(len(clean_df)),
        "clean_columns": int(clean_df.shape[1]),
        "dropped_duration": dropped_duration,
        "dropped_columns": dropped_columns,
        "sentinel_pdays_count": sentinel_pdays_count,
        "never_contacted_rate": never_rate,
        "unknown_counts": unknown_counts,
        "has_target": has_target,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "remaining_features": remaining_features,
    }
    return clean_df, audit


def build_preprocessor_pipelines(
    X_df: pd.DataFrame,
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Build algorithm-specific ColumnTransformer pipelines for LR and DT.

    Returns
    -------
    lr_preprocessor : ColumnTransformer
        OneHotEncoder(drop="first", handle_unknown="ignore") for categorical features,
        median imputation + RobustScaler for continuous numeric features, and passthrough
        for binary engineered flags (`never_contacted_before`).
    dt_preprocessor : ColumnTransformer
        OneHotEncoder(handle_unknown="ignore") for categorical features and median imputation
        + passthrough for numeric features so decision tree splits preserve original units.
    """
    all_cols = list(X_df.columns)
    binary_cols = [c for c in BINARY_ENGINEERED_COLUMNS if c in all_cols]
    numeric_cols = [
        c
        for c in X_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if c not in binary_cols
    ]
    categorical_cols = [c for c in all_cols if c not in numeric_cols and c not in binary_cols]

    lr_transformers: list[tuple[str, Any, list[str]]] = []
    dt_transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        lr_num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        dt_num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        lr_transformers.append(("num", lr_num_pipe, numeric_cols))
        dt_transformers.append(("num", dt_num_pipe, numeric_cols))

    if binary_cols:
        bin_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        lr_transformers.append(("bin", bin_pipe, binary_cols))
        dt_transformers.append(("bin", bin_pipe, binary_cols))

    if categorical_cols:
        lr_cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "ohe",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        dt_cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        lr_transformers.append(("cat", lr_cat_pipe, categorical_cols))
        dt_transformers.append(("cat", dt_cat_pipe, categorical_cols))

    lr_preprocessor = ColumnTransformer(
        transformers=lr_transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    dt_preprocessor = ColumnTransformer(
        transformers=dt_transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return lr_preprocessor, dt_preprocessor
