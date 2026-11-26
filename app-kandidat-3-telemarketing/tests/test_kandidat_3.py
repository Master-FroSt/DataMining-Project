"""Comprehensive unit, leakage-guard, modeling, and Streamlit AppTest suite for Kandidat 3."""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
_APP_ROOT_STR = str(APP_ROOT)
_APP_ROOT_RESOLVED = APP_ROOT.resolve()


def _isolate_sys_path() -> None:
    """Keep this app's root first in sys.path and remove sibling app-kandidat-* roots."""
    cleaned = [
        p
        for p in sys.path
        if p != _APP_ROOT_STR
        and not (
            Path(p).name.startswith("app-kandidat-")
            and Path(p).resolve() != _APP_ROOT_RESOLVED
        )
    ]
    sys.path[:] = [_APP_ROOT_STR, *cleaned]


def _clear_src_modules() -> None:
    """Remove any cached src or src.* entries from sys.modules."""
    for mod_name in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[mod_name]


_isolate_sys_path()
_clear_src_modules()

import src  # noqa: E402, F401
from src.data_loader import generate_bank_sample_data, load_or_download_bank_data  # noqa: E402
from src.modeling_engine import (  # noqa: E402
    assign_priority_badge,
    compute_threshold_metrics,
    score_batch_leads,
    score_single_lead,
    simulate_campaign_quota,
    train_and_evaluate_models,
)
from src.preprocessing import audit_and_remove_leakage, build_preprocessor_pipelines  # noqa: E402

_OWN_SRC_MODULES = {
    k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")
}

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


@pytest.fixture(autouse=True)
def _restore_kandidat_3_src_modules() -> None:
    """Ensure sys.path and sys.modules['src*'] resolve to Kandidat 3 during combined pytest runs."""
    _isolate_sys_path()
    _clear_src_modules()
    sys.modules.update(_OWN_SRC_MODULES)


def test_zero_emojis_in_all_python_files() -> None:
    """Ensure strict compliance with Zero-Emoji rule across all .py files."""
    py_files = list(APP_ROOT.rglob("*.py"))
    assert len(py_files) >= 5
    for py_path in py_files:
        text = py_path.read_text(encoding="utf-8")
        matches = EMOJI_PATTERN.findall(text)
        assert not matches, f"Emoji character(s) found in {py_path}: {matches}"


@pytest.mark.parametrize(
    "variant,expected_cols,sentinel_val",
    [
        ("bank-full", 17, -1),
        ("bank-additional-full", 21, 999),
    ],
)
def test_generate_bank_sample_data_schemas(
    variant: str,
    expected_cols: int,
    sentinel_val: int,
) -> None:
    """Verify both UCI variants generate expected schema, imbalance rate, sentinels, and duration."""
    df = generate_bank_sample_data(variant=variant, n_rows=2000, random_state=42)
    assert len(df) == 2000
    assert df.shape[1] == expected_cols
    assert "duration" in df.columns
    assert "y" in df.columns

    pos_rate = float((df["y"] == "yes").mean())
    assert 0.08 <= pos_rate <= 0.16

    assert (df["pdays"] == sentinel_val).sum() > 500
    assert (df["education"] == "unknown").sum() > 10


def test_load_or_download_bank_data_sample_and_parquet_cache(tmp_path: Path) -> None:
    """Verify sample flag and local Parquet cache path work without network errors."""
    sample_df = load_or_download_bank_data(tmp_path, variant="bank-full", use_sample=True)
    assert len(sample_df) == 6000

    # Pre-seed local parquet cache and ensure loader reads directly from it
    seeded_df = generate_bank_sample_data(variant="bank-additional-full", n_rows=1200, random_state=7)
    parquet_file = tmp_path / "bank-additional-full.parquet"
    seeded_df.to_parquet(parquet_file, index=False)

    loaded_df = load_or_download_bank_data(tmp_path, variant="bank-additional-full", use_sample=False)
    assert len(loaded_df) == 1200
    assert "euribor3m" in loaded_df.columns


@pytest.mark.parametrize("variant", ["bank-full", "bank-additional-full"])
def test_audit_and_remove_leakage_and_sentinels(variant: str) -> None:
    """Verify duration is stripped, pdays sentinels engineered, unknown preserved, and y encoded."""
    raw_df = generate_bank_sample_data(variant=variant, n_rows=1000, random_state=99)
    assert "duration" in raw_df.columns

    clean_df, audit = audit_and_remove_leakage(raw_df, variant=variant)
    assert audit["dropped_duration"] is True
    assert "duration" not in clean_df.columns
    assert "never_contacted_before" in clean_df.columns
    assert set(clean_df["never_contacted_before"].unique()).issubset({0, 1})
    assert (clean_df["pdays"] >= 0).all()
    assert (clean_df["pdays"] < 999).all()
    assert set(clean_df["y"].unique()) == {0, 1}
    assert "unknown" in clean_df["education"].values


def test_build_preprocessor_pipelines_differential_behavior() -> None:
    """Verify LR scales numerics with drop='first' while DT keeps unscaled numerics."""
    raw_df = generate_bank_sample_data(variant="bank-full", n_rows=800, random_state=12)
    clean_df, _ = audit_and_remove_leakage(raw_df, variant="bank-full")
    X = clean_df.drop(columns=["y"])

    lr_prep, dt_prep = build_preprocessor_pipelines(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X_lr = lr_prep.fit_transform(X)
        X_dt = dt_prep.fit_transform(X)

    assert not np.isnan(X_lr).any()
    assert not np.isnan(X_dt).any()
    # LR drops first category per categorical feature, so it has fewer columns than DT
    assert X_lr.shape[1] < X_dt.shape[1]

    dt_names = list(dt_prep.get_feature_names_out())
    bal_idx = dt_names.index("balance")
    np.testing.assert_allclose(X_dt[:, bal_idx], X["balance"].to_numpy(dtype=float))


@pytest.mark.parametrize("strategy", ["class_weight", "smote", "none"])
def test_train_and_evaluate_models_strategies_and_leakage_isolation(strategy: str) -> None:
    """Verify training pipeline across all imbalance strategies and zero sklearn warnings."""
    raw_df = generate_bank_sample_data(variant="bank-full", n_rows=1500, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = train_and_evaluate_models(
            raw_df,
            variant="bank-full",
            imbalance_strategy=strategy,
            test_size=0.20,
            dt_max_depth=5,
            dt_criterion="gini",
            lr_c=1.0,
            lr_penalty="l2",
            random_state=42,
        )

    assert bundle["leakage_audit"]["dropped_duration"] is True
    assert "duration" not in bundle["feature_columns"]
    assert bundle["train_size"] == 1200
    assert bundle["test_size"] == 300
    # Holdout test set must retain authentic ~11.7% minority ratio (never resampled by SMOTE)
    assert 0.08 <= bundle["positive_prevalence"] <= 0.16

    for model_name in ("Decision Tree", "Logistic Regression"):
        ev = bundle["eval_results"][model_name]
        cv = bundle["cv_metrics"][model_name]
        assert 0.55 <= ev["roc_auc"] <= 0.99
        assert 0.12 <= ev["pr_auc"] <= 0.99
        assert 0.50 <= ev["alift"] <= 1.00
        assert 0.50 <= cv["cv_roc_auc_mean"] <= 0.99

    lr_coefs = bundle["lr_coefficients"]
    assert "odds_ratio" in lr_coefs.columns
    assert (lr_coefs["odds_ratio"] > 0).all()
    assert np.isclose(bundle["dt_feature_importance"]["importance"].sum(), 1.0, atol=1e-6)


def test_train_and_evaluate_additional_variant_and_l1_penalty() -> None:
    """Verify bank-additional-full variant and L1 Logistic Regression execute cleanly."""
    raw_df = generate_bank_sample_data(variant="bank-additional-full", n_rows=1500, random_state=17)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = train_and_evaluate_models(
            raw_df,
            variant="bank-additional-full",
            imbalance_strategy="class_weight",
            dt_max_depth=5,
            dt_criterion="entropy",
            lr_c=0.5,
            lr_penalty="l1",
            random_state=17,
        )

    assert "euribor3m" in bundle["feature_columns"]
    assert "balance" not in bundle["feature_columns"]
    assert bundle["eval_results"]["Logistic Regression"]["roc_auc"] > 0.60


def test_threshold_metrics_and_single_lead_and_batch_scoring() -> None:
    """Verify Confusion Matrix math, single lead explainability, and batch ranking/quota simulation."""
    raw_df = generate_bank_sample_data(variant="bank-full", n_rows=1500, random_state=21)
    bundle = train_and_evaluate_models(raw_df, variant="bank-full", imbalance_strategy="class_weight")

    m = compute_threshold_metrics(
        bundle["y_test"],
        bundle["lr_test_probs"],
        threshold=0.50,
        call_cost=2.5,
        conversion_profit=80.0,
    )
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == bundle["test_size"]
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["specificity"] <= 1.0
    assert np.isclose(m["optimal_threshold_cost_benefit"], 2.5 / 80.0, atol=1e-6)

    assert assign_priority_badge(0.75, 0.50) == "High Priority - Immediate Call"
    assert assign_priority_badge(0.40, 0.50) == "Medium Priority - Secondary Queue"
    assert assign_priority_badge(0.10, 0.50) == "Low Priority - Skip to Avoid Customer Fatigue"

    lead_input = {
        "age": 64,
        "job": "retired",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 4500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 14,
        "month": "oct",
        "duration": 999,  # Should be stripped cleanly by anti-leakage guard
        "campaign": 1,
        "pdays": 60,
        "previous": 2,
        "poutcome": "success",
    }
    scored_lead = score_single_lead(bundle, lead_input, threshold=0.50)
    assert scored_lead["leakage_audit"]["dropped_duration"] is True
    assert 0.0 <= scored_lead["dt_prob"] <= 1.0
    assert 0.0 <= scored_lead["lr_prob"] <= 1.0
    assert len(scored_lead["dt_decision_path"]) >= 2
    assert not scored_lead["lr_local_contributions"]["top_drivers"].empty

    batch_df = bundle["X_test"].head(150).copy()
    batch_df["duration"] = 300
    scored_batch, summary = score_batch_leads(bundle, batch_df, threshold=0.50, model_choice="Logistic Regression")
    assert summary["audit"]["dropped_duration"] is True
    assert "duration" not in scored_batch.columns
    assert scored_batch["priority_rank"].tolist() == list(range(1, 151))
    assert scored_batch["active_probability"].is_monotonic_decreasing

    quota_res = simulate_campaign_quota(
        scored_batch,
        quota_pct=20.0,
        baseline_prevalence=bundle["positive_prevalence"],
    )
    assert quota_res["quota_calls"] == 30
    assert quota_res["total_leads"] == 150


def test_streamlit_apptest_smoke_run() -> None:
    """Verify full Streamlit dashboard renders cleanly via AppTest and supports variant switching."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_ROOT / "app.py"), default_timeout=45)
    at.run()
    assert not at.exception, f"Streamlit AppTest raised exceptions: {at.exception}"
    assert len(at.title) >= 1
    assert "Bank Telemarketing Predictive Lead Scoring Dashboard" in at.title[0].value

    # Switch dataset variant in sidebar to bank-additional-full and verify clean re-run
    at.sidebar.selectbox[0].select("bank-additional-full.csv (21 Attributes + Macroeconomic)").run()
    assert not at.exception, f"Streamlit AppTest raised exceptions on variant switch: {at.exception}"
    assert len(at.metric) >= 5
    assert at.metric[0].value == "bank-additional-full"
