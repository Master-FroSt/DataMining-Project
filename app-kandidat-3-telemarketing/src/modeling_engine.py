"""Training, cross-validation, threshold evaluation, lift analysis, and explainability engine."""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import audit_and_remove_leakage, build_preprocessor_pipelines


def _normalize_imbalance_strategy(strategy: str) -> str:
    """Normalize imbalance handling option to canonical internal key."""
    key = str(strategy).strip().lower()
    if key in {"class_weight", "class weighting", "class weighting (balanced)", "balanced"}:
        return "class_weight"
    if key in {"smote", "smote (training fold only)"}:
        return "smote"
    if key in {"none", "none (baseline)", "baseline"}:
        return "none"
    raise ValueError(
        f"Unsupported imbalance_strategy '{strategy}'. Expected 'class_weight', 'smote', or 'none'."
    )


def _build_lr_estimator(
    lr_c: float,
    lr_penalty: str,
    class_weight: str | None,
    random_state: int,
) -> LogisticRegression:
    """Construct scikit-learn 1.9 compatible LogisticRegression without deprecation warnings."""
    penalty_norm = str(lr_penalty).strip().lower()
    if penalty_norm == "l1":
        return LogisticRegression(
            C=float(lr_c),
            l1_ratio=1.0,
            solver="liblinear",
            max_iter=1500,
            class_weight=class_weight,
            random_state=random_state,
        )
    if penalty_norm == "l2":
        return LogisticRegression(
            C=float(lr_c),
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=1500,
            class_weight=class_weight,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported lr_penalty '{lr_penalty}'. Expected 'l1' or 'l2'.")


def _build_model_pipeline(
    preprocessor: Any,
    estimator: Any,
    use_smote: bool,
    random_state: int,
) -> ImbPipeline:
    """Create an imblearn Pipeline ensuring SMOTE runs strictly inside training folds."""
    steps: list[tuple[str, Any]] = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps=steps)


def compute_gain_lift_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute Cumulative Gain, Lift Curve, decile breakdown, and ALIFT."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    n_total = len(y_arr)
    total_positives = int(np.sum(y_arr))
    base_rate = float(total_positives / n_total) if n_total > 0 else 0.0

    order = np.argsort(-p_arr, kind="mergesort")
    y_sorted = y_arr[order]
    p_sorted = p_arr[order]

    cum_positives = np.cumsum(y_sorted)
    pop_frac = np.concatenate(([0.0], np.arange(1, n_total + 1, dtype=float) / n_total))
    if total_positives > 0:
        gain_curve = np.concatenate(([0.0], cum_positives / total_positives))
    else:
        gain_curve = np.zeros_like(pop_frac)

    # ALIFT: Area under Cumulative Gain Curve (trapezoid), plus normalized CAP ratio
    alift = float(np.trapezoid(gain_curve, pop_frac))
    ideal_area = 1.0 - 0.5 * base_rate if total_positives > 0 else 0.5
    denom = ideal_area - 0.5
    normalized_alift = float((alift - 0.5) / denom) if denom > 1e-9 else 0.0

    # Decile table (Decile 1 = Top 10% ... Decile 10 = Bottom 10%)
    decile_indices = np.array_split(np.arange(n_total), 10)
    decile_rows = []
    cum_leads = 0
    cum_conv = 0
    for idx, chunk in enumerate(decile_indices, start=1):
        leads_in_decile = int(len(chunk))
        conv_in_decile = int(np.sum(y_sorted[chunk])) if leads_in_decile > 0 else 0
        cum_leads += leads_in_decile
        cum_conv += conv_in_decile
        decile_rate = float(conv_in_decile / leads_in_decile) if leads_in_decile > 0 else 0.0
        decile_lift = float(decile_rate / base_rate) if base_rate > 0 else 0.0
        cum_gain = float(cum_conv / total_positives) if total_positives > 0 else 0.0
        cum_frac = float(cum_leads / n_total) if n_total > 0 else 0.0
        cum_lift = float(cum_gain / cum_frac) if cum_frac > 0 else 0.0
        decile_rows.append(
            {
                "decile": idx,
                "decile_label": f"Top {idx * 10}%",
                "bucket_label": f"Decile {idx}",
                "leads_count": leads_in_decile,
                "conversions": conv_in_decile,
                "conversion_rate": decile_rate,
                "decile_lift": decile_lift,
                "cumulative_leads": cum_leads,
                "cumulative_pct": cum_frac,
                "cumulative_conversions": cum_conv,
                "cumulative_gain": cum_gain,
                "cumulative_lift": cum_lift,
                "min_score": float(np.min(p_sorted[chunk])) if leads_in_decile > 0 else 0.0,
                "max_score": float(np.max(p_sorted[chunk])) if leads_in_decile > 0 else 0.0,
            }
        )

    decile_df = pd.DataFrame(decile_rows)
    # Sample ~101 curve points cleanly for light Plotly rendering
    sample_idx = np.unique(np.linspace(0, len(pop_frac) - 1, min(101, len(pop_frac)), dtype=int))
    top_20_row = decile_df.loc[decile_df["decile"] == 2].iloc[0]
    top_50_row = decile_df.loc[decile_df["decile"] == 5].iloc[0]

    return {
        "pop_frac": pop_frac[sample_idx],
        "gain_curve": gain_curve[sample_idx],
        "alift": alift,
        "normalized_alift": normalized_alift,
        "decile_df": decile_df,
        "top_20_gain": float(top_20_row["cumulative_gain"]),
        "top_20_lift": float(top_20_row["cumulative_lift"]),
        "top_50_gain": float(top_50_row["cumulative_gain"]),
        "top_50_lift": float(top_50_row["cumulative_lift"]),
    }


def compute_threshold_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
    call_cost: float = 2.5,
    conversion_profit: float = 80.0,
) -> dict[str, Any]:
    """Compute Confusion Matrix and business-aligned classification metrics at a given threshold."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    tau = float(np.clip(threshold, 0.0, 1.0))

    y_pred = (p_arr >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
    tp_i, fp_i, tn_i, fn_i = int(tp), int(fp), int(tn), int(fn)
    total = len(y_arr)

    precision = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
    recall = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
    specificity = float(tn_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else 0.0
    fpr = float(1.0 - specificity)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    beta_sq = 4.0  # F2-score emphasizes Recall (missing a deposit costs more than one wasted call)
    f2 = (
        float((1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall))
        if (beta_sq * precision + recall) > 0
        else 0.0
    )
    accuracy = float((tp_i + tn_i) / total) if total > 0 else 0.0
    pct_called = float((tp_i + fp_i) / total) if total > 0 else 0.0

    expected_net_profit = float(tp_i * float(conversion_profit) - (tp_i + fp_i) * float(call_cost))
    baseline_profit = float(int(np.sum(y_arr)) * float(conversion_profit) - total * float(call_cost))
    optimal_tau = (
        float(np.clip(float(call_cost) / float(conversion_profit), 0.01, 0.99))
        if float(conversion_profit) > 0
        else 0.50
    )

    return {
        "threshold": tau,
        "tp": tp_i,
        "fp": fp_i,
        "tn": tn_i,
        "fn": fn_i,
        "total_samples": total,
        "total_calls": tp_i + fp_i,
        "captured_conversions": tp_i,
        "missed_conversions": fn_i,
        "avoided_unproductive_calls": tn_i,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "pct_called": pct_called,
        "expected_net_profit": expected_net_profit,
        "baseline_profit_call_all": baseline_profit,
        "profit_gain_vs_blanket": expected_net_profit - baseline_profit,
        "optimal_threshold_cost_benefit": optimal_tau,
    }


def _evaluate_model_holdout(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    call_cost: float = 2.5,
    conversion_profit: float = 80.0,
) -> dict[str, Any]:
    """Compute ROC, PR-AUC, Gain/Lift, ALIFT, and default threshold metrics."""
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    # Downsample curves to <= 200 points for crisp, fast UI plotting
    roc_idx = np.unique(np.linspace(0, len(fpr) - 1, min(200, len(fpr)), dtype=int))
    pr_idx = np.unique(np.linspace(0, len(prec_curve) - 1, min(200, len(prec_curve)), dtype=int))

    gain_lift = compute_gain_lift_metrics(y_test, y_prob)
    default_thresh = compute_threshold_metrics(
        y_test,
        y_prob,
        threshold=0.50,
        call_cost=call_cost,
        conversion_profit=conversion_profit,
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "alift": gain_lift["alift"],
        "normalized_alift": gain_lift["normalized_alift"],
        "top_20_gain": gain_lift["top_20_gain"],
        "top_20_lift": gain_lift["top_20_lift"],
        "top_50_gain": gain_lift["top_50_gain"],
        "top_50_lift": gain_lift["top_50_lift"],
        "roc_curve": {"fpr": fpr[roc_idx], "tpr": tpr[roc_idx]},
        "pr_curve": {"precision": prec_curve[pr_idx], "recall": rec_curve[pr_idx]},
        "gain_lift": gain_lift,
        "default_threshold_metrics": default_thresh,
    }


def _extract_raw_schema(X: pd.DataFrame) -> dict[str, Any]:
    """Capture original column schemas, categories, and numeric bounds for UI controls."""
    schema: dict[str, Any] = {"columns": list(X.columns), "categorical": {}, "numeric": {}}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            schema["numeric"][col] = {
                "min": float(series.min()) if len(series) else 0.0,
                "max": float(series.max()) if len(series) else 100.0,
                "median": float(series.median()) if len(series) else 0.0,
            }
        else:
            cats = sorted(X[col].astype(str).unique().tolist())
            schema["categorical"][col] = cats
    return schema


def train_and_evaluate_models(
    df: pd.DataFrame,
    variant: str = "bank-full",
    imbalance_strategy: str = "class_weight",
    test_size: float = 0.20,
    dt_max_depth: int = 6,
    dt_criterion: str = "gini",
    lr_c: float = 1.0,
    lr_penalty: str = "l2",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Decision Tree and Logistic Regression on pre-call telemarketing data."""
    t_start = time.perf_counter()
    strategy = _normalize_imbalance_strategy(imbalance_strategy)
    criterion_norm = str(dt_criterion).strip().lower()
    if criterion_norm not in {"gini", "entropy", "log_loss"}:
        raise ValueError(f"Unsupported dt_criterion '{dt_criterion}'.")

    clean_df, leakage_audit = audit_and_remove_leakage(df, variant=variant)
    if "y" not in clean_df.columns:
        raise ValueError("Target column 'y' is required for training and evaluation.")

    X = clean_df.drop(columns=["y"])
    y = clean_df["y"].to_numpy(dtype=int)
    raw_schema = _extract_raw_schema(X)

    # Stratified Train-Test Split BEFORE any encoding, scaling, or resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        stratify=y,
        random_state=random_state,
    )

    use_smote = strategy == "smote"
    cw = "balanced" if strategy == "class_weight" else None

    lr_prep_cv, dt_prep_cv = build_preprocessor_pipelines(X_train)
    lr_prep_final, dt_prep_final = build_preprocessor_pipelines(X_train)

    dt_est_cv = DecisionTreeClassifier(
        criterion=criterion_norm,
        max_depth=int(dt_max_depth),
        min_samples_leaf=20,
        class_weight=cw,
        random_state=random_state,
    )
    dt_est_final = DecisionTreeClassifier(
        criterion=criterion_norm,
        max_depth=int(dt_max_depth),
        min_samples_leaf=20,
        class_weight=cw,
        random_state=random_state,
    )

    lr_est_cv = _build_lr_estimator(lr_c, lr_penalty, cw, random_state)
    lr_est_final = _build_lr_estimator(lr_c, lr_penalty, cw, random_state)

    dt_cv_pipe = _build_model_pipeline(dt_prep_cv, dt_est_cv, use_smote, random_state)
    lr_cv_pipe = _build_model_pipeline(lr_prep_cv, lr_est_cv, use_smote, random_state)
    dt_pipeline = _build_model_pipeline(dt_prep_final, dt_est_final, use_smote, random_state)
    lr_pipeline = _build_model_pipeline(lr_prep_final, lr_est_final, use_smote, random_state)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "pr_auc": "average_precision"}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories in columns")
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
        dt_cv_res = cross_validate(dt_cv_pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        lr_cv_res = cross_validate(lr_cv_pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)

        dt_pipeline.fit(X_train, y_train)
        lr_pipeline.fit(X_train, y_train)

        dt_test_probs = dt_pipeline.predict_proba(X_test)[:, 1]
        lr_test_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    dt_eval = _evaluate_model_holdout(y_test, dt_test_probs)
    lr_eval = _evaluate_model_holdout(y_test, lr_test_probs)

    # Extract DT Feature Importances
    dt_feat_names = list(dt_pipeline.named_steps["prep"].get_feature_names_out())
    dt_importances = dt_pipeline.named_steps["clf"].feature_importances_
    dt_importance_df = (
        pd.DataFrame({"feature": dt_feat_names, "importance": dt_importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Extract LR Coefficients and Odds Ratios e^(beta_j)
    lr_feat_names = list(lr_pipeline.named_steps["prep"].get_feature_names_out())
    lr_coefs = lr_pipeline.named_steps["clf"].coef_[0]
    lr_intercept = float(lr_pipeline.named_steps["clf"].intercept_[0])
    lr_coef_df = pd.DataFrame(
        {
            "feature": lr_feat_names,
            "coefficient": lr_coefs,
            "abs_coefficient": np.abs(lr_coefs),
            "odds_ratio": np.exp(np.clip(lr_coefs, -20.0, 20.0)),
            "direction": np.where(lr_coefs >= 0, "Positive Driver", "Negative Driver"),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    cv_metrics = {
        "Decision Tree": {
            "cv_roc_auc_mean": float(np.mean(dt_cv_res["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(dt_cv_res["test_roc_auc"])),
            "cv_f1_mean": float(np.mean(dt_cv_res["test_f1"])),
            "cv_f1_std": float(np.std(dt_cv_res["test_f1"])),
            "cv_pr_auc_mean": float(np.mean(dt_cv_res["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(dt_cv_res["test_pr_auc"])),
        },
        "Logistic Regression": {
            "cv_roc_auc_mean": float(np.mean(lr_cv_res["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(lr_cv_res["test_roc_auc"])),
            "cv_f1_mean": float(np.mean(lr_cv_res["test_f1"])),
            "cv_f1_std": float(np.std(lr_cv_res["test_f1"])),
            "cv_pr_auc_mean": float(np.mean(lr_cv_res["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(lr_cv_res["test_pr_auc"])),
        },
    }

    training_time_sec = float(time.perf_counter() - t_start)

    return {
        "variant": variant,
        "imbalance_strategy": strategy,
        "leakage_audit": leakage_audit,
        "feature_columns": list(X.columns),
        "raw_schema": raw_schema,
        "dt_pipeline": dt_pipeline,
        "lr_pipeline": lr_pipeline,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test,
        "dt_test_probs": dt_test_probs,
        "lr_test_probs": lr_test_probs,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "positive_prevalence": float(np.mean(y_test)),
        "cv_metrics": cv_metrics,
        "eval_results": {
            "Decision Tree": dt_eval,
            "Logistic Regression": lr_eval,
        },
        "dt_feature_importance": dt_importance_df,
        "lr_coefficients": lr_coef_df,
        "lr_intercept": lr_intercept,
        "training_time_sec": training_time_sec,
        "hyperparameters": {
            "dt_max_depth": int(dt_max_depth),
            "dt_criterion": criterion_norm,
            "lr_c": float(lr_c),
            "lr_penalty": str(lr_penalty).lower(),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
    }


def assign_priority_badge(prob: float, threshold: float) -> str:
    """Map conversion probability and active threshold to operational Call Priority Badge."""
    p = float(prob)
    tau = float(threshold)
    if p >= tau:
        return "High Priority - Immediate Call"
    if p >= max(0.0, tau * 0.65):
        return "Medium Priority - Secondary Queue"
    return "Low Priority - Skip to Avoid Customer Fatigue"


def _prepare_inference_frame(
    raw_df: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply anti-leakage guard and align schema to trained feature columns."""
    clean_df, audit = audit_and_remove_leakage(raw_df, variant=model_bundle["variant"])
    expected_cols = model_bundle["feature_columns"]
    aligned = clean_df.copy()

    # Preserve target if present for batch evaluation, separate before scoring
    has_y = "y" in aligned.columns
    y_true = aligned["y"].to_numpy(dtype=int) if has_y else None

    for col in expected_cols:
        if col not in aligned.columns:
            if col in model_bundle["raw_schema"]["numeric"]:
                aligned[col] = model_bundle["raw_schema"]["numeric"][col]["median"]
            else:
                aligned[col] = "unknown"

    X_infer = aligned[expected_cols].copy()
    return X_infer, {"audit": audit, "has_y": has_y, "y_true": y_true}


def _format_tree_rule(
    feat_name: str,
    actual_val: float,
    threshold_val: float,
    goes_left: bool,
    dt_raw_cols: list[str],
) -> str:
    """Format Decision Tree split rule cleanly for both numeric and one-hot categorical nodes."""
    is_binary_ohe = feat_name not in dt_raw_cols and abs(threshold_val - 0.5) < 1e-6
    if is_binary_ohe:
        if goes_left:
            return f"{feat_name} = 0 (condition absent)"
        return f"{feat_name} = 1 (condition active)"
    op = "<=" if goes_left else ">"
    return f"{feat_name} {op} {threshold_val:.2f} (observed: {actual_val:.2f})"


def score_single_lead(
    model_bundle: dict[str, Any],
    lead_dict: dict[str, Any],
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Score a single prospect with Decision Tree and Logistic Regression plus explainability."""
    raw_lead_df = pd.DataFrame([lead_dict])
    X_lead, meta = _prepare_inference_frame(raw_lead_df, model_bundle)

    dt_pipe = model_bundle["dt_pipeline"]
    lr_pipe = model_bundle["lr_pipeline"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories in columns")
        dt_prob = float(dt_pipe.predict_proba(X_lead)[0, 1])
        lr_prob = float(lr_pipe.predict_proba(X_lead)[0, 1])

        # LR Local Log-Odds Contributions (beta_j * x_transformed_j)
        lr_prep = lr_pipe.named_steps["prep"]
        lr_clf = lr_pipe.named_steps["clf"]
        x_lr_trans = lr_prep.transform(X_lead)[0]
        lr_feat_names = list(lr_prep.get_feature_names_out())

        # DT Decision Path Trace
        dt_prep = dt_pipe.named_steps["prep"]
        dt_clf = dt_pipe.named_steps["clf"]
        x_dt_trans = dt_prep.transform(X_lead)
        dt_feat_names = list(dt_prep.get_feature_names_out())

    coefs = lr_clf.coef_[0]
    intercept = float(lr_clf.intercept_[0])
    contribs = coefs * x_lr_trans
    total_logit = float(intercept + np.sum(contribs))

    contrib_df = pd.DataFrame(
        {
            "feature": lr_feat_names,
            "transformed_value": x_lr_trans,
            "coefficient": coefs,
            "log_odds_contribution": contribs,
        }
    )
    nonzero_df = contrib_df.loc[contrib_df["log_odds_contribution"].abs() > 1e-6].copy()
    if nonzero_df.empty:
        nonzero_df = contrib_df.copy()

    top_positive = (
        nonzero_df.loc[nonzero_df["log_odds_contribution"] > 0]
        .sort_values("log_odds_contribution", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_negative = (
        nonzero_df.loc[nonzero_df["log_odds_contribution"] < 0]
        .sort_values("log_odds_contribution", ascending=True)
        .head(5)
        .reset_index(drop=True)
    )
    top_drivers = pd.concat([top_positive, top_negative], ignore_index=True).sort_values(
        "log_odds_contribution", ascending=True
    )

    # Extract DT Root-to-Leaf Path
    tree_ = dt_clf.tree_
    node_indicator = dt_clf.decision_path(x_dt_trans)
    leaf_id = int(dt_clf.apply(x_dt_trans)[0])
    node_indices = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    dt_raw_cols = model_bundle["feature_columns"]

    decision_path_steps: list[dict[str, Any]] = []
    for depth_idx, node_id in enumerate(node_indices):
        node_vals = tree_.value[node_id][0]
        node_weight_sum = float(np.sum(node_vals))
        node_pos_prob = float(node_vals[1] / node_weight_sum) if node_weight_sum > 0 else 0.0
        n_node_samples = int(tree_.n_node_samples[node_id])

        if node_id == leaf_id:
            decision_path_steps.append(
                {
                    "step": depth_idx + 1,
                    "node_id": int(node_id),
                    "is_leaf": True,
                    "rule": f"Leaf Node #{int(node_id)}: Predicted Conversion Probability = {node_pos_prob:.1%}",
                    "node_samples": n_node_samples,
                    "node_positive_rate": node_pos_prob,
                }
            )
        else:
            feat_idx = int(tree_.feature[node_id])
            feat_name = dt_feat_names[feat_idx]
            thresh = float(tree_.threshold[node_id])
            obs_val = float(x_dt_trans[0, feat_idx])
            goes_left = obs_val <= thresh
            rule_text = _format_tree_rule(feat_name, obs_val, thresh, goes_left, dt_raw_cols)
            decision_path_steps.append(
                {
                    "step": depth_idx + 1,
                    "node_id": int(node_id),
                    "is_leaf": False,
                    "feature": feat_name,
                    "rule": rule_text,
                    "node_samples": n_node_samples,
                    "node_positive_rate": node_pos_prob,
                }
            )

    leaf_vals = tree_.value[leaf_id][0]
    leaf_weight_sum = float(np.sum(leaf_vals))
    leaf_positive_prob = float(leaf_vals[1] / leaf_weight_sum) if leaf_weight_sum > 0 else dt_prob
    leaf_samples = int(tree_.n_node_samples[leaf_id])

    return {
        "threshold": float(threshold),
        "dt_prob": dt_prob,
        "lr_prob": lr_prob,
        "dt_priority": assign_priority_badge(dt_prob, threshold),
        "lr_priority": assign_priority_badge(lr_prob, threshold),
        "leakage_audit": meta["audit"],
        "lr_local_contributions": {
            "intercept": intercept,
            "total_logit": total_logit,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_drivers": top_drivers,
        },
        "dt_decision_path": decision_path_steps,
        "leaf_id": leaf_id,
        "leaf_samples": leaf_samples,
        "leaf_positive_prob": leaf_positive_prob,
    }


def score_batch_leads(
    model_bundle: dict[str, Any],
    batch_df: pd.DataFrame,
    threshold: float = 0.50,
    model_choice: str = "Logistic Regression",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score, rank, and assign deciles to a batch of pre-call leads with anti-leakage guard."""
    X_batch, meta = _prepare_inference_frame(batch_df, model_bundle)
    dt_pipe = model_bundle["dt_pipeline"]
    lr_pipe = model_bundle["lr_pipeline"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories in columns")
        dt_probs = dt_pipe.predict_proba(X_batch)[:, 1]
        lr_probs = lr_pipe.predict_proba(X_batch)[:, 1]

    active_choice = str(model_choice).strip()
    active_probs = dt_probs if active_choice == "Decision Tree" else lr_probs

    scored = X_batch.copy()
    scored["dt_probability"] = dt_probs
    scored["lr_probability"] = lr_probs
    scored["active_probability"] = active_probs
    if meta["has_y"] and meta["y_true"] is not None:
        scored["actual_y"] = meta["y_true"]

    scored = scored.sort_values("active_probability", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    n_rows = len(scored)
    scored["priority_rank"] = np.arange(1, n_rows + 1)

    decile_num = np.minimum(10, (np.arange(n_rows) * 10) // max(1, n_rows) + 1)
    scored["decile"] = [f"Top {d * 10}%" for d in decile_num]
    scored["priority_action"] = [
        assign_priority_badge(p, threshold) for p in scored["active_probability"].to_numpy()
    ]

    summary: dict[str, Any] = {
        "audit": meta["audit"],
        "total_leads": n_rows,
        "has_actual_target": bool(meta["has_y"]),
        "model_choice": active_choice,
        "threshold": float(threshold),
        "high_priority_count": int((scored["active_probability"] >= float(threshold)).sum()),
        "mean_predicted_prob": float(scored["active_probability"].mean()),
    }
    if meta["has_y"] and meta["y_true"] is not None:
        summary["gain_lift"] = compute_gain_lift_metrics(
            scored["actual_y"].to_numpy(),
            scored["active_probability"].to_numpy(),
        )
    return scored, summary


def simulate_campaign_quota(
    scored_df: pd.DataFrame,
    quota_pct: float,
    baseline_prevalence: float,
    call_cost: float = 2.5,
    conversion_profit: float = 80.0,
) -> dict[str, Any]:
    """Simulate call-capacity quota impact on conversions, Lift, avoided calls, and net profit."""
    n_total = len(scored_df)
    if n_total == 0:
        return {
            "quota_pct": float(quota_pct),
            "quota_calls": 0,
            "total_leads": 0,
            "captured_conversions": 0.0,
            "projected_conversion_rate": 0.0,
            "lift_vs_random": 0.0,
            "cumulative_gain": 0.0,
            "avoided_unproductive_calls": 0.0,
            "expected_net_profit": 0.0,
            "profit_vs_blanket": 0.0,
            "is_actual_ground_truth": False,
        }

    pct = float(np.clip(quota_pct, 1.0, 100.0))
    quota_calls = max(1, min(n_total, int(round(n_total * pct / 100.0))))
    selected = scored_df.iloc[:quota_calls]
    skipped = scored_df.iloc[quota_calls:]

    if "actual_y" in scored_df.columns:
        total_pos = float(scored_df["actual_y"].sum())
        captured = float(selected["actual_y"].sum())
        conv_rate = float(captured / quota_calls)
        base_rate = float(total_pos / n_total) if n_total > 0 else float(baseline_prevalence)
        lift = float(conv_rate / base_rate) if base_rate > 0 else 0.0
        gain = float(captured / total_pos) if total_pos > 0 else 0.0
        avoided = float((skipped["actual_y"] == 0).sum())
        net_profit = float(captured * float(conversion_profit) - quota_calls * float(call_cost))
        blanket_profit = float(total_pos * float(conversion_profit) - n_total * float(call_cost))
        is_actual = True
    else:
        raw_probs = scored_df["active_probability"].to_numpy(dtype=float)
        mean_p = float(np.clip(np.mean(raw_probs), 1e-4, 1.0 - 1e-4))
        target_p = float(np.clip(baseline_prevalence, 1e-4, 1.0 - 1e-4))
        if mean_p > target_p * 1.4:
            clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
            odds_raw = clipped / (1.0 - clipped)
            odds_factor = (target_p / (1.0 - target_p)) / (mean_p / (1.0 - mean_p))
            cal_odds = odds_raw * odds_factor
            proj_probs = cal_odds / (1.0 + cal_odds)
        else:
            proj_probs = raw_probs

        total_pos = float(np.sum(proj_probs))
        captured = float(np.sum(proj_probs[:quota_calls]))
        conv_rate = float(captured / quota_calls)
        base_rate = float(total_pos / n_total) if n_total > 0 else target_p
        lift = float(conv_rate / base_rate) if base_rate > 0 else 0.0
        gain = float(captured / total_pos) if total_pos > 0 else 0.0
        avoided = float(np.sum(1.0 - proj_probs[quota_calls:]))
        net_profit = float(captured * float(conversion_profit) - quota_calls * float(call_cost))
        blanket_profit = float(total_pos * float(conversion_profit) - n_total * float(call_cost))
        is_actual = False

    return {
        "quota_pct": pct,
        "quota_calls": int(quota_calls),
        "total_leads": int(n_total),
        "captured_conversions": captured,
        "projected_conversion_rate": conv_rate,
        "lift_vs_random": lift,
        "cumulative_gain": gain,
        "avoided_unproductive_calls": avoided,
        "expected_net_profit": net_profit,
        "profit_vs_blanket": net_profit - blanket_profit,
        "is_actual_ground_truth": is_actual,
    }

