"""Interactive 3-Module Bank Telemarketing Predictive Lead Scoring Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_or_download_bank_data
from src.modeling_engine import (
    compute_threshold_metrics,
    score_batch_leads,
    score_single_lead,
    simulate_campaign_quota,
    train_and_evaluate_models,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

DT_COLOR = "#0F766E"
LR_COLOR = "#1D4ED8"
BASELINE_COLOR = "#64748B"
POS_COLOR = "#15803D"
NEG_COLOR = "#B91C1C"
WARN_COLOR = "#B45309"

VARIANT_LABELS = {
    "bank-full.csv (17 Attributes)": "bank-full",
    "bank-additional-full.csv (21 Attributes + Macroeconomic)": "bank-additional-full",
}

IMBALANCE_LABELS = {
    "Class Weighting (Balanced)": "class_weight",
    "SMOTE (Training Fold Only)": "smote",
    "None (Baseline)": "none",
}

st.set_page_config(
    page_title="Bank Telemarketing Predictive Lead Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_custom_styles() -> None:
    """Apply clean corporate banking UI styling without emojis."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .priority-badge {
            padding: 0.65rem 0.95rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
            border-left: 5px solid;
        }
        .badge-high {
            background-color: #ECFDF5;
            color: #065F46;
            border-color: #059669;
        }
        .badge-medium {
            background-color: #FFFBEB;
            color: #92400E;
            border-color: #D97706;
        }
        .badge-low {
            background-color: #FEF2F2;
            color: #991B1B;
            border-color: #DC2626;
        }
        .guard-banner {
            padding: 0.75rem 1rem;
            border-radius: 6px;
            background-color: #EFF6FF;
            color: #1E3A8A;
            border-left: 5px solid #2563EB;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading Bank Marketing dataset...")
def load_cached_bank_data(variant: str, use_sample: bool) -> tuple[pd.DataFrame, str]:
    """Cache raw dataset loading by variant and data source."""
    df = load_or_download_bank_data(DATA_DIR, variant=variant, use_sample=use_sample)
    source_note = str(df.attrs.get("data_source", variant))
    return df, source_note


@st.cache_resource(show_spinner="Training Decision Tree and Logistic Regression pipelines...")
def train_cached_bundle(
    variant: str,
    use_sample: bool,
    imbalance_strategy: str,
    dt_max_depth: int,
    dt_criterion: str,
    lr_c: float,
    lr_penalty: str,
) -> dict[str, Any]:
    """Train and cache end-to-end leakage-free DT and LR pipelines and holdout benchmarks."""
    raw_df, source_note = load_cached_bank_data(variant=variant, use_sample=use_sample)
    bundle = train_and_evaluate_models(
        raw_df,
        variant=variant,
        imbalance_strategy=imbalance_strategy,
        test_size=0.20,
        dt_max_depth=dt_max_depth,
        dt_criterion=dt_criterion,
        lr_c=lr_c,
        lr_penalty=lr_penalty,
        random_state=42,
    )
    bundle["data_source_note"] = source_note
    return bundle


def _render_priority_badge(label: str, model_name: str) -> None:
    """Render styled Call Priority Badge cleanly."""
    if label.startswith("High Priority"):
        css_cls = "badge-high"
    elif label.startswith("Medium Priority"):
        css_cls = "badge-medium"
    else:
        css_cls = "badge-low"
    st.markdown(
        f'<div class="priority-badge {css_cls}"><strong>{model_name} Action:</strong> {label}</div>',
        unsafe_allow_html=True,
    )


def _build_probability_gauge(prob: float, threshold: float, title: str, color: str) -> go.Figure:
    """Create Plotly conversion probability gauge with active threshold marker."""
    pct = prob * 100.0
    thresh_pct = threshold * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "valueformat": ".1f"},
            delta={
                "reference": thresh_pct,
                "valueformat": ".1f",
                "suffix": " pp vs Threshold",
            },
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, max(0.0, thresh_pct * 0.65)], "color": "#FEE2E2"},
                    {"range": [max(0.0, thresh_pct * 0.65), thresh_pct], "color": "#FEF3C7"},
                    {"range": [thresh_pct, 100], "color": "#DCFCE7"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.85,
                    "value": thresh_pct,
                },
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=55, b=10), template="plotly_white")
    return fig


def _build_confusion_matrix_figure(metrics: dict[str, Any], title: str, colorscale: str) -> go.Figure:
    """Create Confusion Matrix heatmap for a given model and threshold."""
    z = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]
    text_labels = [
        [f"TN: {metrics['tn']:,}", f"FP: {metrics['fp']:,}"],
        [f"FN: {metrics['fn']:,}", f"TP: {metrics['tp']:,}"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted: No (Skip)", "Predicted: Yes (Call)"],
            y=["Actual: No", "Actual: Yes"],
            text=text_labels,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=colorscale,
            showscale=False,
        )
    )
    fig.update_layout(
        title=title,
        height=310,
        margin=dict(l=20, r=20, t=48, b=25),
        template="plotly_white",
        yaxis={"autorange": "reversed"},
    )
    return fig


def _render_sidebar() -> dict[str, Any]:
    """Render Sidebar Control Center and return configuration dictionary."""
    if "active_threshold" not in st.session_state:
        st.session_state["active_threshold"] = 0.50

    with st.sidebar:
        st.header("Control Center")
        st.caption("Pre-Call Telemarketing Lead Scoring Configuration")

        variant_label = st.selectbox(
            "Dataset Variant Switcher",
            options=list(VARIANT_LABELS.keys()),
            index=0,
        )
        variant = VARIANT_LABELS[variant_label]

        source_label = st.radio(
            "Dataset Source",
            options=["Built-in Benchmark Sample", "UCI Full Download + Parquet"],
            index=0,
            help="Built-in sample starts instantly offline. UCI Full downloads official archive and caches to Parquet.",
        )
        use_sample = source_label == "Built-in Benchmark Sample"

        st.subheader("Class Imbalance & Models")
        imbalance_label = st.selectbox(
            "Class Imbalance Strategy",
            options=list(IMBALANCE_LABELS.keys()),
            index=0,
        )
        imbalance_strategy = IMBALANCE_LABELS[imbalance_label]

        dt_max_depth = st.slider("Decision Tree max_depth", min_value=2, max_value=14, value=6, step=1)
        dt_criterion = st.selectbox("Decision Tree criterion", options=["gini", "entropy"], index=0)
        lr_c = st.select_slider(
            "Logistic Regression C (Inverse Regularization)",
            options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            value=1.0,
        )
        lr_penalty = st.selectbox("Logistic Regression Regularization", options=["l2", "l1"], index=0)

        st.subheader("Business Assumptions & Cutoff")
        call_cost = st.number_input(
            "Cost per Outbound Call (C_call, EUR)",
            min_value=0.10,
            max_value=50.0,
            value=2.50,
            step=0.50,
        )
        conversion_profit = st.number_input(
            "Net Profit per Deposit (V_profit, EUR)",
            min_value=5.0,
            max_value=1000.0,
            value=80.0,
            step=5.0,
        )
        theoretical_tau = float(np.clip(call_cost / conversion_profit, 0.01, 0.99))
        st.caption(f"Theoretical Cost-Benefit Threshold (C_call / V_profit): {theoretical_tau:.3f}")

        active_threshold = st.slider(
            "Global Active Probability Threshold",
            min_value=0.05,
            max_value=0.95,
            step=0.01,
            key="active_threshold",
        )

    return {
        "variant": variant,
        "variant_label": variant_label,
        "use_sample": use_sample,
        "imbalance_strategy": imbalance_strategy,
        "imbalance_label": imbalance_label,
        "dt_max_depth": dt_max_depth,
        "dt_criterion": dt_criterion,
        "lr_c": lr_c,
        "lr_penalty": lr_penalty,
        "call_cost": float(call_cost),
        "conversion_profit": float(conversion_profit),
        "theoretical_tau": theoretical_tau,
        "active_threshold": float(active_threshold),
    }


def _render_module_1(bundle: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Modul 1: Single Lead Scoring & Call Decision Simulator."""
    st.subheader("Modul 1: Single Lead Scoring & Call Decision Simulator")
    st.caption(
        "Pre-call individual prospect evaluation comparing Decision Tree (CART) and Logistic Regression."
    )

    variant = bundle["variant"]
    schema = bundle["raw_schema"]
    cats = schema["categorical"]
    nums = schema["numeric"]

    with st.form("single_lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 1. Demographic & Financial Profile")
            age = st.slider("Age (years)", min_value=18, max_value=95, value=42)
            job = st.selectbox("Job", options=cats.get("job", ["management", "technician", "retired"]))
            marital = st.selectbox("Marital Status", options=cats.get("marital", ["married", "single", "divorced"]))
            education = st.selectbox(
                "Education",
                options=cats.get("education", ["secondary", "tertiary", "primary", "unknown"]),
            )
            default_val = st.radio("Credit in Default", options=["no", "yes"], horizontal=True, index=0)
            housing_val = st.radio("Housing Loan", options=["no", "yes"], horizontal=True, index=0)
            loan_val = st.radio("Personal Loan", options=["no", "yes"], horizontal=True, index=0)
            if variant == "bank-full":
                balance = st.number_input(
                    "Average Yearly Balance (EUR)",
                    min_value=-10000,
                    max_value=150000,
                    value=1450,
                    step=100,
                )
            else:
                balance = None

        with col2:
            st.markdown("#### 2. Campaign History & Contact Plan")
            contact = st.selectbox("Contact Channel", options=cats.get("contact", ["cellular", "telephone"]))
            month_opts = cats.get("month", ["may", "apr", "oct", "sep", "mar"])
            default_month_idx = month_opts.index("oct") if "oct" in month_opts else 0
            month = st.selectbox("Target Campaign Month", options=month_opts, index=default_month_idx)

            if variant == "bank-full":
                day = st.slider("Contact Day of Month", min_value=1, max_value=31, value=15)
                day_of_week = None
            else:
                day = None
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=cats.get("day_of_week", ["mon", "tue", "wed", "thu", "fri"]),
                )

            campaign = st.number_input(
                "Contacts During Current Campaign (campaign)",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
            )
            never_contacted = st.checkbox(
                "Never Contacted Before (sets sentinel pdays automatically)",
                value=True,
            )
            pdays_input = st.number_input(
                "Days Since Last Prior Contact (ignored if Never Contacted)",
                min_value=0,
                max_value=900,
                value=90,
                step=5,
            )
            previous_input = st.number_input(
                "Prior Campaign Contacts (previous)",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )
            pout_opts = cats.get("poutcome", ["unknown", "failure", "other", "success"])
            poutcome = st.selectbox("Previous Campaign Outcome (poutcome)", options=pout_opts)

        with col3:
            st.markdown("#### 3. Macroeconomic Contextual Panel")
            if variant == "bank-additional-full":
                st.info("Active Model Predictors (bank-additional-full.csv)")
            else:
                st.info(
                    "Contextual Macroeconomic Scenario (Reference Only for bank-full.csv; switch variant in sidebar to activate as model predictors)"
                )

            emp_var_rate = st.slider(
                "Employment Variation Rate (emp.var.rate)",
                min_value=-3.5,
                max_value=1.5,
                value=float(nums.get("emp.var.rate", {}).get("median", -1.8)),
                step=0.1,
            )
            cons_price_idx = st.number_input(
                "Consumer Price Index (cons.price.idx)",
                min_value=92.0,
                max_value=95.0,
                value=float(nums.get("cons.price.idx", {}).get("median", 93.05)),
                step=0.05,
            )
            cons_conf_idx = st.slider(
                "Consumer Confidence Index (cons.conf.idx)",
                min_value=-51.0,
                max_value=-26.0,
                value=float(nums.get("cons.conf.idx", {}).get("median", -40.0)),
                step=0.5,
            )
            euribor3m = st.slider(
                "Euribor 3-Month Rate (euribor3m, %)",
                min_value=0.60,
                max_value=5.10,
                value=float(nums.get("euribor3m", {}).get("median", 1.35)),
                step=0.05,
            )
            nr_employed = st.number_input(
                "Number of Employees (nr.employed, thousands)",
                min_value=4950.0,
                max_value=5250.0,
                value=float(nums.get("nr.employed", {}).get("median", 5099.1)),
                step=10.0,
            )

        submitted = st.form_submit_button("Score Single Lead (Pre-Call Prediction)", width="stretch")

    sentinel_val = -1 if variant == "bank-full" else 999
    lead_dict: dict[str, Any] = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default_val,
        "housing": housing_val,
        "loan": loan_val,
        "contact": contact,
        "month": month,
        "campaign": int(campaign),
        "pdays": int(sentinel_val if never_contacted else pdays_input),
        "previous": int(0 if never_contacted else previous_input),
        "poutcome": poutcome,
    }
    if variant == "bank-full":
        lead_dict["balance"] = int(balance if balance is not None else 0)
        lead_dict["day"] = int(day if day is not None else 15)
    else:
        lead_dict["day_of_week"] = day_of_week
        lead_dict["emp.var.rate"] = float(emp_var_rate)
        lead_dict["cons.price.idx"] = float(cons_price_idx)
        lead_dict["cons.conf.idx"] = float(cons_conf_idx)
        lead_dict["euribor3m"] = float(euribor3m)
        lead_dict["nr.employed"] = float(nr_employed)

    result = score_single_lead(bundle, lead_dict, threshold=cfg["active_threshold"])

    if submitted:
        st.success("Pre-call lead scoring completed successfully without post-call leakage features.")

    st.markdown("### Conversion Probability & Call Priority Recommendation")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.plotly_chart(
            _build_probability_gauge(
                result["dt_prob"],
                cfg["active_threshold"],
                "Decision Tree (CART) Conversion Probability",
                DT_COLOR,
            ),
            width="stretch",
        )
        _render_priority_badge(result["dt_priority"], "Decision Tree")

    with g_col2:
        st.plotly_chart(
            _build_probability_gauge(
                result["lr_prob"],
                cfg["active_threshold"],
                "Logistic Regression Conversion Probability",
                LR_COLOR,
            ),
            width="stretch",
        )
        _render_priority_badge(result["lr_priority"], "Logistic Regression")

    st.markdown("### Local Explainability & Feature Contributions")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### Logistic Regression: Local Log-Odds Drivers")
        drivers_df = result["lr_local_contributions"]["top_drivers"]
        if not drivers_df.empty:
            bar_colors = [
                POS_COLOR if val >= 0 else NEG_COLOR for val in drivers_df["log_odds_contribution"]
            ]
            fig_local = go.Figure(
                go.Bar(
                    x=drivers_df["log_odds_contribution"],
                    y=drivers_df["feature"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.3f}" for v in drivers_df["log_odds_contribution"]],
                    textposition="outside",
                )
            )
            fig_local.update_layout(
                title="Top Positive & Negative Log-Odds Contributions (beta_j * x_j)",
                xaxis_title="Log-Odds Contribution",
                yaxis_title="Feature",
                height=360,
                margin=dict(l=20, r=40, t=45, b=30),
                template="plotly_white",
            )
            st.plotly_chart(fig_local, width="stretch")

            top_pos = result["lr_local_contributions"]["top_positive"]
            top_neg = result["lr_local_contributions"]["top_negative"]
            pos_txt = ", ".join(top_pos["feature"].head(2).tolist()) if not top_pos.empty else "none"
            neg_txt = ", ".join(top_neg["feature"].head(2).tolist()) if not top_neg.empty else "none"
            st.caption(
                f"Automated Explanation: Top positive probability drivers are [{pos_txt}], while main dampening drivers are [{neg_txt}]."
            )

    with exp_col2:
        st.markdown("#### Decision Tree: Root-to-Leaf Decision Path Trace")
        path_df = pd.DataFrame(result["dt_decision_path"])
        display_path = path_df[["step", "rule", "node_samples", "node_positive_rate"]].rename(
            columns={
                "step": "Step",
                "rule": "Decision Rule / Terminal Leaf",
                "node_samples": "Train Samples in Node",
                "node_positive_rate": "Node Positive Rate",
            }
        )
        display_path["Node Positive Rate"] = display_path["Node Positive Rate"].map(lambda x: f"{x:.1%}")
        st.dataframe(display_path, width="stretch", hide_index=True)
        st.caption(
            f"Terminal Leaf #{result['leaf_id']} contains {result['leaf_samples']:,} training samples with leaf conversion probability {result['leaf_positive_prob']:.1%}."
        )


def _render_module_2(bundle: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Modul 2: Batch Lead Prioritization & Campaign Planner."""
    st.subheader("Modul 2: Batch Lead Prioritization & Campaign Planner")
    st.caption(
        "Rank prospect batches descending by predicted conversion probability, optimize Call Capacity Quota, and export prioritized call lists."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        uploaded_file = st.file_uploader(
            "Upload Prospect Batch CSV (Pre-Call Features; optional target 'y')",
            type=["csv"],
            key="batch_uploader",
        )
    with ctrl2:
        ranking_model = st.selectbox(
            "Primary Ranking Model",
            options=["Logistic Regression", "Decision Tree"],
            index=0,
            key="batch_model_choice",
        )
    with ctrl3:
        st.write("")
        use_holdout_btn = st.button("Load Holdout Test Batch", width="stretch")

    # Prepare default holdout batch including synthetic duration demo if no file uploaded
    if "batch_source_mode" not in st.session_state:
        st.session_state["batch_source_mode"] = "holdout"
    if use_holdout_btn:
        st.session_state["batch_source_mode"] = "holdout"

    if uploaded_file is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            batch_input_df = pd.read_csv(uploaded_file)
        batch_label = f"Uploaded CSV ({uploaded_file.name}, {len(batch_input_df):,} rows)"
    else:
        batch_input_df = bundle["X_test"].copy()
        batch_input_df["y"] = bundle["y_test"]
        # Add illustrative duration column on holdout demo to showcase anti-leakage guard in action
        batch_input_df["duration"] = 180
        batch_label = f"Holdout Test Batch ({len(batch_input_df):,} unseen rows)"

    scored_df, batch_summary = score_batch_leads(
        bundle,
        batch_input_df,
        threshold=cfg["active_threshold"],
        model_choice=ranking_model,
    )
    audit = batch_summary["audit"]

    if audit["dropped_duration"]:
        st.markdown(
            f'<div class="guard-banner"><strong>Anti-Leakage Guard Triggered ({batch_label}):</strong> Post-call column <code>duration</code> was detected and automatically removed prior to pre-call lead scoring.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="guard-banner"><strong>Anti-Leakage Guard Verified ({batch_label}):</strong> No post-call <code>duration</code> column present; pre-call prediction integrity preserved.</div>',
            unsafe_allow_html=True,
        )

    quota_pct = st.slider(
        "Call Capacity Quota (% of Batch Leads to Contact)",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        key="batch_quota_slider",
    )

    sim = simulate_campaign_quota(
        scored_df=scored_df,
        quota_pct=float(quota_pct),
        baseline_prevalence=bundle["positive_prevalence"],
        call_cost=cfg["call_cost"],
        conversion_profit=cfg["conversion_profit"],
    )

    st.caption(
        f"Selected Call Quota: {sim['quota_calls']:,} of {sim['total_leads']:,} leads ({sim['quota_pct']:.0f}%) | Source Mode: {'Actual Holdout Ground Truth' if sim['is_actual_ground_truth'] else 'Calibrated Pre-Call Projection'}"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Captured Conversions", f"{sim['captured_conversions']:,.1f}" if not sim["is_actual_ground_truth"] else f"{int(sim['captured_conversions']):,}")
    k2.metric(
        "Projected Conversion Rate",
        f"{sim['projected_conversion_rate']:.1%}",
        delta=f"{(sim['projected_conversion_rate'] - bundle['positive_prevalence']) * 100:+.1f} pp vs baseline",
    )
    k3.metric("Lift vs Random Calling", f"{sim['lift_vs_random']:.2f}x")
    k4.metric("Avoided Unproductive Calls", f"{int(round(sim['avoided_unproductive_calls'])):,}")
    k5.metric(
        "Net Profit Impact (EUR)",
        f"{sim['expected_net_profit']:,.0f}",
        delta=f"{sim['profit_vs_blanket']:+,.0f} vs blanket calling",
    )

    chart_col, decile_col = st.columns([1.15, 1])

    with chart_col:
        st.markdown("#### Cumulative Gain Curve with Call Capacity Quota Marker")
        if "gain_lift" in batch_summary:
            gl = batch_summary["gain_lift"]
            pop_pct = gl["pop_frac"] * 100.0
            gain_pct = gl["gain_curve"] * 100.0
            decile_tbl = gl["decile_df"]
        else:
            probs = scored_df["active_probability"].to_numpy()
            cum_p = np.cumsum(probs)
            total_p = float(np.sum(probs)) if np.sum(probs) > 0 else 1.0
            gain_full = np.concatenate(([0.0], cum_p / total_p)) * 100.0
            pop_full = np.linspace(0.0, 100.0, len(gain_full))
            idx_samp = np.unique(np.linspace(0, len(gain_full) - 1, min(101, len(gain_full)), dtype=int))
            pop_pct = pop_full[idx_samp]
            gain_pct = gain_full[idx_samp]
            decile_tbl = None

        fig_gain = go.Figure()
        fig_gain.add_trace(
            go.Scatter(
                x=pop_pct,
                y=gain_pct,
                mode="lines",
                name=f"{ranking_model} Gain",
                line=dict(color=LR_COLOR if ranking_model == "Logistic Regression" else DT_COLOR, width=3),
            )
        )
        fig_gain.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode="lines",
                name="Random Calling Baseline",
                line=dict(color=BASELINE_COLOR, dash="dash", width=2),
            )
        )
        fig_gain.add_vline(
            x=quota_pct,
            line_dash="dot",
            line_color=WARN_COLOR,
            annotation_text=f"Quota {quota_pct}%",
        )
        fig_gain.add_trace(
            go.Scatter(
                x=[quota_pct],
                y=[sim["cumulative_gain"] * 100.0],
                mode="markers",
                name="Selected Quota Point",
                marker=dict(color=WARN_COLOR, size=11, symbol="diamond"),
            )
        )
        fig_gain.update_layout(
            xaxis_title="Percentage of Leads Contacted (%)",
            yaxis_title="Cumulative Gain (% of Total Conversions Captured)",
            height=360,
            margin=dict(l=20, r=20, t=35, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_gain, width="stretch")

    with decile_col:
        st.markdown("#### Decile Lead Ranking Summary (Top 10% to Top 100%)")
        if decile_tbl is not None:
            disp_decile = decile_tbl[
                ["decile_label", "leads_count", "conversions", "conversion_rate", "decile_lift", "cumulative_gain"]
            ].rename(
                columns={
                    "decile_label": "Segment",
                    "leads_count": "Leads",
                    "conversions": "Conversions",
                    "conversion_rate": "Conv Rate",
                    "decile_lift": "Decile Lift",
                    "cumulative_gain": "Cum Gain",
                }
            )
            disp_decile["Conv Rate"] = disp_decile["Conv Rate"].map(lambda x: f"{x:.1%}")
            disp_decile["Decile Lift"] = disp_decile["Decile Lift"].map(lambda x: f"{x:.2f}x")
            disp_decile["Cum Gain"] = disp_decile["Cum Gain"].map(lambda x: f"{x:.1%}")
            st.dataframe(disp_decile, width="stretch", hide_index=True, height=340)
        else:
            st.info("Decile conversion breakdown shows actual conversions when target column 'y' is present.")

    st.markdown("#### Prioritized Call List Preview & Export")
    preview_cols = [
        "priority_rank",
        "decile",
        "priority_action",
        "active_probability",
        "lr_probability",
        "dt_probability",
        "age",
        "job",
        "contact",
        "month",
        "campaign",
        "poutcome",
    ]
    available_preview = [c for c in preview_cols if c in scored_df.columns]
    if "actual_y" in scored_df.columns:
        available_preview.append("actual_y")
    st.dataframe(scored_df[available_preview].head(100), width="stretch", hide_index=True)

    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prioritized Call List (CSV)",
        data=csv_bytes,
        file_name=f"prioritized_leads_{bundle['variant']}.csv",
        mime="text/csv",
    )


def _sync_playground_threshold(new_val: float) -> None:
    """Callback updating global active_threshold safely."""
    st.session_state["active_threshold"] = float(new_val)


def _render_module_3(bundle: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Modul 3: Model Comparison & Evaluation Benchmark."""
    st.subheader("Modul 3: Model Comparison & Evaluation Benchmark")
    st.caption(
        "Side-by-side holdout evaluation, 5-Fold Stratified Cross-Validation, Interactive Probability Threshold tuning, and global feature explainability."
    )

    y_test = bundle["y_test"]
    dt_probs = bundle["dt_test_probs"]
    lr_probs = bundle["lr_test_probs"]
    dt_eval = bundle["eval_results"]["Decision Tree"]
    lr_eval = bundle["eval_results"]["Logistic Regression"]
    cv_dt = bundle["cv_metrics"]["Decision Tree"]
    cv_lr = bundle["cv_metrics"]["Logistic Regression"]

    st.markdown("### Interactive Threshold Tuning Playground")
    t_col1, t_col2, t_col3 = st.columns([2, 1, 1])
    with t_col1:
        play_threshold = st.slider(
            "Playground Probability Threshold (0.05 to 0.95)",
            min_value=0.05,
            max_value=0.95,
            value=cfg["active_threshold"],
            step=0.01,
            key="m3_playground_threshold",
        )
    with t_col2:
        st.metric("Cost-Benefit Optimal Threshold", f"{cfg['theoretical_tau']:.3f}")
    with t_col3:
        st.write("")
        st.button(
            "Apply Threshold to Global Config",
            on_click=_sync_playground_threshold,
            args=(play_threshold,),
            width="stretch",
        )

    dt_thresh = compute_threshold_metrics(
        y_test,
        dt_probs,
        threshold=play_threshold,
        call_cost=cfg["call_cost"],
        conversion_profit=cfg["conversion_profit"],
    )
    lr_thresh = compute_threshold_metrics(
        y_test,
        lr_probs,
        threshold=play_threshold,
        call_cost=cfg["call_cost"],
        conversion_profit=cfg["conversion_profit"],
    )

    st.markdown("### Side-by-Side Holdout & Cross-Validation Benchmark")
    bench_rows = [
        {
            "Metric": "5-Fold Stratified CV ROC-AUC",
            "Decision Tree (CART)": f"{cv_dt['cv_roc_auc_mean']:.4f} +/- {cv_dt['cv_roc_auc_std']:.4f}",
            "Logistic Regression": f"{cv_lr['cv_roc_auc_mean']:.4f} +/- {cv_lr['cv_roc_auc_std']:.4f}",
        },
        {
            "Metric": "5-Fold Stratified CV F1-Score",
            "Decision Tree (CART)": f"{cv_dt['cv_f1_mean']:.4f} +/- {cv_dt['cv_f1_std']:.4f}",
            "Logistic Regression": f"{cv_lr['cv_f1_mean']:.4f} +/- {cv_lr['cv_f1_std']:.4f}",
        },
        {
            "Metric": "Holdout ROC-AUC",
            "Decision Tree (CART)": f"{dt_eval['roc_auc']:.4f}",
            "Logistic Regression": f"{lr_eval['roc_auc']:.4f}",
        },
        {
            "Metric": "Holdout PR-AUC (Average Precision)",
            "Decision Tree (CART)": f"{dt_eval['pr_auc']:.4f}",
            "Logistic Regression": f"{lr_eval['pr_auc']:.4f}",
        },
        {
            "Metric": "Holdout ALIFT (Area under Cumulative Gain)",
            "Decision Tree (CART)": f"{dt_eval['alift']:.4f} (Norm: {dt_eval['normalized_alift']:.3f})",
            "Logistic Regression": f"{lr_eval['alift']:.4f} (Norm: {lr_eval['normalized_alift']:.3f})",
        },
        {
            "Metric": "Top 20% Decile Lift / Gain",
            "Decision Tree (CART)": f"{dt_eval['top_20_lift']:.2f}x ({dt_eval['top_20_gain']:.1%})",
            "Logistic Regression": f"{lr_eval['top_20_lift']:.2f}x ({lr_eval['top_20_gain']:.1%})",
        },
        {
            "Metric": f"Precision (@ tau={play_threshold:.2f})",
            "Decision Tree (CART)": f"{dt_thresh['precision']:.4f}",
            "Logistic Regression": f"{lr_thresh['precision']:.4f}",
        },
        {
            "Metric": f"Recall / Sensitivity (@ tau={play_threshold:.2f})",
            "Decision Tree (CART)": f"{dt_thresh['recall']:.4f}",
            "Logistic Regression": f"{lr_thresh['recall']:.4f}",
        },
        {
            "Metric": f"Specificity (@ tau={play_threshold:.2f})",
            "Decision Tree (CART)": f"{dt_thresh['specificity']:.4f}",
            "Logistic Regression": f"{lr_thresh['specificity']:.4f}",
        },
        {
            "Metric": f"F1-Score / F2-Score (@ tau={play_threshold:.2f})",
            "Decision Tree (CART)": f"{dt_thresh['f1']:.4f} / {dt_thresh['f2']:.4f}",
            "Logistic Regression": f"{lr_thresh['f1']:.4f} / {lr_thresh['f2']:.4f}",
        },
        {
            "Metric": f"Leads Called % / Expected Net Profit (EUR)",
            "Decision Tree (CART)": f"{dt_thresh['pct_called']:.1%} / {dt_thresh['expected_net_profit']:,.0f}",
            "Logistic Regression": f"{lr_thresh['pct_called']:.1%} / {lr_thresh['expected_net_profit']:,.0f}",
        },
    ]
    st.dataframe(pd.DataFrame(bench_rows), width="stretch", hide_index=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(
            _build_confusion_matrix_figure(
                dt_thresh,
                f"Decision Tree Confusion Matrix (tau={play_threshold:.2f})",
                "Teal",
            ),
            width="stretch",
        )
    with cm2:
        st.plotly_chart(
            _build_confusion_matrix_figure(
                lr_thresh,
                f"Logistic Regression Confusion Matrix (tau={play_threshold:.2f})",
                "Blues",
            ),
            width="stretch",
        )

    st.markdown("### Comparative Evaluation Curves (Holdout Test Set)")
    c_roc, c_pr, c_lift = st.columns(3)

    with c_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=dt_eval["roc_curve"]["fpr"],
                y=dt_eval["roc_curve"]["tpr"],
                mode="lines",
                name=f"Decision Tree (AUC={dt_eval['roc_auc']:.3f})",
                line=dict(color=DT_COLOR, width=2.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=lr_eval["roc_curve"]["fpr"],
                y=lr_eval["roc_curve"]["tpr"],
                mode="lines",
                name=f"Logistic Regression (AUC={lr_eval['roc_auc']:.3f})",
                line=dict(color=LR_COLOR, width=2.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random (0.500)",
                line=dict(color=BASELINE_COLOR, dash="dash"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (Recall)",
            height=350,
            margin=dict(l=15, r=15, t=45, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.22),
        )
        st.plotly_chart(fig_roc, width="stretch")

    with c_pr:
        prev = bundle["positive_prevalence"]
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=dt_eval["pr_curve"]["recall"],
                y=dt_eval["pr_curve"]["precision"],
                mode="lines",
                name=f"Decision Tree (AP={dt_eval['pr_auc']:.3f})",
                line=dict(color=DT_COLOR, width=2.5),
            )
        )
        fig_pr.add_trace(
            go.Scatter(
                x=lr_eval["pr_curve"]["recall"],
                y=lr_eval["pr_curve"]["precision"],
                mode="lines",
                name=f"Logistic Regression (AP={lr_eval['pr_auc']:.3f})",
                line=dict(color=LR_COLOR, width=2.5),
            )
        )
        fig_pr.add_hline(
            y=prev,
            line_dash="dash",
            line_color=BASELINE_COLOR,
            annotation_text=f"Baseline ({prev:.1%})",
        )
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=350,
            margin=dict(l=15, r=15, t=45, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.22),
        )
        st.plotly_chart(fig_pr, width="stretch")

    with c_lift:
        dt_dec = dt_eval["gain_lift"]["decile_df"]
        lr_dec = lr_eval["gain_lift"]["decile_df"]
        fig_lift = go.Figure()
        fig_lift.add_trace(
            go.Scatter(
                x=dt_dec["decile"],
                y=dt_dec["decile_lift"],
                mode="lines+markers",
                name="Decision Tree Lift",
                line=dict(color=DT_COLOR, width=2.5),
            )
        )
        fig_lift.add_trace(
            go.Scatter(
                x=lr_dec["decile"],
                y=lr_dec["decile_lift"],
                mode="lines+markers",
                name="Logistic Regression Lift",
                line=dict(color=LR_COLOR, width=2.5),
            )
        )
        fig_lift.add_hline(
            y=1.0,
            line_dash="dash",
            line_color=BASELINE_COLOR,
            annotation_text="Random (1.0x)",
        )
        fig_lift.update_layout(
            title="Decile Lift Curve",
            xaxis_title="Decile (1 = Top 10% Highest Scores)",
            yaxis_title="Decile Lift Multiplier",
            xaxis=dict(tickmode="linear", dtick=1),
            height=350,
            margin=dict(l=15, r=15, t=45, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.22),
        )
        st.plotly_chart(fig_lift, width="stretch")

    st.markdown("### Global Feature Explainability: Decision Tree vs Logistic Regression")
    g1, g2 = st.columns(2)

    with g1:
        dt_imp = bundle["dt_feature_importance"].head(12).sort_values("importance", ascending=True)
        fig_dt_imp = px.bar(
            dt_imp,
            x="importance",
            y="feature",
            orientation="h",
            title="Decision Tree Top 12 Feature Importances (Impurity Reduction)",
            labels={"importance": "Relative Importance", "feature": "Feature"},
            color_discrete_sequence=[DT_COLOR],
            template="plotly_white",
        )
        fig_dt_imp.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_dt_imp, width="stretch")

    with g2:
        lr_coefs = bundle["lr_coefficients"].copy()
        top_pos_lr = lr_coefs.loc[lr_coefs["coefficient"] > 0].head(7)
        top_neg_lr = lr_coefs.loc[lr_coefs["coefficient"] < 0].head(7)
        top_lr = pd.concat([top_pos_lr, top_neg_lr], ignore_index=True).sort_values(
            "coefficient", ascending=True
        )
        colors_lr = [POS_COLOR if c >= 0 else NEG_COLOR for c in top_lr["coefficient"]]
        fig_lr_coef = go.Figure(
            go.Bar(
                x=top_lr["coefficient"],
                y=top_lr["feature"],
                orientation="h",
                marker_color=colors_lr,
                text=[f"OR={or_val:.2f}x" for or_val in top_lr["odds_ratio"]],
                textposition="outside",
            )
        )
        fig_lr_coef.update_layout(
            title="Logistic Regression Top Coefficients and Odds Ratios (exp(beta))",
            xaxis_title="Standardized / Log-Odds Coefficient (beta_j)",
            yaxis_title="Feature",
            height=420,
            margin=dict(l=20, r=60, t=50, b=30),
            template="plotly_white",
        )
        st.plotly_chart(fig_lr_coef, width="stretch")


def main() -> None:
    """Render Bank Telemarketing Predictive Lead Scoring Dashboard."""
    _inject_custom_styles()
    cfg = _render_sidebar()

    bundle = train_cached_bundle(
        variant=cfg["variant"],
        use_sample=cfg["use_sample"],
        imbalance_strategy=cfg["imbalance_strategy"],
        dt_max_depth=cfg["dt_max_depth"],
        dt_criterion=cfg["dt_criterion"],
        lr_c=cfg["lr_c"],
        lr_penalty=cfg["lr_penalty"],
    )

    st.title("Bank Telemarketing Predictive Lead Scoring Dashboard")
    st.caption(
        "Pre-Call Classification Decision Support System comparing Decision Tree (CART) and Logistic Regression with strict Anti-Leakage Guard."
    )

    audit = bundle["leakage_audit"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dataset Variant", bundle["variant"])
    m2.metric("Total Samples (80/20 Split)", f"{bundle['train_size'] + bundle['test_size']:,}")
    m3.metric("Positive Conversion Rate", f"{bundle['positive_prevalence']:.2%}")
    m4.metric("Anti-Leakage (duration Dropped)", "Yes" if audit["dropped_duration"] else "Verified Clean")
    m5.metric("Training & 5-Fold CV Time", f"{bundle['training_time_sec']:.2f}s")

    st.caption(
        f"Data Source: {bundle['data_source_note']} | Imbalance Strategy: {cfg['imbalance_label']} | Sentinel pdays Normalized: {audit['sentinel_pdays_count']:,} rows"
    )

    if cfg["imbalance_strategy"] in {"class_weight", "smote"}:
        st.info(
            "Probability Calibration Shift Notice: Class Weighting and SMOTE balance the training prior toward 50%, shifting raw predicted probabilities upward compared to the ~11.7% real-world baseline. Use ROC-AUC, PR-AUC, Lift Curve, and Threshold Tuning rather than assuming 0.50 represents an uncalibrated conversion rate."
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "Modul 1: Single Lead Scoring & Call Decision Simulator",
            "Modul 2: Batch Lead Prioritization & Campaign Planner",
            "Modul 3: Model Comparison & Evaluation Benchmark",
        ]
    )

    with tab1:
        _render_module_1(bundle, cfg)
    with tab2:
        _render_module_2(bundle, cfg)
    with tab3:
        _render_module_3(bundle, cfg)


if __name__ == "__main__":
    main()
