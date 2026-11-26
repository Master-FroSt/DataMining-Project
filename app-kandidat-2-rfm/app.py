"""Kandidat 2: Interactive RFM & K-Means Customer Segmentation Dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import load_or_download_retail_data, load_uploaded_retail_data
from src.kmeans_engine import evaluate_k_range, fit_kmeans_and_profile
from src.rfm_pipeline import (
    RFM_COLUMNS,
    clean_transactions_for_rfm,
    compute_rfm_table,
    transform_and_scale_rfm,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

PALETTE = [
    "#1D4ED8",
    "#059669",
    "#D97706",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#DB2777",
    "#4B5563",
    "#65A30D",
    "#EA580C",
]

OUTLIER_OPTIONS = {
    "Winsorizing (3 x IQR)": "winsorize_3iqr",
    "Winsorizing (1st - 99th Percentile)": "winsorize_p99",
    "IQR Filtering (1.5 x IQR Drop)": "iqr_1.5",
    "None": "none",
}

SKEW_OPTIONS = {
    "Log Transformation ln(1+x)": "log1p",
    "Box-Cox Transformation": "boxcox",
    "None": "none",
}

SCALER_OPTIONS = {
    "StandardScaler (Z-Score)": "standard",
    "MinMaxScaler (0-1)": "minmax",
}


st.set_page_config(
    page_title="Kandidat 2 - RFM & K-Means Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .header-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: #F8FAFC;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .header-card h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .header-card p {
        margin: 0.35rem 0 0 0;
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    .skew-badge {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #1D4ED8;
        border-radius: 6px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        color: #0F172A;
    }
    .crm-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-top: 4px solid #1D4ED8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        color: #0F172A;
    }
    .crm-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 1.02rem;
        color: #0F172A;
    }
    .crm-meta {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1D4ED8;
        margin-bottom: 0.55rem;
    }
    .crm-row {
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _cached_load_dataset(
    source_mode: str,
    uploaded_bytes: bytes | None,
    uploaded_name: str | None,
) -> tuple[pd.DataFrame, str, str]:
    if source_mode == "Custom Upload (CSV / Excel / Parquet)":
        if uploaded_bytes is None or not uploaded_name:
            raise ValueError("Please upload a transaction file to continue.")
        df = load_uploaded_retail_data(uploaded_bytes, uploaded_name)
        return df, "custom_upload", ""
    use_sample = source_mode == "Built-in Sample (Fast Benchmark)"
    df = load_or_download_retail_data(DATA_DIR, use_sample=use_sample)
    source_tag = str(df.attrs.get("data_source", "built_in_sample"))
    fallback_reason = str(df.attrs.get("fallback_reason", ""))
    return df, source_tag, fallback_reason


@st.cache_data(show_spinner=False)
def _cached_clean_and_rfm(
    raw_df: pd.DataFrame,
    country_filter: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.Timestamp]:
    clean_df, audit = clean_transactions_for_rfm(raw_df, country_filter=country_filter)
    rfm_df, t_ref = compute_rfm_table(clean_df)
    return clean_df, audit, rfm_df, t_ref


@st.cache_data(show_spinner=False)
def _cached_transform_rfm(
    rfm_df: pd.DataFrame,
    outlier_method: str,
    skew_method: str,
    scaler_method: str,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    return transform_and_scale_rfm(
        rfm_df,
        outlier_method=outlier_method,
        skew_method=skew_method,
        scaler_method=scaler_method,
    )


@st.cache_data(show_spinner=False)
def _cached_evaluate_k(
    X_scaled: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eval_df = evaluate_k_range(
        X_scaled,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    meta = {
        "recommended_k": int(eval_df.attrs["recommended_k"]),
        "elbow_k": int(eval_df.attrs["elbow_k"]),
        "silhouette_best_k": int(eval_df.attrs["silhouette_best_k"]),
        "dbi_best_k": int(eval_df.attrs["dbi_best_k"]),
    }
    return eval_df, meta


@st.cache_data(show_spinner=False)
def _cached_fit_profile(
    rfm_filtered_df: pd.DataFrame,
    X_scaled: np.ndarray,
    k_selected: int,
    random_state: int = 42,
) -> dict:
    return fit_kmeans_and_profile(
        rfm_filtered_df,
        X_scaled,
        k=k_selected,
        random_state=random_state,
    )


def _persona_color_map(personas: list[str]) -> dict[str, str]:
    return {persona: PALETTE[idx % len(PALETTE)] for idx, persona in enumerate(personas)}


def main() -> None:
    st.markdown(
        """
        <div class="header-card">
            <h1>Kandidat 2: Customer Segmentation Using RFM & K-Means Clustering</h1>
            <p>End-to-end CRISP-DM & KDD workflow: Transaction Cleaning, RFM Feature Engineering, Skewness & Outlier Control, Multi-Metric K-Means Optimization, and CRM Action Center</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Sidebar Control Center")

        st.subheader("1. Dataset Source")
        source_mode = st.radio(
            "Select Transaction Source",
            options=[
                "Built-in Sample (Fast Benchmark)",
                "UCI Online Retail (Download + Parquet)",
                "Custom Upload (CSV / Excel / Parquet)",
            ],
            index=0,
        )

        uploaded_bytes: bytes | None = None
        uploaded_name: str | None = None
        if source_mode == "Custom Upload (CSV / Excel / Parquet)":
            uploaded_file = st.file_uploader(
                "Upload Online Retail Transaction Log",
                type=["csv", "xlsx", "xls", "parquet"],
            )
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = uploaded_file.name

    try:
        with st.spinner("Loading transaction dataset..."):
            raw_df, source_tag, fallback_reason = _cached_load_dataset(
                source_mode,
                uploaded_bytes,
                uploaded_name,
            )
    except Exception as exc:
        st.info(f"Waiting for valid dataset input: {exc}")
        return

    if source_tag == "fallback_sample":
        st.warning(
            f"UCI Online Retail download unavailable ({fallback_reason}). "
            "Gracefully using built-in 12-month multi-segment benchmark sample."
        )
    elif source_tag == "parquet_cache":
        st.caption("Dataset Status: Loaded UCI Online Retail from local Parquet cache.")
    elif source_tag == "uci_download":
        st.caption("Dataset Status: Downloaded UCI Online Retail and cached to Parquet.")
    elif source_tag == "custom_upload":
        st.caption(f"Dataset Status: Loaded custom file '{uploaded_name}'.")
    else:
        st.caption("Dataset Status: Running on Built-in 12-Month Multi-Segment Benchmark Sample.")

    country_counts = (
        raw_df["Country"].dropna().astype(str).str.strip().value_counts()
        if "Country" in raw_df.columns
        else pd.Series(dtype=int)
    )
    country_options = ["All Countries"] + list(country_counts.index)

    with st.sidebar:
        country_filter = st.selectbox("Country Filter", options=country_options, index=0)

        st.subheader("2. Preprocessing Pipeline")
        outlier_label = st.selectbox(
            "Outlier Handling",
            options=list(OUTLIER_OPTIONS.keys()),
            index=0,
        )
        skew_label = st.selectbox(
            "Skewness Transformation",
            options=list(SKEW_OPTIONS.keys()),
            index=0,
        )
        scaler_label = st.selectbox(
            "Feature Scaler",
            options=list(SCALER_OPTIONS.keys()),
            index=0,
        )

    outlier_method = OUTLIER_OPTIONS[outlier_label]
    skew_method = SKEW_OPTIONS[skew_label]
    scaler_method = SCALER_OPTIONS[scaler_label]

    try:
        with st.spinner("Cleaning transactions and computing RFM metrics..."):
            _, audit, rfm_raw_df, t_ref = _cached_clean_and_rfm(raw_df, country_filter)
    except Exception as exc:
        st.error(
            f"No valid customer transactions remained after cleaning for '{country_filter}': {exc}"
        )
        return

    if len(rfm_raw_df) < 4:
        st.warning(
            f"Only {len(rfm_raw_df)} unique customers found for '{country_filter}'. "
            "Please select 'All Countries' or a larger market with at least 4 customers."
        )
        return

    X_scaled, rfm_filtered_df, diagnostics = _cached_transform_rfm(
        rfm_raw_df,
        outlier_method=outlier_method,
        skew_method=skew_method,
        scaler_method=scaler_method,
    )

    max_k_allowed = min(10, len(rfm_filtered_df) - 1)
    if max_k_allowed < 2:
        st.warning("Not enough customers remain after filtering to fit K-Means clustering.")
        return

    with st.sidebar:
        st.subheader("3. K-Means Modeling")
        default_k = min(4, max_k_allowed)
        k_selected = st.slider(
            "Optimal Cluster Count (k)",
            min_value=2,
            max_value=max_k_allowed,
            value=default_k,
            step=1,
            help="Locks the final k used for Segment Profiling and Customer Lookup in Module 3.",
        )
        st.caption("Initialization: k-means++ | random_state=42 | n_init=10")

    profile_bundle = _cached_fit_profile(
        rfm_filtered_df,
        X_scaled,
        k_selected=int(k_selected),
        random_state=42,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Modul 1: RFM Distribution & Preprocessing Monitor",
            "Modul 2: Cluster Optimization & Evaluation Lab",
            "Modul 3: Segment Profiling & Marketing Action Center",
        ]
    )

    # ------------------------------------------------------------------
    # MODULE 1: RFM Distribution & Preprocessing Monitor
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Active Customer & Transaction Cleaning KPIs")
        span_days = (
            int((audit["date_max"] - audit["date_min"]).days)
            if audit["date_min"] is not None and audit["date_max"] is not None
            else 0
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                "Clean vs Raw Transactions",
                f"{audit['clean_rows']:,} / {audit['raw_rows']:,}",
                delta=f"{audit['retention_rate_pct']:.1f}% Retained",
            )
        with kpi2:
            st.metric(
                "Active Customers (Modeling)",
                f"{diagnostics['rows_after']:,}",
                delta=(
                    f"-{diagnostics['rows_dropped']} IQR Dropped"
                    if diagnostics["rows_dropped"] > 0
                    else f"{audit['unique_invoices']:,} Unique Invoices"
                ),
            )
        with kpi3:
            st.metric(
                "Reference Date (T_ref)",
                t_ref.strftime("%Y-%m-%d"),
                delta=f"{span_days} Days Span",
            )
        with kpi4:
            st.metric(
                "Median R / F / M Baseline",
                f"{rfm_raw_df['Recency'].median():.0f}d | {rfm_raw_df['Frequency'].median():.0f}x | GBP {rfm_raw_df['Monetary'].median():,.0f}",
                delta=f"Total GBP {audit['total_revenue']:,.0f}",
            )

        with st.expander("Transaction Cleaning Audit Breakdown", expanded=False):
            audit_rows = [
                {"Audit Step": "Raw Input Rows", "Row Count": audit["raw_rows"]},
                {
                    "Audit Step": "Removed: Missing CustomerID",
                    "Row Count": -audit["removed_missing_customer"],
                },
                {
                    "Audit Step": "Removed: Cancellation Invoices ('C')",
                    "Row Count": -audit["removed_cancellations"],
                },
                {
                    "Audit Step": "Removed: Quantity <= 0",
                    "Row Count": -audit["removed_non_positive_qty"],
                },
                {
                    "Audit Step": "Removed: UnitPrice <= 0",
                    "Row Count": -audit["removed_non_positive_price"],
                },
                {
                    "Audit Step": "Removed: Admin StockCodes (POST, M, DOT, etc.)",
                    "Row Count": -audit["removed_admin_codes"],
                },
                {
                    "Audit Step": "Removed: Exact Duplicate Rows",
                    "Row Count": -audit["removed_duplicates"],
                },
                {
                    "Audit Step": f"Removed: Country Filter ({audit['country_filter']})",
                    "Row Count": -audit["removed_country_filter"],
                },
                {"Audit Step": "Final Clean Transaction Rows", "Row Count": audit["clean_rows"]},
            ]
            st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)

        st.divider()
        st.subheader("Before vs After Transformation Histograms & Skewness Monitor")
        st.caption(
            f"Active Preprocessing: Outlier={outlier_label} | Skewness={skew_label} | Scaler={scaler_label}"
        )

        scaled_df = diagnostics["rfm_scaled_df"]
        skew_before = diagnostics["skewness_before"]
        skew_after = diagnostics["skewness_after"]
        skew_delta = diagnostics["skewness_delta"]
        capped_map = diagnostics["outliers_capped"]

        for col_name in RFM_COLUMNS:
            sb = skew_before[col_name]
            sa = skew_after[col_name]
            sd = skew_delta[col_name]
            capped_cnt = capped_map.get(col_name, 0)
            bc_lam = diagnostics["boxcox_lambdas"].get(col_name)
            lam_note = f" | Box-Cox lambda={bc_lam:.3f}" if bc_lam is not None else ""
            cap_note = f" | Winsorized Capped={capped_cnt}" if capped_cnt > 0 else ""

            st.markdown(
                f"""
                <div class="skew-badge">
                    <strong>{col_name}</strong> &mdash;
                    Before Skewness: <strong>{sb:.3f}</strong> |
                    After Skewness: <strong>{sa:.3f}</strong> |
                    |Skew| Change: <strong>{sd:+.3f}</strong>{cap_note}{lam_note}
                </div>
                """,
                unsafe_allow_html=True,
            )

            left_col, right_col = st.columns(2)
            with left_col:
                fig_before = px.histogram(
                    rfm_raw_df,
                    x=col_name,
                    nbins=40,
                    title=f"{col_name} - Raw Distribution (Skew = {sb:.2f})",
                    color_discrete_sequence=["#64748B"],
                )
                fig_before.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=42, b=20),
                    yaxis_title="Customers",
                )
                st.plotly_chart(fig_before, width="stretch")

            with right_col:
                fig_after = px.histogram(
                    scaled_df,
                    x=col_name,
                    nbins=40,
                    title=f"{col_name} - Transformed & Scaled (Skew = {sa:.2f})",
                    color_discrete_sequence=["#1D4ED8"],
                )
                fig_after.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=42, b=20),
                    yaxis_title="Customers",
                )
                st.plotly_chart(fig_after, width="stretch")

        st.subheader("Comparative Boxplot Grid (Before vs After Outlier Handling & Scaling)")
        box_fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Raw Recency",
                "Raw Frequency",
                "Raw Monetary",
                "Scaled Recency",
                "Scaled Frequency",
                "Scaled Monetary",
            ],
            vertical_spacing=0.18,
        )
        for col_idx, col_name in enumerate(RFM_COLUMNS, start=1):
            box_fig.add_trace(
                go.Box(
                    y=rfm_raw_df[col_name],
                    name=f"Raw {col_name}",
                    marker_color="#64748B",
                    boxmean=True,
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            box_fig.add_trace(
                go.Box(
                    y=scaled_df[col_name],
                    name=f"Scaled {col_name}",
                    marker_color="#1D4ED8",
                    boxmean=True,
                    showlegend=False,
                ),
                row=2,
                col=col_idx,
            )
        box_fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(box_fig, width="stretch")

    # ------------------------------------------------------------------
    # MODULE 2: Cluster Optimization & Evaluation Lab
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Multi-Metric K-Means Optimization (k-means++)")

        if max_k_allowed > 2:
            k_range = st.slider(
                "Evaluation Range (k_min to k_max)",
                min_value=2,
                max_value=max_k_allowed,
                value=(2, max_k_allowed),
                step=1,
            )
            k_min_eval, k_max_eval = int(k_range[0]), int(k_range[1])
            if k_min_eval == k_max_eval:
                k_min_eval = 2
                k_max_eval = max(3, k_max_eval) if max_k_allowed >= 3 else 2
        else:
            k_min_eval, k_max_eval = 2, 2

        eval_df, eval_meta = _cached_evaluate_k(
            X_scaled,
            k_min=k_min_eval,
            k_max=k_max_eval,
            random_state=42,
        )

        rec_k = eval_meta["recommended_k"]
        elbow_k = eval_meta["elbow_k"]
        sil_k = eval_meta["silhouette_best_k"]
        dbi_k = eval_meta["dbi_best_k"]

        st.info(
            f"Consensus Optimal Cluster Count: k = {rec_k} | "
            f"Elbow Inflection: k = {elbow_k} | "
            f"Peak Silhouette Score: k = {sil_k} | "
            f"Lowest Davies-Bouldin Index: k = {dbi_k} | "
            f"Active Sidebar Selection: k = {k_selected}"
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            elbow_fig = go.Figure()
            elbow_fig.add_trace(
                go.Scatter(
                    x=eval_df["k"],
                    y=eval_df["WCSS"],
                    mode="lines+markers+text",
                    name="WCSS (Inertia)",
                    line=dict(color="#1D4ED8", width=3),
                    marker=dict(size=9),
                    text=[f"{v:,.0f}" for v in eval_df["WCSS"]],
                    textposition="top right",
                )
            )
            elbow_fig.add_vline(
                x=elbow_k,
                line_dash="dash",
                line_color="#059669",
                annotation_text=f"Elbow k={elbow_k}",
                annotation_position="top left",
            )
            if k_selected != elbow_k:
                elbow_fig.add_vline(
                    x=k_selected,
                    line_dash="dot",
                    line_color="#DC2626",
                    annotation_text=f"Active k={k_selected}",
                    annotation_position="bottom right",
                )
            elbow_fig.update_layout(
                title="Elbow Method Chart (WCSS / Inertia vs k)",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="WCSS (Inertia)",
                height=390,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(elbow_fig, width="stretch")

        with chart_col2:
            dual_fig = make_subplots(specs=[[{"secondary_y": True}]])
            dual_fig.add_trace(
                go.Scatter(
                    x=eval_df["k"],
                    y=eval_df["Silhouette Score"],
                    mode="lines+markers",
                    name="Silhouette Score (Maximize)",
                    line=dict(color="#059669", width=3),
                    marker=dict(size=9),
                ),
                secondary_y=False,
            )
            dual_fig.add_trace(
                go.Scatter(
                    x=eval_df["k"],
                    y=eval_df["Davies-Bouldin Index"],
                    mode="lines+markers",
                    name="Davies-Bouldin Index (Minimize)",
                    line=dict(color="#D97706", width=3, dash="dash"),
                    marker=dict(size=9, symbol="diamond"),
                ),
                secondary_y=True,
            )
            dual_fig.add_vline(
                x=rec_k,
                line_dash="dot",
                line_color="#1D4ED8",
                annotation_text=f"Consensus k={rec_k}",
            )
            dual_fig.update_layout(
                title="Silhouette Score vs Davies-Bouldin Index Across k",
                xaxis_title="Number of Clusters (k)",
                height=390,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            dual_fig.update_yaxes(title_text="Silhouette Score", secondary_y=False)
            dual_fig.update_yaxes(title_text="Davies-Bouldin Index", secondary_y=True)
            st.plotly_chart(dual_fig, width="stretch")

        st.subheader("Comparative Validation Metrics Table")
        display_eval = eval_df.copy()
        display_eval["Status"] = display_eval.apply(
            lambda r: "Consensus Recommended"
            if bool(r["Recommended"])
            else ("Active Selection" if int(r["k"]) == int(k_selected) else ""),
            axis=1,
        )
        st.dataframe(display_eval, width="stretch", hide_index=True)

        st.subheader(f"Cluster Size Balance (Active k = {k_selected})")
        profile_df = profile_bundle["profile_df"]
        color_map = _persona_color_map(profile_df["Persona"].tolist())

        bal_fig = px.bar(
            profile_df.sort_values("Customer Count", ascending=True),
            x="Customer Count",
            y="Persona",
            orientation="h",
            color="Persona",
            color_discrete_map=color_map,
            text=profile_df.sort_values("Customer Count", ascending=True).apply(
                lambda r: f"{int(r['Customer Count']):,} ({r['Customer Share (%)']:.1f}%)",
                axis=1,
            ),
            title="Customer Population Distribution per Segment",
        )
        bal_fig.update_traces(textposition="outside")
        bal_fig.update_layout(
            height=max(280, 60 * len(profile_df)),
            showlegend=False,
            xaxis_title="Customer Count",
            yaxis_title="",
            margin=dict(l=20, r=80, t=45, b=20),
        )
        st.plotly_chart(bal_fig, width="stretch")

    # ------------------------------------------------------------------
    # MODULE 3: Segment Profiling & Marketing Action Center
    # ------------------------------------------------------------------
    with tab3:
        profile_df = profile_bundle["profile_df"]
        cust_df = profile_bundle["customer_df"]
        snake_df = profile_bundle["snake_df"]
        radar_df = profile_bundle["radar_df"]
        pca_centroids_df = profile_bundle["pca_centroids_df"]
        pca_var = profile_bundle["pca_explained_variance"]
        marketing_matrix_df = profile_bundle["marketing_matrix_df"]
        fit_metrics = profile_bundle["metrics"]
        color_map = _persona_color_map(profile_df["Persona"].tolist())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected k", f"{k_selected}")
        m2.metric("Silhouette Score", f"{fit_metrics['silhouette_score']:.4f}")
        m3.metric("Davies-Bouldin Index", f"{fit_metrics['davies_bouldin_index']:.4f}")
        m4.metric("Calinski-Harabasz Index", f"{fit_metrics['calinski_harabasz_index']:,.1f}")

        st.subheader("Cluster Behavioral Signatures: Snake Plot & Radar Chart")
        sig_col1, sig_col2 = st.columns(2)

        with sig_col1:
            snake_melted = snake_df.melt(
                id_vars=["Cluster", "Persona"],
                value_vars=["Recency", "Frequency", "Monetary"],
                var_name="RFM Dimension",
                value_name="Standardized Z-Score",
            )
            snake_fig = px.line(
                snake_melted,
                x="RFM Dimension",
                y="Standardized Z-Score",
                color="Persona",
                color_discrete_map=color_map,
                markers=True,
                title="Snake Plot (Standardized RFM Centroids vs Population Mean)",
            )
            snake_fig.add_hline(y=0.0, line_dash="dash", line_color="#64748B")
            snake_fig.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="top", y=-0.18),
            )
            st.plotly_chart(snake_fig, width="stretch")

        with sig_col2:
            radar_fig = go.Figure()
            dims = ["Recency (Recentness)", "Frequency", "Monetary"]
            for _, row in radar_df.iterrows():
                persona = str(row["Persona"])
                vals = [float(row[d]) for d in dims]
                vals_closed = vals + [vals[0]]
                dims_closed = dims + [dims[0]]
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=vals_closed,
                        theta=dims_closed,
                        fill="toself",
                        name=persona,
                        line=dict(color=color_map[persona], width=2),
                        opacity=0.65,
                    )
                )
            radar_fig.update_layout(
                title="Radar Chart (Normalized Strength; Recency Inverted)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
                height=390,
                margin=dict(l=40, r=40, t=50, b=20),
                legend=dict(orientation="h", yanchor="top", y=-0.18),
            )
            st.plotly_chart(radar_fig, width="stretch")

        st.subheader("Spatial Cluster Separation: 3D RFM Scatter & 2D PCA Projection")
        spatial_col1, spatial_col2 = st.columns(2)

        with spatial_col1:
            scatter_space = st.radio(
                "3D Scatter Coordinate Space",
                options=["Original Unscaled RFM (Days / Orders / GBP)", "Standardized Z-Score Space"],
                horizontal=True,
                key="scatter_3d_space",
            )
            if scatter_space.startswith("Original"):
                x_col, y_col, z_col = "Recency", "Frequency", "Monetary"
                title_3d = "Interactive 3D RFM Scatter (Unscaled Business Space)"
            else:
                x_col, y_col, z_col = "z_Recency", "z_Frequency", "z_Monetary"
                title_3d = "Interactive 3D RFM Scatter (Standardized Space)"

            fig_3d = px.scatter_3d(
                cust_df,
                x=x_col,
                y=y_col,
                z=z_col,
                color="Persona",
                color_discrete_map=color_map,
                hover_data=["CustomerID", "Country", "Recency", "Frequency", "Monetary"],
                opacity=0.72,
                title=title_3d,
            )
            fig_3d.update_traces(marker=dict(size=4))
            fig_3d.update_layout(
                height=460,
                margin=dict(l=10, r=10, t=45, b=10),
                legend=dict(orientation="h", yanchor="top", y=-0.05),
            )
            st.plotly_chart(fig_3d, width="stretch")

        with spatial_col2:
            var1 = pca_var[0] * 100.0
            var2 = pca_var[1] * 100.0
            fig_pca = px.scatter(
                cust_df,
                x="PCA1",
                y="PCA2",
                color="Persona",
                color_discrete_map=color_map,
                hover_data=["CustomerID", "Country", "Recency", "Frequency", "Monetary"],
                opacity=0.65,
                title="2D PCA Projection with Cluster Centroids",
            )
            fig_pca.add_trace(
                go.Scatter(
                    x=pca_centroids_df["PCA1"],
                    y=pca_centroids_df["PCA2"],
                    mode="markers+text",
                    name="Cluster Centroids",
                    text=[f"C{int(c)}" for c in pca_centroids_df["Cluster"]],
                    textposition="top center",
                    marker=dict(
                        symbol="star",
                        size=18,
                        color="#FACC15",
                        line=dict(width=1.5, color="#0F172A"),
                    ),
                )
            )
            fig_pca.update_layout(
                xaxis_title=f"PCA 1 ({var1:.1f}% Variance)",
                yaxis_title=f"PCA 2 ({var2:.1f}% Variance)",
                height=495,
                margin=dict(l=20, r=20, t=45, b=20),
                legend=dict(orientation="h", yanchor="top", y=-0.16),
            )
            st.plotly_chart(fig_pca, width="stretch")

        st.subheader("Unscaled Business Centroid Profile & Revenue Share Table")
        st.caption(
            "Computed directly from unscaled raw customer RFM metrics per cluster "
            "(avoids lossy inverse-transformation distortion from Winsorizing or Log transforms). "
            "Personas are deterministically ranked by composite health score S_c = z(F_c) + z(M_c) - z(R_c)."
        )
        st.dataframe(profile_df, width="stretch", hide_index=True)

        st.subheader("CRM Marketing Strategy Matrix Cards")
        card_cols = st.columns(2)
        for idx, row in marketing_matrix_df.iterrows():
            target_col = card_cols[idx % 2]
            persona_name = str(row["Persona"])
            accent_color = color_map.get(persona_name, "#1D4ED8")
            with target_col:
                st.markdown(
                    f"""
                    <div class="crm-card" style="border-top-color: {accent_color};">
                        <h4>Cluster {int(row['Cluster'])}: {persona_name}</h4>
                        <div class="crm-meta">
                            Customers: {int(row['Customer Count']):,} ({row['Customer Share (%)']:.1f}%) |
                            Revenue Share: {row['Revenue Share (%)']:.1f}% |
                            Budget: {row['Budget Allocation']}
                        </div>
                        <div class="crm-row"><strong>Primary Objective:</strong> {row['Primary Objective']}</div>
                        <div class="crm-row"><strong>Recommended Actions:</strong> {row['Recommended Actions']}</div>
                        <div class="crm-row"><strong>Incentive Type:</strong> {row['Incentive Type']}</div>
                        <div class="crm-row"><strong>Priority Channel:</strong> {row['Priority Channel']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Customer Lookup & Target Export Table")
        flt1, flt2, flt3 = st.columns([1.2, 1.4, 1.4])
        with flt1:
            cid_search = st.text_input("Search CustomerID", value="", placeholder="e.g. 12347")
        with flt2:
            persona_options = profile_df["Persona"].tolist()
            selected_personas = st.multiselect(
                "Filter by Segment Persona",
                options=persona_options,
                default=persona_options,
            )
        with flt3:
            country_list = sorted(cust_df["Country"].unique().tolist())
            selected_countries = st.multiselect(
                "Filter by Country",
                options=country_list,
                default=country_list,
            )

        min_m = float(cust_df["Monetary"].min())
        max_m = float(cust_df["Monetary"].max())
        if max_m > min_m:
            m_low, m_high = st.slider(
                "Monetary Spend Range (GBP)",
                min_value=float(np.floor(min_m)),
                max_value=float(np.ceil(max_m)),
                value=(float(np.floor(min_m)), float(np.ceil(max_m))),
            )
        else:
            m_low, m_high = min_m, max_m

        filtered_lookup = cust_df.copy()
        if cid_search.strip():
            filtered_lookup = filtered_lookup[
                filtered_lookup["CustomerID"].astype(str).str.contains(cid_search.strip(), case=False, na=False)
            ]
        if selected_personas:
            filtered_lookup = filtered_lookup[filtered_lookup["Persona"].isin(selected_personas)]
        else:
            filtered_lookup = filtered_lookup.iloc[0:0]
        if selected_countries:
            filtered_lookup = filtered_lookup[filtered_lookup["Country"].isin(selected_countries)]
        else:
            filtered_lookup = filtered_lookup.iloc[0:0]

        filtered_lookup = filtered_lookup[
            (filtered_lookup["Monetary"] >= m_low) & (filtered_lookup["Monetary"] <= m_high)
        ].sort_values("Monetary", ascending=False)

        export_cols = [
            "CustomerID",
            "Country",
            "Recency",
            "Frequency",
            "Monetary",
            "Cluster",
            "Persona",
            "Silhouette",
        ]
        lookup_view = filtered_lookup[export_cols].reset_index(drop=True)

        l_kpi1, l_kpi2, l_kpi3 = st.columns(3)
        l_kpi1.metric("Matching Target Customers", f"{len(lookup_view):,}")
        l_kpi2.metric(
            "Filtered Segment Revenue",
            f"GBP {lookup_view['Monetary'].sum():,.2f}" if not lookup_view.empty else "GBP 0.00",
        )
        l_kpi3.metric(
            "Avg Monetary in Selection",
            f"GBP {lookup_view['Monetary'].mean():,.2f}" if not lookup_view.empty else "GBP 0.00",
        )

        st.dataframe(lookup_view, width="stretch", hide_index=True)

        csv_bytes = lookup_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Target Customer List (CSV)",
            data=csv_bytes,
            file_name="rfm_target_customers.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
