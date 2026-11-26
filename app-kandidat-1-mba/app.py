"""Interactive Market Basket Analysis (FP-Growth) Dashboard on UCI Online Retail."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_or_download_retail_data,
    load_uploaded_retail_file,
)
from src.fpgrowth_engine import (
    mine_fpgrowth_rules,
    simulate_bundling_recommendations,
)
from src.preprocessing import (
    build_sparse_basket_matrix,
    clean_retail_transactions,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MIN_ITEM_FREQ_FLOOR = 0.003

st.set_page_config(
    page_title="Market Basket Analysis - FP-Growth Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean enterprise CSS styling without emojis
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1E3A8A;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #0F766E;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    .bundle-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .badge-strong {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-high {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #93C5FD;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-niche {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_kpi_card(label: str, value: str, subtext: str = "") -> None:
    """Render a clean, non-emoji KPI summary card."""
    sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _badge_html(badge_text: str) -> str:
    """Map an affinity badge string to clean CSS badge markup."""
    if badge_text == "Strong Bundle Candidate":
        return f'<span class="badge-strong">{badge_text}</span>'
    if badge_text == "High-Confidence Cross-Sell":
        return f'<span class="badge-high">{badge_text}</span>'
    return f'<span class="badge-niche">{badge_text}</span>'


@st.cache_data(show_spinner=False)
def _load_cached_dataset(
    source_mode: str,
    uploaded_bytes: bytes | None,
    uploaded_name: str | None,
) -> tuple[pd.DataFrame, str, str]:
    """Load raw transactional dataset and return (df, source_tag, fallback_reason)."""
    if source_mode == "Upload Custom CSV / Excel":
        if not uploaded_bytes or not uploaded_name:
            return pd.DataFrame(), "awaiting_upload", ""
        buffer = io.BytesIO(uploaded_bytes)
        buffer.name = uploaded_name
        df = load_uploaded_retail_file(buffer)
        return df, str(df.attrs.get("data_source", "uploaded_file")), ""

    use_sample = source_mode == "Built-in UCI Benchmark Sample (Fast Default)"
    df = load_or_download_retail_data(DATA_DIR, use_sample=use_sample)
    return (
        df,
        str(df.attrs.get("data_source", "sample")),
        str(df.attrs.get("fallback_reason", "")),
    )


@st.cache_data(show_spinner=False)
def _prepare_cached_basket_pipeline(
    source_mode: str,
    file_hash: str,
    uploaded_bytes: bytes | None,
    uploaded_name: str | None,
    drop_missing_customer_id: bool,
    country_filter: str,
    min_item_freq: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run cached Data Cleaning and Sparse One-Hot Encoding Basket Matrix generation."""
    raw_df, _, _ = _load_cached_dataset(source_mode, uploaded_bytes, uploaded_name)
    if raw_df.empty:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    clean_df, cleaning_audit = clean_retail_transactions(
        raw_df,
        drop_missing_customer_id=drop_missing_customer_id,
        country_filter=country_filter,
    )
    basket_df, item_metadata_df = build_sparse_basket_matrix(
        clean_df,
        item_col="Description",
        min_item_freq=min_item_freq,
    )
    return clean_df, cleaning_audit, basket_df, item_metadata_df


@st.cache_data(show_spinner=False)
def _run_cached_fpgrowth(
    pipeline_key: tuple[str, str, bool, str, float],
    _basket_df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_len: int,
    filter_redundant: bool,
) -> dict[str, Any]:
    """Execute cached FP-Growth rule mining keyed by preprocessing tuple and sliders."""
    _ = pipeline_key
    return mine_fpgrowth_rules(
        _basket_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        filter_redundant=filter_redundant,
    )


def _build_cleaning_audit_chart(audit: dict[str, Any]) -> go.Figure:
    """Create a Plotly horizontal waterfall chart summarizing sequential Data Cleaning steps."""
    labels = [
        "Raw Transactions",
        "Cancelled ('C') Removed",
        "Missing Description",
        "Non-Positive Qty/Price",
        "Admin Stock Codes",
        "Missing CustomerID",
        "Exact Duplicates",
        f"Country Filter ({audit.get('country_filter', 'All')})",
        "Final Clean Rows",
    ]
    measures = [
        "absolute",
        "relative",
        "relative",
        "relative",
        "relative",
        "relative",
        "relative",
        "relative",
        "total",
    ]
    values = [
        audit["raw_rows"],
        -audit["cancelled_removed"],
        -audit["missing_description_removed"],
        -audit["non_positive_removed"],
        -audit["admin_stock_codes_removed"],
        -audit["missing_customer_removed"],
        -audit["duplicates_removed"],
        -audit["country_filtered_removed"],
        audit["final_clean_rows"],
    ]

    fig = go.Figure(
        go.Waterfall(
            name="Data Cleaning Audit",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"{abs(v):,}" if m == "relative" else f"{v:,}" for v, m in zip(values, measures)],
            textposition="outside",
            connector={"line": {"color": "#94A3B8", "width": 1.2}},
            decreasing={"marker": {"color": "#DC2626"}},
            increasing={"marker": {"color": "#0F766E"}},
            totals={"marker": {"color": "#1E3A8A"}},
        )
    )
    fig.update_layout(
        title="Data Cleaning Audit Waterfall (Row Retention Trace)",
        yaxis_title="Row Count",
        xaxis_tickangle=-22,
        height=390,
        margin=dict(l=20, r=20, t=50, b=70),
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _build_basket_size_chart(clean_df: pd.DataFrame) -> go.Figure:
    """Build Basket Size Distribution histogram across distinct commercial SKUs per invoice."""
    basket_sizes = clean_df.groupby("InvoiceNo", sort=False)["Description"].nunique()
    clipped_sizes = basket_sizes.clip(upper=30)

    bins = [0, 1, 5, 15, 10_000]
    labels = ["1 Item (Single)", "2-5 Items (Standard)", "6-15 Items (Multi-Pack)", ">15 Items (Wholesaler)"]
    segments = pd.cut(basket_sizes, bins=bins, labels=labels)
    seg_counts = segments.value_counts().reindex(labels, fill_value=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=seg_counts.index.tolist(),
            y=seg_counts.values.tolist(),
            text=[f"{int(v):,}" for v in seg_counts.values],
            textposition="outside",
            marker_color=["#64748B", "#2563EB", "#0D9488", "#1E3A8A"],
            hovertemplate="%{x}<br>Invoices: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Basket Size Distribution (Max Observed: {int(basket_sizes.max())} SKUs, Mean: {float(clipped_sizes.mean()):.2f})",
        xaxis_title="Basket Size Tier (Distinct SKUs per Invoice)",
        yaxis_title="Number of Invoices",
        height=390,
        margin=dict(l=20, r=20, t=50, b=50),
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _build_rule_scatter_plot(rules_df: pd.DataFrame) -> go.Figure:
    """Create interactive Support vs Confidence scatter plot colored by Lift and sized by Leverage."""
    plot_df = rules_df.copy()
    plot_df["conviction_str"] = plot_df["conviction"].map(
        lambda v: "inf" if np.isinf(v) else f"{v:.3f}"
    )
    plot_df["bubble_size"] = np.clip(plot_df["leverage"].fillna(0.0), 0.0, None) + 0.002

    fig = px.scatter(
        plot_df,
        x="support",
        y="confidence",
        color="lift",
        size="bubble_size",
        size_max=22,
        color_continuous_scale="Viridis",
        custom_data=[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "conviction_str",
            "leverage",
        ],
        title="Association Rules Landscape: Support vs Confidence (Color: Lift, Size: Leverage)",
        labels={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift Ratio",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]} -> %{customdata[1]}</b><br>"
            "Support: %{customdata[2]:.4f}<br>"
            "Confidence: %{customdata[3]:.2%}<br>"
            "Lift: %{customdata[4]:.2f}<br>"
            "Conviction: %{customdata[5]}<br>"
            "Leverage: %{customdata[6]:.5f}<extra></extra>"
        )
    )
    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def _build_network_graph(
    rules_df: pd.DataFrame,
    item_metadata_df: pd.DataFrame,
    anchor_items: list[str],
    graph_mode: str,
    max_rules: int = 25,
) -> tuple[go.Figure, int, int]:
    """Build an interactive Plotly + NetworkX Directed Network Graph for product associations."""
    anchor_set = {a.strip() for a in anchor_items if a.strip()}
    if graph_mode == "Ego-Network (Selected Anchor Items)" and anchor_set:
        sub_rules = rules_df.loc[
            rules_df["antecedents_frozenset"].apply(lambda s: bool(s & anchor_set))
            | rules_df["consequents_frozenset"].apply(lambda s: bool(s & anchor_set))
        ].head(max_rules)
        if sub_rules.empty:
            sub_rules = rules_df.head(max_rules)
    else:
        sub_rules = rules_df.head(max_rules)

    graph = nx.DiGraph()
    support_lookup = (
        item_metadata_df.set_index("item")["support"].to_dict()
        if not item_metadata_df.empty
        else {}
    )

    for _, row in sub_rules.iterrows():
        for ant in row["antecedents_frozenset"]:
            for cons in row["consequents_frozenset"]:
                if ant == cons:
                    continue
                if not graph.has_node(ant):
                    graph.add_node(ant, support=float(support_lookup.get(ant, 0.01)))
                if not graph.has_node(cons):
                    graph.add_node(cons, support=float(support_lookup.get(cons, 0.01)))
                existing = graph.get_edge_data(ant, cons)
                if existing is None or row["lift"] > existing["lift"]:
                    graph.add_edge(
                        ant,
                        cons,
                        lift=float(row["lift"]),
                        confidence=float(row["confidence"]),
                        support=float(row["support"]),
                    )

    if graph.number_of_nodes() == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No graph nodes available for current rule selection.", height=420)
        return empty_fig, 0, 0

    k_dist = 1.6 / math.sqrt(max(graph.number_of_nodes(), 1))
    pos = nx.spring_layout(graph, k=k_dist, iterations=60, seed=42)

    fig = go.Figure()
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []

    max_lift = max((d["lift"] for _, _, d in graph.edges(data=True)), default=1.0)
    min_lift = min((d["lift"] for _, _, d in graph.edges(data=True)), default=1.0)
    lift_span = max(max_lift - min_lift, 1e-6)

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        norm_lift = (data["lift"] - min_lift) / lift_span
        width = 1.4 + 3.2 * norm_lift
        color = "#1E3A8A" if norm_lift >= 0.6 else ("#0D9488" if norm_lift >= 0.25 else "#94A3B8")

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Shorten arrow tip slightly so arrowhead stops near target node boundary
        tip_x = x0 + 0.86 * (x1 - x0)
        tip_y = y0 + 0.86 * (y1 - y0)
        fig.add_annotation(
            x=tip_x,
            y=tip_y,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=max(1.2, width * 0.7),
            arrowcolor=color,
            opacity=0.85,
        )
        edge_mid_x.append((x0 + x1) / 2.0)
        edge_mid_y.append((y0 + y1) / 2.0)
        edge_hover.append(
            f"{u} -> {v}<br>Confidence: {data['confidence']:.1%}<br>"
            f"Lift: {data['lift']:.2f}<br>Support: {data['support']:.4f}"
        )

    # Invisible midpoint markers for edge hover inspection
    if edge_mid_x:
        fig.add_trace(
            go.Scatter(
                x=edge_mid_x,
                y=edge_mid_y,
                mode="markers",
                marker=dict(size=10, color="rgba(0,0,0,0)"),
                hoverinfo="text",
                hovertext=edge_hover,
                showlegend=False,
            )
        )

    node_x: list[float] = []
    node_y: list[float] = []
    node_sizes: list[float] = []
    node_colors: list[str] = []
    node_labels: list[str] = []
    node_hover: list[str] = []

    supports = [graph.nodes[n]["support"] for n in graph.nodes()]
    max_sup = max(supports) if supports else 0.05
    min_sup = min(supports) if supports else 0.01
    sup_span = max(max_sup - min_sup, 1e-6)

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        sup_val = graph.nodes[node]["support"]
        norm_s = (sup_val - min_sup) / sup_span
        node_sizes.append(20.0 + 26.0 * norm_s)

        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        if node in anchor_set:
            node_colors.append("#DC2626")
            role = "Selected Anchor Product"
        elif out_deg > 0 and in_deg == 0:
            node_colors.append("#1E3A8A")
            role = "Primary Antecedent"
        elif in_deg > 0 and out_deg == 0:
            node_colors.append("#0F766E")
            role = "Recommended Consequent"
        else:
            node_colors.append("#D97706")
            role = "Bidirectional Hub"

        short_label = node if len(node) <= 24 else f"{node[:21]}..."
        node_labels.append(short_label)
        node_hover.append(
            f"<b>{node}</b><br>Role: {role}<br>Item Support: {sup_val:.2%}<br>"
            f"Out-Links: {out_deg} | In-Links: {in_deg}"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont=dict(size=10, color="#0F172A"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1.5, color="#FFFFFF"),
            ),
            hoverinfo="text",
            hovertext=node_hover,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Directed Product Association Network Graph ({graph_mode})",
        height=540,
        margin=dict(l=15, r=15, t=50, b=15),
        plot_bgcolor="#F8FAFC",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig, graph.number_of_nodes(), graph.number_of_edges()


def main() -> None:
    """Run the Streamlit Market Basket Analysis dashboard."""
    st.title("Market Basket Analysis & Product Bundling Workbench (FP-Growth)")
    st.caption(
        "CRISP-DM / KDD Analytical Workbench for UCI Online Retail Transactions | "
        "Sparse One-Hot Encoding, FP-Tree Mining, Multi-Metric Evaluation, and Cross-Selling Simulation"
    )

    # ------------------------------------------------------------------
    # SIDEBAR CONTROL CENTER
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Control Center")
        st.subheader("1. Dataset Source")
        source_mode = st.selectbox(
            "Select Transaction Source",
            options=[
                "Built-in UCI Benchmark Sample (Fast Default)",
                "UCI Full Dataset (Download & Parquet Cache)",
                "Upload Custom CSV / Excel",
            ],
            index=0,
        )

        uploaded_bytes: bytes | None = None
        uploaded_name: str | None = None
        if source_mode == "Upload Custom CSV / Excel":
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel (.csv, .xlsx)",
                type=["csv", "xlsx", "xls"],
            )
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = uploaded_file.name

        raw_df, data_source_tag, fallback_reason = _load_cached_dataset(
            source_mode,
            uploaded_bytes,
            uploaded_name,
        )

        if data_source_tag == "sample_fallback":
            st.warning(
                "UCI server download failed or timed out; using built-in UCI benchmark sample instead."
            )
            if fallback_reason:
                st.caption(f"Fallback detail: {fallback_reason[:120]}")
        elif data_source_tag == "uci_parquet_cache":
            st.success("Loaded cached UCI Full Dataset from Parquet.")
        elif data_source_tag == "uci_downloaded":
            st.success("Downloaded UCI Full Dataset and cached to Parquet.")

        st.subheader("2. Data Cleaning & Scope")
        if not raw_df.empty and "Country" in raw_df.columns:
            unique_countries = sorted(raw_df["Country"].dropna().astype(str).unique().tolist())
            ordered_countries: list[str] = []
            if "United Kingdom" in unique_countries:
                ordered_countries.append("United Kingdom")
            ordered_countries.append("All Countries")
            for country in unique_countries:
                if country not in ordered_countries:
                    ordered_countries.append(country)
        else:
            ordered_countries = ["United Kingdom", "All Countries"]

        country_filter = st.selectbox("Country Filter", options=ordered_countries, index=0)
        drop_missing_customer_id = st.toggle(
            "Drop Missing CustomerID (~24.9%)",
            value=True,
            help="Exclude guest-checkout/null CustomerID records to align with standard UCI cleaning benchmarks.",
        )

        st.subheader("3. FP-Growth Parameters")
        min_support = st.slider(
            "Minimum Support",
            min_value=0.003,
            max_value=0.050,
            value=0.010,
            step=0.001,
            format="%.3f",
            help="Minimum fraction of valid transactions containing the itemset.",
        )
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            format="%.2f",
            help="Minimum conditional probability P(Consequent | Antecedent).",
        )
        min_lift = st.slider(
            "Minimum Lift Ratio",
            min_value=1.0,
            max_value=10.0,
            value=1.2,
            step=0.1,
            format="%.1f",
            help="Lift > 1.0 confirms positive correlation beyond random chance.",
        )
        max_len = st.slider(
            "Maximum Itemset Length (max_len)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Caps itemset cardinality inside FP-Tree exploration.",
        )
        filter_redundant = st.checkbox(
            "Filter Redundant Rules",
            value=True,
            help="Remove superset antecedent rules whose Confidence is <= a simpler subset rule with the same Consequent.",
        )

    if raw_df.empty:
        st.info(
            "Upload a CSV or Excel file containing columns: InvoiceNo, StockCode, Description, Quantity, UnitPrice "
            "(plus optional InvoiceDate, CustomerID, Country), or switch to the Built-in UCI Benchmark Sample."
        )
        return

    file_hash = hashlib.sha256(uploaded_bytes).hexdigest()[:16] if uploaded_bytes else "builtin"
    with st.spinner("Running Data Cleaning and Sparse One-Hot Encoding..."):
        clean_df, cleaning_audit, basket_df, item_metadata_df = _prepare_cached_basket_pipeline(
            source_mode=source_mode,
            file_hash=file_hash,
            uploaded_bytes=uploaded_bytes,
            uploaded_name=uploaded_name,
            drop_missing_customer_id=drop_missing_customer_id,
            country_filter=country_filter,
            min_item_freq=MIN_ITEM_FREQ_FLOOR,
        )

    if clean_df.empty:
        st.warning("Zero valid transactions remained after applying Data Cleaning and Country filters.")
        return

    pipeline_key = (
        source_mode,
        file_hash,
        drop_missing_customer_id,
        country_filter,
        MIN_ITEM_FREQ_FLOOR,
    )
    with st.spinner("Mining Frequent Itemsets and Association Rules with FP-Growth..."):
        mining_result = _run_cached_fpgrowth(
            pipeline_key=pipeline_key,
            _basket_df=basket_df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            filter_redundant=filter_redundant,
        )

    rules_df: pd.DataFrame = mining_result["rules"]

    # ------------------------------------------------------------------
    # THREE MODULES TAB LAYOUT
    # ------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "Modul 1: Dataset Explorer & Transaction Overview",
            "Modul 2: Dynamic Rule Explorer & Parameter Tuning",
            "Modul 3: Product Bundling & Recommendation Simulator",
        ]
    )

    # ==================================================================
    # MODUL 1: DATASET EXPLORER & TRANSACTION OVERVIEW
    # ==================================================================
    with tab1:
        st.subheader("Transaction Overview & Data Quality KPIs")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            _render_kpi_card(
                "Raw vs Clean Rows",
                f"{cleaning_audit['final_clean_rows']:,}",
                f"From {cleaning_audit['raw_rows']:,} raw ({cleaning_audit['retained_pct']:.1f}% retained)",
            )
        with k2:
            _render_kpi_card(
                "Unique Transactions",
                f"{cleaning_audit['unique_invoices']:,}",
                f"Filter: {cleaning_audit['country_filter']}",
            )
        with k3:
            _render_kpi_card(
                "Commercial SKUs",
                f"{cleaning_audit['unique_skus']:,}",
                f"{basket_df.shape[1]:,} active >= {MIN_ITEM_FREQ_FLOOR:.1%} support",
            )
        with k4:
            _render_kpi_card(
                "Average Basket Size",
                f"{cleaning_audit['avg_basket_size']:.2f} SKUs",
                f"Median: {cleaning_audit['median_basket_size']:.1f} SKUs/invoice",
            )
        with k5:
            _render_kpi_card(
                "Multi-Item Basket Ratio",
                f"{cleaning_audit['multi_item_basket_ratio']:.1%}",
                f"{cleaning_audit['multi_item_baskets']:,} multi-item invoices",
            )

        st.plotly_chart(_build_cleaning_audit_chart(cleaning_audit), width="stretch")

        with st.expander("Detailed Data Cleaning Audit Table", expanded=False):
            audit_rows = [
                ("Raw Input Rows", cleaning_audit["raw_rows"]),
                ("Removed: Cancelled Invoices ('C')", -cleaning_audit["cancelled_removed"]),
                ("Removed: Missing Description", -cleaning_audit["missing_description_removed"]),
                ("Removed: Non-Positive Quantity / UnitPrice", -cleaning_audit["non_positive_removed"]),
                ("Removed: Administrative StockCodes (POST, DOT, M, CRUK, etc.)", -cleaning_audit["admin_stock_codes_removed"]),
                ("Removed: Missing CustomerID Policy", -cleaning_audit["missing_customer_removed"]),
                ("Removed: Exact Duplicate Line-Items", -cleaning_audit["duplicates_removed"]),
                (f"Removed: Country Filter ({cleaning_audit['country_filter']})", -cleaning_audit["country_filtered_removed"]),
                ("Final Clean Rows", cleaning_audit["final_clean_rows"]),
                ("Canonicalized Descriptions per StockCode", cleaning_audit["canonicalized_descriptions"]),
            ]
            st.dataframe(
                pd.DataFrame(audit_rows, columns=["Audit Step", "Row Count / Impact"]),
                width="stretch",
                hide_index=True,
            )

        col_left, col_right = st.columns(2)
        with col_left:
            top_n = st.slider("Top-N Best-Selling Products", min_value=5, max_value=25, value=15, step=1)
            top_items = item_metadata_df.head(top_n).sort_values("transaction_count", ascending=True)
            fig_top = go.Figure(
                go.Bar(
                    x=top_items["transaction_count"],
                    y=top_items["item"],
                    orientation="h",
                    marker_color="#1E3A8A",
                    customdata=np.stack(
                        [top_items["support"], top_items["total_quantity"], top_items["avg_unit_price"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Invoices: %{x:,}<br>"
                        "Support: %{customdata[0]:.2%}<br>"
                        "Total Units Sold: %{customdata[1]:,}<br>"
                        "Avg UnitPrice: GBP %{customdata[2]:.2f}<extra></extra>"
                    ),
                )
            )
            fig_top.update_layout(
                title=f"Top-{top_n} Products by Transaction Frequency",
                xaxis_title="Unique Invoice Count",
                yaxis_title="",
                height=420,
                margin=dict(l=10, r=20, t=50, b=40),
                plot_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_top, width="stretch")

        with col_right:
            st.write("")
            st.write("")
            st.plotly_chart(_build_basket_size_chart(clean_df), width="stretch")

    # ==================================================================
    # MODUL 2: DYNAMIC RULE EXPLORER & PARAMETER TUNING
    # ==================================================================
    with tab2:
        st.subheader("FP-Growth Execution Summary & Rule Landscape")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            _render_kpi_card(
                "Frequent Itemsets",
                f"{mining_result['n_itemsets']:,}",
                f"min_support = {min_support:.3f} | max_len = {max_len}",
            )
        with s2:
            _render_kpi_card(
                "Valid Association Rules",
                f"{mining_result['n_rules']:,}",
                f"conf >= {min_confidence:.0%} | lift >= {min_lift:.1f}",
            )
        with s3:
            _render_kpi_card(
                "Redundant Rules Pruned",
                f"{mining_result['n_redundant_removed']:,}",
                "Subsumption filter active" if filter_redundant else "Redundancy filter off",
            )
        with s4:
            _render_kpi_card(
                "FP-Growth Computation Time",
                f"{mining_result['execution_time_sec']:.4f} s",
                f"Matrix: {basket_df.shape[0]:,} x {basket_df.shape[1]:,} (Sparse)",
            )

        if rules_df.empty:
            st.warning(
                "No association rules matched the current thresholds. "
                "Lower Minimum Support (e.g., to 0.005), Minimum Confidence (e.g., to 0.30), "
                "or Minimum Lift Ratio in the Sidebar Control Center to discover rules."
            )
        else:
            st.plotly_chart(_build_rule_scatter_plot(rules_df), width="stretch")

            st.subheader("Interactive Association Rules Table")
            f_col1, f_col2, f_col3 = st.columns([2.2, 1.2, 1.0])
            with f_col1:
                keyword_query = st.text_input(
                    "Filter Rules by Product Keyword",
                    value="",
                    placeholder="e.g., REGENCY, JUMBO BAG, ALARM CLOCK, HEART",
                ).strip()
            with f_col2:
                sort_metric = st.selectbox(
                    "Sort Rules By",
                    options=["lift", "confidence", "support", "leverage", "conviction"],
                    index=0,
                )
            with f_col3:
                sort_ascending = st.checkbox("Ascending Order", value=False)

            display_df = rules_df.copy()
            if keyword_query:
                q_upper = keyword_query.upper()
                mask = (
                    display_df["antecedents"].str.upper().str.contains(q_upper, regex=False)
                    | display_df["consequents"].str.upper().str.contains(q_upper, regex=False)
                )
                display_df = display_df.loc[mask].copy()

            display_df = display_df.sort_values(
                by=[sort_metric, "confidence", "support"],
                ascending=[sort_ascending, False, False],
            ).reset_index(drop=True)

            if display_df.empty:
                st.info(f"No association rules match keyword '{keyword_query}'.")
            else:
                table_view = display_df[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift",
                        "conviction",
                        "leverage",
                        "rule_len",
                    ]
                ].rename(
                    columns={
                        "antecedents": "Antecedent (Product A)",
                        "consequents": "Consequent (Product B)",
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift",
                        "conviction": "Conviction",
                        "leverage": "Leverage",
                        "rule_len": "Rule Length",
                    }
                )
                st.dataframe(
                    table_view,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Support": st.column_config.NumberColumn(format="%.4f"),
                        "Confidence": st.column_config.NumberColumn(format="%.2%"),
                        "Lift": st.column_config.NumberColumn(format="%.2f"),
                        "Conviction": st.column_config.NumberColumn(format="%.3f"),
                        "Leverage": st.column_config.NumberColumn(format="%.5f"),
                    },
                )

                export_cols = [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift",
                    "conviction",
                    "leverage",
                    "antecedent_support",
                    "consequent_support",
                    "rule_len",
                ]
                csv_payload = display_df[export_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Export Filtered Association Rules (CSV)",
                    data=csv_payload,
                    file_name="fpgrowth_association_rules.csv",
                    mime="text/csv",
                )

    # ==================================================================
    # MODUL 3: PRODUCT BUNDLING & RECOMMENDATION SIMULATOR
    # ==================================================================
    with tab3:
        st.subheader("Product Bundling & Cross-Selling Impact Simulator")
        if rules_df.empty:
            st.warning(
                "No active association rules are available for simulation. "
                "Adjust Minimum Support or Minimum Confidence in the sidebar first."
            )
            return

        # Prefer products with single-item antecedent rules first for intuitive Anchor Product A selection
        single_ant_rules = rules_df.loc[rules_df["antecedent_len"] == 1]
        if not single_ant_rules.empty:
            single_anchors = sorted(
                {
                    item
                    for item_set in single_ant_rules["antecedents_frozenset"]
                    for item in item_set
                }
            )
            multi_only_anchors = sorted(
                {
                    item
                    for item_set in rules_df["antecedents_frozenset"]
                    for item in item_set
                }
                - set(single_anchors)
            )
            anchor_candidates = single_anchors + multi_only_anchors
            top_single_item = sorted(single_ant_rules.iloc[0]["antecedents_frozenset"])[0]
            default_anchor = [top_single_item]
        else:
            anchor_candidates = sorted(
                {
                    item
                    for item_set in rules_df["antecedents_frozenset"]
                    for item in item_set
                }
            )
            default_anchor = sorted(rules_df.iloc[0]["antecedents_frozenset"])

        sim_c1, sim_c2, sim_c3 = st.columns([2.2, 1.0, 1.0])
        with sim_c1:
            selected_anchors = st.multiselect(
                "Select Anchor Product(s) A in Basket",
                options=anchor_candidates,
                default=default_anchor,
                help="Choose one or more items currently in the customer's cart to simulate Product B recommendations.",
            )
        with sim_c2:
            top_k = st.slider("Top-K Recommendations", min_value=3, max_value=20, value=10, step=1)
        with sim_c3:
            discount_pct = st.slider(
                "Bundle Discount (%)",
                min_value=0,
                max_value=35,
                value=10,
                step=1,
            )

        sim_df = simulate_bundling_recommendations(
            rules_df=rules_df,
            item_metadata_df=item_metadata_df,
            anchor_items=selected_anchors,
            discount_pct=float(discount_pct),
            top_k=top_k,
        )

        if not selected_anchors:
            st.info("Select at least one Anchor Product A above to simulate bundling recommendations.")
        elif sim_df.empty:
            st.info(
                "No Product B rules matched the selected anchor basket at current thresholds. "
                "Try adding another item or lowering Minimum Confidence / Minimum Lift."
            )
        else:
            st.markdown("#### Top Recommended Product B Bundles")
            top_cards = sim_df.head(3)
            card_cols = st.columns(len(top_cards))
            for col, (_, rec) in zip(card_cols, top_cards.iterrows()):
                with col:
                    st.markdown(
                        f"""
                        <div class="bundle-card">
                            <div>{_badge_html(rec['affinity_badge'])}</div>
                            <div style="font-weight:700; font-size:0.98rem; color:#0F172A; margin-top:0.45rem;">
                                {rec['recommended_item']}
                            </div>
                            <div style="font-size:0.78rem; color:#475569; margin-bottom:0.5rem;">
                                Triggered by: {rec['triggered_by']}
                            </div>
                            <div style="font-size:0.84rem; color:#1E293B;">
                                <b>Confidence:</b> {rec['conditional_prob']:.1%} (vs Baseline {rec['baseline_prob']:.1%})<br>
                                <b>Lift:</b> {rec['lift']:.2f}x | <b>Conviction:</b> {"inf" if np.isinf(rec["conviction"]) else f"{rec['conviction']:.2f}"}<br>
                                <b>Incremental Basket Value:</b> GBP {rec['estimated_incremental_basket_value']:.2f}
                                <span style="color:#64748B;">(Unit: GBP {rec['unit_price']:.2f})</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Cross-Selling Impact Calculator")
            best_rec = sim_df.iloc[0]
            total_expected_uplift = float(sim_df["estimated_incremental_basket_value"].sum())
            avg_lift = float(sim_df["lift"].mean())

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                _render_kpi_card(
                    "Top Recommendation Uplift",
                    f"{best_rec['conditional_prob']:.1%}",
                    f"+{best_rec['relative_uplift_pct']:.0f}% vs {best_rec['baseline_prob']:.1%} baseline",
                )
            with ic2:
                _render_kpi_card(
                    "Best Single Incremental Value",
                    f"GBP {best_rec['estimated_incremental_basket_value']:.2f}",
                    f"After {discount_pct}% bundle discount (Gross: GBP {best_rec['gross_incremental_value']:.2f})",
                )
            with ic3:
                _render_kpi_card(
                    "Cumulative Top-K Expected Value",
                    f"GBP {total_expected_uplift:.2f}",
                    f"Mean Lift across Top-{len(sim_df)}: {avg_lift:.2f}x",
                )

            # Baseline vs Conditional Probability grouped bar chart
            fig_uplift = go.Figure()
            fig_uplift.add_trace(
                go.Bar(
                    name="Baseline Purchase Probability (Support B)",
                    x=sim_df["recommended_item"],
                    y=sim_df["baseline_prob"],
                    marker_color="#94A3B8",
                    hovertemplate="%{x}<br>Baseline Support: %{y:.2%}<extra></extra>",
                )
            )
            fig_uplift.add_trace(
                go.Bar(
                    name="Conditional Purchase Probability (Confidence A -> B)",
                    x=sim_df["recommended_item"],
                    y=sim_df["conditional_prob"],
                    marker_color="#1E3A8A",
                    hovertemplate="%{x}<br>Conditional Confidence: %{y:.2%}<extra></extra>",
                )
            )
            fig_uplift.update_layout(
                barmode="group",
                title="Cross-Selling Conversion Lift: Baseline vs Conditional Purchase Probability",
                yaxis_title="Purchase Probability",
                yaxis_tickformat=".0%",
                xaxis_tickangle=-18,
                height=380,
                margin=dict(l=20, r=20, t=55, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                plot_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_uplift, width="stretch")

            sim_table = sim_df[
                [
                    "triggered_by",
                    "recommended_item",
                    "affinity_badge",
                    "conditional_prob",
                    "baseline_prob",
                    "lift",
                    "unit_price",
                    "discounted_unit_price",
                    "estimated_incremental_basket_value",
                ]
            ].rename(
                columns={
                    "triggered_by": "Triggered By (Product A)",
                    "recommended_item": "Recommended Product B",
                    "affinity_badge": "Affinity Category",
                    "conditional_prob": "Conditional Prob (Confidence)",
                    "baseline_prob": "Baseline Prob (Support B)",
                    "lift": "Lift",
                    "unit_price": "Unit Price (GBP)",
                    "discounted_unit_price": "Bundle Price (GBP)",
                    "estimated_incremental_basket_value": "Est. Incremental Value (GBP)",
                }
            )
            st.dataframe(
                sim_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Conditional Prob (Confidence)": st.column_config.NumberColumn(format="%.2%"),
                    "Baseline Prob (Support B)": st.column_config.NumberColumn(format="%.2%"),
                    "Lift": st.column_config.NumberColumn(format="%.2f"),
                    "Unit Price (GBP)": st.column_config.NumberColumn(format="GBP %.2f"),
                    "Bundle Price (GBP)": st.column_config.NumberColumn(format="GBP %.2f"),
                    "Est. Incremental Value (GBP)": st.column_config.NumberColumn(format="GBP %.2f"),
                },
            )

        st.markdown("#### Interactive Directed Product Association Network Graph")
        g_col1, g_col2 = st.columns([1.6, 1.0])
        with g_col1:
            graph_mode = st.radio(
                "Network Graph Scope",
                options=[
                    "Ego-Network (Selected Anchor Items)",
                    "Top-N Macro Rule Network",
                ],
                horizontal=True,
            )
        with g_col2:
            max_graph_rules = st.slider(
                "Max Rules Displayed in Graph",
                min_value=5,
                max_value=40,
                value=20,
                step=5,
            )

        net_fig, n_nodes, n_edges = _build_network_graph(
            rules_df=rules_df,
            item_metadata_df=item_metadata_df,
            anchor_items=selected_anchors,
            graph_mode=graph_mode,
            max_rules=max_graph_rules,
        )
        st.caption(
            f"Graph topology: {n_nodes} product nodes and {n_edges} directed association edges "
            "(Red = Selected Anchor, Navy = Antecedent, Teal = Consequent, Amber = Bidirectional Hub)."
        )
        st.plotly_chart(net_fig, width="stretch")


if __name__ == "__main__":
    main()
