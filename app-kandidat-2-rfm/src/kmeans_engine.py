"""K-Means optimization, multi-metric cluster validation, PCA projection, and RFM persona profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.rfm_pipeline import RFM_COLUMNS

# Deterministic business persona ladders ordered strictly from highest RFM health S_c to lowest
_PERSONA_LADDERS: dict[int, list[str]] = {
    2: [
        "Champions / High-Value Buyers",
        "Low-Engagement / Hibernating",
    ],
    3: [
        "Champions / Loyal Customers",
        "Regular / Potential Customers",
        "At-Risk / Lost Customers",
    ],
    4: [
        "Champions / Loyal Customers",
        "Potential Loyalists",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    5: [
        "Champions / VIP",
        "Loyal Customers",
        "Potential Loyalists",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    6: [
        "Champions / VIP",
        "Loyal Customers",
        "Potential Loyalists",
        "Promising / Regular",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    7: [
        "Champions / VIP",
        "Loyal Customers",
        "Potential Loyalists",
        "Promising Newcomers",
        "Need Attention",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    8: [
        "Champions / VIP",
        "Loyal Customers",
        "Recent High-Spenders",
        "Potential Loyalists",
        "Regular / Steady",
        "Need Attention",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    9: [
        "Champions / VIP",
        "Core Loyalists",
        "Loyal Customers",
        "Potential Loyalists",
        "Promising Newcomers",
        "Regular / Steady",
        "Need Attention",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
    10: [
        "Champions / VIP",
        "Core Loyalists",
        "Loyal Customers",
        "Emerging High-Value",
        "Potential Loyalists",
        "Promising Newcomers",
        "Regular / Steady",
        "Need Attention",
        "At-Risk Customers",
        "Hibernating / Lost Customers",
    ],
}

_CRM_PLAYBOOK: dict[str, dict[str, str]] = {
    "Champions": {
        "Primary Objective": "Retention & VIP Advocacy",
        "Recommended Actions": (
            "Invite to VIP loyalty tier, offer early access to seasonal collections, "
            "assign dedicated wholesale account concierge, and encourage referral programs."
        ),
        "Incentive Type": "Exclusivity, Priority Dispatch & Volume Rebates (Avoid Deep Discounts)",
        "Priority Channel": "Dedicated Account Manager / VIP Personalized Email",
        "Budget Allocation": "High Priority (35-40%)",
    },
    "Loyal": {
        "Primary Objective": "Loyalty Lock-In & Basket Expansion",
        "Recommended Actions": (
            "Deploy personalized cross-sell product pairings, reward repeat cadence with "
            "milestone perks, and promote multi-buy wholesale bundles."
        ),
        "Incentive Type": "Loyalty Points Multiplier & Curated Product Bundles",
        "Priority Channel": "Segmented Email Lifecycle & In-Cart Recommendations",
        "Budget Allocation": "Core Retention (25%)",
    },
    "Potential": {
        "Primary Objective": "Conversion to Loyal Repeat Buyers",
        "Recommended Actions": (
            "Send tailored second/third-order follow-ups, recommend complementary SKUs, "
            "and introduce onboarding membership incentives."
        ),
        "Incentive Type": "Next-Order Threshold Coupon & Free Shipping Upgrade",
        "Priority Channel": "Automated Behavioral Email & Retargeting Ads",
        "Budget Allocation": "Growth Expansion (20%)",
    },
    "Promising": {
        "Primary Objective": "Frequency Acceleration & Habit Building",
        "Recommended Actions": (
            "Educate on core catalog categories, highlight trending giftware items, "
            "and trigger time-boxed replenishment reminders."
        ),
        "Incentive Type": "Category Discovery Bundle & Introductory Perk",
        "Priority Channel": "Email Newsletter & Web Personalization",
        "Budget Allocation": "Moderate Growth (15%)",
    },
    "Regular": {
        "Primary Objective": "Order Value Uplift & Engagement",
        "Recommended Actions": (
            "Provide seasonal catalog digests, minimum-spend basket boosters, "
            "and periodic engagement campaigns to prevent drift."
        ),
        "Incentive Type": "Tiered Cart Discount & Seasonal Multi-Pack Offers",
        "Priority Channel": "Scheduled Email Campaigns",
        "Budget Allocation": "Balanced Maintenance (15%)",
    },
    "Need Attention": {
        "Primary Objective": "Early Churn Prevention",
        "Recommended Actions": (
            "Trigger recency-drop alerts, showcase newly arrived stock aligned with past "
            "purchases, and offer gentle re-engagement reminders."
        ),
        "Incentive Type": "Time-Limited Nudge Voucher",
        "Priority Channel": "Automated Triggered Email / SMS",
        "Budget Allocation": "Selective Retention (12%)",
    },
    "At-Risk": {
        "Primary Objective": "Urgent Reactivation & Win-Back",
        "Recommended Actions": (
            "Launch personalized win-back sequence referencing historical favorites, "
            "conduct short friction feedback survey, and pair with time-bound recovery offer."
        ),
        "Incentive Type": "Limited-Time Reactivation Discount & Feedback Reward",
        "Priority Channel": "High-Urgency Win-Back Email & Direct Outreach for High-M",
        "Budget Allocation": "Targeted Recovery (15-20%)",
    },
    "Hibernating": {
        "Primary Objective": "Cost-Efficient Bulk Re-Engagement or Suppression",
        "Recommended Actions": (
            "Include only in low-cost seasonal clearance blasts; suppress from expensive "
            "paid retargeting or direct mailers if unresponsive after two touches."
        ),
        "Incentive Type": "Low-Cost Clearance Notice / Sunset Opt-In",
        "Priority Channel": "Low-Cost Automated Bulk Email",
        "Budget Allocation": "Minimal Spend (5-10%)",
    },
}


def _match_crm_playbook(persona: str) -> dict[str, str]:
    """Match a deterministic persona label to its structured CRM marketing playbook."""
    lowered = persona.lower()
    if "champion" in lowered or "recent high-spenders" in lowered or "emerging high-value" in lowered:
        return _CRM_PLAYBOOK["Champions"]
    if "loyalist" in lowered and "potential" not in lowered:
        return _CRM_PLAYBOOK["Loyal"]
    if "loyal customers" in lowered:
        return _CRM_PLAYBOOK["Loyal"]
    if "potential" in lowered:
        return _CRM_PLAYBOOK["Potential"]
    if "promising" in lowered:
        return _CRM_PLAYBOOK["Promising"]
    if "regular" in lowered:
        return _CRM_PLAYBOOK["Regular"]
    if "attention" in lowered:
        return _CRM_PLAYBOOK["Need Attention"]
    if "at-risk" in lowered:
        return _CRM_PLAYBOOK["At-Risk"]
    return _CRM_PLAYBOOK["Hibernating"]


def _minmax_norm(arr: np.ndarray, invert: bool = False) -> np.ndarray:
    """Normalize a 1D array to [0, 1] safely even when values are constant."""
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float)
    scaled = (arr - lo) / (hi - lo)
    return 1.0 - scaled if invert else scaled


def _detect_elbow_k(k_vals: np.ndarray, wcss_vals: np.ndarray) -> int:
    """Detect discrete geometric elbow/knee point via max perpendicular distance to chord."""
    if len(k_vals) <= 2:
        return int(k_vals[0])
    x = _minmax_norm(k_vals.astype(float))
    y = _minmax_norm(wcss_vals.astype(float))
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    chord = p2 - p1
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-12:
        return int(k_vals[0])
    distances = []
    for xi, yi in zip(x, y):
        pt = np.array([xi, yi])
        dist = abs(chord[0] * (p1[1] - pt[1]) - (p1[0] - pt[0]) * chord[1]) / chord_len
        distances.append(float(dist))
    best_idx = int(np.argmax(distances))
    return int(k_vals[best_idx])


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate K-Means (k-means++) across k in [k_min, k_max] with internal validation metrics.

    Computes WCSS, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index,
    and detects the recommended consensus k (exposed via the 'Recommended' column
    and eval_df.attrs['recommended_k']).
    """
    X = np.asarray(X_scaled, dtype=float)
    n_samples = int(X.shape[0])
    if n_samples < 3:
        raise ValueError("At least 3 samples are required to evaluate K-Means clustering.")
    if k_min < 2:
        raise ValueError("k_min must be at least 2.")

    effective_k_max = min(int(k_max), n_samples - 1)
    if effective_k_max < k_min:
        raise ValueError(
            f"Cannot evaluate k_min={k_min} on {n_samples} samples (need at least k_min + 1 samples)."
        )

    records: list[dict[str, Any]] = []
    for k in range(k_min, effective_k_max + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        wcss = float(model.inertia_)
        n_unique = int(len(np.unique(labels)))
        if n_unique < 2 or n_unique >= n_samples:
            sil = 0.0
            dbi = float("inf")
            chi = 0.0
        else:
            sil = float(silhouette_score(X, labels))
            dbi = float(davies_bouldin_score(X, labels))
            chi = float(calinski_harabasz_score(X, labels))
        records.append(
            {
                "k": int(k),
                "WCSS": round(wcss, 4),
                "Silhouette Score": round(sil, 4),
                "Davies-Bouldin Index": round(dbi, 4) if np.isfinite(dbi) else 999.0,
                "Calinski-Harabasz Index": round(chi, 2),
            }
        )

    eval_df = pd.DataFrame(records)
    k_arr = eval_df["k"].to_numpy(dtype=float)
    wcss_arr = eval_df["WCSS"].to_numpy(dtype=float)
    sil_arr = eval_df["Silhouette Score"].to_numpy(dtype=float)
    dbi_arr = eval_df["Davies-Bouldin Index"].to_numpy(dtype=float)
    chi_arr = eval_df["Calinski-Harabasz Index"].to_numpy(dtype=float)

    elbow_k = _detect_elbow_k(eval_df["k"].to_numpy(dtype=int), wcss_arr)
    sil_best_k = int(eval_df.loc[eval_df["Silhouette Score"].idxmax(), "k"])
    dbi_best_k = int(eval_df.loc[eval_df["Davies-Bouldin Index"].idxmin(), "k"])

    span = max(1.0, float(k_arr[-1] - k_arr[0]))
    elbow_prox = 1.0 - np.abs(k_arr - float(elbow_k)) / span
    consensus_scores = (
        0.35 * _minmax_norm(sil_arr, invert=False)
        + 0.30 * _minmax_norm(dbi_arr, invert=True)
        + 0.20 * elbow_prox
        + 0.15 * _minmax_norm(chi_arr, invert=False)
    )
    eval_df["Consensus Score"] = np.round(consensus_scores, 4)

    best_idx = int(np.argmax(consensus_scores))
    recommended_k = int(eval_df.loc[best_idx, "k"])
    eval_df["Recommended"] = eval_df["k"] == recommended_k

    eval_df.attrs["recommended_k"] = recommended_k
    eval_df.attrs["elbow_k"] = elbow_k
    eval_df.attrs["silhouette_best_k"] = sil_best_k
    eval_df.attrs["dbi_best_k"] = dbi_best_k
    return eval_df


def _zscore_across_clusters(values: np.ndarray) -> np.ndarray:
    """Compute population Z-score across cluster centroids with zero-std guard."""
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=0))
    if std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - float(np.mean(arr))) / std


def _get_persona_ladder(k: int) -> list[str]:
    """Return strictly unique ordered persona names from best RFM health to worst for any k."""
    if k in _PERSONA_LADDERS:
        return list(_PERSONA_LADDERS[k])
    # Deterministic fallback for k > 10
    base = list(_PERSONA_LADDERS[10])
    extra = [f"Segment Tier {idx}" for idx in range(11, k + 1)]
    return base + extra


def fit_kmeans_and_profile(
    rfm_raw_df: pd.DataFrame,
    X_scaled: np.ndarray,
    k: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit final K-Means (k-means++), compute PCA 2D projection, true unscaled centroids,
    deterministic RFM personas via composite score S_c = z(F_c) + z(M_c) - z(R_c),
    and CRM Marketing Action Matrix.
    """
    X = np.asarray(X_scaled, dtype=float)
    if len(rfm_raw_df) != len(X):
        raise ValueError(
            f"Length mismatch: rfm_raw_df has {len(rfm_raw_df)} rows while X_scaled has {len(X)} rows."
        )
    n_samples = int(len(rfm_raw_df))
    if n_samples < 3:
        raise ValueError("At least 3 customers are required for K-Means segmentation.")
    if k < 2 or k >= n_samples:
        raise ValueError(f"k={k} must satisfy 2 <= k < n_samples ({n_samples}).")

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = model.fit_predict(X).astype(int)

    n_unique = int(len(np.unique(labels)))
    if n_unique >= 2:
        wcss = float(model.inertia_)
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        chi = float(calinski_harabasz_score(X, labels))
        sil_samples = silhouette_samples(X, labels)
    else:
        wcss = float(model.inertia_)
        sil = 0.0
        dbi = 999.0
        chi = 0.0
        sil_samples = np.zeros(n_samples, dtype=float)

    # Population Z-scores across active customer matrix for Snake Plot and S_c consistency
    z_matrix = StandardScaler().fit_transform(X)

    # 2D PCA Projection of customers and K-Means centroids
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X)
    pca_centroids = pca.transform(model.cluster_centers_)
    pca_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # Build customer-level output table from unscaled raw RFM values
    cust_df = rfm_raw_df.copy().reset_index(drop=True)
    cust_df["Cluster"] = labels
    cust_df["Silhouette"] = np.round(sil_samples, 4)
    cust_df["PCA1"] = np.round(pca_coords[:, 0], 4)
    cust_df["PCA2"] = np.round(pca_coords[:, 1], 4)
    cust_df["z_Recency"] = z_matrix[:, 0]
    cust_df["z_Frequency"] = z_matrix[:, 1]
    cust_df["z_Monetary"] = z_matrix[:, 2]

    # Compute true unscaled business centroids directly from rfm_raw_df
    total_customers = float(len(cust_df))
    total_revenue = float(cust_df["Monetary"].sum())
    pop_r_mean = float(cust_df["Recency"].mean())
    pop_f_mean = float(cust_df["Frequency"].mean())
    pop_m_mean = float(cust_df["Monetary"].mean())

    agg_df = (
        cust_df.groupby("Cluster", as_index=False)
        .agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Recency_Mean=("Recency", "mean"),
            Recency_Median=("Recency", "median"),
            Frequency_Mean=("Frequency", "mean"),
            Frequency_Median=("Frequency", "median"),
            Monetary_Mean=("Monetary", "mean"),
            Monetary_Median=("Monetary", "median"),
            z_R_mean=("z_Recency", "mean"),
            z_F_mean=("z_Frequency", "mean"),
            z_M_mean=("z_Monetary", "mean"),
        )
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    # Deterministic RFM Persona Mapping via composite health score S_c = z(F_c) + z(M_c) - z(R_c)
    z_r_cluster = _zscore_across_clusters(agg_df["z_R_mean"].to_numpy(dtype=float))
    z_f_cluster = _zscore_across_clusters(agg_df["z_F_mean"].to_numpy(dtype=float))
    z_m_cluster = _zscore_across_clusters(agg_df["z_M_mean"].to_numpy(dtype=float))
    health_scores_arr = z_f_cluster + z_m_cluster - z_r_cluster
    agg_df["Health Score"] = np.round(health_scores_arr, 4)

    # Sort descending by S_c with deterministic tie-breakers
    ranked = agg_df.sort_values(
        by=["Health Score", "Recency_Mean", "Monetary_Mean", "Frequency_Mean", "Cluster"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)

    ladder = _get_persona_ladder(len(ranked))
    persona_map: dict[int, str] = {}
    health_scores: dict[int, float] = {}
    for idx, row in ranked.iterrows():
        cid = int(row["Cluster"])
        persona_map[cid] = ladder[idx]
        health_scores[cid] = float(row["Health Score"])

    cust_df["Persona"] = cust_df["Cluster"].map(persona_map)
    cust_df["Health Score"] = cust_df["Cluster"].map(health_scores)

    ranked["Persona"] = ranked["Cluster"].map(persona_map)
    ranked["Customer Share (%)"] = np.round(100.0 * ranked["Customer_Count"] / total_customers, 2)
    ranked["Revenue Share (%)"] = (
        np.round(100.0 * ranked["Total_Revenue"] / total_revenue, 2)
        if total_revenue > 0
        else 0.0
    )
    ranked["R Ratio vs Pop"] = (
        np.round(ranked["Recency_Mean"] / pop_r_mean, 2) if pop_r_mean > 0 else 1.0
    )
    ranked["F Ratio vs Pop"] = (
        np.round(ranked["Frequency_Mean"] / pop_f_mean, 2) if pop_f_mean > 0 else 1.0
    )
    ranked["M Ratio vs Pop"] = (
        np.round(ranked["Monetary_Mean"] / pop_m_mean, 2) if pop_m_mean > 0 else 1.0
    )

    profile_df = pd.DataFrame(
        {
            "Cluster": ranked["Cluster"].astype(int),
            "Persona": ranked["Persona"],
            "Health Score (S_c)": ranked["Health Score"],
            "Customer Count": ranked["Customer_Count"].astype(int),
            "Customer Share (%)": ranked["Customer Share (%)"],
            "Total Revenue (GBP)": np.round(ranked["Total_Revenue"], 2),
            "Revenue Share (%)": ranked["Revenue Share (%)"],
            "Recency Mean (Days)": np.round(ranked["Recency_Mean"], 2),
            "Recency Median (Days)": np.round(ranked["Recency_Median"], 2),
            "Frequency Mean": np.round(ranked["Frequency_Mean"], 2),
            "Frequency Median": np.round(ranked["Frequency_Median"], 2),
            "Monetary Mean (GBP)": np.round(ranked["Monetary_Mean"], 2),
            "Monetary Median (GBP)": np.round(ranked["Monetary_Median"], 2),
            "R Ratio vs Pop": ranked["R Ratio vs Pop"],
            "F Ratio vs Pop": ranked["F Ratio vs Pop"],
            "M Ratio vs Pop": ranked["M Ratio vs Pop"],
        }
    ).reset_index(drop=True)

    # Snake Plot DataFrame (population Z-scores per cluster)
    snake_df = pd.DataFrame(
        {
            "Cluster": ranked["Cluster"].astype(int),
            "Persona": ranked["Persona"],
            "Recency": np.round(ranked["z_R_mean"], 4),
            "Frequency": np.round(ranked["z_F_mean"], 4),
            "Monetary": np.round(ranked["z_M_mean"], 4),
        }
    ).reset_index(drop=True)

    # Radar Chart DataFrame ([0, 1] strength profile with Recency inverted so higher = more recent)
    radar_df = pd.DataFrame(
        {
            "Cluster": ranked["Cluster"].astype(int),
            "Persona": ranked["Persona"],
            "Recency (Recentness)": np.round(
                _minmax_norm(ranked["z_R_mean"].to_numpy(dtype=float), invert=True), 4
            ),
            "Frequency": np.round(
                _minmax_norm(ranked["z_F_mean"].to_numpy(dtype=float), invert=False), 4
            ),
            "Monetary": np.round(
                _minmax_norm(ranked["z_M_mean"].to_numpy(dtype=float), invert=False), 4
            ),
        }
    ).reset_index(drop=True)

    # PCA Centroids DataFrame aligned with profile order
    pca_centroids_raw = pd.DataFrame(
        {
            "Cluster": np.arange(k, dtype=int),
            "PCA1": np.round(pca_centroids[:, 0], 4),
            "PCA2": np.round(pca_centroids[:, 1], 4),
        }
    )
    pca_centroids_df = (
        profile_df[["Cluster", "Persona", "Customer Count", "Revenue Share (%)"]]
        .merge(pca_centroids_raw, on="Cluster", how="left")
        .reset_index(drop=True)
    )

    # Structured CRM Marketing Action Matrix
    matrix_rows: list[dict[str, Any]] = []
    for _, prof_row in profile_df.iterrows():
        persona = str(prof_row["Persona"])
        playbook = _match_crm_playbook(persona)
        matrix_rows.append(
            {
                "Cluster": int(prof_row["Cluster"]),
                "Persona": persona,
                "Customer Count": int(prof_row["Customer Count"]),
                "Customer Share (%)": float(prof_row["Customer Share (%)"]),
                "Revenue Share (%)": float(prof_row["Revenue Share (%)"]),
                "Primary Objective": playbook["Primary Objective"],
                "Recommended Actions": playbook["Recommended Actions"],
                "Incentive Type": playbook["Incentive Type"],
                "Priority Channel": playbook["Priority Channel"],
                "Budget Allocation": playbook["Budget Allocation"],
            }
        )
    marketing_matrix_df = pd.DataFrame(matrix_rows)

    return {
        "k": int(k),
        "model": model,
        "labels": labels,
        "metrics": {
            "wcss": round(wcss, 4),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(dbi, 4) if np.isfinite(dbi) else 999.0,
            "calinski_harabasz_index": round(chi, 2),
        },
        "customer_df": cust_df,
        "profile_df": profile_df,
        "snake_df": snake_df,
        "radar_df": radar_df,
        "pca_centroids_df": pca_centroids_df,
        "pca_explained_variance": pca_var,
        "marketing_matrix_df": marketing_matrix_df,
        "persona_map": persona_map,
        "health_scores": health_scores,
    }
