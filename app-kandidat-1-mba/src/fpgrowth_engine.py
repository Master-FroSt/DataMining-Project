"""FP-Growth rule mining, redundancy pruning, and product bundling simulation engine."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

RULE_OUTPUT_COLUMNS: list[str] = [
    "antecedents",
    "consequents",
    "rule_label",
    "antecedent_support",
    "consequent_support",
    "support",
    "confidence",
    "lift",
    "conviction",
    "leverage",
    "antecedent_len",
    "consequent_len",
    "rule_len",
    "antecedents_frozenset",
    "consequents_frozenset",
]

SIMULATION_OUTPUT_COLUMNS: list[str] = [
    "triggered_by",
    "recommended_item",
    "affinity_badge",
    "conditional_prob",
    "baseline_prob",
    "relative_uplift_pct",
    "lift",
    "support",
    "conviction",
    "leverage",
    "unit_price",
    "discounted_unit_price",
    "gross_incremental_value",
    "estimated_incremental_basket_value",
]


def _format_itemset(items: frozenset[str] | set[str]) -> str:
    """Format a frozenset of product labels into a deterministic readable string."""
    return ", ".join(sorted(str(item) for item in items))


def _prune_redundant_rules(rules_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rules whose superset antecedent offers no higher confidence than a subset.

    A rule ``X2 -> Y`` is redundant if there exists another rule ``X1 -> Y`` in ``rules_df``
    with the identical consequent ``Y``, ``X1`` is a strict subset of ``X2``, and
    ``confidence(X2 -> Y) <= confidence(X1 -> Y)``.
    """
    if rules_df.empty:
        return rules_df, 0

    keep_mask = np.ones(len(rules_df), dtype=bool)
    grouped_indices = rules_df.groupby("consequents_frozenset", sort=False).indices

    ants = rules_df["antecedents_frozenset"].tolist()
    confs = rules_df["confidence"].tolist()

    for idx_group in grouped_indices.values():
        if len(idx_group) < 2:
            continue
        # Check each rule in the same-consequent group against strictly shorter subset antecedents
        sorted_group = sorted(idx_group, key=lambda i: (len(ants[i]), -confs[i]))
        for pos_b in range(len(sorted_group)):
            i_b = sorted_group[pos_b]
            ant_b = ants[i_b]
            conf_b = confs[i_b]
            if len(ant_b) <= 1:
                continue
            for pos_a in range(pos_b):
                i_a = sorted_group[pos_a]
                ant_a = ants[i_a]
                if len(ant_a) >= len(ant_b):
                    break
                if ant_a.issubset(ant_b) and conf_b <= confs[i_a] + 1e-12:
                    keep_mask[i_b] = False
                    break

    pruned_df = rules_df.loc[keep_mask].copy()
    removed_count = int((~keep_mask).sum())
    return pruned_df, removed_count


def mine_fpgrowth_rules(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.50,
    min_lift: float = 1.2,
    max_len: int = 3,
    filter_redundant: bool = True,
) -> dict[str, Any]:
    """Run FP-Growth frequent itemset mining and multi-metric association rule extraction.

    Args:
        basket_df: Boolean sparse DataFrame indexed by invoice with item columns.
        min_support: Minimum support threshold in ``(0, 1]``.
        min_confidence: Minimum confidence threshold in ``[0, 1]``.
        min_lift: Minimum lift ratio threshold.
        max_len: Maximum itemset cardinality explored by FP-Growth.
        filter_redundant: If True, prune superset antecedent rules with <= subset confidence.

    Returns:
        Dictionary containing ``frequent_itemsets``, ``rules``, ``execution_time_sec``,
        counts, and mining ``status``.
    """
    start_time = time.perf_counter()
    empty_rules = pd.DataFrame(columns=RULE_OUTPUT_COLUMNS)
    empty_itemsets = pd.DataFrame(columns=["support", "itemsets", "itemsets_str", "length"])

    if basket_df is None or basket_df.empty or basket_df.shape[1] == 0:
        return {
            "frequent_itemsets": empty_itemsets,
            "rules": empty_rules,
            "execution_time_sec": round(time.perf_counter() - start_time, 4),
            "n_itemsets": 0,
            "n_rules": 0,
            "n_raw_rules": 0,
            "n_redundant_removed": 0,
            "status": "empty_basket",
        }

    safe_support = float(max(min_support, 1e-6))
    safe_max_len = int(max_len) if max_len and max_len >= 2 else None

    frequent_itemsets = fpgrowth(
        basket_df,
        min_support=safe_support,
        use_colnames=True,
        max_len=safe_max_len,
    )

    if frequent_itemsets.empty:
        return {
            "frequent_itemsets": empty_itemsets,
            "rules": empty_rules,
            "execution_time_sec": round(time.perf_counter() - start_time, 4),
            "n_itemsets": 0,
            "n_rules": 0,
            "n_raw_rules": 0,
            "n_redundant_removed": 0,
            "status": "no_itemsets",
        }

    frequent_itemsets = frequent_itemsets.copy()
    frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].map(_format_itemset)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].map(len)
    frequent_itemsets = frequent_itemsets.sort_values(
        by=["support", "length", "itemsets_str"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    has_multi_itemsets = bool((frequent_itemsets["length"] >= 2).any())
    if not has_multi_itemsets:
        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": empty_rules,
            "execution_time_sec": round(time.perf_counter() - start_time, 4),
            "n_itemsets": int(len(frequent_itemsets)),
            "n_rules": 0,
            "n_raw_rules": 0,
            "n_redundant_removed": 0,
            "status": "no_multi_itemsets",
        }

    raw_rules = association_rules(
        frequent_itemsets,
        num_itemsets=len(basket_df),
        metric="confidence",
        min_threshold=float(min_confidence),
    )

    if raw_rules.empty:
        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": empty_rules,
            "execution_time_sec": round(time.perf_counter() - start_time, 4),
            "n_itemsets": int(len(frequent_itemsets)),
            "n_rules": 0,
            "n_raw_rules": 0,
            "n_redundant_removed": 0,
            "status": "no_rules",
        }

    filtered = raw_rules.loc[raw_rules["lift"] >= float(min_lift)].copy()
    if filtered.empty:
        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": empty_rules,
            "execution_time_sec": round(time.perf_counter() - start_time, 4),
            "n_itemsets": int(len(frequent_itemsets)),
            "n_rules": 0,
            "n_raw_rules": int(len(raw_rules)),
            "n_redundant_removed": 0,
            "status": "no_rules",
        }

    filtered = filtered.rename(
        columns={
            "antecedent support": "antecedent_support",
            "consequent support": "consequent_support",
            "antecedents": "antecedents_frozenset",
            "consequents": "consequents_frozenset",
        }
    )
    filtered["antecedents"] = filtered["antecedents_frozenset"].map(_format_itemset)
    filtered["consequents"] = filtered["consequents_frozenset"].map(_format_itemset)
    filtered["rule_label"] = filtered["antecedents"] + " -> " + filtered["consequents"]
    filtered["antecedent_len"] = filtered["antecedents_frozenset"].map(len)
    filtered["consequent_len"] = filtered["consequents_frozenset"].map(len)
    filtered["rule_len"] = filtered["antecedent_len"] + filtered["consequent_len"]

    n_pre_prune = int(len(filtered))
    n_redundant_removed = 0
    if filter_redundant:
        filtered, n_redundant_removed = _prune_redundant_rules(filtered)

    filtered = (
        filtered[RULE_OUTPUT_COLUMNS]
        .sort_values(by=["lift", "confidence", "support", "rule_label"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )

    elapsed = round(time.perf_counter() - start_time, 4)
    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": filtered,
        "execution_time_sec": elapsed,
        "n_itemsets": int(len(frequent_itemsets)),
        "n_rules": int(len(filtered)),
        "n_raw_rules": n_pre_prune,
        "n_redundant_removed": n_redundant_removed,
        "status": "ok" if not filtered.empty else "no_rules",
    }


def _classify_affinity_badge(confidence: float, lift: float) -> str:
    """Assign a clean, non-emoji business affinity category to a recommendation rule."""
    if confidence >= 0.65 and lift >= 3.0:
        return "Strong Bundle Candidate"
    if confidence >= 0.50:
        return "High-Confidence Cross-Sell"
    return "Niche Affinity"


def simulate_bundling_recommendations(
    rules_df: pd.DataFrame,
    item_metadata_df: pd.DataFrame,
    anchor_items: list[str],
    discount_pct: float = 10.0,
    top_k: int = 10,
) -> pd.DataFrame:
    """Simulate Product B bundling recommendations and expected incremental basket value.

    Args:
        rules_df: Output ``rules`` DataFrame from ``mine_fpgrowth_rules``.
        item_metadata_df: Item metadata DataFrame containing ``item`` and ``avg_unit_price``.
        anchor_items: Selected anchor item(s) currently in the shopping basket.
        discount_pct: Promotional bundle discount percentage applied to Product B (0..100).
        top_k: Maximum number of ranked Product B recommendations to return.

    Returns:
        Ranked DataFrame of Product B bundling recommendations and uplift metrics.
    """
    empty_sim = pd.DataFrame(columns=SIMULATION_OUTPUT_COLUMNS)
    if rules_df is None or rules_df.empty or not anchor_items:
        return empty_sim

    anchor_set = {str(item).strip() for item in anchor_items if str(item).strip()}
    if not anchor_set:
        return empty_sim

    price_lookup: dict[str, float] = {}
    if item_metadata_df is not None and not item_metadata_df.empty:
        price_lookup = (
            item_metadata_df.set_index("item")["avg_unit_price"].astype(float).to_dict()
        )

    matched_mask = rules_df.apply(
        lambda row: row["antecedents_frozenset"].issubset(anchor_set)
        and row["consequents_frozenset"].isdisjoint(anchor_set),
        axis=1,
    )
    matched = rules_df.loc[matched_mask].copy()
    if matched.empty:
        return empty_sim

    # Keep the strongest triggering rule per unique consequent recommendation
    matched = matched.sort_values(
        by=["confidence", "lift", "support"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["consequents"], keep="first")

    safe_discount = float(np.clip(discount_pct, 0.0, 95.0))
    discount_multiplier = 1.0 - (safe_discount / 100.0)

    records: list[dict[str, Any]] = []
    for _, row in matched.iterrows():
        consequent_items = row["consequents_frozenset"]
        unit_price = float(sum(price_lookup.get(item, 0.0) for item in consequent_items))
        discounted_price = unit_price * discount_multiplier
        cond_prob = float(row["confidence"])
        base_prob = float(row["consequent_support"])
        lift_val = float(row["lift"])
        rel_uplift = ((cond_prob - base_prob) / base_prob * 100.0) if base_prob > 0 else 0.0
        gross_inc = cond_prob * unit_price
        net_inc = cond_prob * discounted_price

        records.append(
            {
                "triggered_by": row["antecedents"],
                "recommended_item": row["consequents"],
                "affinity_badge": _classify_affinity_badge(cond_prob, lift_val),
                "conditional_prob": cond_prob,
                "baseline_prob": base_prob,
                "relative_uplift_pct": round(rel_uplift, 1),
                "lift": lift_val,
                "support": float(row["support"]),
                "conviction": float(row["conviction"]),
                "leverage": float(row["leverage"]),
                "unit_price": round(unit_price, 2),
                "discounted_unit_price": round(discounted_price, 2),
                "gross_incremental_value": round(gross_inc, 2),
                "estimated_incremental_basket_value": round(net_inc, 2),
            }
        )

    sim_df = pd.DataFrame(records, columns=SIMULATION_OUTPUT_COLUMNS)
    sim_df = sim_df.sort_values(
        by=["conditional_prob", "lift", "estimated_incremental_basket_value"],
        ascending=[False, False, False],
    ).head(int(max(1, top_k))).reset_index(drop=True)
    return sim_df
