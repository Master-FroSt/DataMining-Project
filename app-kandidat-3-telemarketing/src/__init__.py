"""Bank Telemarketing Predictive Lead Scoring package."""

from src.data_loader import (
    SUPPORTED_VARIANTS,
    generate_bank_sample_data,
    load_or_download_bank_data,
)
from src.modeling_engine import (
    assign_priority_badge,
    compute_threshold_metrics,
    score_batch_leads,
    score_single_lead,
    simulate_campaign_quota,
    train_and_evaluate_models,
)
from src.preprocessing import (
    audit_and_remove_leakage,
    build_preprocessor_pipelines,
)

__all__ = [
    "SUPPORTED_VARIANTS",
    "assign_priority_badge",
    "audit_and_remove_leakage",
    "build_preprocessor_pipelines",
    "compute_threshold_metrics",
    "generate_bank_sample_data",
    "load_or_download_bank_data",
    "score_batch_leads",
    "score_single_lead",
    "simulate_campaign_quota",
    "train_and_evaluate_models",
]
