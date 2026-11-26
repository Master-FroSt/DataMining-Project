"""Kandidat 2 RFM & K-Means Customer Segmentation package."""

from src.data_loader import generate_rfm_retail_sample_data, load_or_download_retail_data
from src.kmeans_engine import evaluate_k_range, fit_kmeans_and_profile
from src.rfm_pipeline import (
    ADMIN_STOCK_CODES,
    clean_transactions_for_rfm,
    compute_rfm_table,
    transform_and_scale_rfm,
)

__all__ = [
    "ADMIN_STOCK_CODES",
    "clean_transactions_for_rfm",
    "compute_rfm_table",
    "evaluate_k_range",
    "fit_kmeans_and_profile",
    "generate_rfm_retail_sample_data",
    "load_or_download_retail_data",
    "transform_and_scale_rfm",
]
