"""Integrity, schema, portability, and statistical unit tests for the Big Walk KDD pipeline."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "big_walk_kdd_pipeline.ipynb"
RAW_PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "big_walk_reviews_raw.parquet"
CLEAN_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "big_walk_reviews_clean.parquet"
CANDIDATES_CSV_PATH = PROJECT_ROOT / "data" / "annotation" / "annotation_candidates.csv"
ANNOTATION_GUIDE_PATH = PROJECT_ROOT / "data" / "annotation" / "ANNOTATION_GUIDE.md"
ROADMAP_IT2_PATH = PROJECT_ROOT / "plans" / "roadmap_iteration_2_modules_c_to_g.md"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)

FORBIDDEN_IMPORTS = {
    "scratch",
    "src",
    "torch",
    "transformers",
    "sentence_transformers",
    "bertopic",
    "vaderSentiment",
    "nbconvert",
    "nbformat",
    "statsmodels",
    "ruptures",
}

FORBIDDEN_SYMBOLS = {
    "TruncatedSVD",
    "Bertopic",
    "BERTopic",
    "SentenceTransformer",
    "AutoModel",
    "AutoTokenizer",
}

EXPECTED_PHASES = {
    "P0_Pre_Controversy",
    "P1_IGF_Trigger",
    "P2_FAQ_Backlash_Peak",
    "P3_CounterWave_Debate",
    "P4_PostCrisis_Recovery",
}

STANCE_CLASSES = {
    "Gameplay_Genuine",
    "Anti_AI_Backlash",
    "Pro_Dev_Clarification",
}


def _load_notebook() -> dict[str, Any]:
    assert NOTEBOOK_PATH.exists(), f"Missing notebook at {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _extract_notebook_functions() -> dict[str, Any]:
    """Compile only imports, constants, classes, and functions from notebook code cells."""
    nb = _load_notebook()
    ns: dict[str, Any] = {}
    allowed_top_nodes = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        tree = ast.parse(src)
        filtered_body = []
        for node in tree.body:
            if isinstance(node, allowed_top_nodes):
                filtered_body.append(node)
            elif isinstance(node, ast.Assign):
                # Keep uppercase constant assignments and simple configuration scalars
                if all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                    filtered_body.append(node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.isupper():
                    filtered_body.append(node)
        sub_mod = ast.Module(body=filtered_body, type_ignores=[])
        ast.fix_missing_locations(sub_mod)
        exec(compile(sub_mod, "<notebook-defs>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def nb_doc() -> dict[str, Any]:
    return _load_notebook()


@pytest.fixture(scope="module")
def nb_funcs() -> dict[str, Any]:
    return _extract_notebook_functions()


def test_notebook_valid_nbformat_and_execution(nb_doc: dict[str, Any]) -> None:
    assert nb_doc.get("nbformat") == 4
    assert isinstance(nb_doc.get("nbformat_minor"), int) and nb_doc["nbformat_minor"] >= 0
    cells = nb_doc.get("cells", [])
    assert len(cells) >= 10

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    assert len(code_cells) >= 6, "Expected Setup + 5 KDD Chapter code cells"
    assert len(md_cells) >= 6, "Expected academic Markdown cells per chapter"

    expected_exec = 1
    png_count = 0
    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") == expected_exec, (
            f"Code cell {idx} execution_count={cell.get('execution_count')} != {expected_exec}"
        )
        expected_exec += 1

        outputs = cell.get("outputs", [])
        assert len(outputs) > 0, f"Code cell {idx} has empty outputs"
        for out in outputs:
            assert out.get("output_type") != "error", f"Code cell {idx} recorded error: {out}"
            if out.get("output_type") == "stream":
                text = "".join(out.get("text", []))
                assert "Traceback (most recent call last)" not in text
            if out.get("output_type") == "display_data":
                data = out.get("data", {})
                if "image/png" in data:
                    assert len(data["image/png"]) > 100
                    png_count += 1

    assert png_count >= 5, f"Expected at least 5 rendered PNG plots, found {png_count}"


def test_notebook_self_contained_and_classical_only(nb_doc: dict[str, Any]) -> None:
    for idx, cell in enumerate(nb_doc["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in FORBIDDEN_IMPORTS, f"Forbidden import '{alias.name}' in cell {idx}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    assert top not in FORBIDDEN_IMPORTS, f"Forbidden import from '{node.module}' in cell {idx}"
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_SYMBOLS, f"Forbidden symbol '{alias.name}' in cell {idx}"
            elif isinstance(node, ast.Name):
                assert node.id not in FORBIDDEN_SYMBOLS, f"Forbidden symbol reference '{node.id}' in cell {idx}"


def test_zero_emojis_across_notebook_and_artifacts(nb_doc: dict[str, Any]) -> None:
    for idx, cell in enumerate(nb_doc["cells"]):
        src = "".join(cell.get("source", []))
        assert not EMOJI_PATTERN.search(src), f"Emoji found in notebook cell {idx}"
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                txt = "".join(out.get("text", []))
                assert not EMOJI_PATTERN.search(txt), f"Emoji found in output of cell {idx}"

    for artifact in [Path(__file__), ANNOTATION_GUIDE_PATH, ROADMAP_IT2_PATH]:
        text = artifact.read_text(encoding="utf-8")
        assert not EMOJI_PATTERN.search(text), f"Emoji found in {artifact}"


def test_multinomial_nb_non_negative_matrix(nb_funcs: dict[str, Any]) -> None:
    build_prep = nb_funcs["build_feature_preprocessor"]
    beh_cols = nb_funcs["BEHAVIORAL_FEATURES"]

    rng = np.random.default_rng(42)
    n_train, n_test = 40, 15
    corpus_templates = [
        "coop walkie puzzle compass explore friend",
        "genai ai_slop igf disqualification refund trash",
        "shadow_caster 10_lines 0_1_percent read faq review_bomb",
    ]
    train_df = pd.DataFrame(
        {
            "text_clean": [corpus_templates[i % 3] for i in range(n_train)],
            "playtime_at_review_hrs": rng.uniform(0.1, 20.0, size=n_train),
            "is_refund_window": rng.choice([0.0, 1.0], size=n_train),
            "stopped_after_review": rng.choice([0.0, 1.0], size=n_train),
            "was_edited": rng.choice([0.0, 1.0], size=n_train),
            "funny_ratio": rng.uniform(0.0, 0.5, size=n_train),
            "hourly_velocity": rng.uniform(1.0, 50.0, size=n_train),
            "word_count": rng.integers(3, 40, size=n_train).astype(float),
            "caps_ratio": rng.uniform(0.0, 0.3, size=n_train),
            "exclamation_count": rng.integers(0, 4, size=n_train).astype(float),
            "vader_compound": rng.uniform(-0.6, 0.9, size=n_train),
        }
    )
    # Include an extreme negative value in test_df below train_df.min() to verify clipping
    test_df = train_df.iloc[:n_test].copy().reset_index(drop=True)
    test_df.loc[0, "vader_compound"] = -0.99
    test_df.loc[1, "playtime_at_review_hrs"] = 0.0

    prep = build_prep(regime="Multimodal", model_name="MultinomialNB", behavioral_cols=beh_cols)
    X_train = prep.fit_transform(train_df)
    X_test = prep.transform(test_df)

    assert float(X_train.min()) >= 0.0
    assert float(X_test.min()) >= 0.0

    y_train = np.array(["Gameplay_Genuine", "Anti_AI_Backlash", "Pro_Dev_Clarification"] * 13 + ["Gameplay_Genuine"])
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert len(preds) == n_test


def test_pelt_and_cusum_synthetic_step_change(nb_funcs: dict[str, Any]) -> None:
    pelt = nb_funcs["detect_changepoints_pelt"]
    cusum = nb_funcs["detect_changepoints_cusum"]

    step = np.concatenate([np.full(30, 0.10), np.full(30, 0.85)])
    res_l2 = pelt(step, model="l2", min_size=5)
    res_rbf = pelt(step, model="rbf", min_size=5)
    res_cusum = cusum(step, min_size=5)

    assert 30 in res_l2["changepoints"], f"PELT L2 missed step at 30: {res_l2}"
    assert 30 in res_rbf["changepoints"], f"PELT RBF missed step at 30: {res_rbf}"
    assert 30 in res_cusum["changepoints"], f"CUSUM missed step at 30: {res_cusum}"
    assert res_cusum["p_value_approx"] < 0.01


def test_mcnemar_exact_binomial_vs_chi2_and_holm_bonferroni(nb_funcs: dict[str, Any]) -> None:
    mcnemar = nb_funcs["mcnemar_pair_test"]
    holm = nb_funcs["holm_bonferroni_adjust"]

    # Case 1: discordant = 10 (< 25) -> Exact Binomial
    yt_small = np.array([0] * 30)
    pa_small = np.array([0] * 28 + [1] * 2)
    pb_small = np.array([0] * 18 + [1] * 12)  # b=10, c=0
    out_small = mcnemar(yt_small, pa_small, pb_small)
    assert out_small["discordant"] == 10
    assert out_small["test_method"] == "Exact Binomial"
    expected_binom_p = float(stats.binomtest(0, n=10, p=0.5, alternative="two-sided").pvalue)
    assert out_small["raw_p"] == pytest.approx(expected_binom_p, abs=1e-12)

    # Case 2: discordant = 30 (>= 25) -> Continuity-Corrected Chi2
    yt_big = np.array([0] * 60)
    pa_big = np.array([0] * 55 + [1] * 5)
    pb_big = np.array([0] * 25 + [1] * 35)  # b=30, c=0
    out_big = mcnemar(yt_big, pa_big, pb_big)
    assert out_big["discordant"] == 30
    assert out_big["test_method"] == "Continuity-Corrected Chi2"
    expected_chi2 = ((30.0 - 0.0 - 1.0) ** 2) / 30.0
    expected_chi2_p = float(stats.chi2.sf(expected_chi2, df=1))
    assert out_big["statistic"] == pytest.approx(expected_chi2, abs=1e-12)
    assert out_big["raw_p"] == pytest.approx(expected_chi2_p, abs=1e-12)

    # Case 3: Holm-Bonferroni adjustment
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_data_files_and_schemas() -> None:
    for p in [RAW_PARQUET_PATH, CLEAN_PARQUET_PATH, CANDIDATES_CSV_PATH, ANNOTATION_GUIDE_PATH, ROADMAP_IT2_PATH]:
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"

    raw_df = pd.read_parquet(RAW_PARQUET_PATH)
    assert len(raw_df) >= 30000
    assert raw_df["recommendationid"].astype(str).is_unique
    for col in ["recommendationid", "review", "timestamp_created", "timestamp_updated", "voted_up", "author_playtime_at_review", "author_playtime_forever"]:
        assert col in raw_df.columns, f"Raw parquet missing '{col}'"

    clean_df = pd.read_parquet(CLEAN_PARQUET_PATH)
    assert 30000 <= len(clean_df) <= len(raw_df)
    required_clean_cols = [
        "recommendationid", "review", "text_vader", "text_clean",
        "timestamp_created", "timestamp_updated", "created_utc", "phase",
        "voted_up", "votes_up", "votes_funny", "weighted_vote_score",
        "steam_purchase", "received_for_free",
        "playtime_at_review_hrs", "post_review_playtime_hrs", "playtime_bucket",
        "is_refund_window", "stopped_after_review", "was_edited",
        "funny_ratio", "hourly_velocity", "word_count", "char_count",
        "caps_ratio", "exclamation_count", "question_count",
        "vader_neg", "vader_neu", "vader_pos", "vader_compound", "vader_label",
    ]
    for col in required_clean_cols:
        assert col in clean_df.columns, f"Clean parquet missing '{col}'"

    assert set(clean_df["phase"].unique()).issubset(EXPECTED_PHASES)
    assert clean_df["vader_compound"].between(-1.0, 1.0).all()
    assert set(clean_df["vader_label"].unique()).issubset({"positive", "neutral", "negative"})
    assert (clean_df["playtime_at_review_hrs"] >= 0.0).all()
    assert (clean_df["post_review_playtime_hrs"] >= 0.0).all()

    cand_df = pd.read_csv(CANDIDATES_CSV_PATH)
    assert len(cand_df) >= 300
    for col in ["recommendationid", "phase", "voted_up", "text_vader", "text_clean", "suggested_label", "gold_label"]:
        assert col in cand_df.columns, f"Candidates CSV missing '{col}'"
    assert set(cand_df["suggested_label"].unique()) == STANCE_CLASSES

    guide_text = ANNOTATION_GUIDE_PATH.read_text(encoding="utf-8")
    for cls_name in STANCE_CLASSES:
        assert cls_name in guide_text

    roadmap_text = ROADMAP_IT2_PATH.read_text(encoding="utf-8")
    for keyword in ["TruncatedSVD", "big_walk_reviews_clean.parquet", "Module C", "Module D", "Module E", "Module F", "Module G"]:
        assert keyword in roadmap_text
