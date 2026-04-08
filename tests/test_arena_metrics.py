# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random

from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics

SCORES_POOL = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>>A"), ("B>>A", "A>>B")]

ANSWER_POOL = [
    "Yes.",
    "### Short\nUse `x` and **bold** here.\n1. first\n2. second",
    "# Title\n\n" + "More words. " * 25,
    "```python\nprint('hi')\n```\n\nPlain follow-up line.",
    "- a\n- b\n- c\n\n" + "Dense text block " * 15,
    "## Notes\n\n*italic style* and a [link](https://example.com) plus trailing detail.",
]


def _make_prediction(
    gen_base_score,
    base_gen_score,
    category=None,
    generation=None,
    baseline_answer=None,
):
    """Helper to create a prediction dict with judgment scores."""
    pred = {
        "judgement-gen-base": f"[[{gen_base_score}]]",
        "judgement-base-gen": f"[[{base_gen_score}]]",
        "generation": generation if generation is not None else "Placeholder for the model answer.",
        "baseline_answer": baseline_answer if baseline_answer is not None else "Placeholder for the baseline answer.",
    }
    if category is not None:
        pred["category"] = category
    return pred


def _fill_two_categories(m):
    random.seed(42)
    for _ in range(50):
        score = random.choice(SCORES_POOL)
        m.update(
            [
                _make_prediction(
                    score[0],
                    score[1],
                    category="hard_prompt",
                    generation=random.choice(ANSWER_POOL),
                    baseline_answer=random.choice(ANSWER_POOL),
                )
            ]
        )
    for _ in range(25):
        score = random.choice(SCORES_POOL)
        m.update(
            [
                _make_prediction(
                    score[0],
                    score[1],
                    category="creative_writing",
                    generation=random.choice(ANSWER_POOL),
                    baseline_answer=random.choice(ANSWER_POOL),
                )
            ]
        )


def test_arena_metrics_per_category_scoring_v2():
    """Test that arena-hard-v2 with multiple categories produces per-category scores."""
    m = ArenaMetrics()

    _fill_two_categories(m)

    assert m.total == 75
    assert set(m.categories) == {"hard_prompt", "creative_writing"}

    metrics = m.get_metrics(style_control=False)

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 75
    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]
    assert "invalid_scores" in metrics["pass@1"]

    # Check per-category metrics exist
    assert "category_hard_prompt" in metrics["pass@1"]
    assert metrics["pass@1"]["category_hard_prompt"]["num_entries"] == 50
    assert "score" in metrics["pass@1"]["category_hard_prompt"]
    assert "95_CI" in metrics["pass@1"]["category_hard_prompt"]

    assert "category_creative_writing" in metrics["pass@1"]
    assert metrics["pass@1"]["category_creative_writing"]["num_entries"] == 25
    assert "score" in metrics["pass@1"]["category_creative_writing"]
    assert "95_CI" in metrics["pass@1"]["category_creative_writing"]


def test_arena_metrics_single_category_v1():
    """Test that arena-hard-v1 with single category does not produce per-category breakdown."""
    m = ArenaMetrics()

    random.seed(42)

    # All entries have same category (v1 scenario)
    for _ in range(50):
        score = random.choice(SCORES_POOL)
        m.update([_make_prediction(score[0], score[1], category="arena-hard-v0.1")])

    assert m.total == 50
    assert set(m.categories) == {"arena-hard-v0.1"}

    metrics = m.get_metrics(style_control=False)

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 50
    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]

    # Check no per-category breakdown for single category
    has_category_keys = any(k.startswith("category_") for k in metrics["pass@1"].keys())
    assert not has_category_keys


def test_arena_metrics_legacy_data_no_category():
    """Test that legacy data without category field works correctly."""
    m = ArenaMetrics()

    random.seed(42)

    # Data without category field
    for _ in range(30):
        score = random.choice(SCORES_POOL)
        m.update([_make_prediction(score[0], score[1])])  # No category

    assert m.total == 30
    assert set(m.categories) == {None}

    metrics = m.get_metrics(style_control=False)

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 30
    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]

    # Check no per-category breakdown
    has_category_keys = any(k.startswith("category_") for k in metrics["pass@1"].keys())
    assert not has_category_keys


def test_arena_metrics_score_parsing():
    """Test that judge scores are correctly parsed."""
    m = ArenaMetrics()

    # Test various score formats
    test_cases = [
        ("A>>B", "A>>B"),
        ("A>B", "A>B"),
        ("A=B", "A=B"),
        ("B>A", "B>A"),
        ("B>>A", "B>>A"),
    ]

    for gen_base, base_gen in test_cases:
        m.reset()
        m.update([_make_prediction(gen_base, base_gen, category="test")])
        assert m.scores[0] == [gen_base, base_gen]


def test_arena_metrics_invalid_score_handling():
    """Test that invalid scores are handled correctly."""
    m = ArenaMetrics()

    # Invalid score format
    pred = {
        "judgement-gen-base": "No valid score here",
        "judgement-base-gen": "Also invalid",
        "category": "test",
        "generation": "Placeholder for the model answer.",
        "baseline_answer": "Placeholder for the baseline answer.",
    }
    m.update([pred])

    assert m.scores[0] == [None, None]


def test_arena_metrics_style_control_default():
    """Test that arena-metrics handles style control default correctly."""
    m = ArenaMetrics()

    _fill_two_categories(m)

    metrics = m.get_metrics()

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]


def test_arena_metrics_style_control_norm():
    """Test that arena-metrics handles style control normalization factors correctly."""
    m = ArenaMetrics()

    _fill_two_categories(m)

    # with empirical normalization factors
    with_empirical = m.get_metrics(
        style_control=True,
        category_style_control_normalization_factors=None,
        category_style_control_coefs=None,
    )

    # with fixed normalization factors
    fixed_norm = {
        "hard_prompt": {
            "mean": np.array([0.0, 0.0, 0.0, 0.0]),
            "std": np.array([1.0, 1.0, 1.0, 1.0]),
        },
        "creative_writing": {
            "mean": np.array([0.0, 0.0, 0.0, 0.0]),
            "std": np.array([1.0, 1.0, 1.0, 1.0]),
        },
    }
    with_fixed_norm = m.get_metrics(
        style_control=True,
        category_style_control_normalization_factors=fixed_norm,
        category_style_control_coefs=None,
    )


def test_arena_metrics_style_std_zero():
    """Test that arena-metrics handles standard deviation of 0 correctly."""
    m = ArenaMetrics()

    random.seed(42)
    for _ in range(50):
        score = random.choice(SCORES_POOL)
        m.update(
            [
                _make_prediction(
                    score[0],
                    score[1],
                    category="hard_prompt",
                )
            ]
        )
    for _ in range(25):
        score = random.choice(SCORES_POOL)
        m.update(
            [
                _make_prediction(
                    score[0],
                    score[1],
                    category="creative_writing",
                )
            ]
        )
    metrics = m.get_metrics(
        style_control=True,
        category_style_control_normalization_factors=None,
        category_style_control_coefs=None,
    )
    p1 = metrics["pass@1"]

    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]


def test_arena_metrics_style_control_coefs():
    """Test that arena-metrics handles style control coefficients correctly."""
    m = ArenaMetrics()

    _fill_two_categories(m)

    fixed_coefs = {
        "hard_prompt": np.array([0.1, 0.2, 0.3, 0.4]),
        "creative_writing": np.array([0.5, 0.4, 0.3, 0.2]),
    }
    with_fixed_coefs = m.get_metrics(
        style_control=True,
        category_style_control_normalization_factors=None,
        category_style_control_coefs=fixed_coefs,
    )
    p1 = with_fixed_coefs["pass@1"]
    assert "score" in with_fixed_coefs["pass@1"]
    assert "invalid_scores" in with_fixed_coefs["pass@1"]
    assert "category_hard_prompt" in with_fixed_coefs["pass@1"]
    assert "category_creative_writing" in with_fixed_coefs["pass@1"]


def test_arena_metrics_style_control_norm_and_coefs():
    """Test that arena-metrics handles style control normalization factors and coefficients correctly."""
    m = ArenaMetrics()

    _fill_two_categories(m)

    fixed_norm = {
        "hard_prompt": {
            "mean": np.array([0.1, 0.0, 0.0, 0.0]),
            "std": np.array([0.5, 0.01, 0.02, 0.02]),
        },
        "creative_writing": {
            "mean": np.array([0.05, 0.0, 0.0, 0.0]),
            "std": np.array([0.4, 0.01, 0.02, 0.02]),
        },
    }
    fixed_coefs = {
        "hard_prompt": np.array([0.1, 0.2, -0.1, 0.05]),
        "creative_writing": np.array([0.2, -0.1, 0.1, 0.0]),
    }
    with_fixed_norm_and_coefs = m.get_metrics(
        style_control=True,
        category_style_control_normalization_factors=fixed_norm,
        category_style_control_coefs=fixed_coefs,
    )
    assert "score" in with_fixed_norm_and_coefs["pass@1"]
    assert with_fixed_norm_and_coefs["pass@1"]["invalid_scores"] == 0
