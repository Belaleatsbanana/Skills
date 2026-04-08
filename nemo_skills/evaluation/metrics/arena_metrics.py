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

import logging
import re
from collections import defaultdict

import numpy as np

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Pre-computed style-control normalization factors and regression coefficients
DEFAULT_CATEGORY_STYLE_CONTROL_NORMALIZATION_FACTORS = {
    "hard_prompt": {
        "mean": np.array([0.0904, 0.0020, 0.0075, 0.0091]),
        "std": np.array([0.3343, 0.0044, 0.0116, 0.0119]),
    },
    "creative_writing": {
        "mean": np.array([0.0243, 0.0005, -0.0016, -0.0024]),
        "std": np.array([0.3345, 0.0022, 0.0100, 0.0161]),
    },
}
DEFAULT_CATEGORY_STYLE_CONTROL_COEFS = {
    "hard_prompt": np.array([0.4332, 0.1713, 0.1071, 0.1268]),
    "creative_writing": np.array([0.3337, 0.1287, -0.3389, 0.0034]),
}


def categories_covered_for_style_control(categories, norm_factors, coefs):
    """Return True if norm/coef dicts contain every category."""
    if norm_factors is not None and not all(c in norm_factors for c in categories):
        return False
    if coefs is not None and not all(c in coefs for c in categories):
        return False
    return True


class ArenaMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def _get_judge_score(self, judgment):
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        pattern = re.compile("\[\[([AB<>=]+)\]\]")
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["judgement-gen-base"] = "Rating: [[A>>B]]"
        prediction["judgement-base-gen"] = "Rating: [[B>>A]]"
        return prediction

    def get_style_metadata(self, answer):
        # adapted from https://github.com/lmarena/arena-hard-auto/blob/main/utils/add_markdown_info.py

        def count_markdown_elements(markdown_text, suffix):
            counters = {
                f"header_count{suffix}": {
                    "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
                    "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
                    "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
                    "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
                    "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
                    "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
                },
                f"list_count{suffix}": {
                    "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
                    "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
                },
                f"bold_count{suffix}": {
                    "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
                    "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
                },
            }
            return counters

        def remove_pattern(answer, pattern):
            blocks = pattern.findall(answer)
            for block in blocks:
                answer = answer.replace(block, "")
            return answer

        def get_num_tokens(answer):
            import tiktoken

            encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(answer, disallowed_special=()))

        metadata = {"token_len": get_num_tokens(answer)}
        return metadata | count_markdown_elements(
            remove_pattern(answer, re.compile("```([^`]*)```")),
            suffix="",
        )

    def get_style_features(self, baseline: str, answer: str) -> np.array:
        # adapted from https://github.com/lmarena/arena-hard-auto/blob/main/show_result.py
        baseline_style_metadata = self.get_style_metadata(baseline)
        answer_style_metadata = self.get_style_metadata(answer)
        baseline_feature_tensor = np.array(
            [v if isinstance(v, int) else sum(v.values()) for k, v in baseline_style_metadata.items()]
        )
        answer_feature_tensor = np.array(
            [v if isinstance(v, int) else sum(v.values()) for k, v in answer_style_metadata.items()]
        )
        # model features are normalized against baseline features
        normalized_feature_tensor = np.zeros_like(answer_feature_tensor, dtype=float)
        normalized_feature_tensor[0] = (answer_feature_tensor[0] - baseline_feature_tensor[0]) / (
            answer_feature_tensor[0] + baseline_feature_tensor[0]
        )
        model_md_density = answer_feature_tensor[1:] / (answer_feature_tensor[0] + 1)
        baseline_md_density = baseline_feature_tensor[1:] / (baseline_feature_tensor[0] + 1)
        normalized_feature_tensor[1:] = (model_md_density - baseline_md_density) / (
            model_md_density + baseline_md_density + 1
        )
        return normalized_feature_tensor

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        super().update(predictions)
        self.scores.append([])
        self.agg_mode = f"pass@{len(predictions)}"

        # Track category for per-category scoring (defaults to None for v1 compatibility)
        category = predictions[0].get("category")
        self.categories.append(category)

        if len(predictions) > 1:
            judge_scores = [self._get_judge_score(elem["judgement-gen-base"]) for elem in predictions]
            # adding the best score out of all the generations
            possible_scores = ["A>>B", "A>B", "A=B", "B>A", "B>>A"]
            for possible_score in possible_scores:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += predictions[best_id].get("num_generated_tokens", 0)
                    self.style_features.append(
                        self.get_style_features(
                            predictions[best_id]["baseline_answer"],
                            predictions[best_id]["generation"],
                        )
                    )
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score

            judge_scores = [self._get_judge_score(elem["judgement-base-gen"]) for elem in predictions]
            # second score is grading swapped answers, so we iterate from the end
            for possible_score in possible_scores[::-1]:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += predictions[best_id].get("num_generated_tokens", 0)
                    self.style_features.append(
                        self.get_style_features(
                            predictions[best_id]["baseline_answer"],
                            predictions[best_id]["generation"],
                        )
                    )
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score
        else:
            self.lengths += predictions[0].get("num_generated_tokens", 0)
            self.style_features.append(
                self.get_style_features(
                    predictions[0]["baseline_answer"],
                    predictions[0]["generation"],
                )
            )
            self.scores[-1] = [
                self._get_judge_score(predictions[0]["judgement-gen-base"]),
                self._get_judge_score(predictions[0]["judgement-base-gen"]),
            ]

    def get_metrics(
        self,
        style_control=True,
        category_style_control_normalization_factors=DEFAULT_CATEGORY_STYLE_CONTROL_NORMALIZATION_FACTORS,
        category_style_control_coefs=DEFAULT_CATEGORY_STYLE_CONTROL_COEFS,
    ):
        """
        Args:
            style_control (bool): Whether to use style (length and markdown) control.
                Legacy categories (e.g. missing ``category``, ``arena-hard-v0.1``) don't use style
                control.
            category_style_control_normalization_factors (dict | None): Per-category mean/std for
                style features. If None, uses empirical mean/std from the current data.
            category_style_control_coefs (dict | None): Per-category fixed regression coefficients.
                If None, fits coefficients from the current data.
        """
        from nemo_skills.evaluation.evaluator.arena import get_aggregate_score

        unique_categories = sorted(set(self.categories))
        if style_control and not categories_covered_for_style_control(
            unique_categories,
            category_style_control_normalization_factors,
            category_style_control_coefs,
        ):
            LOG.info(
                "Disabling style control: not all categories %s have entries in the "
                "normalization and coefficient tables.",
                unique_categories,
            )
            style_control = False

        # Group by category
        category_scores = defaultdict(list)
        for score, category in zip(self.scores, self.categories, strict=True):
            category_scores[category].append(score)

        category_style_features = defaultdict(list)
        for style_feature, category in zip(self.style_features, self.categories, strict=True):
            category_style_features[category].append(style_feature)

        overall_metrics = {"num_entries": self.total}
        self.update_common_metrics(overall_metrics)
        if not unique_categories:
            metrics_dict = {self.agg_mode: overall_metrics}
            return metrics_dict

        multi_category = len(unique_categories) > 1

        category_style_features_mean_std = {}
        if style_control:
            for category in unique_categories:
                raw_features = np.array(category_style_features[category])
                if category_style_control_normalization_factors is not None:
                    mean = category_style_control_normalization_factors[category]["mean"]
                    std = category_style_control_normalization_factors[category]["std"]
                else:
                    mean = raw_features.mean(axis=0)
                    std = raw_features.std(axis=0)
                if (std == 0).any():
                    LOG.warning(
                        "Category %r: zero standard deviation in style features; "
                        "skipping style control for this category only.",
                        category,
                    )
                    category_style_features_mean_std[category] = None
                else:
                    category_style_features_mean_std[category] = (mean, std)

        for category in unique_categories:
            scores = category_scores[category]
            style_features = None
            style_features_coefs = None
            if style_control and category_style_features_mean_std[category] is not None:
                mean, std = category_style_features_mean_std[category]
                raw_features = np.array(category_style_features[category])
                style_features = (raw_features - mean) / std
                if category_style_control_coefs is not None:
                    style_features_coefs = category_style_control_coefs[category]
            cat_metrics = {"num_entries": len(scores)}
            cat_metrics.update(get_aggregate_score(scores, style_features, style_features_coefs))
            if multi_category:
                overall_metrics[f"category_{category}"] = cat_metrics
            else:
                overall_metrics.update(cat_metrics)

        if multi_category:
            # assuming equal weight for each category
            score_sum = 0
            se_squared_sum = 0
            invalid_scores_sum = 0
            for category in unique_categories:
                score_sum += overall_metrics[f"category_{category}"]["score"]
                lower, upper = overall_metrics[f"category_{category}"]["95_CI"]
                se = (upper - lower) / 2 * 1.96
                se_squared_sum += se**2
                invalid_scores_sum += overall_metrics[f"category_{category}"]["invalid_scores"]
            se = np.sqrt(se_squared_sum)
            overall_metrics["95_CI"] = (- se * 1.96, se * 1.96)
            overall_metrics["score"] = score_sum / len(unique_categories)
            overall_metrics["invalid_scores"] = invalid_scores_sum

        metrics_dict = {}
        metrics_dict[self.agg_mode] = overall_metrics
        # arena metrics have their own confidence estimation, so not doing std metrics here
        return metrics_dict

    def reset(self):
        super().reset()
        self.scores = []  # list of lists
        self.categories = []  # list of category strings
        self.lengths = 0
        self.style_features = []
        # TODO: the class should support pass@k, but this forces it to report as pass@1.
        #       There is some error here for k>1
        self.agg_mode = "pass@1"
