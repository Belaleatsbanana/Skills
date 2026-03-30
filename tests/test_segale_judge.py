# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the SEGALE judge script (segale_judge.py).

All SEGALE model calls are mocked so no GPU or model downloads are needed.

Tests cover:
  - NeMo-Skills records are correctly converted to SEGALE format
  - Scores are correctly written back to every sentence record in the doc
  - Records without doc_id are passed through unchanged
  - .done marker is created after processing
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixture data: 5 sentences across 2 documents
#
# Mirrors the real document-level MT output format:
#   - seg_id=0 carries the full document translation in `generation`
#   - all other segments have generation="" (the model was given the full
#     document as one prompt and produced one generation, stored on the first
#     record; subsequent records preserve the source/reference structure)
# ---------------------------------------------------------------------------
DOC1_GENERATION = "The cat sat on the mat. She observed it carefully. Then she left it in peace."
DOC2_GENERATION = "It was a bright and sunny day. Children were playing in the park."

RECORDS = [
    # doc1 — 3 source/reference sentences, one record per document
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "text": "Die Katze saß auf der Matte. Sie beobachtete es aufmerksam. Dann ließ sie es in Ruhe.",
        "source_sentences": [
            "Die Katze saß auf der Matte.",
            "Sie beobachtete es aufmerksam.",
            "Dann ließ sie es in Ruhe.",
        ],
        "reference_sentences": [
            "The cat sat on the mat.",
            "She watched it carefully.",
            "Then she left it alone.",
        ],
        "generation": DOC1_GENERATION,
        "doc_id": "doc1",
        "seg_id": 1,
    },
    # doc2 — 2 source/reference sentences
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "text": "Es war ein heller Tag. Kinder spielten im Park.",
        "source_sentences": [
            "Es war ein heller Tag.",
            "Kinder spielten im Park.",
        ],
        "reference_sentences": [
            "It was a bright day.",
            "Children played in the park.",
        ],
        "generation": DOC2_GENERATION,
        "doc_id": "doc2",
        "seg_id": 1,
    },
]

# Records with no doc_id (graceful-degradation fixture)
RECORDS_NO_DOC_ID = [{k: v for k, v in r.items() if k != "doc_id"} for r in RECORDS]

# Pre-baked aligned spans that the mock alignment step will return.
# These represent what segale-align produces after segmenting and aligning
# the full document generation — NOT the raw per-segment generation fields.
# 3 spans for doc1, 2 spans for doc2 (1:1 alignment for simplicity).
MOCK_ALIGNED_SPANS = [
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][0],
        "ref": RECORDS[0]["reference_sentences"][0],
        "tgt": "The cat sat on the mat.",
        "seg_id": 1,
    },
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][1],
        "ref": RECORDS[0]["reference_sentences"][1],
        "tgt": "She observed it carefully.",
        "seg_id": 2,
    },
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][2],
        "ref": RECORDS[0]["reference_sentences"][2],
        "tgt": "Then she left it in peace.",
        "seg_id": 3,
    },
    {
        "doc_id": "doc2",
        "sys_id": "nemo_skills",
        "src": RECORDS[1]["source_sentences"][0],
        "ref": RECORDS[1]["reference_sentences"][0],
        "tgt": "It was a bright and sunny day.",
        "seg_id": 1,
    },
    {
        "doc_id": "doc2",
        "sys_id": "nemo_skills",
        "src": RECORDS[1]["source_sentences"][1],
        "ref": RECORDS[1]["reference_sentences"][1],
        "tgt": "Children were playing in the park.",
        "seg_id": 2,
    },
]

# Score values the mock scoring step will return (one per aligned span).
MOCK_COMET_SCORES = [0.88, 0.90, 0.85, 0.91, 0.89]
MOCK_COMET_QE_SCORES = [0.83, 0.86, 0.80, 0.87, 0.84]
MOCK_METRICX_SCORES = [1.2, 1.1, 1.3, 0.9, 1.0]
MOCK_METRICX_QE_SCORES = [1.5, 1.4, 1.6, 1.1, 1.2]

# Expected doc-level averages computed from mock scores above.
# doc1: spans 0,1,2 — doc2: spans 3,4
DOC1_COMET = sum(MOCK_COMET_SCORES[:3]) / 3
DOC1_COMET_QE = sum(MOCK_COMET_QE_SCORES[:3]) / 3
DOC1_METRICX = sum(MOCK_METRICX_SCORES[:3]) / 3
DOC1_METRICX_QE = sum(MOCK_METRICX_QE_SCORES[:3]) / 3
DOC2_COMET = sum(MOCK_COMET_SCORES[3:]) / 2
DOC2_COMET_QE = sum(MOCK_COMET_QE_SCORES[3:]) / 2
DOC2_METRICX = sum(MOCK_METRICX_SCORES[3:]) / 2
DOC2_METRICX_QE = sum(MOCK_METRICX_QE_SCORES[3:]) / 2


# ---------------------------------------------------------------------------
# Helpers to build fake SEGALE module stubs
# ---------------------------------------------------------------------------


def _make_segale_align_stub():
    """Return a minimal segale_align module stub."""
    stub = types.ModuleType("segale_align")
    stub.VERBOSE = 0
    stub.SPACY = "ersatz"
    stub.STOP_JUMP = 0.15
    stub.COST_MIN = 0.30
    stub.COST_MAX = 0.30
    stub.init_config = MagicMock()
    stub.load_alternative_model = MagicMock(return_value=(None, MagicMock()))
    stub.merge_system_entries = MagicMock(return_value=[])
    stub.merge_ref_entries = MagicMock(return_value=[])
    stub.combine_system_ref = MagicMock(return_value=[])
    stub.prepare_doc_windows = MagicMock(return_value=None)
    return stub


def _make_laser_stub():
    """Return a minimal laser_encoders stub."""
    stub = types.ModuleType("laser_encoders")
    stub.LaserEncoderPipeline = MagicMock(return_value=MagicMock())
    return stub


def _make_segale_eval_stub():
    """Return a segale_eval stub whose scoring functions echo mock scores."""
    stub = types.ModuleType("segale_eval")
    stub.run_comet_evaluation = MagicMock(return_value=MOCK_COMET_SCORES)
    stub.run_comet_qe_evaluation = MagicMock(return_value=MOCK_COMET_QE_SCORES)
    stub.run_metricx_evaluation = MagicMock(return_value=MOCK_METRICX_SCORES)
    stub.run_metricx_qe_evaluation = MagicMock(return_value=MOCK_METRICX_QE_SCORES)
    return stub


# ---------------------------------------------------------------------------
# Import segale_judge after stubbing out heavy dependencies
# ---------------------------------------------------------------------------


def _import_judge_with_stubs():
    """Import segale_judge with SEGALE library stubs injected into sys.modules."""
    # Remove cached module if already imported in a previous test.
    sys.modules.pop("segale_judge", None)

    # Inject stubs before import so the lazy `import segale_align` inside the
    # functions resolves to our stubs.
    sys.modules["segale_align"] = _make_segale_align_stub()
    sys.modules["laser_encoders"] = _make_laser_stub()
    sys.modules["segale_eval"] = _make_segale_eval_stub()

    # Add the evaluator directory to sys.path so `import segale_judge` works.
    evaluator_dir = str(Path(__file__).parent.parent / "nemo_skills" / "evaluation" / "evaluator")
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)

    import segale_judge

    return segale_judge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSegaleEntriesDocFormat:
    """_build_segale_entries_doc_format converts NeMo-Skills records to SEGALE format."""

    def setup_method(self):
        self.judge = _import_judge_with_stubs()

    def test_system_entries_one_per_record(self):
        sys_entries, _, _ = self.judge._build_segale_entries_doc_format(RECORDS)
        assert len(sys_entries) == 2  # doc1, doc2

        assert sys_entries[0]["doc_id"] == "doc1"
        assert sys_entries[0]["src"] == RECORDS[0]["text"]
        assert sys_entries[0]["tgt"] == DOC1_GENERATION
        assert sys_entries[0]["sys_id"] == "nemo_skills"

        assert sys_entries[1]["doc_id"] == "doc2"
        assert sys_entries[1]["tgt"] == DOC2_GENERATION

    def test_ref_entries_per_sentence(self):
        _, ref_entries, _ = self.judge._build_segale_entries_doc_format(RECORDS)
        assert len(ref_entries) == 5  # 3 for doc1 + 2 for doc2

        assert ref_entries[0]["src"] == "Die Katze saß auf der Matte."
        assert ref_entries[0]["tgt"] == "The cat sat on the mat."
        assert ref_entries[0]["sys_id"] == "reference"
        assert ref_entries[0]["seg_id"] == 1

    def test_ref_entries_seg_ids_are_1_indexed(self):
        _, ref_entries, _ = self.judge._build_segale_entries_doc_format(RECORDS)
        doc1_refs = [e for e in ref_entries if e["doc_id"] == "doc1"]
        assert [e["seg_id"] for e in doc1_refs] == [1, 2, 3]

    def test_system_entry_seg_id_is_1(self):
        sys_entries, _, _ = self.judge._build_segale_entries_doc_format(RECORDS)
        assert all(e["seg_id"] == 1 for e in sys_entries)

    def test_ref_sys_id_is_reference(self):
        _, ref_entries, _ = self.judge._build_segale_entries_doc_format(RECORDS)
        assert all(e["sys_id"] == "reference" for e in ref_entries)


class TestRunScoring:
    """_run_scoring returns correct doc-level averages from span scores."""

    def setup_method(self):
        self.judge = _import_judge_with_stubs()

    def test_doc1_comet_average(self):
        scores = self.judge._run_scoring(MOCK_ALIGNED_SPANS)
        assert abs(scores["doc1"]["segale_comet"] - DOC1_COMET) < 1e-9

    def test_doc2_comet_average(self):
        scores = self.judge._run_scoring(MOCK_ALIGNED_SPANS)
        assert abs(scores["doc2"]["segale_comet"] - DOC2_COMET) < 1e-9

    def test_all_metrics_present(self):
        scores = self.judge._run_scoring(MOCK_ALIGNED_SPANS)
        for doc_id in ("doc1", "doc2"):
            for field in ("segale_comet", "segale_comet_qe", "segale_metricx", "segale_metricx_qe", "segale_bleu"):
                assert field in scores[doc_id], f"{field} missing for {doc_id}"

    def test_segale_bleu_is_positive(self):
        scores = self.judge._run_scoring(MOCK_ALIGNED_SPANS)
        for doc_id in ("doc1", "doc2"):
            assert scores[doc_id]["segale_bleu"] >= 0, f"segale_bleu should be >= 0 for {doc_id}"

    def test_total_seg_count(self):
        scores = self.judge._run_scoring(MOCK_ALIGNED_SPANS)
        assert scores["doc1"]["segale_total_seg"] == 3
        assert scores["doc2"]["segale_total_seg"] == 2


class TestProcessFile:
    """process_file writes scores back to every sentence in a doc."""

    def setup_method(self):
        self.judge = _import_judge_with_stubs()

    def _write_jsonl(self, path, records):
        with open(path, "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _read_jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_scores_written_to_all_sentences_in_doc(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        self._write_jsonl(input_file, RECORDS)

        with (
            patch.object(self.judge, "_run_alignment", return_value=MOCK_ALIGNED_SPANS),
            patch.object(self.judge, "_run_scoring", wraps=self.judge._run_scoring),
        ):
            self.judge.process_file(
                input_file=input_file,
                output_file=output_file,
                segmenter="ersatz",
                task_lang="en",
                embedding_model=None,
                proc_device="cpu",
            )

        out = self._read_jsonl(output_file)
        doc1_rows = [r for r in out if r.get("doc_id") == "doc1"]
        doc2_rows = [r for r in out if r.get("doc_id") == "doc2"]

        # Every sentence in doc1 should carry the same doc-level score.
        for row in doc1_rows:
            assert abs(row["segale_comet"] - DOC1_COMET) < 1e-9
        for row in doc2_rows:
            assert abs(row["segale_comet"] - DOC2_COMET) < 1e-9

    def test_done_marker_created(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        self._write_jsonl(input_file, RECORDS)

        with patch.object(self.judge, "_run_alignment", return_value=MOCK_ALIGNED_SPANS):
            self.judge.process_file(
                input_file=input_file,
                output_file=output_file,
                segmenter="ersatz",
                task_lang="en",
                embedding_model=None,
                proc_device="cpu",
            )

        assert Path(str(output_file) + ".done").exists()

    def test_no_doc_id_copies_file_unchanged(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        self._write_jsonl(input_file, RECORDS_NO_DOC_ID)

        self.judge.process_file(
            input_file=input_file,
            output_file=output_file,
            segmenter="ersatz",
            task_lang="en",
            embedding_model=None,
            proc_device="cpu",
        )

        out = self._read_jsonl(output_file)
        assert len(out) == len(RECORDS_NO_DOC_ID)
        for row in out:
            assert "segale_comet" not in row

    def test_empty_alignment_copies_file_unchanged(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        self._write_jsonl(input_file, RECORDS)

        with patch.object(self.judge, "_run_alignment", return_value=[]):
            self.judge.process_file(
                input_file=input_file,
                output_file=output_file,
                segmenter="ersatz",
                task_lang="en",
                embedding_model=None,
                proc_device="cpu",
            )

        out = self._read_jsonl(output_file)
        for row in out:
            assert "segale_comet" not in row
