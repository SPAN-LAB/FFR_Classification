from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from server_analysis.submit_analysis import build_manifest
from src.analysis.job_result import write_analysis_results
from src.analysis.runner import ConditionResult, run_analysis_conditions
from src.analysis.settings import MODEL_TRAINING_OPTIONS, SUBAVERAGE_VALUES
from src.analysis.utils import get_result_records
from src.core import AnalysisPipeline, EEGSubject, EEGTrial


class AnalysisWorkflowTests(unittest.TestCase):
    class _FakePipeline:
        def __init__(self):
            self.subjects = [SimpleNamespace(folds=None, trials=[])]

        def load_subjects(self, _subject):
            return self

        def trim_by_timestamp(self, *, start_time, end_time):
            return self

        def deepcopy(self):
            return deepcopy(self)

        def subaverage(self, *, size):
            return self

        def fold(self, *, num_folds):
            self.subjects[0].folds = [[object(), object()] for _ in range(num_folds)]
            return self

        def evaluate_model(self, *, model_name, training_options):
            return self

        def evaluate_model_with_training_amount(
            self,
            *,
            model_name,
            training_options,
            training_amount,
        ):
            return self

    class _FakeModel:
        needs_all_subjects = False

        def train(self, *, trials=None):
            return None

    def _successful_result(self) -> ConditionResult:
        subject = EEGSubject(source_filepath="4T1002.mat")
        trials = [
            EEGTrial(
                subject=subject,
                data=np.array([0.0, 1.0]),
                trial_index=index,
                timestamps=np.array([0.0, 1.0]),
                raw_label=label,
                prediction=prediction,
                prediction_distribution={1: 0.8, 2: 0.2},
            )
            for index, (label, prediction) in enumerate([(1, 1), (2, 1)])
        ]
        subject.trials = trials
        subject.folds = [[trials[0]], [trials[1]]]
        subject.setup_labels_map()
        pipeline = AnalysisPipeline()
        pipeline.subjects = [subject]
        return ConditionResult(
            value=5,
            status="success",
            pipeline=pipeline,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
        )

    def test_aggregate_results_round_trip_for_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "FFNN.4T1002.subaverage"
            write_analysis_results(
                prefix,
                condition_results=[self._successful_result()],
                metadata={
                    "subject": "4T1002",
                    "model": "FFNN",
                    "analysis": "subaverage",
                },
            )

            records = get_result_records(directory)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["value"], 5)
        self.assertEqual(records[0]["accuracy"], 0.5)
        self.assertEqual(records[0]["subject"], "4T1002")

    def test_manifest_expands_subjects_and_analysis_types(self):
        rows = build_manifest(
            models=["FFNN"],
            subjects=["A.mat", "B.mat"],
            analyses=["subaverage", "data_amount"],
            git_commit="abc123",
        )
        self.assertEqual(len(rows), 4)
        self.assertIn("FFNN A.mat subaverage abc123", rows)
        self.assertIn("FFNN B.mat data_amount abc123", rows)

    def test_all_means_all_configured_models(self):
        rows = build_manifest(
            models=["all"],
            subjects=["A.mat"],
            analyses=["subaverage"],
            git_commit="abc123",
        )
        self.assertEqual(len(rows), len(MODEL_TRAINING_OPTIONS))

    def test_default_subaverage_values_include_unaveraged_case(self):
        self.assertEqual(SUBAVERAGE_VALUES[0], 1)
        self.assertEqual(SUBAVERAGE_VALUES[-1], 125)

    def test_data_amount_values_stop_at_available_training_pool(self):
        with (
            patch(
                "src.analysis.runner.AnalysisPipeline",
                self._FakePipeline,
            ),
            patch(
                "src.analysis.runner.find_model",
                return_value=self._FakeModel,
            ),
        ):
            results = run_analysis_conditions(
                model_name="Fake",
                subject_filepath="subject.mat",
                analysis="data_amount",
                data_amount_min=2,
                data_amount_stride=3,
            )

        self.assertEqual([result.value for result in results], [2, 5, 8])
        self.assertTrue(all(result.status == "success" for result in results))


if __name__ == "__main__":
    unittest.main()
