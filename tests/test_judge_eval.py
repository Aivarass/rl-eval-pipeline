import pytest
from eval_pipeline.judge_result import JudgeResult
from eval_pipeline.judge_eval import JudgeEval


def make_eval_result(confidence, human_bug, llm_bug, severity="high", category="error_handling"):
    return {
        "human": {"is_genuine_bug": human_bug, "severity": severity, "category": category},
        "llm": JudgeResult(
            is_genuine_bug=llm_bug,
            confidence=confidence,
            severity=severity,
            category=category,
            root_cause="test",
        ),
    }


class TestCalibration:

    def test_perfectly_calibrated(self):
        """High confidence results are all correct, low confidence are mixed."""
        results = [
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.9, human_bug=False, llm_bug=False),
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.9, human_bug=False, llm_bug=False),
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=True),
        ]
        report = JudgeEval.measure_calibration(results, n_bins=5)
        high_bin = next(b for b in report["bins"] if b["avg_confidence"] >= 0.8)
        assert high_bin["actual_accuracy"] == 1.0

    def test_overconfident_judge(self):
        """Judge says 0.9 confidence but is wrong half the time."""
        results = [
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=False),
            make_eval_result(confidence=0.9, human_bug=False, llm_bug=False),
            make_eval_result(confidence=0.9, human_bug=False, llm_bug=True),
        ]
        report = JudgeEval.measure_calibration(results, n_bins=5)
        high_bin = next(b for b in report["bins"] if b["avg_confidence"] >= 0.8)
        assert high_bin["actual_accuracy"] == 0.5
        assert high_bin["gap"] == pytest.approx(0.4, abs=0.05)
        assert report["expected_calibration_error"] > 0.3

    def test_ece_is_zero_for_perfect_calibration(self):
        """If avg_confidence matches actual_accuracy in every bin, ECE = 0."""
        results = [
            make_eval_result(confidence=0.5, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.5, human_bug=True, llm_bug=False),
        ]
        report = JudgeEval.measure_calibration(results, n_bins=5)
        bin_data = report["bins"][0]
        assert bin_data["avg_confidence"] == pytest.approx(bin_data["actual_accuracy"], abs=0.01)

    def test_bins_cover_full_range(self):
        """Results spread across confidence range produce multiple bins."""
        results = [
            make_eval_result(confidence=0.1, human_bug=False, llm_bug=False),
            make_eval_result(confidence=0.5, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.9, human_bug=True, llm_bug=True),
        ]
        report = JudgeEval.measure_calibration(results, n_bins=5)
        assert len(report["bins"]) == 3

    def test_empty_results(self):
        report = JudgeEval.measure_calibration([], n_bins=5)
        assert report["expected_calibration_error"] == 0
        assert report["bins"] == []

    def test_all_bins_have_required_fields(self):
        results = [
            make_eval_result(confidence=0.7, human_bug=True, llm_bug=True),
            make_eval_result(confidence=0.3, human_bug=False, llm_bug=False),
        ]
        report = JudgeEval.measure_calibration(results, n_bins=5)
        for b in report["bins"]:
            assert "bin" in b
            assert "count" in b
            assert "avg_confidence" in b
            assert "actual_accuracy" in b
            assert "gap" in b
