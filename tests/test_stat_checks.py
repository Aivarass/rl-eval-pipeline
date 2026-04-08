import pytest
from eval_pipeline.judge_result import JudgeResult
from eval_pipeline.stat_checks import StatChecks


def make_evaluation(is_bug=True, severity="high", category="error_handling", episode=100):
    return {
        "discovery": {"episode": episode},
        "llm": JudgeResult(
            is_genuine_bug=is_bug,
            confidence=0.9,
            severity=severity,
            category=category,
            root_cause="test",
        ),
    }


class TestDistributionSummary:

    def test_counts_bugs_and_non_bugs(self):
        evaluations = [
            make_evaluation(is_bug=True),
            make_evaluation(is_bug=True),
            make_evaluation(is_bug=False),
        ]
        stats = StatChecks()
        result = stats.distribution_summary(evaluations)
        assert result["bugs"] == 2
        assert result["non_bugs"] == 1
        assert result["total"] == 3
        assert result["bug_rate"] == pytest.approx(2 / 3)

    def test_tracks_all_four_severities(self):
        evaluations = [
            make_evaluation(severity="critical"),
            make_evaluation(severity="high"),
            make_evaluation(severity="medium"),
            make_evaluation(severity="low"),
        ]
        stats = StatChecks()
        result = stats.distribution_summary(evaluations)
        assert result["severity"]["critical"] == 1
        assert result["severity"]["high"] == 1
        assert result["severity"]["medium"] == 1
        assert result["severity"]["low"] == 1

    def test_tracks_all_categories(self):
        evaluations = [
            make_evaluation(category="data_integrity"),
            make_evaluation(category="error_handling"),
            make_evaluation(category="state_management"),
            make_evaluation(category="validation"),
            make_evaluation(category="other"),
        ]
        stats = StatChecks()
        result = stats.distribution_summary(evaluations)
        for cat in ["data_integrity", "error_handling", "state_management", "validation", "other"]:
            assert result["category"][cat] == 1

    def test_empty_input(self):
        stats = StatChecks()
        result = stats.distribution_summary([])
        assert result["total"] == 0
        assert result["bug_rate"] == 0

    def test_no_accumulated_state_between_calls(self):
        stats = StatChecks()
        batch_1 = [make_evaluation(is_bug=True)]
        batch_2 = [make_evaluation(is_bug=False)]

        result_1 = stats.distribution_summary(batch_1)
        result_2 = stats.distribution_summary(batch_2)

        assert result_1["bugs"] == 1
        assert result_2["bugs"] == 0
        assert result_2["non_bugs"] == 1


class TestOutlierSessions:

    def test_detects_high_fp_bucket(self):
        evaluations = []
        # Episodes 0-99: mostly bugs (low FP)
        for i in range(10):
            evaluations.append(make_evaluation(is_bug=True, episode=i))
        # Episodes 100-199: all false positives (high FP)
        for i in range(10):
            evaluations.append(make_evaluation(is_bug=False, episode=100 + i))

        stats = StatChecks()
        outliers = stats.outlier_sessions(evaluations, bucket_size=100)
        assert len(outliers) > 0
        assert outliers[0]["episodes"] == "100-200"

    def test_no_outliers_when_uniform(self):
        evaluations = [make_evaluation(is_bug=True, episode=i) for i in range(10)]
        stats = StatChecks()
        outliers = stats.outlier_sessions(evaluations, bucket_size=100)
        assert len(outliers) == 0

    def test_skips_small_buckets(self):
        evaluations = [
            make_evaluation(is_bug=False, episode=500),
            make_evaluation(is_bug=False, episode=501),
        ]
        stats = StatChecks()
        outliers = stats.outlier_sessions(evaluations, bucket_size=100)
        assert len(outliers) == 0

    def test_empty_input(self):
        stats = StatChecks()
        outliers = stats.outlier_sessions([])
        assert outliers == []
