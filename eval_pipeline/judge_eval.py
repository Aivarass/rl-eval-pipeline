import logging
from collections import Counter

from eval_pipeline.judge_result import JudgeResult
from eval_pipeline.llm_judge import LlmJudge

logger = logging.getLogger(__name__)


class JudgeEval:

    def __init__(self, judge: LlmJudge = None):
        self.judge = judge or LlmJudge()

    def measure_consistency(self, discovery, n_runs=5):
        """Run the same discovery through the judge N times and measure agreement.

        Returns per-field agreement rates and the raw results for inspection.
        """
        results = []
        for i in range(n_runs):
            result = self.judge.query_llm(discovery)
            results.append(result)

        bug_votes = Counter(r.is_genuine_bug for r in results)
        severity_votes = Counter(r.severity for r in results)
        category_votes = Counter(r.category for r in results)

        bug_agreement = bug_votes.most_common(1)[0][1] / n_runs
        severity_agreement = severity_votes.most_common(1)[0][1] / n_runs
        category_agreement = category_votes.most_common(1)[0][1] / n_runs

        confidences = [r.confidence for r in results]

        return {
            "n_runs": n_runs,
            "bug_detection": {
                "agreement": round(bug_agreement, 2),
                "votes": dict(bug_votes),
                "majority": bug_votes.most_common(1)[0][0],
            },
            "severity": {
                "agreement": round(severity_agreement, 2),
                "votes": dict(severity_votes),
                "majority": severity_votes.most_common(1)[0][0],
            },
            "category": {
                "agreement": round(category_agreement, 2),
                "votes": dict(category_votes),
                "majority": category_votes.most_common(1)[0][0],
            },
            "confidence": {
                "mean": round(sum(confidences) / len(confidences), 3),
                "min": round(min(confidences), 3),
                "max": round(max(confidences), 3),
                "spread": round(max(confidences) - min(confidences), 3),
            },
        }

    def measure_batch_consistency(self, discoveries, n_runs=5):
        """Run consistency measurement across multiple discoveries.

        Returns per-discovery results and aggregate agreement.
        """
        per_discovery = []
        total_bug_agreement = 0
        total_severity_agreement = 0
        total_category_agreement = 0

        for discovery in discoveries:
            discovery_id = discovery.get("discovery_id", "unknown")
            result = self.measure_consistency(discovery, n_runs=n_runs)
            result["discovery_id"] = discovery_id
            per_discovery.append(result)
            total_bug_agreement += result["bug_detection"]["agreement"]
            total_severity_agreement += result["severity"]["agreement"]
            total_category_agreement += result["category"]["agreement"]

        count = len(discoveries)
        return {
            "n_discoveries": count,
            "n_runs_per_discovery": n_runs,
            "avg_bug_agreement": round(total_bug_agreement / count, 3) if count else 0,
            "avg_severity_agreement": round(total_severity_agreement / count, 3) if count else 0,
            "avg_category_agreement": round(total_category_agreement / count, 3) if count else 0,
            "per_discovery": per_discovery,
        }

    @staticmethod
    def measure_calibration(eval_results, n_bins=5):
        """Measure whether judge confidence scores are well-calibrated.

        Bins results by confidence level and computes actual accuracy per bin.
        A well-calibrated judge at confidence=0.9 should be correct ~90% of the time.

        eval_results: list of {"human": {"is_genuine_bug": bool, ...}, "llm": JudgeResult}
        """
        bin_width = 1.0 / n_bins
        bins = {}

        for r in eval_results:
            confidence = r["llm"].confidence
            human_bug = r["human"]["is_genuine_bug"]
            llm_bug = r["llm"].is_genuine_bug
            correct = human_bug == llm_bug

            bin_idx = min(int(confidence / bin_width), n_bins - 1)
            bin_lower = round(bin_idx * bin_width, 2)
            bin_upper = round(bin_lower + bin_width, 2)
            bin_label = f"{bin_lower:.1f}-{bin_upper:.1f}"

            if bin_label not in bins:
                bins[bin_label] = {"total": 0, "correct": 0, "sum_confidence": 0.0}

            bins[bin_label]["total"] += 1
            bins[bin_label]["sum_confidence"] += confidence
            if correct:
                bins[bin_label]["correct"] += 1

        calibration = []
        total_calibration_error = 0
        total_items = 0

        for label in sorted(bins.keys()):
            b = bins[label]
            actual_accuracy = b["correct"] / b["total"]
            avg_confidence = b["sum_confidence"] / b["total"]
            gap = abs(avg_confidence - actual_accuracy)
            total_calibration_error += gap * b["total"]
            total_items += b["total"]

            calibration.append({
                "bin": label,
                "count": b["total"],
                "avg_confidence": round(avg_confidence, 3),
                "actual_accuracy": round(actual_accuracy, 3),
                "gap": round(gap, 3),
            })

        ece = round(total_calibration_error / total_items, 3) if total_items else 0

        return {
            "expected_calibration_error": ece,
            "bins": calibration,
        }
