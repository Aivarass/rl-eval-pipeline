import logging
from collections import Counter

from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

SEVERITIES = ["critical", "high", "medium", "low"]
CATEGORIES = ["data_integrity", "error_handling", "state_management", "validation", "other"]


class StatChecks:

    def distribution_summary(self, evaluations):
        bugs = 0
        non_bugs = 0
        severity_counts = Counter()
        category_counts = Counter()

        for item in evaluations:
            llm = item["llm"]
            if llm.is_genuine_bug:
                bugs += 1
            else:
                non_bugs += 1
            severity_counts[llm.severity] += 1
            category_counts[llm.category] += 1

        total = bugs + non_bugs
        return {
            "total": total,
            "bugs": bugs,
            "non_bugs": non_bugs,
            "bug_rate": bugs / total if total > 0 else 0,
            "severity": {s: severity_counts.get(s, 0) for s in SEVERITIES},
            "category": {c: category_counts.get(c, 0) for c in CATEGORIES},
        }

    def batch_comparison(self, batch_a, batch_b):
        counts_a = Counter()
        counts_b = Counter()
        for item in batch_a:
            cat = item["llm"].category if item["llm"].category in CATEGORIES else "other"
            counts_a[cat] += 1
        for item in batch_b:
            cat = item["llm"].category if item["llm"].category in CATEGORIES else "other"
            counts_b[cat] += 1

        table = [
            [counts_a.get(c, 0) for c in CATEGORIES],
            [counts_b.get(c, 0) for c in CATEGORIES],
        ]
        chi2, p_value, dof, expected = chi2_contingency(table)
        drift = p_value < 0.05
        return {
            "category_drift": drift,
            "p_value": round(p_value, 4),
            "batch_a_distribution": dict(counts_a),
            "batch_b_distribution": dict(counts_b),
            "message": "significant shift in category distribution" if drift else "no significant drift detected",
        }

    def outlier_sessions(self, evaluations, bucket_size=100):
        buckets = {}

        for item in evaluations:
            episode = item["discovery"]["episode"]
            bucket_key = (episode // bucket_size) * bucket_size
            bucket_label = f"{bucket_key}-{bucket_key + bucket_size}"

            if bucket_label not in buckets:
                buckets[bucket_label] = {"total": 0, "false_positives": 0}

            buckets[bucket_label]["total"] += 1
            if not item["llm"].is_genuine_bug:
                buckets[bucket_label]["false_positives"] += 1

        total_items = sum(b["total"] for b in buckets.values())
        if total_items == 0:
            return []

        overall_fp_rate = sum(b["false_positives"] for b in buckets.values()) / total_items

        outliers = []
        for label, counts in buckets.items():
            if counts["total"] < 3:
                continue
            fp_rate = counts["false_positives"] / counts["total"]
            if fp_rate > overall_fp_rate + 0.3:
                outliers.append({
                    "episodes": label,
                    "false_positive_rate": round(fp_rate, 2),
                    "total": counts["total"],
                    "flag": "abnormally high noise compared to average",
                })

        return outliers
