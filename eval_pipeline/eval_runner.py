import json
import logging
import os
from eval_pipeline.judge_eval import JudgeEval
from eval_pipeline.llm_judge import LlmJudge

logger = logging.getLogger(__name__)


class EvalRunner:

    def __init__(self, data_dir=None):
        self.judge = LlmJudge()
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
        else:
            self.data_dir = data_dir
        self.results = []

    def load_examples(self):
        examples = []
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_dir, filename), "r") as f:
                    examples.append(json.load(f))
        return examples

    def run(self):
        examples = self.load_examples()
        self.results = []

        for example in examples:
            discovery = example["discovery"]
            human = example["human_label"]
            llm_result = self.judge.query_llm(discovery)

            self.results.append({
                "discovery_id": discovery["discovery_id"],
                "human": human,
                "llm": llm_result,
            })

        return self.compute_metrics()

    def compute_metrics(self):
        total = len(self.results)
        if total == 0:
            return {"error": "no examples"}

        tp = fp = fn = tn = 0
        severity_correct = 0
        category_correct = 0
        severity_groups = {}
        category_groups = {}
        total_confidence = 0

        for r in self.results:
            human_bug = r["human"]["is_genuine_bug"]
            llm_bug = r["llm"].is_genuine_bug

            if llm_bug and human_bug:
                tp += 1
            elif llm_bug and not human_bug:
                fp += 1
            elif not llm_bug and human_bug:
                fn += 1
            else:
                tn += 1

            # Severity
            human_sev = r["human"]["severity"]
            llm_sev = r["llm"].severity
            sev_match = human_sev == llm_sev
            if sev_match:
                severity_correct += 1
            if human_sev not in severity_groups:
                severity_groups[human_sev] = {"total": 0, "correct": 0}
            severity_groups[human_sev]["total"] += 1
            if sev_match:
                severity_groups[human_sev]["correct"] += 1

            # Category
            human_cat = r["human"]["category"]
            llm_cat = r["llm"].category
            cat_match = human_cat == llm_cat
            if cat_match:
                category_correct += 1
            if human_cat not in category_groups:
                category_groups[human_cat] = {"total": 0, "correct": 0}
            category_groups[human_cat]["total"] += 1
            if cat_match:
                category_groups[human_cat]["correct"] += 1

            total_confidence += r["llm"].confidence

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "total": total,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "severity_accuracy": {k: round(v["correct"] / v["total"], 3) for k, v in severity_groups.items()},
            "category_accuracy": {k: round(v["correct"] / v["total"], 3) for k, v in category_groups.items()},
            "severity_overall": f"{severity_correct}/{total}",
            "category_overall": f"{category_correct}/{total}",
            "avg_confidence": round(total_confidence / total, 3),
            "calibration": JudgeEval.measure_calibration(self.results),
            "details": self.build_details(),
        }

    def build_details(self):
        details = []
        for r in self.results:
            details.append({
                "id": r["discovery_id"],
                "bug": "✓" if r["llm"].is_genuine_bug == r["human"]["is_genuine_bug"] else "✗",
                "sev": "✓" if r["llm"].severity == r["human"]["severity"] else "✗",
                "cat": "✓" if r["llm"].category == r["human"]["category"] else "✗",
                "conf": r["llm"].confidence,
                "human": f"{r['human']['severity']}/{r['human']['category']}",
                "llm": f"{r['llm'].severity}/{r['llm'].category}",
            })
        return details

    def print_report(self, metrics):
        print(f"\n{'=' * 60}")
        print("LLM JUDGE EVALUATION REPORT")
        print(f"{'=' * 60}")
        print(f"Examples:       {metrics['total']}")
        print(f"Avg confidence: {metrics['avg_confidence']}")

        print(f"\n--- Bug Detection ---")
        print(f"Accuracy:       {metrics['accuracy']}")
        print(f"Precision:      {metrics['precision']}")
        print(f"Recall:         {metrics['recall']}")
        print(f"F1:             {metrics['f1']}")
        cm = metrics["confusion"]
        print(f"Confusion:      TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")

        print(f"\n--- Severity ({metrics['severity_overall']}) ---")
        for sev, acc in metrics["severity_accuracy"].items():
            print(f"  {sev:<12}: {acc:.1%}")

        print(f"\n--- Category ({metrics['category_overall']}) ---")
        for cat, acc in metrics["category_accuracy"].items():
            print(f"  {cat:<20}: {acc:.1%}")

        print(f"\n--- Confidence Calibration (ECE={metrics['calibration']['expected_calibration_error']}) ---")
        for b in metrics["calibration"]["bins"]:
            print(f"  {b['bin']}: n={b['count']} avg_conf={b['avg_confidence']} "
                  f"actual_acc={b['actual_accuracy']} gap={b['gap']}")

        print(f"\n--- Details ---")
        for d in metrics["details"]:
            print(f"  [{d['id']}] bug:{d['bug']} sev:{d['sev']} cat:{d['cat']} "
                  f"conf:{d['conf']} | {d['human']} -> {d['llm']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    runner = EvalRunner()
    metrics = runner.run()
    runner.print_report(metrics)