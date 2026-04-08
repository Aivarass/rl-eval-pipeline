import json
import os
import pytest

from eval_pipeline.eval_runner import EvalRunner
from eval_pipeline.llm_judge import LlmJudge


def load_example(filename):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "golden", filename)
    with open(path, "r") as f:
        return json.load(f)


class TestEvalRunner:
    @pytest.fixture
    def judge(self):
        return LlmJudge()

    @pytest.mark.parametrize("filename", [f"example_{i}.json" for i in range(1, 11)])
    def test_genuine_bugs(self, judge, filename):
        example = load_example(filename)
        result = judge.query_llm(example["discovery"])
        human = example["human_label"]
        assert result.is_genuine_bug == human["is_genuine_bug"]
        assert result.confidence >= 0.5

    def test_full_eval(self):
        runner = EvalRunner()
        metrics = runner.run()
        runner.print_report(metrics)
        assert metrics["accuracy"] >= 0.7
        assert metrics["f1"] >= 0.7
