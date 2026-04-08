import json
import os
import pytest
from eval_pipeline.llm_judge import LlmJudge

def load_discovery(category, filename):
    path = os.path.join(os.path.dirname(__file__), "..", "data", category, filename)
    with open(path, "r") as f:
        return json.load(f)

class TestLlmJudge:

    @pytest.fixture
    def judge(self):
        return LlmJudge()

    @pytest.mark.parametrize("filename", [f"example_{i}.json" for i in range(1, 11)])
    def test_genuine_bugs(self, judge, filename):
        discovery = load_discovery("bugs", filename)
        response = judge.query_llm(discovery)
        assert response.is_genuine_bug is True
        assert response.severity in ("high", "critical")
        assert response.root_cause and len(response.root_cause) > 0

    @pytest.mark.parametrize("filename", [f"example_{i}.json" for i in range(1, 7)])
    def test_noise(self, judge, filename):
        discovery = load_discovery("noise", filename)
        response = judge.query_llm(discovery)
        assert response.is_genuine_bug is False
        assert response.severity == "low"
        assert response.root_cause and len(response.root_cause) > 0

    @pytest.mark.parametrize("filename", [f"example_{i}.json" for i in range(1, 11)])
    def test_non_bugs(self, judge, filename):
        discovery = load_discovery("nonbugs", filename)
        response = judge.query_llm(discovery)
        assert response.is_genuine_bug is False
        assert response.severity == "low"
        assert response.root_cause and len(response.root_cause) > 0