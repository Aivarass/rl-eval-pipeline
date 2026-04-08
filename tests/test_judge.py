from eval_pipeline.llm_judge import LlmJudge


class TestJudge:



    def test_judge(self):
        discovery = {
            "discovery_id": "test-001",
            "timestamp": "2026-04-07T10:00:00",
            "episode": 100,
            "api_sequence": [
                {"method": "POST", "endpoint": "/items", "status": 201},
                {"method": "POST", "endpoint": "/prices", "status": 201},
                {"method": "DELETE", "endpoint": "/prices", "status": 500},
            ],
            "final_status": 500,
            "reward": 10.0,
            "state_features": {"hasValidItemId": 1, "hasValidPriceId": 1, "hasValidDiscountId": 0, "hasValidPointsId": 0},
        }

        judge = LlmJudge()
        result = judge.query_llm(discovery)
        assert result.is_genuine_bug is not None
        assert 0.0 <= result.confidence <= 1.0

if __name__ == '__main__':
    obj = TestJudge()
    obj.test_judge()