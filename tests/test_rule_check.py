import pytest
from eval_pipeline.rule_check import RuleCheck


def make_discovery(**overrides):
    base = {
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
        "state_features": {
            "hasValidItemId": 1,
            "hasValidPriceId": 1,
            "hasValidDiscountId": 0,
            "hasValidPointsId": 0,
        },
    }
    base.update(overrides)
    return base


class TestInputValidation:

    def setup_method(self):
        self.checker = RuleCheck()

    def test_valid_discovery_passes(self):
        data, warnings = self.checker.input_completeness_validation(make_discovery())
        assert data["discovery_id"] == "test-001"
        assert isinstance(warnings, list)

    def test_accepts_json_string(self):
        import json
        raw = json.dumps(make_discovery())
        data, warnings = self.checker.input_completeness_validation(raw)
        assert data["discovery_id"] == "test-001"

    def test_rejects_invalid_json_string(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            self.checker.input_completeness_validation("not json")

    def test_rejects_missing_required_fields(self):
        discovery = make_discovery()
        del discovery["episode"]
        with pytest.raises(ValueError, match="Missing fields"):
            self.checker.input_completeness_validation(discovery)

    def test_rejects_non_string_discovery_id(self):
        with pytest.raises(ValueError, match="discovery_id must be string"):
            self.checker.input_completeness_validation(make_discovery(discovery_id=123))

    def test_rejects_non_int_episode(self):
        with pytest.raises(ValueError, match="episode must be int"):
            self.checker.input_completeness_validation(make_discovery(episode="100"))

    def test_rejects_empty_api_sequence(self):
        with pytest.raises(ValueError, match="api_sequence must not be empty"):
            self.checker.input_completeness_validation(make_discovery(api_sequence=[]))

    def test_rejects_non_int_final_status(self):
        with pytest.raises(ValueError, match="final_status must be int"):
            self.checker.input_completeness_validation(make_discovery(final_status="500"))

    def test_rejects_non_numeric_reward(self):
        with pytest.raises(ValueError, match="reward must be numeric"):
            self.checker.input_completeness_validation(make_discovery(reward="high"))

    def test_rejects_missing_state_features(self):
        with pytest.raises(ValueError, match="state_features missing"):
            self.checker.input_completeness_validation(make_discovery(state_features={"hasValidItemId": 1}))

    def test_warns_on_final_status_mismatch(self):
        data, warnings = self.checker.input_completeness_validation(make_discovery(final_status=200))
        assert any("doesn't match" in w for w in warnings)

    def test_rejects_api_step_missing_fields(self):
        bad_sequence = [{"method": "POST", "endpoint": "/items"}]
        with pytest.raises(ValueError, match="missing fields"):
            self.checker.input_completeness_validation(make_discovery(api_sequence=bad_sequence))


class TestDependencyValidation:

    def setup_method(self):
        self.checker = RuleCheck()

    def test_valid_dependency_order(self):
        sequence = [
            {"method": "POST", "endpoint": "/items", "status": 201},
            {"method": "POST", "endpoint": "/prices", "status": 201},
            {"method": "POST", "endpoint": "/discounts", "status": 201},
        ]
        warnings = self.checker.validate_api_sequence(sequence)
        assert len(warnings) == 0

    def test_warns_on_missing_parent(self):
        sequence = [
            {"method": "POST", "endpoint": "/prices", "status": 201},
        ]
        warnings = self.checker.validate_api_sequence(sequence)
        assert any("/items" in w for w in warnings)

    def test_warns_on_delete_before_create(self):
        sequence = [
            {"method": "DELETE", "endpoint": "/items", "status": 500},
        ]
        warnings = self.checker.validate_api_sequence(sequence)
        assert any("never created" in w for w in warnings)

    def test_delete_removes_from_created(self):
        sequence = [
            {"method": "POST", "endpoint": "/items", "status": 201},
            {"method": "DELETE", "endpoint": "/items", "status": 204},
            {"method": "GET", "endpoint": "/items", "status": 404},
        ]
        warnings = self.checker.validate_api_sequence(sequence)
        assert any("never created" in w for w in warnings)


class TestDuplicateDetection:

    def setup_method(self):
        self.checker = RuleCheck()

    def test_first_occurrence_not_duplicate(self):
        sequence = [{"method": "POST", "endpoint": "/items", "status": 201}]
        assert self.checker.is_duplicate(sequence) is False

    def test_second_occurrence_is_duplicate(self):
        sequence = [{"method": "POST", "endpoint": "/items", "status": 201}]
        self.checker.is_duplicate(sequence)
        assert self.checker.is_duplicate(sequence) is True

    def test_normalises_consecutive_repeats(self):
        seq_short = [
            {"method": "GET", "endpoint": "/items", "status": 200},
            {"method": "DELETE", "endpoint": "/items", "status": 500},
        ]
        seq_long = [
            {"method": "GET", "endpoint": "/items", "status": 200},
            {"method": "GET", "endpoint": "/items", "status": 200},
            {"method": "GET", "endpoint": "/items", "status": 200},
            {"method": "DELETE", "endpoint": "/items", "status": 500},
        ]
        assert self.checker.is_duplicate(seq_short) is False
        assert self.checker.is_duplicate(seq_long) is True

    def test_different_sequences_not_duplicate(self):
        seq_a = [{"method": "POST", "endpoint": "/items", "status": 201}]
        seq_b = [{"method": "POST", "endpoint": "/prices", "status": 201}]
        assert self.checker.is_duplicate(seq_a) is False
        assert self.checker.is_duplicate(seq_b) is False
