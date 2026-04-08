import pytest
from eval_pipeline.judge_result import JudgeResult


class TestJudgeResult:

    def test_valid_response(self):
        raw = '{"is_genuine_bug": true, "confidence": 0.9, "severity": "high", "category": "error_handling", "root_cause": "Server crash on delete"}'
        result = JudgeResult.from_response(raw)
        assert result.is_genuine_bug is True
        assert result.confidence == 0.9
        assert result.severity == "high"
        assert result.category == "error_handling"
        assert result.root_cause == "Server crash on delete"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"is_genuine_bug": false, "confidence": 0.3, "severity": "low", "category": "validation", "root_cause": "Bad input"}\n```'
        result = JudgeResult.from_response(raw)
        assert result.is_genuine_bug is False

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            JudgeResult.from_response("not json at all")

    def test_rejects_missing_fields(self):
        raw = '{"is_genuine_bug": true, "confidence": 0.5}'
        with pytest.raises(ValueError, match="Missing fields"):
            JudgeResult.from_response(raw)

    def test_rejects_non_bool_bug(self):
        raw = '{"is_genuine_bug": "yes", "confidence": 0.5, "severity": "low", "category": "other", "root_cause": "test"}'
        with pytest.raises(ValueError, match="is_genuine_bug must be bool"):
            JudgeResult.from_response(raw)

    def test_rejects_confidence_out_of_range(self):
        raw = '{"is_genuine_bug": true, "confidence": 1.5, "severity": "low", "category": "other", "root_cause": "test"}'
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            JudgeResult.from_response(raw)

    def test_rejects_negative_confidence(self):
        raw = '{"is_genuine_bug": true, "confidence": -0.1, "severity": "low", "category": "other", "root_cause": "test"}'
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            JudgeResult.from_response(raw)

    def test_rejects_invalid_severity(self):
        raw = '{"is_genuine_bug": true, "confidence": 0.5, "severity": "extreme", "category": "other", "root_cause": "test"}'
        with pytest.raises(ValueError, match="Invalid severity"):
            JudgeResult.from_response(raw)

    def test_rejects_invalid_category(self):
        raw = '{"is_genuine_bug": true, "confidence": 0.5, "severity": "low", "category": "unknown", "root_cause": "test"}'
        with pytest.raises(ValueError, match="Invalid category"):
            JudgeResult.from_response(raw)

    def test_rejects_empty_root_cause(self):
        raw = '{"is_genuine_bug": true, "confidence": 0.5, "severity": "low", "category": "other", "root_cause": ""}'
        with pytest.raises(ValueError, match="root_cause must be a non-empty string"):
            JudgeResult.from_response(raw)

    def test_accepts_boundary_confidence_values(self):
        for conf in [0.0, 1.0]:
            raw = f'{{"is_genuine_bug": true, "confidence": {conf}, "severity": "low", "category": "other", "root_cause": "test"}}'
            result = JudgeResult.from_response(raw)
            assert result.confidence == conf

    def test_all_valid_severities(self):
        for sev in JudgeResult.VALID_SEVERITIES:
            raw = f'{{"is_genuine_bug": true, "confidence": 0.5, "severity": "{sev}", "category": "other", "root_cause": "test"}}'
            result = JudgeResult.from_response(raw)
            assert result.severity == sev

    def test_all_valid_categories(self):
        for cat in JudgeResult.VALID_CATEGORIES:
            raw = f'{{"is_genuine_bug": true, "confidence": 0.5, "severity": "low", "category": "{cat}", "root_cause": "test"}}'
            result = JudgeResult.from_response(raw)
            assert result.category == cat
