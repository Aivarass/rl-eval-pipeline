import json
from dataclasses import dataclass


@dataclass
class JudgeResult:
    is_genuine_bug: bool
    confidence: float
    severity: str
    category: str
    root_cause: str
    novelty: float = 1.0

    VALID_SEVERITIES = {"critical", "high", "medium", "low"}
    VALID_CATEGORIES = {"data_integrity", "error_handling", "state_management", "validation", "other"}

    @staticmethod
    def from_response(response_text: str) -> "JudgeResult":
        try:
            cleaned = response_text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

        required = {"is_genuine_bug", "confidence", "severity", "category", "root_cause"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        # Validate types and values
        if not isinstance(data["is_genuine_bug"], bool):
            raise ValueError(f"is_genuine_bug must be bool, got {type(data['is_genuine_bug'])}")

        if not isinstance(data["confidence"], (int, float)) or not 0.0 <= data["confidence"] <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {data['confidence']}")

        if data["severity"] not in JudgeResult.VALID_SEVERITIES:
            raise ValueError(f"Invalid severity: {data['severity']}")

        if data["category"] not in JudgeResult.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {data['category']}")

        if not isinstance(data["root_cause"], str) or not data["root_cause"]:
            raise ValueError(f"root_cause must be a non-empty string")

        novelty = data.get("novelty", 1.0)
        if not isinstance(novelty, (int, float)) or not 0.0 <= novelty <= 1.0:
            raise ValueError(f"novelty must be 0.0-1.0, got {novelty}")

        result = {k: data[k] for k in required}
        result["novelty"] = float(novelty)
        return JudgeResult(**result)