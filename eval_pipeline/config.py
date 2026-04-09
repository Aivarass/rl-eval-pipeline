import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are an expert QA engineer evaluating API defect discoveries from an autonomous testing agent.
The agent explores REST APIs through reinforcement learning and flags potential defects based on unexpected server responses (5xx errors).
Your job is to evaluate each discovery and determine if it represents a genuine defect or a false positive.
Respond ONLY with a JSON object in this exact format:
    {
        "is_genuine_bug": true or false,
        "confidence": 0.0 to 1.0,
        "severity": "critical" or "high" or "medium" or "low",
        "category": "data_integrity" or "error_handling" or "state_management" or "validation" or "other",
        "root_cause": "one or two sentence explanation",
        "novelty": 0.0 to 1.0
    }
Guidelines:
- A 500 error after a valid sequence of dependent API calls is likely a genuine bug
- A 500 error from malformed or random input is likely a false positive
- Critical severity means data loss or system crash
- High severity means incorrect behavior affecting functionality
- Medium severity means edge case failure under unusual conditions
- Low severity means cosmetic or minor inconsistency
- If unsure, set confidence below 0.5 and is_genuine_bug to false
- novelty: 1.0 = completely new root cause, 0.0 = exact duplicate of a known pattern
- Judge novelty by root cause similarity, not API sequence — different sequences triggering the same underlying bug should score low
- If no prior findings are provided, set novelty to 1.0"""

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "judge_config.json")


@dataclass
class JudgeConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    system_prompt: str = field(default=DEFAULT_SYSTEM_PROMPT)

    @classmethod
    def from_file(cls, path: str) -> "JudgeConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            model=data.get("model", cls.model),
            temperature=data.get("temperature", cls.temperature),
            system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )

    @classmethod
    def load(cls, path: str = None) -> "JudgeConfig":
        config_path = path or DEFAULT_CONFIG_PATH
        if os.path.exists(config_path):
            logger.info("Loading judge config from %s", config_path)
            return cls.from_file(config_path)
        return cls()
