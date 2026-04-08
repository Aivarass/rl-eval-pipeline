import json
import logging
import os

from openai import OpenAI

from eval_pipeline.config import JudgeConfig
from eval_pipeline.judge_result import JudgeResult

logger = logging.getLogger(__name__)


class LlmJudge:

    def __init__(self, config: JudgeConfig = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.config = config or JudgeConfig.load()

    def query_llm(self, discovery_json):
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Evaluate this discovery:\n{json.dumps(discovery_json, indent=2)}"
                }
            ]
        )

        result = JudgeResult.from_response(response.choices[0].message.content)
        return result
