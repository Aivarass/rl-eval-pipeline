import logging

from eval_pipeline.llm_judge import LlmJudge
from eval_pipeline.rule_check import RuleCheck
from eval_pipeline.stat_checks import StatChecks

logger = logging.getLogger(__name__)


class QualityReport:

    def __init__(self, discoveries):
        self.discoveries = discoveries
        self.rule_check = RuleCheck()
        self.llm_judge = LlmJudge()
        self.stat_checks = StatChecks()
        self.accumulating_batch = []

    def execute_single_pipeline(self, discovery, terminal):
        validated = self.execute_rules_based_check(discovery)
        if validated is None:
            return None, None
        result = self.llm_judge.query_llm(validated["data"])
        evaluation = {"discovery": validated["data"], "llm": result}
        self.accumulating_batch.append(evaluation)
        summary = None
        if terminal:
            summary = self.execute_statistical_check(self.accumulating_batch)
        return evaluation, summary

    def execute_batch_pipeline(self):
        validated, skipped = self.execute_rules_based_checks()
        evaluations = self.execute_llm_judge_analysis(validated)
        summary = self.execute_statistical_check(evaluations)
        return {
            "total_input": len(self.discoveries),
            "validated_count": len(validated),
            "skipped_count": skipped,
            "evaluated_count": len(evaluations),
            "evaluations": evaluations,
            "summary": summary,
        }

    def execute_rules_based_check(self, discovery):
        try:
            data, warnings = self.rule_check.input_completeness_validation(discovery)
        except ValueError as e:
            logger.warning("Skipping invalid discovery: %s", e)
            return None
        if warnings and warnings[0].startswith("SKIPPED"):
            return None
        return {"data": data, "warnings": warnings}

    def execute_rules_based_checks(self):
        validated = []
        skipped = 0
        for discovery in self.discoveries:
            result = self.execute_rules_based_check(discovery)
            if result is None:
                skipped += 1
            else:
                validated.append(result)
        return validated, skipped

    def execute_llm_judge_analysis(self, validated):
        evaluations = []
        for discovery in validated:
            try:
                result = self.llm_judge.query_llm(discovery["data"])
                evaluations.append({
                    "discovery": discovery["data"],
                    "llm": result,
                })
            except (ValueError, Exception) as e:
                logger.error("LLM judge failed for discovery: %s", e)
        return evaluations

    def execute_statistical_check(self, evaluations):
        summary = self.stat_checks.distribution_summary(evaluations)
        summary["outlier_sessions"] = self.stat_checks.outlier_sessions(evaluations)
        return summary