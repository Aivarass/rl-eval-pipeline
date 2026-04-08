import json
import logging

logger = logging.getLogger(__name__)


class RuleCheck:

    def __init__(self):
        self.seen_sequences = set()

    def input_completeness_validation(self, discovery_json):

        if isinstance(discovery_json, dict):
            data = discovery_json
        else:
            try:
                data = json.loads(discovery_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")

        required = {"discovery_id", "timestamp", "episode", "api_sequence", "final_status", "reward", "state_features"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        if not isinstance(data["discovery_id"], str):
            raise ValueError(f"discovery_id must be string, got {type(data['discovery_id'])}")

        if not isinstance(data["timestamp"], str):
            raise ValueError(f"timestamp must be string, got {type(data['timestamp'])}")

        if not isinstance(data["episode"], int):
            raise ValueError(f"episode must be int, got {type(data['episode'])}")



        if not isinstance(data["api_sequence"], list):
            raise ValueError(f"api_sequence must be list, got {type(data['api_sequence'])}")
        if len(data["api_sequence"]) == 0:
            raise ValueError("api_sequence must not be empty")
        for i, step in enumerate(data["api_sequence"]):
            if not isinstance(step, dict):
                raise ValueError(f"api_sequence[{i}] must be dict, got {type(step)}")
            step_required = {"method", "endpoint", "status"}
            step_missing = step_required - step.keys()
            if step_missing:
                raise ValueError(f"api_sequence[{i}] missing fields: {step_missing}")
            if not isinstance(step["status"], int):
                raise ValueError(f"api_sequence[{i}].status must be int, got {type(step['status'])}")

        if not isinstance(data["final_status"], int):
            raise ValueError(f"final_status must be int, got {type(data['final_status'])}")

        warnings = self.validate_api_sequence(data["api_sequence"])

        if self.is_duplicate(data["api_sequence"]):
            return data, ["SKIPPED: duplicate sequence"]

        if warnings:
            logger.warning("Sequence warnings: %s", warnings)

        last_status = data["api_sequence"][-1]["status"]
        if data["final_status"] != last_status:
            warnings.append(f"final_status {data['final_status']} doesn't match last api_sequence status {last_status}")

        if not isinstance(data["reward"], (int, float)):
            raise ValueError(f"reward must be numeric, got {type(data['reward'])}")

        if not isinstance(data["state_features"], dict):
            raise ValueError(f"state_features must be dict, got {type(data['state_features'])}")
        expected_features = {"hasValidItemId", "hasValidPriceId", "hasValidDiscountId", "hasValidPointsId"}
        feature_missing = expected_features - data["state_features"].keys()
        if feature_missing:
            raise ValueError(f"state_features missing: {feature_missing}")

        return data, warnings

    def validate_api_sequence(self, api_sequence):
        DEPENDENCY_ORDER = {
            "/items": 0,
            "/prices": 1,
            "/discounts": 2,
            "/points": 3,
        }

        created = set()
        warnings = []

        for i, step in enumerate(api_sequence):
            endpoint = step["endpoint"]
            method = step["method"]
            status = step["status"]

            # Check dependency: can't create a child before its parent exists
            if method == "POST" and status == 201:
                required_parents = {k for k, v in DEPENDENCY_ORDER.items()
                                    if v < DEPENDENCY_ORDER.get(endpoint, 0)}
                missing_parents = required_parents - created
                if missing_parents:
                    warnings.append(
                        f"api_sequence[{i}]: {method} {endpoint} created without parents: {missing_parents}"
                    )
                created.add(endpoint)

            # Check: can't DELETE/GET/PUT/PATCH something that was never created
            if method in ("DELETE", "GET", "PUT", "PATCH"):
                if endpoint not in created:
                    warnings.append(
                        f"api_sequence[{i}]: {method} {endpoint} but it was never created"
                    )

            # Check: DELETE removes from created set
            if method == "DELETE" and status in (200, 204):
                created.discard(endpoint)

        return warnings

    def is_duplicate(self, api_sequence):
        # Collapse consecutive repeated calls
        normalised = [api_sequence[0]]
        for step in api_sequence[1:]:
            if (step["method"], step["endpoint"]) != (normalised[-1]["method"], normalised[-1]["endpoint"]):
                normalised.append(step)

        key = tuple(
            (step["method"], step["endpoint"], step["status"])
            for step in normalised
        )
        if key in self.seen_sequences:
            return True
        self.seen_sequences.add(key)
        return False