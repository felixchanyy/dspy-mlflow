from __future__ import annotations
from typing import Any
import dspy

try:
    from team_a_dspy.services.sandbox_es_client import SandboxESClient
except ModuleNotFoundError:
    from services.sandbox_es_client import SandboxESClient

def normalize_query_dsl(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("query_dsl")
    if isinstance(nested, dict):
        return nested
    return payload

# Fix: Deleted the _run_async helper method

class ExecutionAwareESMetric:
    """
    Stronger optimization metric:
    - executes query
    - validates schema
    - scores semantic closeness to expected query
    """

    def __init__(self, sandbox_client: SandboxESClient):
        self.sandbox_client = sandbox_client

    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
        candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))

        if not candidate:
            return 0.0

        # Fix: Call the evaluation directly (synchronously) without asyncio.run()
        result = self.sandbox_client.evaluate_query_dsl(
            query_dsl=candidate,
            expected_query_dsl=gold,
        )

        return float(result.get("score", 0.0))

def metric_exact_query_dsl(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
    candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))
    return 1.0 if gold == candidate else 0.0