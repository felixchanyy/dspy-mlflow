from __future__ import annotations
from copy import deepcopy
from typing import Any

from services.es_client import ESClient
from services.config import settings

class SandboxESClient(ESClient):
    """
    Client for interacting with the sandbox Elasticsearch instance.
    Includes advanced validation and execution-aware metric support.
    """
    SAFE_MAX_SIZE = 100
    SAFE_MAX_AGG_SIZE = 100

    AGG_FIELD_TYPES = {
        "terms", "avg", "sum", "min", "max", "cardinality", "value_count",
        "date_histogram", "histogram", "stats", "extended_stats", "percentiles"
    }

    BOOL_KEYS = {"must", "should", "filter", "must_not"}

    def __init__(
        self,
        host: str | None = None,
        username: str | None = None,
        password: str | None = None,
        index: str | None = None,
        verify_ssl: bool | None = None,
    ):
        super().__init__(
            host or settings.sandbox_es_host,
            username if username is not None else settings.sandbox_es_username,
            password if password is not None else settings.sandbox_es_password,
            index or settings.sandbox_es_index,
            verify_ssl if verify_ssl is not None else settings.sandbox_es_verify_ssl,
        )
        self._flat_mapping_cache = None

    def get_flat_mapping(self) -> dict[str, str]:
        """Returns the flattened schema mapping required by the optimizer."""
        if self._flat_mapping_cache is None:
            self._flat_mapping_cache = self.flatten_es_mapping()
        return self._flat_mapping_cache

    # -------------------------------------------------------------------------
    # SYNCHRONOUS VALIDATION: Required by JudgeDSPY & standard API routes
    # -------------------------------------------------------------------------
    def validate_query_dsl(self, query_dsl: dict):
        body = query_dsl.get("query_dsl", query_dsl) if isinstance(query_dsl, dict) else {}
        
        try:
            response = self.es.indices.validate_query(
                index=self.index,
                body=body,
                explain=True
            )
            return {
                "is_valid": response.body.get("valid", False),
                "feedback": response.body.get("explanations", response.body.get("error", "No explanation provided"))
            }
        except Exception as e:
            return {
                "is_valid": False,
                "feedback": str(e)
            }

    # -------------------------------------------------------------------------
    # ASYNCHRONOUS EVALUATION: Required by ExecutionAwareESMetric in Optimizer
    # -------------------------------------------------------------------------
    def evaluate_query_dsl(self, query_dsl: dict[str, Any], expected_query_dsl: dict[str, Any] | None = None) -> dict[str, Any]:
        query = query_dsl.get("query_dsl", query_dsl) if isinstance(query_dsl, dict) else {}
        if not query:
            return {"score": 0.0}
        
        valid_fields = set(self.get_flat_mapping().keys())
        referenced_fields = self.extract_referenced_fields(query)
        
        unknown_fields = {field for field in referenced_fields if field not in valid_fields}
        schema_score = 0.0 if unknown_fields else 1.0
        
        execution_score = 0.0
        try:
            # Cap sizes for safety
            safe_query = deepcopy(query)
            if isinstance(safe_query.get("size"), int):
                safe_query["size"] = min(safe_query["size"], self.SAFE_MAX_SIZE)
                
            self.es.search(index=self.index, body=safe_query)
            execution_score = 1.0
        except Exception:
            execution_score = 0.0
            
        exact_match_score = 0.0
        if expected_query_dsl:
            expected = expected_query_dsl.get("query_dsl", expected_query_dsl) if isinstance(expected_query_dsl, dict) else {}
            exact_match_score = 1.0 if expected == query else 0.0
            
        final_score = (0.3 * schema_score) + (0.4 * execution_score) + (0.3 * exact_match_score)
        
        return {
            "score": round(final_score, 4),
            "is_valid": bool(execution_score > 0 and schema_score > 0)
        }

    # -------------------------------------------------------------------------
    # UTILITIES: Used by the optimizer to parse the JSON tree
    # -------------------------------------------------------------------------
    def extract_referenced_fields(self, query_dsl: dict[str, Any]) -> set[str]:
        """Recursively extract all Elasticsearch fields referenced in the query."""
        fields: set[str] = set()

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in self.BOOL_KEYS:
                        visit(value)
                    elif key in {"term", "range", "match", "match_phrase", "wildcard", "prefix", "regexp"} and isinstance(value, dict):
                        for field_name in value.keys():
                            fields.add(field_name)
                            visit(value[field_name])
                    elif key == "terms" and isinstance(value, dict):
                        if "field" in value:
                            fields.add(value["field"])
                        else:
                            for field_name in value.keys():
                                fields.add(field_name)
                    elif key in self.AGG_FIELD_TYPES and isinstance(value, dict):
                        if "field" in value and isinstance(value["field"], str):
                            fields.add(value["field"])
                        visit(value)
                    else:
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(query_dsl)
        return {field for field in fields if isinstance(field, str) and field.strip()}