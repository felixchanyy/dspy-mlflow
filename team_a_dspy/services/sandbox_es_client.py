from services.es_client import ESClient
from services.config import settings

class SandboxESClient(ESClient):
    """
    Client for interacting with the sandbox Elasticsearch instance.
    This is a subclass of ESClient configured to connect to the sandbox ES instance using settings from the config.
    """
    def __init__(self):
        super().__init__(settings.sandbox_es_host, settings.sandbox_es_username, settings.sandbox_es_password, settings.sandbox_es_index, settings.sandbox_es_verify_ssl)

    def validate_query_dsl(self, query_dsl: dict):
        # Safely extract the query body whether it's wrapped or not
        body = query_dsl.get("query_dsl", query_dsl) if isinstance(query_dsl, dict) else {}
        
        response = self.es.indices.validate_query(
            index=self.index,
            body=body,
            explain=True
        )

        return {
            "is_valid": response.body.get("valid", False),
            "feedback": response.body.get("explanations", response.body.get("error", "No explanation provided"))
        }