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
        """
        Validates the generated Query DSL by executing it against the sandbox ES instance and checking for errors.
        
        Args:
            query_dsl (dict): The generated Elasticsearch Query DSL to be validated.
        Returns:
            dict: A dictionary containing the validation results, including whether the query is valid and any feedback or error messages from Elasticsearch.
        """
        response = self.es.indices.validate_query(
            index=self.index,
            body=query_dsl.get("query_dsl", {}),
            explain=True
        )

        return {
        "is_valid": response.body.get("valid", False),
        "feedback": response.body.get("explanations", response.body.get("error", "No explanation provided"))
        }
