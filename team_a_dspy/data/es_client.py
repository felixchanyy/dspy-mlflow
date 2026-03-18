from typing import Any, Dict
from elasticsearch import AsyncElasticsearch
from config import settings

class ESClient:
    """Thin wrapper around AsyncElasticsearch with response normalisation."""

    def __init__(self) -> None:
        self.client = AsyncElasticsearch(
            hosts=[settings.es_host],
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=settings.es_verify_ssl,
            request_timeout=settings.es_request_timeout_seconds,
            # Added headers to fix the 'media_type_header_exception'
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
    async def ping(self) -> bool:
        return await self.client.ping()

    @staticmethod
    def _to_dict(response: Any) -> Dict[str, Any]:
        return response.body if hasattr(response, "body") else response

    async def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.search(index=index, body=query)
        return self._to_dict(response)

    async def get_index_mapping(self, index: str | None = None) -> Dict[str, Any]:
        response = await self.client.indices.get_mapping(index=index or settings.es_index)
        return self._to_dict(response)

# Create a single global instance
es_client = ESClient()