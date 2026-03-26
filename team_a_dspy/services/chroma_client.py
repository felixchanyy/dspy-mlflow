from chromadb import HttpClient, PersistentClient

from services.config import settings


class ChromaClient:
    """
    Client for interacting with ChromaDB vector store.
    This is used to store and retrieve vector embeddings for gdelt metadata fields.
    """
    def __init__(self, dev: bool = False) -> None:
        self.collection_name = settings.chroma_collection_name
        if dev:
            client = PersistentClient(path=settings.chroma_persistent_path)
        else:
            client = HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )

        self.collection = client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, interpreted_fields: list[dict] | dict) -> None:
        if isinstance(interpreted_fields, dict):
            interpreted_fields = [interpreted_fields]

        # Bulk extract into lists for ChromaDB
        ids = [doc["field_name"] for doc in interpreted_fields]
        metadatas = [{
            "field_name": doc["field_name"],
            "field_type": doc["field_type"]
        } for doc in interpreted_fields]
        documents = [str(doc["interpretation"]) for doc in interpreted_fields]

        # Perform a single bulk upsert (much faster and avoids the string-iteration bug)
        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )
    
    def query(self, query_text: str, k: int = 6) -> dict:
        """
        Query the ChromaDB collection for the most relevant interpretations based on the query text.
        Returns a dictionary with the following structure:
        {
            "ids": [list of document ids],
            "documents": [list of document contents],
            "metadatas": [list of document metadatas],
        }
        """
        results = self.collection.query(
            query_texts=query_text,
            n_results=k,
        )
        return results