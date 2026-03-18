import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

class SchemaRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.schema_embedding_model)
        
        self.client = chromadb.HttpClient(
            host=settings.chroma_host, 
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name=settings.chroma_collection)
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB collection... Error: {e}")
            self.collection = None

    def search(self, question: str, k: int = 8) -> str:
        if not self.collection:
            return "(Schema retrieval unavailable)"
            
        query_embedding = self.embeddings.embed_query(question)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents"]
        )
        
        docs = result.get("documents", [[]])[0]
        return "\n".join(docs) if docs else "(No schema context found)"