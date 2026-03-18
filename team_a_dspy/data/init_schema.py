import hashlib
import chromadb
import asyncio
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from data.es_client import es_client

def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()

def _mapping_to_chunks(mapping_response: dict, index_name: str) -> list:
    # flattening function to convert ES mapping to a list of chunks with metadata
    index_payload = mapping_response.get(index_name)
    if index_payload is None and mapping_response:
        index_payload = next(iter(mapping_response.values()))
    if not isinstance(index_payload, dict):
        return []

    mappings = index_payload.get("mappings", {})
    properties = mappings.get("properties", {})
    chunks = []

    chunks.append({
        "id": _stable_id("overview"),
        "document": f"Index {index_name} schema overview. Use only fields that appear in this schema. For terms aggregations prefer keyword fields or keyword subfields when available.",
        "metadata": {"kind": "overview", "index": index_name}
    })

    def walk(node: dict, prefix: str = ""):
        for field_name, spec in node.items():
            if not isinstance(spec, dict): continue
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            field_type = spec.get("type") or ("object" if "properties" in spec else "unknown")
            subfields = spec.get("fields", {}) if isinstance(spec.get("fields"), dict) else {}
            
            parts = [f"Index: {index_name}", f"Field: {full_name}", f"Type: {field_type}"]
            
            chunks.append({
                "id": _stable_id(full_name),
                "document": ". ".join(parts),
                "metadata": {"kind": "field", "field": full_name, "type": field_type, "index": index_name}
            })
            
            for subfield_name, sub_spec in subfields.items():
                sub_type = sub_spec.get("type", "unknown") if isinstance(sub_spec, dict) else "unknown"
                subfield_path = f"{full_name}.{subfield_name}"
                chunks.append({
                    "id": _stable_id(subfield_path),
                    "document": f"Index: {index_name}. Field: {subfield_path}. Type: {sub_type}. Usage: Use this field for exact matches, filters, sorting, and terms aggregations.",
                    "metadata": {"kind": "subfield", "field": subfield_path, "type": sub_type, "index": index_name}
                })

            child_properties = spec.get("properties")
            if isinstance(child_properties, dict): walk(child_properties, full_name)

    if isinstance(properties, dict): walk(properties)
    return chunks

async def initialize_schema():
    print("⏳ Connecting to local ChromaDB...")
    
    client = chromadb.HttpClient(
        host=settings.chroma_host, 
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    collection = client.get_or_create_collection(name=settings.chroma_collection)
    
    if collection.count() > 0:
        print(f"✅ Schema already exists in local ChromaDB ({collection.count()} chunks).")
        return

    print(f"🔍 Fetching live mapping from Elasticsearch using ESClient ({settings.es_host})...")
    
    try:
        mapping_response = await es_client.get_index_mapping(index=settings.es_index)
    except Exception as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")
        raise e

    chunks = _mapping_to_chunks(mapping_response, settings.es_index)
    if not chunks: return

    print(f"🧠 Embedding {len(chunks)} schema fields (This may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.schema_embedding_model)
    
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["document"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings.embed_documents([chunk["document"] for chunk in chunks])
    )
    print("✅ Successfully synchronized live Elasticsearch mapping to local ChromaDB!")

if __name__ == "__main__":
    asyncio.run(initialize_schema())