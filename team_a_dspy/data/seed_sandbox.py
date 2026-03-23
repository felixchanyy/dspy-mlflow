import asyncio
from elasticsearch import Elasticsearch
from config import settings
from data.es_client import es_client

async def seed_sandbox_es():
    print("🔍 Fetching live mapping and sample data from Production Elasticsearch...")
    
    index_name = settings.es_index
    
    # 1. Fetch Mapping and 50 Real Documents from Production
    try:
        # Fetch exact mapping
        prod_mapping_response = await es_client.get_index_mapping(index=index_name)
        prod_mappings = prod_mapping_response.get(index_name, {}).get("mappings", {})
        
        # Fetch 50 real documents
        prod_data_response = await es_client.search(
            index=index_name, 
            query={"size": 50, "query": {"match_all": {}}}
        )
        real_docs = prod_data_response.get("hits", {}).get("hits", [])
        print(f"✅ Fetched mapping and {len(real_docs)} documents from Production.")
        
    except Exception as e:
        print(f"❌ Failed to fetch from Production ES: {e}")
        return

    if not real_docs:
        print("⚠️ No documents found in Production. Sandbox will be empty.")
        return

    # 2. Connect to local Sandbox ES
    print(f"🌱 Connecting to Sandbox Elasticsearch ({settings.sandbox_es_host})...")
    sandbox_es = Elasticsearch(settings.sandbox_es_host)
    
    # 3. Recreate the Sandbox Index using the Production Mapping
    if sandbox_es.indices.exists(index=index_name):
        sandbox_es.indices.delete(index=index_name)
        
    print(f"Creating sandbox index '{index_name}' with production mapping...")
    sandbox_es.indices.create(index=index_name, mappings=prod_mappings)
    
    # 4. Insert the real documents into the Sandbox
    print(f"💾 Inserting {len(real_docs)} real documents into Sandbox...")
    for doc in real_docs:
        source = doc.get("_source", {})
        doc_id = doc.get("_id")
        # Push to sandbox
        sandbox_es.index(index=index_name, id=doc_id, document=source)
        
    # Force a refresh so data is immediately searchable for our evaluation
    sandbox_es.indices.refresh(index=index_name)
    print("✅ Sandbox ES successfully mirrored with real production sample!")

if __name__ == "__main__":
    asyncio.run(seed_sandbox_es())