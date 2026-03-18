import time
import dspy
import asyncio
from config import settings
from data.init_schema import initialize_schema

# Import our new dedicated module
from modules.es_query_builder import ESQueryBuilder

# --- 1. SEED THE SCHEMA DATABASE ---
async def boot_sequence():
    print("Booting up DSPy environment...")
    for _ in range(5):
        try:
            await initialize_schema()
            return True
        except Exception as e:
            print(f"Waiting for ChromaDB/Elasticsearch to be ready... ({e})")
            await asyncio.sleep(3) 
    return False

if not asyncio.run(boot_sequence()):
    print("❌ Could not initialize schema. Exiting.")
    exit(1)
# -----------------------------------

# 2. Configure DSPy to use the local LLM
lm = dspy.LM(
    model=f"openai/{settings.llm_model_name}",
    api_base=settings.llm_base_url,
    api_key=settings.llm_api_key,
    temperature=0.0,
)
dspy.configure(lm=lm)


# 3. CLI Interactive Loop
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 DSPy Chain-of-Thought Query DSL (Team A)")
    print("="*60)
    
    # Instantiate the new module
    generator = ESQueryBuilder()
    
    while True:
        try:
            q = input("\n📝 Ask a question (or type 'exit'): ")
            if q.lower() in ['exit', 'quit']:
                break
            if not q.strip():
                continue
                
            print("\n🔄 Generating Elasticsearch JSON Query...")
            result = generator(q)
            
            print("\n" + "-"*40)
            print("🧠 LLM CHAIN OF THOUGHT (Reasoning):")
            print("-" * 40)
            print(result.reasoning)  
            
            print("\n" + "-"*40)
            print("⚙️  GENERATED ES JSON QUERY:")
            print("-" * 40)
            print(result.es_query)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")