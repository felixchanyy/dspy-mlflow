import dspy
import json
import asyncio
from elasticsearch import Elasticsearch
from config import settings
from data.init_schema import initialize_schema
from data.seed_sandbox import seed_sandbox_es
from modules.es_query_builder import ESQueryBuilder

lm = dspy.LM(
    model=f"openai/{settings.llm_model_name}",
    api_base=settings.llm_base_url,
    api_key=settings.llm_api_key,
    temperature=0.0,
)
dspy.configure(lm=lm)
sandbox_es = Elasticsearch(settings.sandbox_es_host)

def es_execution_metric(example, pred, trace=None):
    """
    Evaluates the generated Query DSL based on two criteria:
    1. Is it valid JSON?
    2. Does it execute successfully against the Sandbox ES without syntax errors?
    """
    # Criterion 1: Valid JSON Parse
    try:
        # Strip potential markdown formatting (```json ... ```) the LLM might have hallucinated
        clean_json = pred.es_query.replace("```json", "").replace("```", "").strip()
        query_dsl = json.loads(clean_json)
    except Exception as e:
        print(f"❌ Metric Fail: Invalid JSON format. Error: {e}")
        return False 
        
    # Criterion 2: Execution against Sandbox ES
    try:
        res = sandbox_es.search(index=settings.es_index, body=query_dsl)
        
        # Check if the query returned the expected type of result (Hits vs Aggregations)
        if example.expected_type == "hits":
            success = len(res.get('hits', {}).get('hits', [])) >= 0
        elif example.expected_type == "aggs":
            success = "aggregations" in res
        else:
            success = True
            
        return success
        
    except Exception as e:
        # If ES throws a BadRequestError, the LLM hallucinated a field or wrote bad syntax
        print(f"❌ Metric Fail: Elasticsearch Execution Error: {e}")
        return False

# --- 3. DEFINE THE DATASET ---
dataset = [
    dspy.Example(
        question="Find all articles mentioning John Doe.", 
        expected_type="hits"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What are the top 5 most mentioned people?", 
        expected_type="aggs"
    ).with_inputs("question"),
    
    dspy.Example(
        question="How many documents are from the US?", 
        expected_type="hits"
    ).with_inputs("question"),
]

# --- 4. RUN EVALUATION ---
async def run_eval():
    print("⏳ Initializing environment...")
    await initialize_schema()       # Load schema from prod into Chroma
    await seed_sandbox_es()         # <-- ADD AWAIT HERE: Seed local Sandbox with real data
    
    print("\n🚀 Starting Evaluation Pipeline...\n")
    builder = ESQueryBuilder()
    
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=es_execution_metric,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    evaluator(builder)

if __name__ == "__main__":
    asyncio.run(run_eval())