from fastapi import Depends, FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.concurrency import asynccontextmanager, run_in_threadpool # Essential for non-blocking I/O
from pydantic import BaseModel
import mlflow
import mlflow.dspy
import nest_asyncio
import time
from services.dspy_client import DSPYClient
from services.es_client import ESClient
from services.chroma_client import ChromaClient
from services.sandbox_es_client import SandboxESClient
from services.config import settings
from services.judge_dspy import JudgeDSPY
from elasticsearch import helpers

import os
# Suppress GitPython warnings since git isn't installed in the container
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

nest_asyncio.apply()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MLflow with global autologging
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.dspy.autolog(log_traces=True, log_compiles=True) # [2, 3]
    
    es_client = ESClient(
        host=settings.es_host,
        username=settings.es_username,
        password=settings.es_password,
        index=settings.es_index,
        verify_ssl=settings.es_verify_ssl
    )
    chroma_client = ChromaClient(dev=False)
    sandbox_es_client = SandboxESClient()
    dspy_judge = JudgeDSPY(es_client=sandbox_es_client)
    dspy_client = DSPYClient(es_client=es_client, chroma_client=chroma_client, judge_dspy=dspy_judge)
    
    app.state.es_client = es_client
    app.state.dspy_client = dspy_client
    app.state.dspy_judge = dspy_judge
    
    app.state.sandbox_es_client = sandbox_es_client
    
    yield
    dspy_client.close() 

app = FastAPI(title="GDELT Text-to-Query-DSL", lifespan=lifespan)

def get_dspy_client(request: Request) -> DSPYClient:
    return request.app.state.dspy_client

def get_dspy_judge(request: Request) -> JudgeDSPY:
    return request.app.state.dspy_judge

def get_es_client(request: Request) -> ESClient:
    return request.app.state.es_client

def get_sandbox_es_client(request: Request) -> SandboxESClient:
    return request.app.state.sandbox_es_client

def require_dev_mode() -> None:
    if not settings.dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )

app = FastAPI(title="GDELT Text-to-Query-DSL", lifespan=lifespan)

class QueryRequest(BaseModel):
    query_text: str

class QueryResponse(BaseModel):
    query_dsl: dict

def run_mlflow_eval(dspy_client, query_text: str):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_id="0")
    eval_dataset = [{
        "inputs": {"query_text": "Which is the safest country in the Middle East to travel to last week and provide figures?"},
        "expectations": {
            "generated_dsl_query": {
            "size": 0,
            "query": {
                "bool": {
                "must": [
                    {
                    "terms": {
                        "V2Locations.CountryCode.keyword": [
                        "IS",
                        "IZ",
                        "JO",
                        "SA",
                        "AE",
                        "QA",
                        "KW",
                        "OM",
                        "BH",
                        "LB",
                        "SY",
                        "YM",
                        "IR",
                        "TU"
                        ]
                    }
                    },
                    {
                    "range": {
                        "V21Date": {
                        "gte": "now-7d/d",
                        "lte": "now/d"
                        }
                    }
                    }
                ]
                }
            },
            "aggs": {
                "by_country": {
                "terms": {
                    "field": "V2Locations.CountryCode.keyword",
                    "size": 20
                },
                "aggs": {
                    "avg_tone": {
                    "avg": {
                        "field": "V15Tone.Tone"
                    }
                    },
                    "avg_negative_score": {
                    "avg": {
                        "field": "V15Tone.NegativeScore"
                    }
                    },
                    "avg_positive_score": {
                    "avg": {
                        "field": "V15Tone.PositiveScore"
                    }
                    },
                    "conflict_articles": {
                    "filter": {
                        "terms": {
                        "V2EnhancedThemes.V2Theme": [
                            "ARMEDCONFLICT",
                            "CRISISLEX_CRISISLEXREC",
                            "MILITARY"
                        ]
                        }
                    }
                    }
                }
                }
            }
            }
        }
    }]
    try:
        mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=dspy_client.generate_query_dsl,
        )
        print("Background MLflow evaluation completed successfully.")
    except Exception as e:
        print(f"MLflow Background Eval Error: {e}")

@app.post("/generate_query", response_model=QueryResponse, dependencies=[Depends(require_dev_mode)])
async def generate_query(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client) # Removed BackgroundTasks here
):
    query_dsl = await run_in_threadpool(dspy_client.generate_query_dsl, query.query_text)
    
    return QueryResponse(query_dsl=query_dsl)

@app.post("/search")
async def search(query: QueryRequest, dspy_client: DSPYClient = Depends(get_dspy_client), es_client: ESClient = Depends(get_es_client)):
    try:
        # 1. Generate Query DSL (Non-blocking)
        query_dsl = await run_in_threadpool(dspy_client.generate_query_dsl, query.query_text)
        # 2. Execute Search with Error Handling 
        search_results = await run_in_threadpool(es_client.search, query_dsl)
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")

@app.post("/evaluate_query", response_model=dict, dependencies=[Depends(require_dev_mode)])
async def evaluate_query(
    query: QueryResponse,
    dspy_judge: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="evaluate_query"):
        # Offload syntax checking to threadpool
        evaluation_result = await run_in_threadpool(dspy_judge._evaluate_query_dsl_syntax, query.query_dsl)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_metric("is_valid", 1 if evaluation_result.get("is_valid") else 0)
        mlflow.log_param("feedback", evaluation_result.get("feedback", ""))
    return evaluation_result

@app.post("/search_and_aggregate", response_model=dict)
async def search_and_aggregate(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client),
    judge_dspy: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="search_and_aggregate"):
        query_dsl = await run_in_threadpool(dspy_client.generate_query_dsl, query.query_text)
        search_results = await run_in_threadpool(es_client.search, query_dsl)
        docs = search_results.get("hits", {}).get("hits", [])
        
        aggregations = await run_in_threadpool(judge_dspy._aggregate_es_documents, docs)
        
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "executed_query_dsl.json")
        mlflow.log_dict(aggregations, "aggregations.json")
    return aggregations

@app.post("/evaluate_relevance", response_model=dict)
async def evaluate_relevance(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client),
    judge_dspy: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="evaluate_relevance"):
        query_dsl = await run_in_threadpool(dspy_client.generate_query_dsl, query.query_text)
        search_results = await run_in_threadpool(es_client.search, query_dsl)
        docs = search_results.get("hits", {}).get("hits", [])
        
        aggregations = await run_in_threadpool(judge_dspy._aggregate_es_documents, docs)
        relevance_evaluation = await run_in_threadpool(judge_dspy.compute_relevance_score, query.query_text, aggregations)
        
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "executed_query_dsl.json")
        mlflow.log_dict(aggregations, "aggregations.json")
        mlflow.log_metric("relevance_score", relevance_evaluation.get("relevance_score", 0))
    return relevance_evaluation

@app.get("/initialize", dependencies=[Depends(require_dev_mode)])
async def initialize(
    sandbox_es_client: SandboxESClient = Depends(get_sandbox_es_client),
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    dspy_client.startup()
    sample_docs = dspy_client.fetch_samples()
    def push_to_dev_es(sandbox_es_client: SandboxESClient, docs: list[dict]):
        """
        Pushes the sample documents to the sandbox ES instance.
        """
        if not docs:
            return
        actions = [
            {
                "_index": settings.sandbox_es_index,
                "_id": doc.get("_id"),
                "_source": doc.get("_source"),
            }
            for doc in docs
        ]
        
        # --- UPDATE THIS SECTION ---
        from elasticsearch.helpers import BulkIndexError
        
        try:
            success, failed = helpers.bulk(sandbox_es_client.es, actions)
            print(f"Succeeded: {success}, Failed: {failed}")
        except BulkIndexError as e:
            print(f"Failed to index {len(e.errors)} documents.")
            # Print the first error to see exactly what Elasticsearch is complaining about
            print("Reason for first failure:", e.errors[0])

@app.get("/load_example", dependencies=[Depends(require_dev_mode)])
async def load_example(
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    example_query = "Find all events related to natural disasters in 2020."
    query_dsl = dspy_client.generate_query_dsl(example_query)
    return {"query": example_query, "generated_query_dsl": query_dsl}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}