import argparse
import json
from pathlib import Path
import sys

import dspy

# Allow running this file directly from anywhere in the repo.
THIS_FILE = Path(__file__).resolve()
TEAM_A_ROOT = THIS_FILE.parents[1]
if str(TEAM_A_ROOT) not in sys.path:
    sys.path.insert(0, str(TEAM_A_ROOT))

from services.chroma_client import ChromaClient
from services.config import settings
from signatures.es_query_generator import NLToQuerySignature
from signatures.schema_interpreter import SchemaRetriever


class OptimizableNLToQueryDSL(dspy.Module):
    """
    Lightweight optimization target that focuses on first-pass DSL generation.
    The production module can keep validation/refinement logic; this module is
    only for prompt optimization.
    """

    def __init__(self, chroma_client: ChromaClient):
        super().__init__()
        self.schema_retriever = SchemaRetriever(chroma_client=chroma_client)
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)

    def forward(self, nl_query: str):
        es_schema = self.schema_retriever(nl_query=nl_query)
        generated_query = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        return dspy.Prediction(query_dsl=generated_query.query_dsl)


def load_jsonl_examples(path: Path) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                dspy.Example(
                    nl_query=row["nl_query"],
                    expected_query_dsl=row["expected_query_dsl"],
                ).with_inputs("nl_query")
            )
    return examples


def normalize_query_dsl(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    return payload.get("query_dsl", payload)


def metric_exact_query_dsl(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    expected = normalize_query_dsl(example.expected_query_dsl)
    predicted = normalize_query_dsl(pred.query_dsl)
    return 1.0 if expected == predicted else 0.0


def configure_lm() -> None:
    lm = dspy.LM(
        base_url=settings.llm_base_url,
        model=f"openai/{settings.llm_model_name}",
        api_key=settings.llm_api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset", default=str(TEAM_A_ROOT / "data/optimizer_trainset.jsonl"))
    parser.add_argument("--output", default=str(TEAM_A_ROOT / "optimizers/artifacts/optimized_query_generator.json"))
    args = parser.parse_args()

    trainset_path = Path(args.trainset).resolve()
    output_path = Path(args.output).resolve()

    if not trainset_path.exists():
        raise FileNotFoundError(f"Trainset not found: {trainset_path}")

    configure_lm()

    student = OptimizableNLToQueryDSL(
        chroma_client=ChromaClient(dev=settings.dev)
    )

    print(f"Loading dataset from {trainset_path}...")
    dataset = load_jsonl_examples(trainset_path)

    if not dataset:
        raise ValueError("Trainset is empty")

    trainset, devset = dataset[:30], dataset[30:]
    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    optimized = dspy.BootstrapFewShot(metric=metric_exact_query_dsl)\
                   .compile(student=student, trainset=trainset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(output_path))

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
