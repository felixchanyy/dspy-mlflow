import re
import json
import dspy

from services.chroma_client import ChromaClient
from services.judge_dspy import JudgeDSPY
from signatures.schema_interpreter import SchemaRetriever

class NLToQueryDSL(dspy.Module):
    def __init__(self, chroma_client: ChromaClient, dspy_judge: JudgeDSPY):
        super().__init__()
        self.chroma_client = chroma_client
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)
        self.schema_retriever = SchemaRetriever(chroma_client=chroma_client)
        self.dspy_judge = dspy_judge
        self.refiner = dspy.ChainOfThought(RefinerSignature)

    def forward(self, nl_query: str):
        schema_prediction = self.schema_retriever(nl_query=nl_query)
        es_schema = schema_prediction.schema

        generated_query = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        
        # Helper function to strip markdown and parse JSON
        def extract_json(text: str) -> dict:
            if not text: return {}
            # Strip out markdown formatting if the LLM includes it
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            try:
                return json.loads(text.strip())
            except Exception:
                return {"error": "Failed to parse JSON", "raw_output": text}

        # Parse the string into a dictionary manually
        current_query_dsl = extract_json(generated_query.query_dsl)

        # 2. Fix the logic bug: move the evaluation INSIDE the loop
        for attempt in range(3):
            response = self.dspy_judge.evaluate_query_dsl(current_query_dsl)
            print(f"Validation attempt {attempt+1}: is_valid={response['is_valid']}, feedback={response['feedback']}")
            
            if response['is_valid']:
                # 3. Wrap the return value
                return dspy.Prediction(query_dsl=current_query_dsl)
                
            refined_query = self.refiner(
                nl_query=nl_query, 
                es_schema=es_schema, 
                failed_query_dsl=str(current_query_dsl), 
                feedback=str(response['feedback'])
            )
            current_query_dsl = refined_query.query_dsl
        
        # 3. Wrap the return value
        return dspy.Prediction(query_dsl=current_query_dsl)
    
class NLToQuerySignature(dspy.Signature):
    """
    Convert a natural language query into an Elasticsearch Query DSL format.
    This module takes a natural language query as input and generates a corresponding Elasticsearch Query DSL JSON object.
    Use the ES schema of the GDELT index to inform the generation of the Query DSL, ensuring that field names and types are correctly referenced in the output.
    If the natural language query references requirements that cannot be fulfilled based on the ES schema, rely on the existing fields in the schema to generate the most relevant Query DSL possible, and do not include any fields that are not present in the ES schema.
    Do not include any fields in the output that are not present in the ES schema. 
    Do not use any scripts.
    If feedback is provided from the judge on a previous iteration of the generated Query DSL, use that feedback to refine and improve the Query DSL in subsequent iterations.
    The output should be a valid Elasticsearch Query DSL that can be directly used to query the GDELT index.
    """
    nl_query: str = dspy.InputField(desc="A natural language query describing the search criteria.")
    es_schema: str = dspy.InputField(desc="The Elasticsearch schema...")
    prev_query: dict = dspy.InputField(desc="The previously generated Query DSL, if this is a refinement iteration. For the initial generation, this will be empty or null.", required=False)
    feedback: str = dspy.InputField(desc="Feedback from the judge on the previous Query DSL, if this is a refinement iteration. For the initial generation, this will be empty or null.", required=False)

    query_dsl: str = dspy.OutputField(
        desc="A JSON object representing the Elasticsearch Query DSL. Output ONLY raw JSON, without any markdown formatting or backticks."
    )

class RefinerSignature(dspy.Signature):
    """
    Act as an Elasticsearch Expert.
    Analyze the provided Query DSL and the resulting error message.
    Fix the DSL to make it valid and semantically correct.
    """
    nl_query = dspy.InputField(desc="The original natural language query.")
    es_schema = dspy.InputField(desc="The Elasticsearch schema of the GDELT index, including field names, types and descriptions.")
    failed_query_dsl = dspy.InputField(desc="The failed Elasticsearch Query DSL.")
    feedback = dspy.InputField(desc="The feedback returned from Elasticsearch when the failed_query_dsl was executed.")

    query_dsl = dspy.OutputField(desc="A refined version of the Query DSL that addresses the issues identified in the feedback and is expected to be valid and semantically correct.")