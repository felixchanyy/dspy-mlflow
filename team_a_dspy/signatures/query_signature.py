import dspy

class TextToESQuery(dspy.Signature):
    """Translate a natural language question into a valid Elasticsearch JSON Query DSL for the GDELT GKG index. 
    Review the schema context carefully to choose the correct field names and types.
    For top-N or ranking questions, use aggregations and set the top-level size to 0.
    Return ONLY the raw JSON query object. Do not include markdown formatting, backticks, or explanations in the final output field."""
    
    question = dspy.InputField(desc="The user's natural language question.")
    schema_context = dspy.InputField(desc="Relevant schema fields from the GKG index retrieved from ChromaDB.")
    es_query = dspy.OutputField(desc="The exact Elasticsearch JSON Query DSL. Valid JSON only.")