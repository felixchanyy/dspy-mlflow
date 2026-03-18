import dspy
from data.schema_retriever import SchemaRetriever
from signatures.query_signature import TextToESQuery

class ESQueryBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = SchemaRetriever()
        self.generate_query = dspy.ChainOfThought(TextToESQuery)

    def forward(self, question):
        schema_context = self.retriever.search(question)
        prediction = self.generate_query(
            question=question, 
            schema_context=schema_context
        )
        
        # Safely fetch the thought process (DSPy 2.5+ uses 'reasoning', older used 'rationale')
        thought_process = getattr(prediction, 'reasoning', getattr(prediction, 'rationale', 'No reasoning generated.'))
        
        return dspy.Prediction(
            reasoning=thought_process, # <-- Updated key
            es_query=prediction.es_query,
            schema_context=schema_context
        )