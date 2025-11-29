from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from typing import List
from app.core.config import settings

class Reranker:
    def __init__(self, base_retriever):
        """
        Wraps a standard retriever with Cohere's Rerank model.
        """
        if not settings.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is missing in .env")
            
        self.compressor = CohereRerank(
            cohere_api_key=settings.COHERE_API_KEY.get_secret_value(), # type: ignore
            top_n=5, # Only keep the top 5 most relevant chunks
            model="rerank-english-v3.0"
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )

    def rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Manually rerank a list of documents.
        """
        # Explicitly convert Sequence to List
        return list(self.compressor.compress_documents(documents, query))