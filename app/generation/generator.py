from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings
from app.retrieval.vector import VectorStore
from app.retrieval.query_transformer import QueryTransformer

class RAGGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Question: {question} 

            Context: {context} 

            Answer:"""
        )
        
        self.vector_store = VectorStore()
        # Restore the public retriever attribute so Service doesn't crash
        self.retriever = self.vector_store.as_retriever() 
        self.transformer = QueryTransformer()

    def format_docs(self, docs):
        # Flatten and deduplicate
        unique_docs = {}
        for doc in docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        return "\n\n".join(doc.page_content for doc in unique_docs.values())

    def retrieve_multi_query(self, question: str):
        """
        Public method to get docs using Multi-Query strategy.
        """
        queries = self.transformer.generate_queries(question)
        all_docs = []
        for q in queries:
            docs = self.retriever.invoke(q)
            all_docs.extend(docs)
        return all_docs

    def get_chain(self):
        rag_chain = (
            # Use the multi-query retriever in the chain
            {"context": lambda x: self.format_docs(self.retrieve_multi_query(x)), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain