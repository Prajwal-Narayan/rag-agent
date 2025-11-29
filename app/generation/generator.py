from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings
from app.retrieval.vector import VectorStore
from app.retrieval.query_transformer import QueryTransformer
from app.retrieval.reranker import Reranker

class RAGGenerator:
    def __init__(self): 
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value() # type: ignore
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
        # Raw retriever
        self.base_retriever = self.vector_store.as_retriever()
        
        self.transformer = QueryTransformer()
        # 👇 Initialize Reranker
        self.reranker = Reranker(self.base_retriever)

    def format_docs(self, docs):
        # Flatten and deduplicate
        unique_docs = {}
        for doc in docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        return "\n\n".join(doc.page_content for doc in unique_docs.values())

    def retrieve_multi_query(self, question: str):
        """
        1. Generate 5 queries
        2. Retrieve docs for ALL 5 (Broad Recall)
        3. Rerank them (High Precision)
        """
        # Step 1: Broad Search
        queries = self.transformer.generate_queries(question)
        all_docs = []
        for q in queries:
            docs = self.base_retriever.invoke(q)
            all_docs.extend(docs)
        
        # Step 2: Deduplicate BEFORE Reranking (Save API costs)
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        deduped_docs = list(unique_docs.values())
        
        # Step 3: Rerank (The Filter)
        if not deduped_docs:
            return []
            
        print(f"Reranking {len(deduped_docs)} documents...")
        ranked_docs = self.reranker.rerank_documents(deduped_docs, question)
        print(f"Kept top {len(ranked_docs)} relevant docs.")
        
        return ranked_docs

    def get_chain(self):
        rag_chain = (
            {"context": lambda x: self.format_docs(self.retrieve_multi_query(x)), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain