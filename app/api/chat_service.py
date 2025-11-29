from app.generation.generator import RAGGenerator
from app.retrieval.router import IntentRouter
from langchain_openai import ChatOpenAI
from app.core.config import settings

class ChatService:
    def __init__(self):
        self.generator = RAGGenerator()
        self.chain = self.generator.get_chain()
        self.router = IntentRouter()
        
        # Simple LLM for general chat (Cheaper/Faster than RAG)
        self.general_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value() # type: ignore
        )

    async def chat(self, question: str):
        # 1. Route the Request
        route = self.router.route(question)
        
        # 2. Handle General Chat (Fast Path)
        if route == "general_chat":
            print(f"Destination: general_chat (Skipping Search)")
            response = self.general_llm.invoke(question)
            return {
                "answer": response.content,
                "sources": ["General Conversation (No Search Performed)"]
            }

        # 3. Handle RAG (Deep Path)
        print(f"Destination: vector_store (Searching DB)")
        
        # Generate Answer (Uses Multi-Query internally)
        response_text = self.chain.invoke(question)
        
        # Retrieve Sources for Transparency
        source_docs = self.generator.retrieve_multi_query(question)
        unique_sources = list(set([doc.page_content for doc in source_docs]))
        
        return {
            "answer": response_text,
            "sources": unique_sources
        }