from app.generation.generator import RAGGenerator

class ChatService:
    def __init__(self):
        self.generator = RAGGenerator()
        self.chain = self.generator.get_chain()

    async def chat(self, question: str):
        # 1. Generate Answer (This runs Multi-Query internally)
        response_text = self.chain.invoke(question)
        
        # 2. Retrieve Sources (Explicitly call Multi-Query so UI sees what LLM saw)
        source_docs = self.generator.retrieve_multi_query(question)
        
        # Deduplicate sources for the UI
        unique_sources = list(set([doc.page_content for doc in source_docs]))
        
        return {
            "answer": response_text,
            "sources": unique_sources
        }