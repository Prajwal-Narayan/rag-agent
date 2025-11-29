from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.schemas.models import RouteQuery
from typing import cast 

class IntentRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value() # type: ignore
        )
        
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at routing a user question to a vectorstore or general chat."),
            ("human", "{question}"),
        ])
        
        self.router_chain = self.prompt | self.structured_llm

    def route(self, question: str) -> str:
        """
        Returns 'vector_store' or 'general_chat'
        """
        print(f"Routing Question: {question}")
        
        # Explicitly cast the result 
        result = cast(RouteQuery, self.router_chain.invoke({"question": question}))
        
        return result.datasource