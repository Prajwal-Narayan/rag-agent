from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

class QueryTransformer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value() # type: ignore
        )
        
        # Multi-Query Prompt
        self.multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
            
            Original question: {question}"""
        )
        
        self.chain = (
            self.multi_query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

    def generate_queries(self, question: str) -> List[str]:
        """
        Returns a list of 5 variations of the input question.
        """
        print(f"🔄 Generating variations for: {question}")
        return self.chain.invoke({"question": question})