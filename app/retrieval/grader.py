from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.core.config import settings
from typing import cast

class GradeResult(BaseModel):
    """Binary score for relevance check."""
    score: str = Field(description="yes or no")

class DocumentGrader:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value() #type: ignore
        )
        self.structured_llm = self.llm.with_structured_output(GradeResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def grade_document(self, question: str, document: str) -> str:
        """
        Returns 'yes' if relevant, 'no' if not.
        """
        result = cast(GradeResult, self.chain.invoke({"question": question, "document": document}))
        return result.score