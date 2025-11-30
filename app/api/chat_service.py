from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Import our Tools
from app.generation.generator import RAGGenerator
from app.retrieval.router import IntentRouter
from app.retrieval.grader import DocumentGrader
from app.core.config import settings 
from app.schemas.state import GraphState
from langchain_openai import ChatOpenAI
import app.core.database as db_module
from langchain_core.messages import HumanMessage

class ChatService:
    def __init__(self):
        # Initialize Tools
        self.generator = RAGGenerator()
        self.chain = self.generator.get_chain()
        self.router = IntentRouter()
        self.grader = DocumentGrader()
        self.general_llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            api_key=settings.OPENAI_API_KEY.get_secret_value() #type:ignore
        )
        
        # Build the Graph once on initialization
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph State Machine."""
        workflow = StateGraph(GraphState)

        # 1. Define Nodes
        workflow.add_node("router_node", self.route_query)
        workflow.add_node("general_chat_node", self.run_general_chat)
        workflow.add_node("retrieve_node", self.retrieve)
        workflow.add_node("grade_node", self.grade_documents)
        workflow.add_node("generate_node", self.generate)

        # 2. Define Entry Point
        workflow.set_entry_point("router_node")

        # 3. Define Conditional Edges (The Logic)
        workflow.add_conditional_edges(
            "router_node",
            self.decide_route,
            {
                "general_chat": "general_chat_node",
                "vector_store": "retrieve_node"
            }
        )

        # 4. Define Standard Edges (The Flow)
        workflow.add_edge("general_chat_node", END)
        workflow.add_edge("retrieve_node", "grade_node")
        workflow.add_edge("grade_node", "generate_node") # V1 Loop (Simple)
        workflow.add_edge("generate_node", END)

        # 5. Compile with Memory

        
        # 👇 FIX 1: Check if checkpointer is ready (Runtime check)
        if db_module.checkpointer is None:
            raise RuntimeError("Database checkpointer is not initialized. Did init_db run?")

        # 👇 FIX 2: Use db_module.checkpointer
        return workflow.compile(checkpointer=db_module.checkpointer)

    # --- NODE FUNCTIONS ---

    def route_query(self, state: GraphState):
        """Node: Analyzing user intent..."""
        question = state["question"]
        return {"question": question}

    def decide_route(self, state: GraphState) -> Literal["general_chat", "vector_store"]:
        """Edge Logic: Where do we go next?"""
        question = state["question"]
        route = self.router.route(question)
        return route #type:ignore

    async def run_general_chat(self, state: GraphState):
        """Node: Simple LLM chat with Memory."""
        # 1. Get the full history from State
        messages = state["messages"]
        # 2. Invoke LLM with the WHOLE history (Context)
        response = self.general_llm.invoke(messages)
        # 3. Return the AI's response (LangGraph adds this to history automatically)
        return {"generation": response.content, "messages": [response]}

    def retrieve(self, state: GraphState):
        """Node: Multi-Query Search + Rerank."""
        question = state["question"]
        documents = self.generator.retrieve_multi_query(question)
        # Convert objects to strings for the State
        doc_texts = [doc.page_content for doc in documents]
        return {"documents": doc_texts}

    def grade_documents(self, state: GraphState):
        """Node: Filtering irrelevant docs."""
        question = state["question"]
        documents = state["documents"]
        
        relevant_docs = []
        for doc in documents:
            score = self.grader.grade_document(question, doc)
            if score == "yes":
                relevant_docs.append(doc)
        
        # If no docs are relevant, we could loop back here (CRAG). 
        # For now, we pass empty list to Generator which handles fallback.
        return {"documents": relevant_docs}

    async def generate(self, state: GraphState):
        """Node: Generating final answer."""
        question = state["question"]
        documents = state["documents"]
        
        # If we have docs, use RAG chain. Else, say "I don't know".
        if documents:
            context = "\n\n".join(documents)
            # Use the chain we built in Phase 3
            # We bypass the retriever step in the chain since we already have docs
            chain_input = {"context": context, "question": question}
            response = self.chain.invoke(chain_input) # This might crash if chain expects retriever.
            # FIX: We invoke LLM directly for simplicity here
            rag_prompt = self.generator.prompt.format(context=context, question=question)
            response = self.generator.llm.invoke(rag_prompt).content
        else:
            response = "I couldn't find relevant documents to answer your question."

        return {"generation": response}

    # --- PUBLIC API ---

    async def chat(self, question: str, thread_id: str = "default_user"):
        config = {"configurable": {"thread_id": thread_id}}
        
        # 👇 FIX: Add the new message to the history list explicitly
        inputs = {
            "question": question,
            "messages": [HumanMessage(content=question)] 
        }
        
        result = await self.graph.ainvoke(inputs, config=config) #type:ignore
        
        return {
            "answer": result["generation"],
            "sources": result.get("documents", [])
        }