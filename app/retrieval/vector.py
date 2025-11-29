from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class VectorStore:
    def __init__(self):
        # Initialize Embeddings
        self.embedding_fn = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model = settings.EMBEDDING_MODEL
        )

        # Initialize Vector DB
        self.vector_db = Chroma(
            persist_directory= settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embedding_fn,
            collection_name= settings.COLLECTION_NAME
        )

    def add_documents(self, documents):
        """
        Ingest documents into vector store
        """
        if not documents:
            print("No documents to add!")
            return
        
        self.vector_db.add_documents(documents)
        print(f"Added {len(documents)} documents to ChromaDB collection: {settings.COLLECTION_NAME}")

    def as_retriever(self, search_kwargs: dict = {"k":5}):
        """
        Returns the DB as Langchain Retriever
        """
        return self.vector_db.as_retriever(search_kwargs = search_kwargs)
