# Data loading

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from typing import List

class DataLoader:
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_web(url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()
