# Ingest Serive

import shutil
import os
from fastapi import UploadFile
from app.ingestion.loader import DataLoader
from app.ingestion.splitter import TextSplitter
from app.retrieval.vector import VectorStore

class IngestService:
    def __init__(self):
        self.splitter = TextSplitter()
        self.vector_store = VectorStore()

    async def process_pdf(self, file: UploadFile):
        # Save file locally temporarily
        temp_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok= True)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            raw_docs = DataLoader.load_pdf(temp_path)
            chunks = self.splitter.split_documents(raw_docs)
            self.vector_store.add_documents(documents= chunks) 

            return {
                "status":"success",
                "filename":file.filename,
                "chunks_processed":len(chunks)
            }
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)