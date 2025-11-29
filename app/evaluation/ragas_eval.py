from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from app.api.chat_service import ChatService
import os

# Disable Ragas analytics to prevent timeout/errors
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

class RagasEvaluator:
    def __init__(self):
        self.chat_service = ChatService()

    async def run_benchmark(self, test_questions: list[str]):
        """
        Runs the RAG pipeline on a list of questions and evaluates the results.
        """
        answers = []
        contexts = []

        print(f"Starting Benchmark for {len(test_questions)} questions...")

        # 1. Generate Answers using YOUR System
        for q in test_questions:
            result = await self.chat_service.chat(q)
            answers.append(result["answer"])
            contexts.append(result["sources"])

        # 2. Prepare Data for Ragas
        data = {
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
            # We don't have ground_truth for dynamic questions, 
            # so we only test Faithfulness and Relevancy
        }
        dataset = Dataset.from_dict(data)

        # 3. Run Evaluation (The "Judge")
        print("Running Ragas Evaluator...")
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
        )
        
        return results