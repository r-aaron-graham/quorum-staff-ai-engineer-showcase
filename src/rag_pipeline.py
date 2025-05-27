import os
from typing import Optional, Dict, Any

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.vector_store import VectorStore


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.

    This pipeline:
    1. Encodes the user query
    2. Retrieves relevant document chunks from a vector store
    3. Generates an answer using an LLM with retrieved context
    """

    def __init__(
        self,
        backend: str = "qdrant",
        vector_index: str = "documents",
        llm_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
        **vector_store_kwargs: Any,
    ):
        # Initialize vector store abstraction
        self.vector_store = VectorStore(
            backend=backend,
            index_name=vector_index,
            **vector_store_kwargs,
        )

        # Initialize LLM (OpenAI)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(
            model_name=llm_model,
            openai_api_key=api_key,
        )

        # Construct RetrievalQA chain from LangChain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # or 'map_reduce', 'refine'
            retriever=self.vector_store,  # our VectorStore implements search()
            return_source_documents=True,
        )

        # Custom prompt to guide generation style
        self.chain.combine_documents_chain.prompt = PromptTemplate(
            template=(
                "You are an expert legislative AI assistant. "
                "Use the following context excerpts to answer the query as accurately as possible.\n"
                "{context}\n"  # retrieved chunks
                "Question: {question}\n"
                "Answer:"),
            input_variables=["context", "question"],
        )

    def run(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Execute the RAG pipeline on a user query.

        Args:
            query (str): The user’s natural-language question.
            top_k (int): Number of documents to retrieve.
            temperature (float): LLM sampling temperature.
            max_tokens (int): Maximum tokens for LLM response.

        Returns:
            Dict[str, Any]: {
                'answer': generated answer,
                'source_documents': [List of retrieved docs],
                'llm_output': raw LLM response metadata
            }
        """
        # Temporary override LLM params
        self.chain.llm.temperature = temperature  # type: ignore
        self.chain.llm.max_tokens = max_tokens  # type: ignore

        # Retrieve and generate
        result = self.chain(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        return {
            "answer": result["result"],
            "source_documents": [
                {"id": doc.metadata.get("id"), "text": doc.page_content}
                for doc in result.get("source_documents", [])
            ],
            "llm_output": result.get("llm_output"),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG pipeline demo.")
    parser.add_argument("--query", type=str, required=True, help="Query string.")
    parser.add_argument("--backend", type=str, default="qdrant", help="Vector store backend.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of docs to retrieve.")
    args = parser.parse_args()

    pipeline = RAGPipeline(backend=args.backend)
    output = pipeline.run(query=args.query, top_k=args.top_k)

    print("Answer:\n", output["answer"])
    print("\nSources:\n", output["source_documents"])
