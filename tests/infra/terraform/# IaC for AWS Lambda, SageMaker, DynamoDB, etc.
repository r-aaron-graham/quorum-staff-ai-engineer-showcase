import pytest

from src.rag_pipeline import RAGPipeline


class DummyDoc:
    def __init__(self, content, id):
        self.page_content = content
        self.metadata = {"id": id}


class DummyChain:
    def __init__(self):
        # Mimic LLM params storage
        self.llm = type("LLM", (), {"temperature": None, "max_tokens": None})()

    def __call__(self, inputs):
        # Return a fixed result structure
        docs = [DummyDoc("chunk1", 1), DummyDoc("chunk2", 2)]
        return {
            "result": "test answer",
            "source_documents": docs,
            "llm_output": {"mocked": True},
        }


def test_rag_pipeline_run(monkeypatch):
    # Monkeypatch VectorStore and OpenAI to avoid real external calls
    monkeypatch.setattr(
        "src.rag_pipeline.VectorStore", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.rag_pipeline.OpenAI", lambda *args, **kwargs: None
    )

    # Instantiate pipeline and replace the chain with our dummy
    pipeline = RAGPipeline()
    pipeline.chain = DummyChain()

    # Execute the pipeline
    output = pipeline.run(query="dummy?", top_k=2, temperature=0.7, max_tokens=128)

    # Validate results
    assert output["answer"] == "test answer"
    assert isinstance(output["source_documents"], list)
    assert len(output["source_documents"]) == 2
    assert output["source_documents"][0]["id"] == 1
    assert output["source_documents"][0]["text"] == "chunk1"
    assert output["llm_output"] == {"mocked": True}
