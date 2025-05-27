from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Quorum AI Services",
    description="Semantic search and summarization APIs for legislative data",
    version="0.1.0",
)

# Initialize backends (could be extended with DI or factory patterns)
vector_store = VectorStore(
    backend="qdrant",
    index_name="documents"
)
rag_pipeline = RAGPipeline(
    backend="qdrant",
    vector_index="documents"
)


class SearchResponseItem(BaseModel):
    id: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResponseItem]


class SummarizeRequest(BaseModel):
    query: str
    top_k: Optional[int] = Query(5, ge=1, le=20)
    temperature: Optional[float] = Query(0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Query(512, ge=32, le=2048)


class SummarizeResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResponseItem]


@app.get("/search", response_model=SearchResponse)
async def semantic_search(
    query: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """
    Perform semantic search over ingested legislative data.
    """
    try:
        hits = vector_store.search(query_text=query, top_k=top_k)
        items = [SearchResponseItem(id=int(hit['id']), score=hit['score'], text=hit['text']) for hit in hits]
        return SearchResponse(query=query, results=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generate a summarized answer to the query using RAG.
    """
    try:
        output = rag_pipeline.run(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        # Map source_documents to response items
        sources = [SearchResponseItem(id=int(doc['id']), score=doc.get('score', 0.0), text=doc['text'])
                   for doc in output['source_documents']]
        return SummarizeResponse(
            query=request.query,
            answer=output['answer'],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
