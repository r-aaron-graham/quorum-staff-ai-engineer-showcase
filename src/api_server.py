import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.sem_search import app as sem_app

# Main API server that mounts individual service routers

def create_app() -> FastAPI:
    app = FastAPI(
        title="Quorum Copilot AI Services",
        description="Unified API server for semantic search and summarization",
        version=os.getenv("API_VERSION", "0.1.0"),
    )

    # CORS configuration
    origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount semantic search endpoints
    app.mount("/alpha", sem_app)

    @app.get("/healthz")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=port, reload=True)
