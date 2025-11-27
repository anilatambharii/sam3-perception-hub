"""
SAM3 Perception API - Main FastAPI Application

Provides REST and gRPC endpoints for:
- Promptable Concept Segmentation (PCS) - text-based segmentation
- Promptable Visual Segmentation (PVS) - click/box/mask prompts
- Video object tracking with stable IDs
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from sam3_perception.api.routes import router as api_router
from sam3_perception.api.websocket import router as ws_router
from sam3_perception.models.sam3_wrapper import SAM3Model
from sam3_perception.utils.config import get_settings
from sam3_perception.utils.logging import setup_logging
from sam3_perception.utils.tracing import setup_tracing

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    logger.info("Starting SAM3 Perception API", version="0.1.0")
    
    # Initialize model
    try:
        app.state.model = SAM3Model(
            model_path=settings.sam3_model_path,
            device=settings.sam3_device,
            inference_provider=settings.inference_provider,
        )
        await app.state.model.load()
        logger.info("SAM3 model loaded successfully", device=settings.sam3_device)
    except Exception as e:
        logger.error("Failed to load SAM3 model", error=str(e))
        raise
    
    # Initialize cache connection
    if settings.redis_url:
        from sam3_perception.utils.cache import init_cache
        app.state.cache = await init_cache(settings.redis_url)
        logger.info("Redis cache connected")
    
    yield
    
    # Cleanup
    logger.info("Shutting down SAM3 Perception API")
    if hasattr(app.state, 'cache'):
        await app.state.cache.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging(settings.log_level)
    
    app = FastAPI(
        title="SAM3 Perception API",
        description="Production-ready perception layer powered by Meta's SAM 3",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup tracing
    if settings.otel_endpoint:
        setup_tracing(app, settings.otel_endpoint)
    
    # Mount Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/ws")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": hasattr(app.state, 'model') and app.state.model.is_loaded,
            "version": "0.1.0"
        }
    
    # Ready check endpoint
    @app.get("/ready")
    async def ready_check():
        if not hasattr(app.state, 'model') or not app.state.model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ready"}
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sam3_perception.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.dev_mode,
    )
