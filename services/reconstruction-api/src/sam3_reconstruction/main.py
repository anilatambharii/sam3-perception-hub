"""
SAM3 Reconstruction API - Main FastAPI Application

Provides REST and gRPC endpoints for:
- 3D Object Reconstruction (SAM 3D Objects)
- 3D Body Estimation (SAM 3D Body)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from sam3_reconstruction.api.routes import router as api_router
from sam3_reconstruction.models.sam3d_wrapper import SAM3DModel
from sam3_reconstruction.utils.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting SAM3 Reconstruction API", version="0.1.0")
    
    # Initialize models
    try:
        app.state.object_model = SAM3DModel(
            model_type="objects",
            model_path=settings.sam3d_objects_path,
            device=settings.sam3d_device,
        )
        await app.state.object_model.load()
        
        app.state.body_model = SAM3DModel(
            model_type="body",
            model_path=settings.sam3d_body_path,
            device=settings.sam3d_device,
        )
        await app.state.body_model.load()
        
        logger.info("SAM3D models loaded successfully")
    except Exception as e:
        logger.error("Failed to load SAM3D models", error=str(e))
        raise
    
    # Initialize cache
    if settings.redis_url:
        from sam3_reconstruction.utils.cache import init_cache
        app.state.cache = await init_cache(settings.redis_url)
    
    yield
    
    logger.info("Shutting down SAM3 Reconstruction API")
    if hasattr(app.state, 'cache'):
        await app.state.cache.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from sam3_reconstruction.utils.logging import setup_logging
    setup_logging(settings.log_level)
    
    app = FastAPI(
        title="SAM3 Reconstruction API",
        description="3D reconstruction powered by Meta's SAM 3D",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Routes
    app.include_router(api_router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "object_model_loaded": hasattr(app.state, 'object_model') and app.state.object_model.is_loaded,
            "body_model_loaded": hasattr(app.state, 'body_model') and app.state.body_model.is_loaded,
            "version": "0.1.0"
        }
    
    @app.get("/ready")
    async def ready_check():
        if not hasattr(app.state, 'object_model') or not app.state.object_model.is_loaded:
            raise HTTPException(status_code=503, detail="Object model not ready")
        if not hasattr(app.state, 'body_model') or not app.state.body_model.is_loaded:
            raise HTTPException(status_code=503, detail="Body model not ready")
        return {"status": "ready"}
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sam3_reconstruction.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.dev_mode,
    )
