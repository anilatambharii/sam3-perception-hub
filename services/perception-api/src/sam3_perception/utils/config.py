"""
Configuration management for SAM3 Perception API.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    port: int = Field(default=8080, description="HTTP server port")
    grpc_port: int = Field(default=50051, description="gRPC server port")
    dev_mode: bool = Field(default=False, description="Enable development mode")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    sam3_model_path: str = Field(
        default="/models/sam3",
        description="Path to SAM3 model checkpoints"
    )
    sam3_device: str = Field(
        default="cuda:0",
        description="Device for model inference"
    )
    inference_provider: str = Field(
        default="local",
        description="Inference provider: local, fal, replicate"
    )
    model_dtype: str = Field(
        default="bfloat16",
        description="Model data type for inference"
    )
    
    # ==========================================================================
    # Cloud Provider Keys
    # ==========================================================================
    fal_api_key: Optional[str] = Field(default=None, description="FAL API key")
    replicate_api_token: Optional[str] = Field(default=None, description="Replicate API token")
    
    # ==========================================================================
    # Privacy & Safety
    # ==========================================================================
    enable_face_blur: bool = Field(
        default=True,
        description="Enable automatic face blurring"
    )
    concept_allowlist: List[str] = Field(
        default=[],
        description="Allowed concepts (empty = allow all)"
    )
    concept_denylist: List[str] = Field(
        default=["weapon", "drug", "explosive"],
        description="Blocked concepts"
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    
    # ==========================================================================
    # Performance
    # ==========================================================================
    max_batch_size: int = Field(default=8, description="Maximum batch size")
    frame_sample_rate: int = Field(
        default=1,
        description="Frame sampling rate for video"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Cache concept embeddings"
    )
    max_image_size: int = Field(
        default=4096,
        description="Maximum image dimension"
    )
    default_confidence_threshold: float = Field(
        default=0.5,
        description="Default confidence threshold"
    )
    
    # ==========================================================================
    # Infrastructure
    # ==========================================================================
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    
    # ==========================================================================
    # Observability
    # ==========================================================================
    log_level: str = Field(default="INFO", description="Logging level")
    otel_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry endpoint"
    )
    
    def is_concept_allowed(self, concept: str) -> bool:
        """Check if a concept is allowed based on allow/deny lists."""
        concept_lower = concept.lower()
        
        # Check denylist first
        if any(denied in concept_lower for denied in self.concept_denylist):
            return False
        
        # If allowlist is empty, allow everything not denied
        if not self.concept_allowlist:
            return True
        
        # Check allowlist
        return any(allowed in concept_lower for allowed in self.concept_allowlist)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
