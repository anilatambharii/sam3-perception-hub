"""Configuration for SAM3 Reconstruction API."""
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    port: int = Field(default=8081)
    grpc_port: int = Field(default=50052)
    dev_mode: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    
    sam3d_objects_path: str = Field(default="/models/sam3d/objects")
    sam3d_body_path: str = Field(default="/models/sam3d/body")
    sam3d_device: str = Field(default="cuda:0")
    
    redis_url: Optional[str] = Field(default="redis://localhost:6379")
    log_level: str = Field(default="INFO")
    otel_endpoint: Optional[str] = Field(default=None)

@lru_cache()
def get_settings() -> Settings:
    return Settings()
