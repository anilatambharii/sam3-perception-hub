"""
Redis cache utilities for embedding and result caching.
"""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class Cache:
    """Redis-based cache for embeddings and results."""
    
    def __init__(self, client: redis.Redis):
        self.client = client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in cache."""
        try:
            await self.client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False
    
    async def close(self) -> None:
        """Close the cache connection."""
        await self.client.close()
    
    @staticmethod
    def make_key(prefix: str, *args: Any) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create hash of arguments for consistent key length
        arg_str = json.dumps(args, sort_keys=True)
        arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()[:16]
        return f"{prefix}:{arg_hash}"


class EmbeddingCache:
    """Specialized cache for concept embeddings."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
        self.prefix = "embedding"
    
    async def get_embedding(self, concept: str) -> Optional[list]:
        """Get cached embedding for a concept."""
        key = Cache.make_key(self.prefix, concept.lower())
        return await self.cache.get(key)
    
    async def set_embedding(
        self,
        concept: str,
        embedding: list,
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Cache an embedding for a concept."""
        key = Cache.make_key(self.prefix, concept.lower())
        return await self.cache.set(key, embedding, ttl)


async def init_cache(redis_url: str) -> Cache:
    """Initialize Redis cache connection."""
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        logger.info("Redis cache connected", url=redis_url.split("@")[-1])
        return Cache(client)
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise
