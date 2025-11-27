"""Reconstruction Tools - Client wrappers for the Reconstruction API."""

from typing import Any, Dict, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)


class ReconstructionTools:
    """Tool wrappers for SAM3 Reconstruction API."""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def reconstruct_object(
        self,
        image_url: str,
        mask_url: str,
        output_format: str = "glb",
        texture_resolution: int = 1024,
    ) -> Dict[str, Any]:
        """Reconstruct 3D object from image and mask."""
        logger.info("Reconstructing object", image_url=image_url)
        
        # Load image and mask
        image_data = await self._load_media(image_url)
        mask_data = await self._load_media(mask_url)
        
        files = {
            "image": ("image.jpg", image_data, "image/jpeg"),
            "mask": ("mask.png", mask_data, "image/png"),
        }
        
        request = {
            "output_format": output_format,
            "texture_resolution": texture_resolution
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/object",
            files=files,
            data={"request": str(request).replace("'", '"')}
        )
        response.raise_for_status()
        
        return {
            "mesh_data": response.content,
            "format": output_format,
            "processing_time_ms": float(response.headers.get("X-Processing-Time-Ms", 0)),
            "vertex_count": int(response.headers.get("X-Vertex-Count", 0)),
        }
    
    async def reconstruct_body(
        self,
        image_url: str,
        output_format: str = "glb",
        include_pose: bool = True,
    ) -> Dict[str, Any]:
        """Reconstruct 3D body mesh from person image."""
        logger.info("Reconstructing body", image_url=image_url)
        
        image_data = await self._load_media(image_url)
        
        files = {"image": ("image.jpg", image_data, "image/jpeg")}
        request = {
            "output_format": output_format,
            "include_pose": include_pose
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/body",
            files=files,
            data={"request": str(request).replace("'", '"')}
        )
        response.raise_for_status()
        
        import json
        pose_data = response.headers.get("X-Pose-Data")
        
        return {
            "mesh_data": response.content,
            "format": output_format,
            "pose": json.loads(pose_data) if pose_data else None,
            "processing_time_ms": float(response.headers.get("X-Processing-Time-Ms", 0)),
        }
    
    async def _load_media(self, url: str) -> bytes:
        """Load media from URL or local path."""
        if url.startswith(("http://", "https://")):
            response = await self.client.get(url)
            return response.content
        else:
            with open(url, "rb") as f:
                return f.read()
    
    async def close(self):
        await self.client.aclose()
