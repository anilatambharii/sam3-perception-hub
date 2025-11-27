"""
Perception Tools - Client wrappers for the Perception API.
"""

from typing import Any, Dict, List, Optional, Tuple
import httpx
import structlog

logger = structlog.get_logger(__name__)


class PerceptionTools:
    """Tool wrappers for SAM3 Perception API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def segment(
        self,
        image_url: str,
        concept: Optional[str] = None,
        points: Optional[List[List[int]]] = None,
        boxes: Optional[List[List[int]]] = None,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Segment objects in an image.
        
        Args:
            image_url: URL or local path to image
            concept: Text concept for PCS mode
            points: Click points for PVS mode
            boxes: Bounding boxes for PVS mode
            confidence_threshold: Minimum confidence
        
        Returns:
            Segmentation result with instances
        """
        logger.info("Segmenting image", image_url=image_url, concept=concept)
        
        # Download image if URL
        if image_url.startswith(("http://", "https://")):
            image_response = await self.client.get(image_url)
            image_data = image_response.content
        else:
            with open(image_url, "rb") as f:
                image_data = f.read()
        
        files = {"image": ("image.jpg", image_data, "image/jpeg")}
        
        if concept:
            # PCS mode
            query = {
                "text": concept,
                "confidence_threshold": confidence_threshold
            }
            response = await self.client.post(
                f"{self.base_url}/api/v1/segment/concept",
                files=files,
                data={"query": str(query).replace("'", '"')}
            )
        else:
            # PVS mode
            query = {}
            if points:
                query["points"] = points
                query["point_labels"] = [1] * len(points)
            if boxes:
                query["boxes"] = boxes
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/segment/visual",
                files=files,
                data={"query": str(query).replace("'", '"')}
            )
        
        response.raise_for_status()
        return response.json()
    
    async def track(
        self,
        video_url: str,
        concepts: List[str],
        sample_rate: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Track objects across video frames.
        
        Args:
            video_url: URL or local path to video
            concepts: List of concepts to track
            sample_rate: Frame sampling rate
        
        Returns:
            List of tracking results per frame
        """
        logger.info("Tracking video", video_url=video_url, concepts=concepts)
        
        # Download video if URL
        if video_url.startswith(("http://", "https://")):
            video_response = await self.client.get(video_url)
            video_data = video_response.content
        else:
            with open(video_url, "rb") as f:
                video_data = f.read()
        
        files = {"video": ("video.mp4", video_data, "video/mp4")}
        request_data = {
            "concepts": concepts,
            "sample_rate": sample_rate,
            "output_format": "json"
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/track",
            files=files,
            data={"request": str(request_data).replace("'", '"')},
            timeout=300.0  # Longer timeout for video
        )
        
        response.raise_for_status()
        
        # Parse NDJSON response
        results = []
        for line in response.text.strip().split("\n"):
            if line:
                import json
                results.append(json.loads(line))
        
        return results
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
