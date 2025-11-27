"""
SAM3 Perception Hub - Python Client Library

A convenient Python client for interacting with SAM3 Perception Hub APIs.

Example usage:
    from sam3_perception_hub import PerceptionClient, ReconstructionClient

    # Perception API
    client = PerceptionClient("http://localhost:8080")
    result = client.segment(
        image="warehouse.jpg",
        query=ConceptQuery(text="forklift")
    )

    # Reconstruction API
    recon = ReconstructionClient("http://localhost:8081")
    mesh = recon.reconstruct_object(
        image="chair.jpg",
        mask=result.instances[0].mask
    )
    mesh.save("chair_3d.glb")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import io

import httpx


__version__ = "0.1.0"
__all__ = [
    "PerceptionClient",
    "ReconstructionClient",
    "AgentClient",
    "ConceptQuery",
    "VisualQuery",
    "SegmentationResult",
    "SegmentationInstance",
    "Mesh3D",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConceptQuery:
    """Query for text-based concept segmentation (PCS)."""
    text: str
    confidence_threshold: float = 0.5
    max_instances: int = 100
    region_of_interest: Optional[Tuple[int, int, int, int]] = None


@dataclass
class VisualQuery:
    """Query for visual prompt segmentation (PVS)."""
    points: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None
    boxes: Optional[List[Tuple[int, int, int, int]]] = None


@dataclass
class Mask:
    """Segmentation mask."""
    rle: Optional[Dict[str, Any]] = None
    polygon: Optional[List[List[Tuple[int, int]]]] = None
    
    def to_rle(self) -> Dict[str, Any]:
        """Convert mask to RLE format."""
        if self.rle:
            return self.rle
        raise ValueError("RLE data not available")
    
    def to_numpy(self) -> "np.ndarray":
        """Convert mask to numpy array."""
        import numpy as np
        # Decode RLE to numpy array
        if self.rle:
            # Implement RLE decoding
            size = self.rle.get("size", [0, 0])
            return np.zeros(size, dtype=np.uint8)
        raise ValueError("Mask data not available")


@dataclass
class SegmentationInstance:
    """A single segmented instance."""
    instance_id: int
    concept: Optional[str] = None
    confidence: float = 1.0
    mask: Optional[Mask] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    area: int = 0
    frame_index: int = 0


@dataclass
class SegmentationResult:
    """Result of a segmentation operation."""
    instances: List[SegmentationInstance] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)
    processing_time_ms: float = 0.0


@dataclass
class TrackingResult:
    """Result of video tracking for a single frame."""
    frame_index: int
    timestamp_ms: float
    masklets: List[SegmentationInstance] = field(default_factory=list)


@dataclass
class Mesh3D:
    """3D mesh with optional texture."""
    data: bytes
    format: str
    vertex_count: int = 0
    face_count: int = 0
    
    def save(self, path: str) -> None:
        """Save mesh to file."""
        with open(path, "wb") as f:
            f.write(self.data)


# =============================================================================
# Perception Client
# =============================================================================

class PerceptionClient:
    """Client for SAM3 Perception API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
    
    def segment(
        self,
        image: Union[str, bytes, "Image.Image"],
        query: Union[ConceptQuery, VisualQuery],
    ) -> SegmentationResult:
        """
        Segment objects in an image.
        
        Args:
            image: Image path, bytes, or PIL Image
            query: ConceptQuery for PCS or VisualQuery for PVS
        
        Returns:
            SegmentationResult with instances
        """
        image_data = self._prepare_image(image)
        
        if isinstance(query, ConceptQuery):
            return self._segment_pcs(image_data, query)
        else:
            return self._segment_pvs(image_data, query)
    
    def _segment_pcs(
        self,
        image_data: bytes,
        query: ConceptQuery,
    ) -> SegmentationResult:
        """Perform PCS segmentation."""
        import json
        
        query_json = json.dumps({
            "text": query.text,
            "confidence_threshold": query.confidence_threshold,
            "max_instances": query.max_instances,
        })
        
        response = self.client.post(
            f"{self.base_url}/api/v1/segment/concept",
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"query": query_json},
        )
        response.raise_for_status()
        
        return self._parse_result(response.json())
    
    def _segment_pvs(
        self,
        image_data: bytes,
        query: VisualQuery,
    ) -> SegmentationResult:
        """Perform PVS segmentation."""
        import json
        
        query_dict = {}
        if query.points:
            query_dict["points"] = [list(p) for p in query.points]
            query_dict["point_labels"] = query.point_labels or [1] * len(query.points)
        if query.boxes:
            query_dict["boxes"] = [list(b) for b in query.boxes]
        
        response = self.client.post(
            f"{self.base_url}/api/v1/segment/visual",
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"query": json.dumps(query_dict)},
        )
        response.raise_for_status()
        
        return self._parse_result(response.json())
    
    def create_tracker(
        self,
        video: Union[str, bytes],
        concepts: List[str],
        sample_rate: int = 1,
    ) -> "VideoTracker":
        """Create a video tracker."""
        return VideoTracker(
            self.base_url,
            self.client,
            video,
            concepts,
            sample_rate,
        )
    
    def _prepare_image(
        self,
        image: Union[str, bytes, "Image.Image"],
    ) -> bytes:
        """Convert image to bytes."""
        if isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            with open(image, "rb") as f:
                return f.read()
        else:
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
    
    def _parse_result(self, data: Dict[str, Any]) -> SegmentationResult:
        """Parse API response to SegmentationResult."""
        instances = []
        for inst_data in data.get("instances", []):
            mask = None
            if inst_data.get("mask_rle"):
                mask = Mask(rle=inst_data["mask_rle"])
            
            instances.append(SegmentationInstance(
                instance_id=inst_data["instance_id"],
                concept=inst_data.get("concept"),
                confidence=inst_data.get("confidence", 1.0),
                mask=mask,
                bbox=tuple(inst_data["bbox"]) if inst_data.get("bbox") else None,
                area=inst_data.get("area", 0),
            ))
        
        return SegmentationResult(
            instances=instances,
            image_size=tuple(data.get("image_size", [0, 0])),
            processing_time_ms=data.get("processing_time_ms", 0.0),
        )
    
    def close(self):
        """Close the client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class VideoTracker:
    """Video object tracker with streaming results."""
    
    def __init__(
        self,
        base_url: str,
        client: httpx.Client,
        video: Union[str, bytes],
        concepts: List[str],
        sample_rate: int,
    ):
        self.base_url = base_url
        self.client = client
        self.video = video
        self.concepts = concepts
        self.sample_rate = sample_rate
        self._results = None
    
    def stream(self):
        """Stream tracking results frame by frame."""
        import json
        
        # Prepare video data
        if isinstance(self.video, str):
            with open(self.video, "rb") as f:
                video_data = f.read()
        else:
            video_data = self.video
        
        request_json = json.dumps({
            "concepts": self.concepts,
            "sample_rate": self.sample_rate,
            "output_format": "ndjson",
        })
        
        response = self.client.post(
            f"{self.base_url}/api/v1/track",
            files={"video": ("video.mp4", video_data, "video/mp4")},
            data={"request": request_json},
            timeout=300.0,
        )
        response.raise_for_status()
        
        # Parse NDJSON
        for line in response.text.strip().split("\n"):
            if line:
                data = json.loads(line)
                masklets = []
                for m in data.get("masklets", []):
                    masklets.append(SegmentationInstance(
                        instance_id=m["instance_id"],
                        concept=m.get("concept"),
                        confidence=m.get("confidence", 1.0),
                        bbox=tuple(m["bbox"]) if m.get("bbox") else None,
                        frame_index=data["frame_index"],
                    ))
                
                yield TrackingResult(
                    frame_index=data["frame_index"],
                    timestamp_ms=data.get("timestamp_ms", 0.0),
                    masklets=masklets,
                )


# =============================================================================
# Reconstruction Client
# =============================================================================

class ReconstructionClient:
    """Client for SAM3 Reconstruction API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
    
    def reconstruct_object(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes, Mask],
        output_format: str = "glb",
        texture_resolution: int = 1024,
    ) -> Mesh3D:
        """
        Reconstruct 3D object from image and mask.
        
        Args:
            image: Image path or bytes
            mask: Mask path, bytes, or Mask object
            output_format: Output format (glb, obj)
            texture_resolution: Texture resolution
        
        Returns:
            Mesh3D object
        """
        import json
        
        image_data = self._load_file(image)
        mask_data = self._load_mask(mask)
        
        request_json = json.dumps({
            "output_format": output_format,
            "texture_resolution": texture_resolution,
        })
        
        response = self.client.post(
            f"{self.base_url}/api/v1/object",
            files={
                "image": ("image.jpg", image_data, "image/jpeg"),
                "mask": ("mask.png", mask_data, "image/png"),
            },
            data={"request": request_json},
        )
        response.raise_for_status()
        
        return Mesh3D(
            data=response.content,
            format=output_format,
            vertex_count=int(response.headers.get("X-Vertex-Count", 0)),
            face_count=int(response.headers.get("X-Face-Count", 0)),
        )
    
    def reconstruct_body(
        self,
        image: Union[str, bytes],
        output_format: str = "glb",
        include_pose: bool = True,
    ) -> Tuple[Mesh3D, Optional[Dict[str, Any]]]:
        """
        Reconstruct 3D body mesh from person image.
        
        Args:
            image: Image path or bytes
            output_format: Output format
            include_pose: Include pose estimation
        
        Returns:
            Tuple of (Mesh3D, pose_data)
        """
        import json
        
        image_data = self._load_file(image)
        
        request_json = json.dumps({
            "output_format": output_format,
            "include_pose": include_pose,
        })
        
        response = self.client.post(
            f"{self.base_url}/api/v1/body",
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"request": request_json},
        )
        response.raise_for_status()
        
        pose_data = None
        if response.headers.get("X-Pose-Data"):
            pose_data = json.loads(response.headers["X-Pose-Data"])
        
        mesh = Mesh3D(
            data=response.content,
            format=output_format,
            vertex_count=int(response.headers.get("X-Vertex-Count", 0)),
            face_count=int(response.headers.get("X-Face-Count", 0)),
        )
        
        return mesh, pose_data
    
    def _load_file(self, file: Union[str, bytes]) -> bytes:
        """Load file to bytes."""
        if isinstance(file, bytes):
            return file
        with open(file, "rb") as f:
            return f.read()
    
    def _load_mask(self, mask: Union[str, bytes, Mask]) -> bytes:
        """Load mask to bytes."""
        if isinstance(mask, bytes):
            return mask
        elif isinstance(mask, str):
            with open(mask, "rb") as f:
                return f.read()
        elif isinstance(mask, Mask):
            # Convert Mask to PNG bytes
            arr = mask.to_numpy()
            from PIL import Image
            img = Image.fromarray(arr * 255)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        else:
            raise ValueError(f"Invalid mask type: {type(mask)}")
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# Agent Client
# =============================================================================

class AgentClient:
    """Client for SAM3 Agent Bridge API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8082",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        response = self.client.get(f"{self.base_url}/tools")
        response.raise_for_status()
        return response.json()["tools"]
    
    def call_tool(
        self,
        name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a tool by name."""
        response = self.client.post(
            f"{self.base_url}/tools/call",
            json={"name": name, "params": params},
        )
        response.raise_for_status()
        return response.json()
    
    def execute(self, instruction: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a natural language workflow."""
        response = self.client.post(
            f"{self.base_url}/workflow",
            json={"instruction": instruction, "context": context or {}},
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
