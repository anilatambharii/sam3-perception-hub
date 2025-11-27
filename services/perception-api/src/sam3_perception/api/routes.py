"""
SAM3 Perception API Routes

REST endpoints for segmentation and tracking operations.
"""

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from sam3_perception.models.sam3_wrapper import (
    ConceptQuery,
    SAM3Model,
    SegmentationResult,
    VisualQuery,
)
from sam3_perception.utils.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["perception"])
settings = get_settings()


# =============================================================================
# Request/Response Models
# =============================================================================

class ConceptQueryRequest(BaseModel):
    """Request model for concept-based segmentation (PCS)."""
    text: str = Field(..., description="Text prompt describing the concept to segment")
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for instances"
    )
    max_instances: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of instances to return"
    )
    region_of_interest: Optional[List[int]] = Field(
        default=None,
        description="Optional ROI as [x1, y1, x2, y2]"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "forklift",
                "confidence_threshold": 0.7,
                "max_instances": 50
            }
        }


class VisualQueryRequest(BaseModel):
    """Request model for visual prompt segmentation (PVS)."""
    points: Optional[List[List[int]]] = Field(
        default=None,
        description="Click points as [[x, y], ...]"
    )
    point_labels: Optional[List[int]] = Field(
        default=None,
        description="Point labels: 1=foreground, 0=background"
    )
    boxes: Optional[List[List[int]]] = Field(
        default=None,
        description="Bounding boxes as [[x1, y1, x2, y2], ...]"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "points": [[256, 256]],
                "point_labels": [1]
            }
        }


class InstanceResponse(BaseModel):
    """Response model for a segmented instance."""
    instance_id: int
    concept: Optional[str] = None
    confidence: float
    bbox: Optional[List[int]] = None
    area: int
    mask_rle: Optional[Dict[str, Any]] = None
    frame_index: int = 0


class SegmentationResponse(BaseModel):
    """Response model for segmentation results."""
    instances: List[InstanceResponse]
    image_size: List[int]
    processing_time_ms: float
    instance_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "instance_id": 1,
                        "concept": "forklift",
                        "confidence": 0.95,
                        "bbox": [100, 100, 300, 300],
                        "area": 40000
                    }
                ],
                "image_size": [1920, 1080],
                "processing_time_ms": 45.2,
                "instance_count": 1
            }
        }


class TrackingRequest(BaseModel):
    """Request model for video tracking."""
    concepts: List[str] = Field(..., description="Concepts to track")
    sample_rate: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Frame sampling rate"
    )
    output_format: str = Field(
        default="json",
        description="Output format: json or ndjson"
    )


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_model(request: Request) -> SAM3Model:
    """Get the loaded SAM3 model from app state."""
    if not hasattr(request.app.state, 'model'):
        raise HTTPException(status_code=503, detail="Model not initialized")
    return request.app.state.model


# =============================================================================
# Helper Functions
# =============================================================================

async def load_image(file: UploadFile) -> np.ndarray:
    """Load and validate uploaded image."""
    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Check image size
        if max(image.size) > settings.max_image_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max dimension: {settings.max_image_size}"
            )
        
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")


def validate_concept(concept: str) -> None:
    """Validate concept against allow/deny lists."""
    if not settings.is_concept_allowed(concept):
        raise HTTPException(
            status_code=400,
            detail=f"Concept '{concept}' is not allowed"
        )


def result_to_response(result: SegmentationResult) -> SegmentationResponse:
    """Convert internal result to API response."""
    instances = []
    for inst in result.instances:
        instances.append(InstanceResponse(
            instance_id=inst.instance_id,
            concept=inst.concept,
            confidence=inst.confidence,
            bbox=list(inst.bbox) if inst.bbox else None,
            area=inst.area,
            mask_rle=inst.mask_rle,
            frame_index=inst.frame_index,
        ))
    
    return SegmentationResponse(
        instances=instances,
        image_size=list(result.image_size),
        processing_time_ms=result.processing_time_ms,
        instance_count=len(instances),
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/segment/concept",
    response_model=SegmentationResponse,
    summary="Segment by concept (PCS)",
    description="Perform Promptable Concept Segmentation using a text prompt",
)
async def segment_by_concept(
    image: UploadFile = File(..., description="Image to segment"),
    query: str = Form(..., description="JSON query object"),
    model: SAM3Model = Depends(get_model),
):
    """
    Segment objects in an image using a text/concept prompt.
    
    This endpoint uses SAM 3's Promptable Concept Segmentation (PCS) mode
    to find all instances of a concept without any visual prompts.
    """
    import json
    
    # Parse query
    try:
        query_data = ConceptQueryRequest.model_validate_json(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    
    # Validate concept
    validate_concept(query_data.text)
    
    # Load image
    img_array = await load_image(image)
    
    # Log request
    logger.info(
        "PCS segmentation request",
        concept=query_data.text,
        image_size=img_array.shape[:2],
    )
    
    # Create query
    concept_query = ConceptQuery(
        text=query_data.text,
        confidence_threshold=query_data.confidence_threshold,
        max_instances=query_data.max_instances,
        region_of_interest=tuple(query_data.region_of_interest) if query_data.region_of_interest else None,
    )
    
    # Run segmentation
    try:
        result = await model.segment_pcs(img_array, text=query_data.text, confidence_threshold=query_data.confidence_threshold)
    except Exception as e:
        logger.error("Segmentation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    # Log result
    logger.info(
        "PCS segmentation complete",
        concept=query_data.text,
        instance_count=len(result.instances),
        processing_time_ms=result.processing_time_ms,
    )
    
    return result_to_response(result)


@router.post(
    "/segment/visual",
    response_model=SegmentationResponse,
    summary="Segment by visual prompt (PVS)",
    description="Perform Promptable Visual Segmentation using points/boxes",
)
async def segment_by_visual(
    image: UploadFile = File(..., description="Image to segment"),
    query: str = Form(..., description="JSON query object"),
    model: SAM3Model = Depends(get_model),
):
    """
    Segment objects using visual prompts (points or bounding boxes).
    
    This endpoint uses SAM 3's Promptable Visual Segmentation (PVS) mode.
    """
    import json
    
    # Parse query
    try:
        query_data = VisualQueryRequest.model_validate_json(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    
    # Validate that at least one prompt type is provided
    if not query_data.points and not query_data.boxes:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'points' or 'boxes' must be provided"
        )
    
    # Load image
    img_array = await load_image(image)
    
    # Log request
    logger.info(
        "PVS segmentation request",
        num_points=len(query_data.points or []),
        num_boxes=len(query_data.boxes or []),
        image_size=img_array.shape[:2],
    )
    
    # Convert query
    points = [tuple(p) for p in query_data.points] if query_data.points else None
    boxes = [tuple(b) for b in query_data.boxes] if query_data.boxes else None
    
    # Run segmentation
    try:
        result = await model.segment_pvs(
            img_array,
            points=points,
            point_labels=query_data.point_labels,
            boxes=boxes,
        )
    except Exception as e:
        logger.error("Segmentation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    logger.info(
        "PVS segmentation complete",
        instance_count=len(result.instances),
        processing_time_ms=result.processing_time_ms,
    )
    
    return result_to_response(result)


@router.post(
    "/segment/auto",
    response_model=SegmentationResponse,
    summary="Automatic segmentation",
    description="Segment all objects in an image automatically",
)
async def segment_auto(
    image: UploadFile = File(..., description="Image to segment"),
    confidence_threshold: float = Form(default=0.5),
    max_instances: int = Form(default=100),
    model: SAM3Model = Depends(get_model),
):
    """
    Automatically segment all objects in an image.
    
    Uses SAM 3's automatic mode to find all segmentable regions.
    """
    img_array = await load_image(image)
    
    logger.info(
        "Auto segmentation request",
        image_size=img_array.shape[:2],
        confidence_threshold=confidence_threshold,
    )
    
    # Use empty text prompt for auto mode
    try:
        result = await model.segment_pcs(
            img_array,
            text="",  # Empty prompt for auto mode
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:
        logger.error("Auto segmentation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    return result_to_response(result)


@router.post(
    "/track",
    summary="Track objects in video",
    description="Track specified concepts across video frames",
)
async def track_video(
    video: UploadFile = File(..., description="Video to process"),
    request: str = Form(..., description="JSON tracking request"),
    model: SAM3Model = Depends(get_model),
):
    """
    Track objects across video frames with stable IDs.
    
    Returns streaming NDJSON with results for each frame.
    """
    import json
    import tempfile
    
    # Parse request
    try:
        track_request = TrackingRequest.model_validate_json(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Validate concepts
    for concept in track_request.concepts:
        validate_concept(concept)
    
    # Save video to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await video.read()
            tmp.write(contents)
            video_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process video: {str(e)}")
    
    logger.info(
        "Video tracking request",
        concepts=track_request.concepts,
        sample_rate=track_request.sample_rate,
    )
    
    async def generate_results():
        """Generate streaming results."""
        try:
            async for frame_result in model.track(
                video_path,
                track_request.concepts,
                track_request.sample_rate,
            ):
                yield json.dumps(frame_result.to_dict()) + "\n"
        except Exception as e:
            logger.error("Tracking failed", error=str(e))
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            # Cleanup temp file
            import os
            try:
                os.unlink(video_path)
            except:
                pass
    
    return StreamingResponse(
        generate_results(),
        media_type="application/x-ndjson",
    )


@router.get(
    "/concepts/validate",
    summary="Validate concept",
    description="Check if a concept is allowed by the current policy",
)
async def validate_concept_endpoint(concept: str):
    """Check if a concept is allowed by the safety policy."""
    is_allowed = settings.is_concept_allowed(concept)
    return {
        "concept": concept,
        "allowed": is_allowed,
        "reason": None if is_allowed else "Concept is in deny list or not in allow list"
    }
