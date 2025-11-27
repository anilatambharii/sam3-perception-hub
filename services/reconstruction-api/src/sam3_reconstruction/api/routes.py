"""
SAM3 Reconstruction API Routes

REST endpoints for 3D reconstruction operations.
"""

import io
import json
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from sam3_reconstruction.models.sam3d_wrapper import (
    BodyReconstructionResult,
    MeshFormat,
    ObjectReconstructionResult,
    SAM3DModel,
)
from sam3_reconstruction.utils.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["reconstruction"])
settings = get_settings()


# =============================================================================
# Request/Response Models
# =============================================================================

class ObjectReconstructionRequest(BaseModel):
    """Request for 3D object reconstruction."""
    output_format: str = Field(
        default="glb",
        description="Output mesh format: glb, obj, ply"
    )
    texture_resolution: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Texture map resolution"
    )


class BodyReconstructionRequest(BaseModel):
    """Request for 3D body reconstruction."""
    output_format: str = Field(
        default="glb",
        description="Output mesh format: glb, obj, fbx"
    )
    include_pose: bool = Field(
        default=True,
        description="Include pose estimation"
    )


class ReconstructionMetadata(BaseModel):
    """Metadata about reconstruction result."""
    vertex_count: int
    face_count: int
    has_texture: bool
    processing_time_ms: float
    confidence: float
    output_format: str


class BodyReconstructionMetadata(ReconstructionMetadata):
    """Metadata for body reconstruction."""
    joint_count: int
    pose_included: bool


# =============================================================================
# Dependencies
# =============================================================================

async def get_object_model(request: Request) -> SAM3DModel:
    """Get the object reconstruction model."""
    if not hasattr(request.app.state, 'object_model'):
        raise HTTPException(status_code=503, detail="Object model not initialized")
    return request.app.state.object_model


async def get_body_model(request: Request) -> SAM3DModel:
    """Get the body reconstruction model."""
    if not hasattr(request.app.state, 'body_model'):
        raise HTTPException(status_code=503, detail="Body model not initialized")
    return request.app.state.body_model


# =============================================================================
# Helper Functions
# =============================================================================

async def load_image(file: UploadFile) -> np.ndarray:
    """Load and validate uploaded image."""
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")


async def load_mask(file: UploadFile) -> np.ndarray:
    """Load and validate mask image."""
    try:
        contents = await file.read()
        mask = Image.open(io.BytesIO(contents)).convert("L")
        return np.array(mask) > 127  # Convert to binary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load mask: {str(e)}")


def get_mesh_content_type(format: str) -> str:
    """Get content type for mesh format."""
    content_types = {
        "glb": "model/gltf-binary",
        "gltf": "model/gltf+json",
        "obj": "text/plain",
        "fbx": "application/octet-stream",
        "ply": "application/octet-stream",
    }
    return content_types.get(format, "application/octet-stream")


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/object",
    summary="Reconstruct 3D object",
    description="Generate 3D mesh from image and segmentation mask",
)
async def reconstruct_object(
    image: UploadFile = File(..., description="Source image"),
    mask: UploadFile = File(..., description="Segmentation mask"),
    request: str = Form(default='{}', description="JSON reconstruction options"),
    model: SAM3DModel = Depends(get_object_model),
):
    """
    Reconstruct a 3D object from an image and segmentation mask.
    
    Uses SAM 3D Objects to generate a textured 3D mesh.
    
    Returns the mesh file in the requested format.
    """
    # Parse request
    try:
        req_data = ObjectReconstructionRequest.model_validate_json(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Load inputs
    img_array = await load_image(image)
    mask_array = await load_mask(mask)
    
    logger.info(
        "Object reconstruction request",
        image_size=img_array.shape[:2],
        output_format=req_data.output_format,
    )
    
    # Validate mask shape
    if mask_array.shape[:2] != img_array.shape[:2]:
        raise HTTPException(
            status_code=400,
            detail=f"Mask size {mask_array.shape[:2]} doesn't match image size {img_array.shape[:2]}"
        )
    
    # Perform reconstruction
    try:
        result = await model.reconstruct_object(
            img_array,
            mask_array,
            texture_resolution=req_data.texture_resolution,
        )
    except Exception as e:
        logger.error("Object reconstruction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")
    
    logger.info(
        "Object reconstruction complete",
        vertex_count=len(result.mesh.vertices),
        processing_time_ms=result.processing_time_ms,
    )
    
    # Export mesh
    if req_data.output_format == "glb":
        mesh_data = result.mesh.to_glb()
        media_type = "model/gltf-binary"
        filename = "reconstructed.glb"
    elif req_data.output_format == "obj":
        mesh_data = result.mesh.to_obj().encode()
        media_type = "text/plain"
        filename = "reconstructed.obj"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {req_data.output_format}"
        )
    
    return Response(
        content=mesh_data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Processing-Time-Ms": str(result.processing_time_ms),
            "X-Vertex-Count": str(len(result.mesh.vertices)),
            "X-Face-Count": str(len(result.mesh.faces)),
        }
    )


@router.post(
    "/object/metadata",
    response_model=ReconstructionMetadata,
    summary="Reconstruct 3D object (metadata only)",
    description="Generate 3D mesh and return metadata without the mesh data",
)
async def reconstruct_object_metadata(
    image: UploadFile = File(..., description="Source image"),
    mask: UploadFile = File(..., description="Segmentation mask"),
    request: str = Form(default='{}', description="JSON reconstruction options"),
    model: SAM3DModel = Depends(get_object_model),
):
    """
    Reconstruct 3D object and return only metadata.
    
    Useful for getting reconstruction info before downloading the full mesh.
    """
    try:
        req_data = ObjectReconstructionRequest.model_validate_json(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    img_array = await load_image(image)
    mask_array = await load_mask(mask)
    
    try:
        result = await model.reconstruct_object(
            img_array,
            mask_array,
            texture_resolution=req_data.texture_resolution,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")
    
    return ReconstructionMetadata(
        vertex_count=len(result.mesh.vertices),
        face_count=len(result.mesh.faces),
        has_texture=result.mesh.texture is not None,
        processing_time_ms=result.processing_time_ms,
        confidence=result.confidence,
        output_format=req_data.output_format,
    )


@router.post(
    "/body",
    summary="Reconstruct 3D body",
    description="Generate 3D body mesh from person image",
)
async def reconstruct_body(
    image: UploadFile = File(..., description="Image containing a person"),
    request: str = Form(default='{}', description="JSON reconstruction options"),
    model: SAM3DModel = Depends(get_body_model),
):
    """
    Reconstruct a 3D body mesh from an image of a person.
    
    Uses SAM 3D Body to generate a rigged body mesh with pose.
    """
    try:
        req_data = BodyReconstructionRequest.model_validate_json(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    img_array = await load_image(image)
    
    logger.info(
        "Body reconstruction request",
        image_size=img_array.shape[:2],
        output_format=req_data.output_format,
    )
    
    try:
        result = await model.reconstruct_body(
            img_array,
            include_pose=req_data.include_pose,
        )
    except Exception as e:
        logger.error("Body reconstruction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")
    
    logger.info(
        "Body reconstruction complete",
        vertex_count=len(result.mesh.vertices),
        processing_time_ms=result.processing_time_ms,
    )
    
    # Export mesh
    if req_data.output_format == "glb":
        mesh_data = result.mesh.to_glb()
        media_type = "model/gltf-binary"
        filename = "body.glb"
    elif req_data.output_format == "obj":
        mesh_data = result.mesh.to_obj().encode()
        media_type = "text/plain"
        filename = "body.obj"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {req_data.output_format}"
        )
    
    # Include pose data in header
    pose_json = json.dumps(result.pose.to_dict())
    
    return Response(
        content=mesh_data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Processing-Time-Ms": str(result.processing_time_ms),
            "X-Vertex-Count": str(len(result.mesh.vertices)),
            "X-Face-Count": str(len(result.mesh.faces)),
            "X-Pose-Data": pose_json,
        }
    )


@router.post(
    "/body/pose",
    summary="Extract body pose only",
    description="Extract 3D pose from person image without full mesh reconstruction",
)
async def extract_body_pose(
    image: UploadFile = File(..., description="Image containing a person"),
    model: SAM3DModel = Depends(get_body_model),
):
    """
    Extract body pose without generating full mesh.
    
    Faster than full reconstruction when only pose is needed.
    """
    img_array = await load_image(image)
    
    try:
        result = await model.reconstruct_body(img_array, include_pose=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose extraction failed: {str(e)}")
    
    return {
        "pose": result.pose.to_dict(),
        "processing_time_ms": result.processing_time_ms,
        "confidence": result.confidence,
    }


@router.get(
    "/formats",
    summary="List supported formats",
    description="Get list of supported mesh export formats",
)
async def list_formats():
    """List all supported mesh export formats."""
    return {
        "object_formats": ["glb", "gltf", "obj", "ply"],
        "body_formats": ["glb", "gltf", "obj", "fbx"],
        "default_format": "glb",
    }
