"""
WebSocket routes for real-time perception streaming.

Enables interactive segmentation and live video tracking.
"""

import asyncio
import json
from typing import Any, Dict

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["websocket"])


class WSMessage(BaseModel):
    """Base WebSocket message."""
    type: str
    payload: Dict[str, Any]


@router.websocket("/segment")
async def websocket_segment(websocket: WebSocket):
    """
    WebSocket endpoint for interactive segmentation.
    
    Supports real-time click-to-segment and concept queries.
    
    Message format:
    {
        "type": "segment_point" | "segment_box" | "segment_concept",
        "payload": { ... }
    }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = WSMessage.model_validate_json(data)
            except ValidationError as e:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": str(e)}
                })
                continue
            
            # Process message
            if message.type == "segment_point":
                # Handle point-based segmentation
                result = await handle_segment_point(websocket, message.payload)
                await websocket.send_json({
                    "type": "segment_result",
                    "payload": result
                })
            
            elif message.type == "segment_box":
                # Handle box-based segmentation
                result = await handle_segment_box(websocket, message.payload)
                await websocket.send_json({
                    "type": "segment_result",
                    "payload": result
                })
            
            elif message.type == "segment_concept":
                # Handle concept-based segmentation
                result = await handle_segment_concept(websocket, message.payload)
                await websocket.send_json({
                    "type": "segment_result",
                    "payload": result
                })
            
            elif message.type == "ping":
                await websocket.send_json({"type": "pong", "payload": {}})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": f"Unknown message type: {message.type}"}
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close(code=1011, reason=str(e))


@router.websocket("/track")
async def websocket_track(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video tracking.
    
    Streams tracking results as frames are processed.
    """
    await websocket.accept()
    logger.info("Tracking WebSocket connection established")
    
    try:
        # Wait for configuration message
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        concepts = config.get("concepts", [])
        sample_rate = config.get("sample_rate", 1)
        
        logger.info("Starting tracking stream", concepts=concepts)
        
        # Get model from app state
        model = websocket.app.state.model
        
        # This would be connected to a live video stream
        # For now, we simulate with periodic updates
        frame_idx = 0
        while True:
            # Check for stop message
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.1
                )
                if json.loads(msg).get("type") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            
            # Send mock tracking update
            result = {
                "type": "tracking_frame",
                "payload": {
                    "frame_index": frame_idx,
                    "timestamp_ms": frame_idx * 33.33,
                    "masklets": [
                        {
                            "instance_id": i,
                            "concept": concept,
                            "confidence": 0.9,
                            "bbox": [100 + frame_idx, 100, 300 + frame_idx, 300]
                        }
                        for i, concept in enumerate(concepts)
                    ]
                }
            }
            await websocket.send_json(result)
            
            frame_idx += 1
            await asyncio.sleep(0.033)  # ~30fps
    
    except WebSocketDisconnect:
        logger.info("Tracking WebSocket disconnected")
    except Exception as e:
        logger.error("Tracking WebSocket error", error=str(e))
        await websocket.close(code=1011, reason=str(e))


async def handle_segment_point(websocket: WebSocket, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle point-based segmentation request."""
    # Extract parameters
    points = payload.get("points", [])
    labels = payload.get("labels", [1] * len(points))
    
    # Get model
    model = websocket.app.state.model
    
    # In production, this would use the actual image from the session
    # For now, return mock result
    return {
        "instances": [
            {
                "instance_id": 1,
                "confidence": 0.95,
                "bbox": [points[0][0] - 50, points[0][1] - 50, points[0][0] + 50, points[0][1] + 50]
            }
        ] if points else [],
        "processing_time_ms": 28.5
    }


async def handle_segment_box(websocket: WebSocket, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle box-based segmentation request."""
    boxes = payload.get("boxes", [])
    
    return {
        "instances": [
            {
                "instance_id": i + 1,
                "confidence": 0.93,
                "bbox": box
            }
            for i, box in enumerate(boxes)
        ],
        "processing_time_ms": 32.1
    }


async def handle_segment_concept(websocket: WebSocket, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle concept-based segmentation request."""
    concept = payload.get("concept", "")
    
    return {
        "instances": [
            {
                "instance_id": 1,
                "concept": concept,
                "confidence": 0.91,
                "bbox": [100, 100, 300, 300]
            }
        ] if concept else [],
        "processing_time_ms": 45.8
    }
