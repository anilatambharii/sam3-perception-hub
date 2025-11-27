"""
SAM3 Agent Bridge - LLM Tool Server

Provides:
- MCP-compatible tool server for LLM agents
- Workflow orchestration for multi-step perception tasks
- Natural language planning engine
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sam3_agent.tools.perception_tools import PerceptionTools
from sam3_agent.tools.reconstruction_tools import ReconstructionTools
from sam3_agent.workflows.orchestrator import WorkflowOrchestrator
from sam3_agent.planning.planner import TaskPlanner

logger = structlog.get_logger(__name__)


class AgentSettings(BaseModel):
    perception_api_url: str = "http://localhost:8080"
    reconstruction_api_url: str = "http://localhost:8081"
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"


settings = AgentSettings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting SAM3 Agent Bridge")
    
    # Initialize tool clients
    app.state.perception = PerceptionTools(settings.perception_api_url)
    app.state.reconstruction = ReconstructionTools(settings.reconstruction_api_url)
    app.state.orchestrator = WorkflowOrchestrator(
        app.state.perception,
        app.state.reconstruction
    )
    app.state.planner = TaskPlanner()
    
    yield
    logger.info("Shutting down SAM3 Agent Bridge")


app = FastAPI(
    title="SAM3 Agent Bridge",
    description="LLM tool server for SAM3 perception workflows",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Tool Definitions (MCP-compatible)
# =============================================================================

class ToolCall(BaseModel):
    """A tool call request."""
    name: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default={}, description="Tool parameters")


class ToolResult(BaseModel):
    """Result of a tool call."""
    success: bool
    result: Any = None
    error: str = None


class WorkflowRequest(BaseModel):
    """Natural language workflow request."""
    instruction: str = Field(..., description="Natural language instruction")
    context: Dict[str, Any] = Field(default={}, description="Additional context")


AVAILABLE_TOOLS = [
    {
        "name": "segment_image",
        "description": "Segment objects in an image using text prompts or visual prompts",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string", "description": "URL or path to image"},
                "concept": {"type": "string", "description": "Text concept to segment"},
                "points": {"type": "array", "description": "Click points [[x,y], ...]"},
                "boxes": {"type": "array", "description": "Bounding boxes [[x1,y1,x2,y2], ...]"},
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "track_video",
        "description": "Track objects across video frames",
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {"type": "string", "description": "URL or path to video"},
                "concepts": {"type": "array", "items": {"type": "string"}, "description": "Concepts to track"},
            },
            "required": ["video_url", "concepts"]
        }
    },
    {
        "name": "reconstruct_3d",
        "description": "Reconstruct 3D mesh from image and mask",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string"},
                "mask_url": {"type": "string"},
                "output_format": {"type": "string", "enum": ["glb", "obj"]}
            },
            "required": ["image_url", "mask_url"]
        }
    },
    {
        "name": "reconstruct_body",
        "description": "Reconstruct 3D body mesh from person image",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string"},
                "include_pose": {"type": "boolean", "default": True}
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "blur_faces",
        "description": "Detect and blur faces in image/video for privacy",
        "parameters": {
            "type": "object",
            "properties": {
                "media_url": {"type": "string"},
                "blur_strength": {"type": "number", "default": 20}
            },
            "required": ["media_url"]
        }
    }
]


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/tools")
async def list_tools():
    """List all available tools (MCP-compatible)."""
    return {"tools": AVAILABLE_TOOLS}


@app.post("/tools/call", response_model=ToolResult)
async def call_tool(request: ToolCall):
    """Execute a tool call."""
    logger.info("Tool call", name=request.name, params=request.params)
    
    try:
        if request.name == "segment_image":
            result = await app.state.perception.segment(**request.params)
        elif request.name == "track_video":
            result = await app.state.perception.track(**request.params)
        elif request.name == "reconstruct_3d":
            result = await app.state.reconstruction.reconstruct_object(**request.params)
        elif request.name == "reconstruct_body":
            result = await app.state.reconstruction.reconstruct_body(**request.params)
        elif request.name == "blur_faces":
            result = await app.state.orchestrator.blur_faces(**request.params)
        else:
            raise ValueError(f"Unknown tool: {request.name}")
        
        return ToolResult(success=True, result=result)
    
    except Exception as e:
        logger.error("Tool call failed", name=request.name, error=str(e))
        return ToolResult(success=False, error=str(e))


@app.post("/workflow")
async def execute_workflow(request: WorkflowRequest):
    """Execute a natural language workflow."""
    logger.info("Workflow request", instruction=request.instruction[:100])
    
    try:
        # Plan the workflow
        plan = await app.state.planner.plan(request.instruction, request.context)
        
        # Execute the plan
        results = await app.state.orchestrator.execute_plan(plan)
        
        return {
            "success": True,
            "plan": plan,
            "results": results
        }
    except Exception as e:
        logger.error("Workflow failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
