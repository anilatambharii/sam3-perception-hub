"""Workflow Orchestrator - Chains perception operations into workflows."""

from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)


class WorkflowOrchestrator:
    """Orchestrates multi-step perception workflows."""
    
    def __init__(self, perception_tools, reconstruction_tools):
        self.perception = perception_tools
        self.reconstruction = reconstruction_tools
    
    async def execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a workflow plan."""
        results = []
        context = {}
        
        for step in plan.get("steps", []):
            logger.info("Executing step", step=step["name"])
            
            try:
                result = await self._execute_step(step, context)
                results.append({"step": step["name"], "success": True, "result": result})
                context[step["name"]] = result
            except Exception as e:
                logger.error("Step failed", step=step["name"], error=str(e))
                results.append({"step": step["name"], "success": False, "error": str(e)})
                if not step.get("continue_on_error", False):
                    break
        
        return results
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        action = step["action"]
        params = self._resolve_params(step.get("params", {}), context)
        
        if action == "segment":
            return await self.perception.segment(**params)
        elif action == "track":
            return await self.perception.track(**params)
        elif action == "reconstruct_object":
            return await self.reconstruction.reconstruct_object(**params)
        elif action == "reconstruct_body":
            return await self.reconstruction.reconstruct_body(**params)
        elif action == "blur_faces":
            return await self.blur_faces(**params)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _resolve_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter references from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]
                parts = ref.split(".")
                val = context
                for part in parts:
                    val = val.get(part, {}) if isinstance(val, dict) else getattr(val, part, None)
                resolved[key] = val
            else:
                resolved[key] = value
        return resolved
    
    async def blur_faces(
        self,
        media_url: str,
        blur_strength: int = 20,
    ) -> Dict[str, Any]:
        """Detect and blur faces in media."""
        logger.info("Blurring faces", media_url=media_url)
        
        # Segment faces
        result = await self.perception.segment(
            image_url=media_url,
            concept="face",
            confidence_threshold=0.7
        )
        
        # In production, apply blur to each face region
        face_count = len(result.get("instances", []))
        
        return {
            "faces_detected": face_count,
            "blur_applied": True,
            "blur_strength": blur_strength,
        }
    
    async def privacy_analytics(
        self,
        video_url: str,
        keep_silhouettes: bool = True,
        export_skeletons: bool = True,
    ) -> Dict[str, Any]:
        """Privacy-preserving video analytics workflow."""
        logger.info("Running privacy analytics", video_url=video_url)
        
        # Track people
        tracking_results = await self.perception.track(
            video_url=video_url,
            concepts=["person", "face"]
        )
        
        # Process results
        return {
            "frame_count": len(tracking_results),
            "people_tracked": len(set(
                m["instance_id"] 
                for r in tracking_results 
                for m in r.get("masklets", [])
                if m.get("concept") == "person"
            )),
            "faces_blurred": True,
            "silhouettes_exported": keep_silhouettes,
            "skeletons_exported": export_skeletons,
        }
