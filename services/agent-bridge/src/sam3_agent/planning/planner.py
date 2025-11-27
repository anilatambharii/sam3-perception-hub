"""Task Planner - Converts natural language instructions to workflow plans."""

from typing import Any, Dict, List
import re
import structlog

logger = structlog.get_logger(__name__)


class TaskPlanner:
    """Plans perception workflows from natural language instructions."""
    
    # Pattern matching for common instructions
    PATTERNS = [
        (r"find\s+(?:all\s+)?(\w+)s?\s+in\s+(?:the\s+)?(\w+)", "segment"),
        (r"segment\s+(\w+)s?\s+(?:in|from)\s+(?:the\s+)?(\w+)", "segment"),
        (r"track\s+(\w+)s?\s+(?:in|across)\s+(?:the\s+)?(\w+)", "track"),
        (r"blur\s+(?:all\s+)?faces?\s+(?:in\s+)?(?:the\s+)?(\w+)?", "blur_faces"),
        (r"reconstruct\s+(?:3d\s+)?(\w+)", "reconstruct"),
        (r"create\s+(?:a\s+)?3d\s+(?:model|mesh)\s+of\s+(?:the\s+)?(\w+)", "reconstruct"),
    ]
    
    async def plan(
        self,
        instruction: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Convert natural language instruction to a workflow plan.
        
        Args:
            instruction: Natural language instruction
            context: Additional context (media URLs, etc.)
        
        Returns:
            Workflow plan with steps
        """
        logger.info("Planning workflow", instruction=instruction[:100])
        
        context = context or {}
        instruction_lower = instruction.lower()
        
        steps = []
        
        # Parse instruction using patterns
        for pattern, action in self.PATTERNS:
            match = re.search(pattern, instruction_lower)
            if match:
                step = self._create_step(action, match, context)
                if step:
                    steps.append(step)
        
        # If no patterns matched, try keyword-based planning
        if not steps:
            steps = self._plan_from_keywords(instruction_lower, context)
        
        # Add post-processing steps if mentioned
        if "blur" in instruction_lower and "face" in instruction_lower:
            if not any(s["action"] == "blur_faces" for s in steps):
                steps.append({
                    "name": "blur_faces",
                    "action": "blur_faces",
                    "params": {"media_url": context.get("image_url") or context.get("video_url", "")},
                })
        
        if "3d" in instruction_lower or "mesh" in instruction_lower:
            if not any(s["action"].startswith("reconstruct") for s in steps):
                steps.append({
                    "name": "reconstruct",
                    "action": "reconstruct_object",
                    "params": {
                        "image_url": context.get("image_url", ""),
                        "mask_url": "$segment.mask_url"  # Reference previous step
                    },
                })
        
        return {
            "instruction": instruction,
            "steps": steps,
            "context": context,
        }
    
    def _create_step(
        self,
        action: str,
        match: re.Match,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a workflow step from a pattern match."""
        groups = match.groups()
        
        if action == "segment":
            concept = groups[0] if groups else "object"
            return {
                "name": f"segment_{concept}",
                "action": "segment",
                "params": {
                    "image_url": context.get("image_url", ""),
                    "concept": concept,
                },
            }
        
        elif action == "track":
            concept = groups[0] if groups else "object"
            return {
                "name": f"track_{concept}",
                "action": "track",
                "params": {
                    "video_url": context.get("video_url", ""),
                    "concepts": [concept],
                },
            }
        
        elif action == "blur_faces":
            return {
                "name": "blur_faces",
                "action": "blur_faces",
                "params": {
                    "media_url": context.get("image_url") or context.get("video_url", ""),
                },
            }
        
        elif action == "reconstruct":
            return {
                "name": "reconstruct",
                "action": "reconstruct_object",
                "params": {
                    "image_url": context.get("image_url", ""),
                    "mask_url": context.get("mask_url", ""),
                },
            }
        
        return None
    
    def _plan_from_keywords(
        self,
        instruction: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Fallback planning based on keywords."""
        steps = []
        
        # Detect concepts mentioned
        common_concepts = [
            "person", "people", "face", "car", "vehicle", "dog", "cat",
            "forklift", "pallet", "box", "product", "building", "tree"
        ]
        
        detected_concepts = [c for c in common_concepts if c in instruction]
        
        if detected_concepts:
            # Determine if image or video
            if "video" in instruction or context.get("video_url"):
                steps.append({
                    "name": "track_objects",
                    "action": "track",
                    "params": {
                        "video_url": context.get("video_url", ""),
                        "concepts": detected_concepts,
                    },
                })
            else:
                for concept in detected_concepts:
                    steps.append({
                        "name": f"segment_{concept}",
                        "action": "segment",
                        "params": {
                            "image_url": context.get("image_url", ""),
                            "concept": concept,
                        },
                    })
        
        return steps
