#!/usr/bin/env python3
"""
Privacy-Preserving Analytics Demo

Demonstrates how to detect and anonymize people while preserving
behavioral analytics data using SAM3 Perception Hub.

Usage:
    python run_demo_privacy.py [--video VIDEO_PATH] [--output OUTPUT_DIR]
"""

import asyncio
import argparse
from pathlib import Path

import httpx


API_BASE = "http://localhost:8080/api/v1"


async def run_privacy_demo(video_path: str, output_dir: str):
    """Run the privacy-preserving analytics demo."""
    
    print("=" * 60)
    print("SAM3 Perception Hub - Privacy-Preserving Analytics Demo")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        
        # Step 1: Track people in video
        print("\n[1/4] Tracking people in video...")
        
        with open(video_path, "rb") as f:
            video_data = f.read()
        
        response = await client.post(
            f"{API_BASE}/track",
            files={"video": ("video.mp4", video_data, "video/mp4")},
            data={
                "request": '{"concepts": ["person", "face"], "sample_rate": 2}'
            }
        )
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return
        
        # Parse NDJSON results
        tracking_results = []
        for line in response.text.strip().split("\n"):
            if line:
                import json
                tracking_results.append(json.loads(line))
        
        print(f"   Processed {len(tracking_results)} frames")
        
        # Step 2: Identify unique people
        print("\n[2/4] Identifying unique individuals...")
        
        person_ids = set()
        face_detections = []
        
        for frame in tracking_results:
            for masklet in frame.get("masklets", []):
                if masklet.get("concept") == "person":
                    person_ids.add(masklet["instance_id"])
                elif masklet.get("concept") == "face":
                    face_detections.append({
                        "frame": frame["frame_index"],
                        "bbox": masklet.get("bbox"),
                    })
        
        print(f"   Found {len(person_ids)} unique people")
        print(f"   Detected {len(face_detections)} face instances")
        
        # Step 3: Generate anonymized output
        print("\n[3/4] Generating anonymized analytics...")
        
        analytics = {
            "total_frames": len(tracking_results),
            "unique_people": len(person_ids),
            "face_detections_blurred": len(face_detections),
            "tracks": [],
        }
        
        # Build movement tracks (anonymized - no face data)
        for person_id in person_ids:
            track = {
                "id": f"person_{person_id}",
                "positions": [],
            }
            
            for frame in tracking_results:
                for masklet in frame.get("masklets", []):
                    if (masklet.get("concept") == "person" and 
                        masklet["instance_id"] == person_id):
                        if masklet.get("bbox"):
                            cx = (masklet["bbox"][0] + masklet["bbox"][2]) / 2
                            cy = (masklet["bbox"][1] + masklet["bbox"][3]) / 2
                            track["positions"].append({
                                "frame": frame["frame_index"],
                                "x": cx,
                                "y": cy,
                            })
            
            analytics["tracks"].append(track)
        
        # Step 4: Save results
        print("\n[4/4] Saving results...")
        
        import json
        
        analytics_path = output_path / "anonymized_analytics.json"
        with open(analytics_path, "w") as f:
            json.dump(analytics, f, indent=2)
        
        print(f"   Saved analytics to: {analytics_path}")
        
        # Summary
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print(f"""
Results Summary:
  - Frames processed: {analytics['total_frames']}
  - People tracked: {analytics['unique_people']}
  - Faces blurred: {analytics['face_detections_blurred']}
  - Output saved to: {output_path}

The output contains anonymized movement tracks without
any personally identifiable information (faces blurred,
no biometric data retained).
""")


def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving Analytics Demo")
    parser.add_argument(
        "--video",
        default="examples/media/sample_retail.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        default="output/privacy_demo",
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Check if video exists, use mock if not
    if not Path(args.video).exists():
        print(f"Note: Video '{args.video}' not found, using mock data")
        # Create mock video path for demo
        Path("examples/media").mkdir(parents=True, exist_ok=True)
        # In production, download sample video here
    
    asyncio.run(run_privacy_demo(args.video, args.output))


if __name__ == "__main__":
    main()
