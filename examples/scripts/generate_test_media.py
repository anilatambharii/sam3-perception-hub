#!/usr/bin/env python3
"""
Generate synthetic test media files for demos.

Creates sample videos and images for testing SAM3 perception demos.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_sample_video(output_path: Path, duration: int = 5, fps: int = 30):
    """Create a sample video with moving objects."""
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating video: {output_path}")
    
    for frame_num in range(fps * duration):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add moving rectangles (simulating people/objects)
        num_objects = 3
        for i in range(num_objects):
            # Object moves across screen
            x_pos = int((frame_num * 5 + i * 300) % (width + 200)) - 100
            y_pos = 200 + i * 150
            
            # Draw rectangle (person/object)
            color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][i]
            cv2.rectangle(frame, 
                         (x_pos, y_pos), 
                         (x_pos + 80, y_pos + 120), 
                         color, -1)
            
            # Add simple "head"
            cv2.circle(frame, (x_pos + 40, y_pos - 20), 20, color, -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Created {output_path} ({duration}s @ {fps}fps)")


def create_sample_image(output_path: Path, width: int = 1280, height: int = 720):
    """Create a sample image with objects."""
    
    # Create image
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw some objects
    objects = [
        # Box (simulating warehouse package)
        {"type": "rect", "coords": (100, 200, 300, 400), "color": (200, 150, 100)},
        {"type": "rect", "coords": (400, 150, 600, 350), "color": (150, 200, 150)},
        {"type": "rect", "coords": (700, 250, 900, 450), "color": (150, 150, 200)},
        
        # Circles (simulating people heads)
        {"type": "circle", "coords": (200, 100, 50), "color": (255, 200, 200)},
        {"type": "circle", "coords": (500, 100, 50), "color": (200, 255, 200)},
    ]
    
    for obj in objects:
        if obj["type"] == "rect":
            draw.rectangle(obj["coords"], fill=obj["color"], outline=(0, 0, 0), width=2)
        elif obj["type"] == "circle":
            x, y, r = obj["coords"]
            draw.ellipse([x-r, y-r, x+r, y+r], fill=obj["color"], outline=(0, 0, 0), width=2)
    
    # Add text
    draw.text((20, 20), "Sample Test Image", fill=(0, 0, 0))
    
    img.save(output_path)
    print(f"✓ Created {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate test media files")
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=Path("examples/media"),
        help="Directory to save media files",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Video duration in seconds",
    )
    args = parser.parse_args()
    
    # Create directories
    args.media_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating test media files...\n")
    
    # Generate videos for each demo
    videos = {
        "sample_retail.mp4": "Privacy-preserving analytics demo",
        "sample_warehouse.mp4": "Warehouse analytics demo",
        "sample_object.mp4": "AR reconstruction demo",
        "sample_content.mp4": "Content production demo",
    }
    
    for video_name, description in videos.items():
        print(f"[{description}]")
        create_sample_video(
            args.media_dir / video_name,
            duration=args.duration,
        )
        print()
    
    # Generate sample images
    images = {
        "sample_image.jpg": "General test image",
        "sample_object.jpg": "Object for 3D reconstruction",
    }
    
    for img_name, description in images.items():
        print(f"[{description}]")
        create_sample_image(args.media_dir / img_name)
        print()
    
    print(f"\n✓ All test media files created in {args.media_dir}")
    print(f"\nYou can now run demos:")
    print(f"  dev.bat demo-privacy")
    print(f"  dev.bat demo-warehouse")
    print(f"  dev.bat demo-ar")
    print(f"  dev.bat demo-content")


if __name__ == "__main__":
    main()
