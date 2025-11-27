#!/usr/bin/env python3
"""
Download SAM3 and SAM3D model checkpoints.

This script downloads the required model files from official sources.

Usage:
    python scripts/download_models.py [--models-dir MODELS_DIR]
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Model registry with download URLs and checksums
MODELS: Dict[str, Dict] = {
    "sam3_hiera_large": {
        "url": "https://huggingface.co/facebook/sam3/resolve/main/sam3_hiera_large.pt",
        "size_mb": 2400,
        "sha256": None,  # Add checksum when available
        "description": "SAM 3 Large model (Hiera backbone)",
    },
    "sam3_hiera_base": {
        "url": "https://huggingface.co/facebook/sam3/resolve/main/sam3_hiera_base.pt",
        "size_mb": 800,
        "sha256": None,
        "description": "SAM 3 Base model (Hiera backbone)",
    },
    "sam3d_objects": {
        "url": "https://huggingface.co/facebook/sam3d/resolve/main/sam3d_objects_v1.pt",
        "size_mb": 1200,
        "sha256": None,
        "description": "SAM 3D Objects model",
    },
    "sam3d_body": {
        "url": "https://huggingface.co/facebook/sam3d/resolve/main/sam3d_body_v1.pt",
        "size_mb": 1000,
        "sha256": None,
        "description": "SAM 3D Body model",
    },
}


def verify_checksum(file_path: Path, expected_sha256: Optional[str]) -> bool:
    """Verify file checksum."""
    if expected_sha256 is None:
        return True
    
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest() == expected_sha256


async def download_model(
    name: str,
    model_info: Dict,
    models_dir: Path,
    progress: Progress,
) -> bool:
    """Download a single model."""
    
    output_path = models_dir / f"{name}.pt"
    
    # Skip if already exists
    if output_path.exists():
        if verify_checksum(output_path, model_info.get("sha256")):
            console.print(f"  [green]✓[/green] {name} already downloaded")
            return True
        else:
            console.print(f"  [yellow]![/yellow] {name} checksum mismatch, re-downloading")
            output_path.unlink()
    
    task = progress.add_task(
        f"[cyan]Downloading {name}...",
        total=model_info["size_mb"],
    )
    
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", model_info["url"]) as response:
                response.raise_for_status()
                
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(
                            task,
                            completed=downloaded / (1024 * 1024),
                        )
        
        # Verify checksum
        if not verify_checksum(output_path, model_info.get("sha256")):
            console.print(f"  [red]✗[/red] {name} checksum verification failed")
            output_path.unlink()
            return False
        
        console.print(f"  [green]✓[/green] {name} downloaded successfully")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to download {name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


async def download_all_models(models_dir: Path, models: list = None):
    """Download all required models."""
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold]SAM3 Perception Hub - Model Downloader[/bold]\n")
    console.print(f"Models directory: {models_dir}\n")
    
    # Filter models if specified
    to_download = MODELS
    if models:
        to_download = {k: v for k, v in MODELS.items() if k in models}
    
    console.print("Models to download:")
    for name, info in to_download.items():
        console.print(f"  • {name}: {info['description']} (~{info['size_mb']} MB)")
    console.print()
    
    # Download with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        
        results = []
        for name, info in to_download.items():
            success = await download_model(name, info, models_dir, progress)
            results.append((name, success))
    
    # Summary
    console.print("\n[bold]Download Summary:[/bold]")
    success_count = sum(1 for _, success in results if success)
    console.print(f"  {success_count}/{len(results)} models ready")
    
    if success_count < len(results):
        console.print("\n[yellow]Some models failed to download. They may not be available yet.[/yellow]")
        console.print("You can use cloud inference (FAL/Replicate) as an alternative.")
    
    return success_count == len(results)


def main():
    parser = argparse.ArgumentParser(description="Download SAM3 model checkpoints")
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Specific models to download (default: all)",
    )
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(download_all_models(
        Path(args.models_dir),
        args.models,
    ))


if __name__ == "__main__":
    main()
