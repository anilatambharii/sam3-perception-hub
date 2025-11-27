#!/usr/bin/env python3
"""
Download SAM3 and SAM3D model checkpoints.

This script downloads the required model files from official sources.

Usage:
    python scripts/download_models.py [--models-dir MODELS_DIR]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console

console = Console()

# Model registry with HuggingFace repo IDs
MODELS: Dict[str, Dict] = {
    "sam3": {
        "repo_id": "facebook/sam3",
        "files": ["sam3.pt"],
        "size_mb": 3450,
        "description": "SAM 3 unified model (Hiera backbone)",
    },
    "sam3d_objects": {
        "repo_id": "facebook/sam-3d-objects",
        "files": None,  # Download entire repo
        "size_mb": 1200,
        "description": "SAM 3D Objects model",
    },
    "sam3d_body": {
        "repo_id": "facebook/sam-3d-body-dinov3",
        "files": None,  # Download entire repo
        "size_mb": 1000,
        "description": "SAM 3D Body model (DINOv3)",
    },
}


def get_hf_command():
    """Get the correct HF CLI command for this platform."""
    # Try different ways to invoke hf CLI
    commands_to_try = [
        ["hf"],  # Direct command
        ["python", "-m", "huggingface_hub.cli"],  # Python module
        [str(Path(sys.executable).parent / "hf.exe")],  # Direct path to exe
    ]
    
    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return None


def check_authentication(hf_cmd):
    """Check if authenticated with HuggingFace."""
    try:
        result = subprocess.run(
            hf_cmd + ["auth", "whoami"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def download_model(
    name: str,
    model_info: Dict,
    models_dir: Path,
    hf_cmd: list,
) -> bool:
    """Download a single model using HuggingFace CLI."""
    
    model_output_dir = models_dir / name
    
    # Check if already exists
    if model_output_dir.exists():
        if model_info["files"]:
            # Check if specific files exist
            all_exist = all((model_output_dir / f).exists() for f in model_info["files"])
            if all_exist:
                console.print(f"  [green]✓[/green] {name} already downloaded")
                return True
        else:
            # Directory exists, assume complete
            console.print(f"  [green]✓[/green] {name} already downloaded")
            return True
    
    try:
        console.print(f"  [cyan]Downloading {name}...[/cyan]")
        
        # Build hf download command
        cmd = hf_cmd + ["download", model_info["repo_id"], "--local-dir", str(model_output_dir)]
        
        # If specific files, add them
        if model_info["files"]:
            cmd.extend(model_info["files"])
        
        # Run download
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            console.print(f"  [green]✓[/green] {name} downloaded successfully")
            return True
        else:
            console.print(f"  [red]✗[/red] Failed to download {name}")
            if result.stderr:
                console.print(f"    Error: {result.stderr.strip()}")
            return False
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to download {name}: {e}")
        return False


def download_all_models(models_dir: Path, models: Optional[list] = None):
    """Download all required models."""
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold]SAM3 Perception Hub - Model Downloader[/bold]\n")
    console.print(f"Models directory: {models_dir}\n")
    
    # Check HF CLI
    hf_cmd = get_hf_command()
    if not hf_cmd:
        console.print("[red]Error: HuggingFace CLI not found![/red]")
        console.print("Please ensure you ran: pip install 'huggingface_hub[cli]'")
        console.print("\nTry running: python -m huggingface_hub.cli auth whoami")
        sys.exit(1)
    
    console.print(f"[dim]Using HF command: {' '.join(hf_cmd)}[/dim]\n")
    
    # Check authentication
    if not check_authentication(hf_cmd):
        console.print("[yellow]Warning: Not logged in to HuggingFace[/yellow]")
        console.print("Run: python -m huggingface_hub.cli auth login")
        console.print("These models require authentication.\n")
    
    # Filter models if specified
    to_download = MODELS
    if models:
        to_download = {k: v for k, v in MODELS.items() if k in models}
    
    console.print("Models to download:")
    for name, info in to_download.items():
        console.print(f"  • {name}: {info['description']} (~{info['size_mb']} MB)")
    console.print()
    
    # Download models
    results = []
    for name, info in to_download.items():
        success = download_model(name, info, models_dir, hf_cmd)
        results.append((name, success))
    
    # Summary
    console.print("\n[bold]Download Summary:[/bold]")
    success_count = sum(1 for _, success in results if success)
    console.print(f"  {success_count}/{len(results)} models ready")
    
    if success_count < len(results):
        console.print("\n[yellow]Some models failed to download.[/yellow]")
        console.print("Common issues:")
        console.print("  1. Not logged in: run 'python -m huggingface_hub.cli auth login'")
        console.print("  2. No access granted: visit model pages and request access")
        console.print("  3. Network issues: check your connection")
        console.print("\nYou can use cloud inference (FAL/Replicate) as an alternative.")
    
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
    
    download_all_models(
        Path(args.models_dir),
        args.models,
    )


if __name__ == "__main__":
    main()
