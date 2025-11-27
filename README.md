# 🎯 SAM3 Perception Hub

<div align="center">

**Open-source reference stack for building AI-native products on top of SAM 3 + SAM 3D**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [API Reference](#-api-reference) • [Enterprise Workflows](#-enterprise-workflows) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

**SAM3 Perception Hub** provides enterprises with a production-ready blueprint to integrate Meta's revolutionary [Segment Anything Model 3 (SAM 3)](https://ai.meta.com/sam3/) and [SAM 3D](https://ai.meta.com/sam3d/) into real-world AI products. This isn't just a model wrapper—it's a full-stack reference platform designed by engineers, for engineers.

### What SAM 3 + SAM 3D Enable

| Capability | Description |
|------------|-------------|
| **Open-Vocabulary Segmentation** | Segment anything with text prompts—no training required |
| **Video Object Tracking** | Stable mask tracking with consistent IDs across frames |
| **3D Object Reconstruction** | Lift 2D segments into textured 3D meshes (GLB/OBJ) |
| **3D Body Estimation** | Generate 3D body meshes with pose parameters from single images |
| **Promptable Concept Segmentation (PCS)** | Detect concepts by name across entire videos |
| **Promptable Visual Segmentation (PVS)** | Click/box/mask prompts for precise selection |

### Why This Project?

- 🏢 **Enterprise-Ready**: Production patterns for batch + streaming pipelines
- 🔐 **Privacy-First**: Built-in face blurring, audit logging, and redaction hooks
- 🤖 **Agent-Native**: LLM tool server for AI agent orchestration
- 📊 **Observable**: Metrics, tracing, and structured logging throughout
- 🔌 **Extensible**: Clean interfaces to add custom detectors and integrations

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SAM3 PERCEPTION HUB                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   Playground    │  │   Enterprise    │  │   LLM Agents    │            │
│  │      (UI)       │  │   Applications  │  │   & Workflows   │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                      │
│           └────────────────────┼────────────────────┘                      │
│                                │                                           │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                        PERCEPTION API (gRPC/REST)                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │     PCS      │  │     PVS      │  │   Tracking   │                │ │
│  │  │  Text-based  │  │ Visual-based │  │   Service    │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                │                                           │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                     RECONSTRUCTION API (gRPC/REST)                    │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                  │ │
│  │  │   SAM 3D Objects     │  │    SAM 3D Body       │                  │ │
│  │  │   2D → 3D Mesh       │  │  Person → Body Mesh  │                  │ │
│  │  └──────────────────────┘  └──────────────────────┘                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                │                                           │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                         AGENT BRIDGE                                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │ Tool Server  │  │   Workflow   │  │   Planning   │                │ │
│  │  │   (MCP)      │  │ Orchestrator │  │    Engine    │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      INFRASTRUCTURE LAYER                             │ │
│  │  Redis (Cache) • Kafka (Events) • PostgreSQL (Metadata) • S3 (Assets)│ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+ (for local inference)
- 16GB+ GPU VRAM recommended

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/sam3-perception-hub.git
cd sam3-perception-hub

# Copy environment template
cp .env.example .env

# Start all services
docker compose up -d

# Open the playground
open http://localhost:3000
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Download model checkpoints
python scripts/download_models.py

# Start services
make dev
```

### Option 3: Cloud Inference (No GPU Required)

```bash
# Use hosted inference providers
export SAM3_INFERENCE_PROVIDER=fal
export FAL_API_KEY=your_key_here

docker compose -f docker-compose.cloud.yml up -d
```

---

## 📚 API Reference

### Perception API

#### Segment with Text Prompt (PCS)

```python
from sam3_perception_hub import PerceptionClient

client = PerceptionClient("http://localhost:8080")

# Segment by concept name
result = client.segment(
    image="warehouse.jpg",
    query=ConceptQuery(
        text="forklift",
        confidence_threshold=0.7
    )
)

for instance in result.instances:
    print(f"Found {instance.concept} with confidence {instance.confidence}")
    mask_rle = instance.mask.to_rle()
```

#### Video Tracking

```python
# Track objects across video frames
tracker = client.create_tracker(
    video="warehouse_feed.mp4",
    concepts=["person", "forklift", "pallet"]
)

for frame_result in tracker.stream():
    for masklet in frame_result.masklets:
        print(f"Frame {frame_result.frame_idx}: {masklet.concept} ID={masklet.instance_id}")
```

### Reconstruction API

#### 3D Object Reconstruction

```python
from sam3_perception_hub import ReconstructionClient

recon = ReconstructionClient("http://localhost:8081")

# Reconstruct object from image + mask
mesh = recon.reconstruct_object(
    image="chair.jpg",
    mask=result.instances[0].mask,
    output_format="glb"
)
mesh.save("chair_3d.glb")
```

---

## 🏭 Enterprise Workflows

### 1. Privacy-Preserving Analytics

```bash
make demo-privacy
```

### 2. Retail & Warehouse Analytics

```bash
make demo-warehouse
```

### 3. AR/UX Prototyping

```bash
make demo-ar
```

### 4. Content Production

```bash
make demo-content
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the AI community**

</div>
