```markdown
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
│  │      (UI)       │  │   Applications  │  │   \& Workflows   │            │
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
│  │  Redis (Cache) -  Kafka (Events) -  PostgreSQL (Metadata) -  S3 (Assets)│ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested with 3.11)
- **NVIDIA GPU with CUDA 12.1+** (for local inference) - 16GB+ VRAM recommended
- **Hugging Face account** with access to gated models
- **Windows 10/11** or **Linux/macOS**

### Setup Instructions

#### 1. Clone and Setup Environment

```


# Clone the repository

git clone https://github.com/anilatambharii/sam3-perception-hub.git
cd sam3-perception-hub

# Create virtual environment

python -m venv .venv

# Activate virtual environment

# Windows:

.venv\Scripts\activate

# Linux/Mac:

source .venv/bin/activate

```

#### 2. Install Dependencies

```


# Install core dependencies

pip install -e ".[dev]"

# Install additional required packages

pip install opencv-python pillow
pip install opentelemetry-exporter-otlp-proto-grpc opentelemetry-instrumentation-fastapi

```

#### 3. Authenticate with Hugging Face

SAM3 models are gated and require access approval:

```


# Install HuggingFace CLI

pip install "huggingface_hub[cli]"

# Login to HuggingFace

hf auth login

# Or on Windows if 'hf' is not found:

python -m huggingface_hub.cli auth login

# Verify authentication

hf auth whoami

```

**Request model access** (required before downloading):
1. Visit [facebook/sam3](https://huggingface.co/facebook/sam3)
2. Visit [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)
3. Visit [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
4. Click "Access repository" and agree to the terms
5. Wait for approval (typically < 1 hour)

#### 4. Download Models

```


# Download SAM3 and SAM3D model checkpoints

python scripts/download_models.py

# Verify downloads

# Windows:

dir models /s

# Linux/Mac:

ls -lR models/

```

**Expected downloads:**
- `models/sam3/sam3.pt` (~3.45 GB)
- `models/sam3d_objects/` (full checkpoint structure)
- `models/sam3d_body/` (full checkpoint structure)

#### 5. Generate Test Media (Optional)

```


# Generate synthetic test videos and images

python scripts/generate_test_media.py

# This creates:

# - examples/media/sample_retail.mp4

# - examples/media/sample_warehouse.mp4

# - examples/media/sample_object.mp4

# - examples/media/sample_image.jpg

```

#### 6. Start Development Servers

**Windows:**
```


# Option 1: Start all services (opens separate windows)

dev.bat dev

# Option 2: Start API services only

dev.bat dev-api

# Option 3: Start UI only

dev.bat dev-ui

```

**Linux/Mac:**
```


# Start all services

make dev

# Or start API services only

make dev-api

```

This starts:
- **Perception API**: http://localhost:8080
- **Reconstruction API**: http://localhost:8081
- **Agent Bridge**: http://localhost:8082
- **Playground UI**: http://localhost:3000

#### 7. Run Tests

```


# Windows:

dev.bat test-unit    \# Unit tests only (faster)
dev.bat test         \# All tests including integration

# Linux/Mac:

make test-unit
make test

```

---

## 🎬 Running Demos

### Privacy-Preserving Analytics Demo

Demonstrates person tracking with automatic face blurring for privacy compliance.

```


# 1. Start API servers (in Terminal 1)

# Windows:

dev.bat dev-api

# Linux/Mac:

make dev-api

# 2. Run demo (in Terminal 2, after APIs are running)

# Windows:

python examples\scripts\run_demo_privacy.py --video examples\media\sample_retail.mp4

# Linux/Mac:

python examples/scripts/run_demo_privacy.py --video examples/media/sample_retail.mp4

```

**What it does:**
- Tracks people across video frames
- Applies face blurring for privacy
- Generates analytics with redacted footage
- Outputs results to `outputs/privacy_demo/`

---

## 📚 API Reference

### Perception API

#### Health Check

```

curl http://localhost:8080/health

```

#### Segment with Text Prompt (PCS)

```

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

```


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

```

from sam3_perception_hub import ReconstructionClient

recon = ReconstructionClient("http://localhost:8081")

# Reconstruct object from image + mask

mesh = recon.reconstruct_object(
image="chair.jpg",
mask=result.instances.mask,
output_format="glb"
)
mesh.save("chair_3d.glb")

```

---

## 🛠️ Development Commands

### Windows (using dev.bat)

```

dev.bat help              \# Show all available commands
dev.bat install           \# Install all dependencies
dev.bat download-models   \# Download model checkpoints
dev.bat dev               \# Start all services
dev.bat dev-api           \# Start API servers only
dev.bat dev-ui            \# Start UI only
dev.bat test              \# Run all tests
dev.bat test-unit         \# Run unit tests only
dev.bat lint              \# Run linters
dev.bat format            \# Format code
dev.bat clean             \# Clean build artifacts
dev.bat demo-privacy      \# Run privacy demo

```

### Linux/Mac (using Makefile)

```

make help                 \# Show all available commands
make install              \# Install all dependencies
make download-models      \# Download model checkpoints
make dev                  \# Start all services
make test                 \# Run all tests
make lint                 \# Run linters
make format               \# Format code
make clean                \# Clean build artifacts
make demo-privacy         \# Run privacy demo

```

---

## 🐛 Troubleshooting

### Models won't download (401 Unauthorized)

**Solution:** Request access to gated models on Hugging Face:
1. Visit the model pages (links in step 3 above)
2. Click "Access repository" and agree to terms
3. Wait for approval (typically < 1 hour)
4. Re-run `python scripts/download_models.py`

### API returns 404 errors in demos

**Solution:** Ensure API servers are running before running demos:
```


# Start APIs first

dev.bat dev-api  \# Windows
make dev-api     \# Linux/Mac

# Then run demo in separate terminal

python examples\scripts\run_demo_privacy.py

```

### Import errors (ModuleNotFoundError)

**Solution:** Install missing dependencies:
```

pip install opentelemetry-exporter-otlp-proto-grpc
pip install opentelemetry-instrumentation-fastapi
pip install opencv-python pillow

```

### `hf` command not found (Windows)

**Solution:** Use Python module form:
```

python -m huggingface_hub.cli auth login
python -m huggingface_hub.cli auth whoami

```

### `make` command not found (Windows)

**Solution:** Use `dev.bat` instead:
```

dev.bat dev     \# instead of make dev
dev.bat test    \# instead of make test

```

---

## 🏭 Production Deployment

### Docker Compose (Recommended)

```


# Copy environment template

cp .env.example .env

# Configure production settings in .env

# - Set INFERENCE_PROVIDER (local/fal/replicate)

# - Add API keys if using cloud providers

# Start all services

docker compose up -d

# View logs

docker compose logs -f

# Stop services

docker compose down

```

### Cloud Inference (No GPU Required)

Use hosted inference providers for production without GPU infrastructure:

```


# Set environment variables

export SAM3_INFERENCE_PROVIDER=fal
export FAL_API_KEY=your_key_here

# Start with cloud config

docker compose -f docker-compose.cloud.yml up -d

```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests: `dev.bat test` or `make test`
5. Format code: `dev.bat format` or `make format`
6. Commit changes: `git commit -m "Add feature"`
7. Push to fork: `git push origin feature/your-feature`
8. Open a Pull Request

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Meta AI Research for [SAM 3](https://ai.meta.com/sam3/) and [SAM 3D](https://ai.meta.com/sam3d/)
- Hugging Face for model hosting infrastructure
- Open-source community for valuable feedback and contributions

---

<div align="center">

**Built with ❤️ for the AI community**

[Report Bug](https://github.com/anilatambharii/sam3-perception-hub/issues) • [Request Feature](https://github.com/anilatambharii/sam3-perception-hub/issues)

</div>
```

**Key changes made:**

1. ✅ **Added Windows-specific instructions** with `dev.bat` commands
2. ✅ **Detailed HuggingFace authentication** steps with gated model access
3. ✅ **Actual model download instructions** reflecting the real process
4. ✅ **Test media generation** section added
5. ✅ **Troubleshooting section** with common issues you encountered
6. ✅ **Corrected API startup** workflow (start APIs before demos)
7. ✅ **Removed non-existent demos** (warehouse, AR, content)
8. ✅ **Added missing dependencies** to install steps
9. ✅ **Platform-specific commands** clearly separated (Windows vs Linux/Mac)
10. ✅ **Realistic prerequisites** based on actual setup experience

This README now accurately reflects the steps needed to get the project running on Windows!

