.PHONY: help install dev test lint format build clean docker-build docker-up docker-down demo-privacy demo-warehouse demo-ar demo-content download-models

# Default target
help:
	@echo "SAM3 Perception Hub - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install all dependencies"
	@echo "  download-models  Download model checkpoints"
	@echo ""
	@echo "Development:"
	@echo "  dev              Start development servers"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  lint             Run linters"
	@echo "  format           Format code"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build all Docker images"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      View service logs"
	@echo ""
	@echo "Demos:"
	@echo "  demo-privacy     Run privacy-preserving analytics demo"
	@echo "  demo-warehouse   Run warehouse analytics demo"
	@echo "  demo-ar          Run AR reconstruction demo"
	@echo "  demo-content     Run content production demo"
	@echo ""
	@echo "Other:"
	@echo "  clean            Clean build artifacts"
	@echo "  docs             Build documentation"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e ".[dev]"
	cd ui/playground && npm install
	pre-commit install

download-models:
	python scripts/download_models.py

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting development servers..."
	@trap 'kill 0' INT; \
	(cd services/perception-api && uvicorn sam3_perception.main:app --reload --port 8080) & \
	(cd services/reconstruction-api && uvicorn sam3_reconstruction.main:app --reload --port 8081) & \
	(cd services/agent-bridge && uvicorn sam3_agent.main:app --reload --port 8082) & \
	(cd ui/playground && npm run dev) & \
	wait

dev-api:
	@trap 'kill 0' INT; \
	(cd services/perception-api && uvicorn sam3_perception.main:app --reload --port 8080) & \
	(cd services/reconstruction-api && uvicorn sam3_reconstruction.main:app --reload --port 8081) & \
	(cd services/agent-bridge && uvicorn sam3_agent.main:app --reload --port 8082) & \
	wait

dev-ui:
	cd ui/playground && npm run dev

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --cov=services --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --slow

benchmark:
	pytest tests/ --benchmark-only --benchmark-json=benchmark_results.json

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check services/ tests/
	mypy services/

format:
	ruff format services/ tests/
	ruff check --fix services/ tests/

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --remove-orphans
	docker system prune -f

# =============================================================================
# Demos
# =============================================================================

demo-privacy:
	@echo "Running Privacy-Preserving Analytics Demo..."
	python examples/scripts/run_demo_privacy.py

demo-warehouse:
	@echo "Running Warehouse Analytics Demo..."
	python examples/scripts/run_demo_warehouse.py

demo-ar:
	@echo "Running AR Reconstruction Demo..."
	python examples/scripts/run_demo_ar.py

demo-content:
	@echo "Running Content Production Demo..."
	python examples/scripts/run_demo_content.py

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Release
# =============================================================================

version:
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*
