# Contributing to SAM3 Perception Hub

Thank you for your interest in contributing! This document provides guidelines
and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/sam3-perception-hub.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `make test`
6. Submit a pull request

## Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Run linting
make lint
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and small
- Write tests for new features

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the CHANGELOG if applicable
5. Request review from maintainers

## Areas for Contribution

- **New Enterprise Recipes**: Robotics, autonomous driving, industrial inspection
- **Data Platform Connectors**: Snowflake, Databricks, Flink integration
- **MLOps Integrations**: MLflow, Weights & Biases, Kubeflow support
- **Documentation**: Tutorials, examples, translations
- **Performance**: Optimization, benchmarking, profiling

## Code of Conduct

Be respectful and inclusive. We welcome contributors from all backgrounds.

## Questions?

Open an issue or reach out to the maintainers.
