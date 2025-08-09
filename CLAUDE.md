# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gsplat is an open-source library for CUDA-accelerated rasterization of Gaussian splats with Python bindings. It provides fast, memory-efficient 3D Gaussian splatting for real-time rendering of radiance fields. The project includes both 3D Gaussian Splatting (3DGS) and 2D Gaussian Splatting (2DGS) implementations.

## Development Commands

### Installation for Development
```bash
# For CUDA development (recommended - uses JIT compilation)
BUILD_NO_CUDA=1 pip install -e .[dev]

# Standard development install (compiles CUDA during install)
pip install -e .[dev]

# Install from source with submodules
git clone --recurse-submodules URL
```

### Code Formatting and Linting
```bash
# Format Python code (required before commits)
black . gsplat/ tests/ examples/ profiling/

# Format C++/CUDA code
./formatter.sh

# Check formatting without making changes
black . gsplat/ tests/ examples/ profiling/ --check
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_basic.py

# Note: GPU tests require CUDA and won't run in CI
```

### Documentation
```bash
# Install doc dependencies
pip install -r docs/requirements.txt

# Build documentation locally
sphinx-build docs/source _build
```

### Example Usage
```bash
cd examples
pip install -r requirements.txt

# Download benchmark data
python datasets/download_dataset.py

# Run basic evaluation
bash benchmarks/basic.sh
```

## Architecture Overview

### Core Components

**CUDA Backend (`gsplat/cuda/`)**
- `csrc/`: C++/CUDA kernels for rasterization, projection, and Gaussian operations
- `_wrapper.py`: Python bindings to CUDA functions
- `_torch_impl.py`: PyTorch implementations for 3DGS operations
- `_torch_impl_2dgs.py`: PyTorch implementations for 2DGS operations

**Rendering Pipeline (`gsplat/rendering.py`)**
- `rasterization()`: Main 3DGS rendering function
- `rasterization_2dgs()`: 2DGS rendering function
- Supports batched rendering over multiple scenes/viewpoints

**Training Strategies (`gsplat/strategy/`)**
- `DefaultStrategy`: Standard 3DGS densification strategy
- `MCMCStrategy`: MCMC-based optimization strategy
- `Strategy`: Base class for custom training strategies

**Key Modules**
- `compression/`: PNG compression utilities for Gaussians
- `optimizers/`: Selective Adam optimizer for sparse parameters
- `distributed.py`: Multi-GPU training support
- `utils.py`: Utility functions for depth, normals, projections

### Data Flow

1. **Input**: Gaussian parameters (means, quaternions, scales, opacities, colors)
2. **Projection**: Transform 3D Gaussians to screen space
3. **Rasterization**: Render pixels using splatted Gaussians
4. **Strategy**: Update Gaussian parameters based on gradients and training strategy

### Key Conventions

- Quaternions are stored as [w, x, y, z] format
- All tensors support batching with `[..., N, ...]` dimensions
- CUDA kernels use JIT compilation for development (stored in `~/.cache/torch_extensions/`)
- Both packed and unpacked tensor formats are supported for memory efficiency

### Testing Patterns

Tests are organized by functionality:
- `test_basic.py`: Core functionality tests
- `test_2dgs.py`: 2D Gaussian Splatting tests
- `test_rasterization.py`: Rendering pipeline tests
- `test_strategy.py`: Training strategy tests

All test functions must be named with `test_*` prefix for pytest discovery.