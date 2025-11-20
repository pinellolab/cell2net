# Cell2Net Installation Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model training)
- At least 16GB RAM for large datasets

### Install from PyPI (Recommended)

```bash
pip install cell2net
```

### Install from GitHub (Latest Development Version)

```bash
pip install git+https://github.com/pinellolab/cell2net.git
```

### Install in Development Mode

For contributors or users wanting to modify the source code:

```bash
git clone https://github.com/pinellolab/cell2net.git
cd cell2net
pip install -e .
```

### Optional Dependencies

For enhanced functionality, install additional packages:

```bash
# For advanced visualization
pip install plotly kaleido

# For motif analysis
pip install logomaker

# For GPU acceleration (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
