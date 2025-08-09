# CUDA 12.4 + 開発ツール一式
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 環境
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV=/opt/venv

# ベース依存
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential ninja-build cmake \
    ffmpeg libgl1 libglib2.0-0 \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# venv 作成
RUN python3.10 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Python 基本ツール
RUN pip install --upgrade pip setuptools wheel packaging rich

# PyTorch (CUDA 12.4 対応の公式ホイール)
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.* torchvision==0.19.*

# ★ gsplat の CUDA 拡張が要求する GLM ヘッダ
#   → Torch 層の後に追加してレイヤ再利用を最大化
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglm-dev && \
    rm -rf /var/lib/apt/lists/*

# CUDA/ビルドの既定（必要なら compose の environment で上書き）
ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG MAX_JOBS=6
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

# C++/CUDA 拡張用ヘッダ（必要に応じて）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# Claude Code など用に Node.js (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# 任意: 開発補助ツール
RUN pip install pytest pre-commit ruff ipython

WORKDIR /workspace
CMD ["/bin/bash"]
