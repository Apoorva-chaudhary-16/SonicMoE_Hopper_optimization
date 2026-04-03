FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Python 3.12 + dev headers + system deps
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    ninja-build \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install PyTorch 2.9.1 + CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install SonicMoE dependencies
RUN pip install --no-cache-dir \
    nvidia-cutlass-dsl==4.4.0 \
    quack-kernels==0.2.5 \
    ninja \
    pytest \
    parameterized \
    rich \
    "cuda-python<13.0.0"

WORKDIR /workspace/sonicmoe

# Create entrypoint script that sets environment variables
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'export TORCH_CUDA_ARCH_LIST="9.0"' >> /entrypoint.sh && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> /entrypoint.sh && \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"' >> /entrypoint.sh && \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"' >> /entrypoint.sh && \
    echo 'echo "Inside Docker container"' >> /entrypoint.sh && \
    echo 'echo "Python: $(python --version)"' >> /entrypoint.sh && \
    echo 'echo "CUDA: $(nvcc --version | grep release)"' >> /entrypoint.sh && \
    echo 'echo ""' >> /entrypoint.sh && \
    echo 'echo "Python headers: $(ls /usr/include/python3.12/Python.h 2>/dev/null || echo NOT FOUND)"' >> /entrypoint.sh && \
    echo 'echo ""' >> /entrypoint.sh && \
    echo 'if [ -f "/workspace/sonicmoe/pyproject.toml" ]; then' >> /entrypoint.sh && \
    echo '    echo "Installing SonicMoE..."' >> /entrypoint.sh && \
    echo '    cd /workspace/sonicmoe && pip install -e . --no-deps 2>&1 | tail -3' >> /entrypoint.sh && \
    echo '    echo ""' >> /entrypoint.sh && \
    echo '    if [ -f "/workspace/sonicmoe/docker_test.py" ]; then' >> /entrypoint.sh && \
    echo '        echo "Running tests..."' >> /entrypoint.sh && \
    echo '        python /workspace/sonicmoe/docker_test.py' >> /entrypoint.sh && \
    echo '    fi' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
