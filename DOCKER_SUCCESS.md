# Docker Setup - SUCCESS! ✅

## Issue Resolved

The `cudaErrorInsufficientDriver (error code: 35)` was caused by **environment variables not persisting** when restarting Docker containers.

### Root Cause
When you use `docker start -ai sonicmoe-dev`, Docker restarts an existing container but **doesn't re-run Dockerfile ENV commands**. The `TORCH_CUDA_ARCH_LIST` environment variable was lost.

### Solution
Created an **entrypoint script** (`/entrypoint.sh`) that sets environment variables every time the container starts:

```dockerfile
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
```

The entrypoint script exports:
- `TORCH_CUDA_ARCH_LIST="9.0"` (critical for Hopper H100)
- `CUDA_HOME=/usr/local/cuda`
- `PATH` and `LD_LIBRARY_PATH` with CUDA paths

## Test Results

All tests passed successfully! 🎉

```
======================================================================
  SUMMARY
======================================================================
  topk                : PASS
  cumsum              : PASS
  forward             : PASS
  correctness         : PASS

  ALL TESTS PASSED!

======================================================================
  Test 5: Performance Benchmark
======================================================================
  Config:   T=8192 H=4096 I=512 E=128 K=8
  Average:  3.17ms +/- 0.03ms
  Min:      3.12ms
  TFLOPS:   260.2
```

### Performance Highlights
- **260.2 TFLOPS** achieved on H100 NVL
- First run (with JIT compilation): 8.8s
- Second run (cached): 2.57ms
- Benchmark average: 3.17ms ± 0.03ms
- All correctness tests passed (max diff = 0.0)

## How to Use

### Quick Reference - Common Commands

```bash
# Start existing container interactively
docker start -ai sonicmoe-dev

# Start existing container in background
docker start sonicmoe-dev

# Stop running container
docker stop sonicmoe-dev

# Execute commands in running container
docker exec -it sonicmoe-dev python /workspace/sonicmoe/docker_test.py

# Open interactive shell in running container
docker exec -it sonicmoe-dev bash

# Check container status
docker ps -a | grep sonicmoe-dev

# View container logs
docker logs sonicmoe-dev

# Remove container (if you want to recreate it)
docker rm sonicmoe-dev

# Remove image (if you want to rebuild)
docker rmi sonicmoe-hopper
```

---

### First-Time Setup

#### 1. Build the image (one-time)
```bash
cd /home/gpu1/testing/SonicMoE_Hopper_optimization
docker build -t sonicmoe-hopper .
```

This creates a Docker **image** named `sonicmoe-hopper`. Think of it as a template.

#### 2. Create a container from the image
```bash
docker run -d --name sonicmoe-dev \
  --gpus all \
  -v "$(pwd):/workspace/sonicmoe" \
  sonicmoe-hopper \
  tail -f /dev/null
```

This creates a **container** named `sonicmoe-dev` from the image. Think of it as an instance.

**Explanation:**
- `-d` = detached mode (runs in background)
- `--name sonicmoe-dev` = name your container
- `--gpus all` = give container access to all GPUs
- `-v "$(pwd):/workspace/sonicmoe"` = mount current directory to /workspace/sonicmoe
- `tail -f /dev/null` = keeps container running

---

### Daily Usage

#### Start the existing container
```bash
# Interactive mode (you'll see output directly)
docker start -ai sonicmoe-dev

# Background mode (container runs in background)
docker start sonicmoe-dev
```

#### Run commands inside the container
```bash
# Execute a single command
docker exec -it sonicmoe-dev python /workspace/sonicmoe/docker_test.py

# Open an interactive shell
docker exec -it sonicmoe-dev bash
```

#### Stop the container when done
```bash
docker stop sonicmoe-dev
```

**Note:** Your container persists! You can start/stop it as many times as needed.

---

### Using the Container with Different Directories

You can create **multiple containers** from the same image, each mounting a different directory.

#### Example 1: Create a container for another project
```bash
# Navigate to different directory
cd /home/gpu1/testing/AnotherProject

# Create new container with different name and mount
docker run -d --name sonicmoe-project2 \
  --gpus all \
  -v "$(pwd):/workspace/sonicmoe" \
  sonicmoe-hopper \
  tail -f /dev/null

# Use it
docker exec -it sonicmoe-project2 bash
```

#### Example 2: Mount additional directories
```bash
# Create container with multiple mounts
docker run -d --name sonicmoe-multi \
  --gpus all \
  -v "/home/gpu1/testing/SonicMoE_Hopper_optimization:/workspace/sonicmoe" \
  -v "/home/gpu1/data:/workspace/data" \
  -v "/home/gpu1/models:/workspace/models" \
  sonicmoe-hopper \
  tail -f /dev/null
```

#### Example 3: Work with a different directory without recreating container
```bash
# Copy files into existing container
docker cp /home/gpu1/testing/NewProject/. sonicmoe-dev:/workspace/newproject/

# Or use the exec command to work with mounted directories
docker exec -it sonicmoe-dev bash
# Inside container: cd /workspace/newproject
```

---

### Container vs Image - What's the Difference?

| Concept | Analogy | Commands |
|---------|---------|----------|
| **Image** | Blueprint/Template | `docker build`, `docker images`, `docker rmi` |
| **Container** | Instance/Running copy | `docker run`, `docker start/stop`, `docker rm` |

- **One image** (sonicmoe-hopper) can create **many containers** (sonicmoe-dev, sonicmoe-project2, etc.)
- Each container is independent with its own mounted directories
- Containers persist until you explicitly remove them with `docker rm`

---

### Best Practices for Future Use

#### 1. List your containers
```bash
# See all containers (running and stopped)
docker ps -a

# See only running containers
docker ps
```

#### 2. Clean up old containers
```bash
# Remove a specific container
docker rm sonicmoe-dev

# Remove all stopped containers
docker container prune
```

#### 3. Update the image
```bash
# If you modify Dockerfile, rebuild the image
docker build -t sonicmoe-hopper .

# Remove old container and create new one
docker rm -f sonicmoe-dev
docker run -d --name sonicmoe-dev \
  --gpus all \
  -v "$(pwd):/workspace/sonicmoe" \
  sonicmoe-hopper \
  tail -f /dev/null
```

#### 4. Save and share your image
```bash
# Save image to file
docker save sonicmoe-hopper -o sonicmoe-hopper.tar

# Load image on another machine
docker load -i sonicmoe-hopper.tar
```

---

### Workflow Examples

#### Workflow 1: Development cycle
```bash
# Start container in background
docker start sonicmoe-dev

# Edit files on host (they're mounted, so changes appear in container)
nano /home/gpu1/testing/SonicMoE_Hopper_optimization/sonicmoe/moe.py

# Test changes in container
docker exec -it sonicmoe-dev python /workspace/sonicmoe/docker_test.py

# Stop when done
docker stop sonicmoe-dev
```

#### Workflow 2: Interactive debugging
```bash
# Start container and get shell
docker start sonicmoe-dev
docker exec -it sonicmoe-dev bash

# Inside container:
cd /workspace/sonicmoe
python -m pdb docker_test.py

# Exit container (container keeps running)
exit

# Stop container
docker stop sonicmoe-dev
```

#### Workflow 3: Multiple projects
```bash
# Project 1
docker start sonicmoe-dev
docker exec sonicmoe-dev python /workspace/sonicmoe/docker_test.py

# Project 2 (simultaneously)
docker start sonicmoe-project2
docker exec sonicmoe-project2 python /workspace/sonicmoe/train.py

# Both can run at same time!
```

---

### Quick Cheat Sheet

| Task | Command |
|------|---------|
| Start existing container (interactive) | `docker start -ai sonicmoe-dev` |
| Start existing container (background) | `docker start sonicmoe-dev` |
| Open shell in running container | `docker exec -it sonicmoe-dev bash` |
| Run command in container | `docker exec sonicmoe-dev <command>` |
| Stop container | `docker stop sonicmoe-dev` |
| Check container status | `docker ps -a` |
| View container logs | `docker logs sonicmoe-dev` |
| Remove container | `docker rm sonicmoe-dev` |
| Rebuild image | `docker build -t sonicmoe-hopper .` |
| Create new container | `docker run -d --name <name> --gpus all -v "$(pwd):/workspace/sonicmoe" sonicmoe-hopper tail -f /dev/null` |

## Environment Details

**Container:**
- Base: nvidia/cuda:12.6.3-devel-ubuntu22.04
- Python: 3.12.13
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- Triton: 3.5.1
- CUTLASS DSL: 4.4.0
- cuda-python: <13.0.0

**GPU:**
- Model: NVIDIA H100 NVL
- Architecture: Hopper (sm_90a)
- Memory: 93.11 GB

## Files Modified

1. **Dockerfile** - Added entrypoint script that preserves environment variables
2. **docker_test.py** - Comprehensive test suite with 5 tests
3. **setup_and_run_docker.sh** - Setup script (optional, manual steps work too)

## Next Steps

Now that the Docker environment works perfectly:

1. **Development**: Use the container for all SonicMoE development
2. **Optimization**: Tune kernel parameters to increase TFLOPS further
3. **Benchmarking**: Run production workloads with larger models
4. **Deployment**: Use this Docker image for reproducible deployments

## Troubleshooting

If you encounter issues:

1. **Check environment variables inside container:**
   ```bash
   docker exec sonicmoe-dev bash -c "echo \$TORCH_CUDA_ARCH_LIST"
   ```
   Should output: `9.0`

2. **Verify CUDA driver version:**
   ```bash
   nvidia-smi
   ```
   Should show driver ≥ 565.57.01

3. **Rebuild image if dependencies change:**
   ```bash
   docker build --no-cache -t sonicmoe-hopper .
   ```

## Success Metrics

✅ All 4 core tests passed  
✅ 260.2 TFLOPS performance on H100  
✅ Zero correctness errors (max diff = 0.0)  
✅ Consistent performance (std = 0.03ms)  
✅ Fast inference (2.57ms after warmup)  

**The optimized FA4 Hopper kernels are working perfectly!**
