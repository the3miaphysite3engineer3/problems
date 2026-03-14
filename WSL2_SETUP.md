# Running Tensara Problems Locally on WSL2

This guide covers setting up your WSL2 environment to compile, verify, and benchmark CUDA solutions for Tensara problems.

## Prerequisites

- **Windows 10 (21H2+) or Windows 11** with WSL2 enabled
- **NVIDIA GPU** with up-to-date Windows drivers (Game Ready or Studio)

> **Important:** You install the NVIDIA driver on **Windows**, not inside WSL2. WSL2 automatically exposes the GPU to Linux.

## 1. Install WSL2

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

After installation, launch Ubuntu from the Start menu and create a user account.

## 2. Install NVIDIA CUDA Toolkit in WSL2

Inside your WSL2 Ubuntu terminal:

```bash
# Remove any existing CUDA packages to avoid conflicts
sudo apt-get remove --purge -y nvidia-cuda-toolkit 2>/dev/null

# Add the NVIDIA package repository (Ubuntu 22.04 example)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the CUDA toolkit (compiler, libraries, headers)
sudo apt-get install -y cuda-toolkit
```

Add CUDA to your PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify the installation:

```bash
nvcc --version      # Should print the CUDA compiler version
nvidia-smi          # Should show your GPU
```

## 3. Install Python and PyTorch

```bash
# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the latest command.
# Example for CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify PyTorch CUDA:

```python
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 4. Clone and Set Up This Repository

```bash
git clone https://github.com/tensara/problems.git
cd problems
```

No additional Python packages are needed beyond PyTorch — the local runner uses only the standard library and torch.

## 5. Write a CUDA Solution

Each problem expects a C function with `extern "C"` linkage matching the problem's parameter signature. For example, for `vector-addition`:

```cuda
// solution.cu
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void vector_addition(const float* d_input1, const float* d_input2,
                                 float* d_output, size_t n) {
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_input1, d_input2, d_output, (int)n);
    cudaDeviceSynchronize();
}
```

Key rules:
- The function name is the problem slug with hyphens replaced by underscores (e.g. `matrix-multiplication` → `matrix_multiplication`).
- Use `extern "C"` so the function is callable via ctypes.
- Parameter names, types, and order must match the `parameters` list in the problem's `def.py`.
- Call `cudaDeviceSynchronize()` at the end of your wrapper function.

You can find the parameter list by reading `problems/<slug>/def.py` or the `parameters` section of `problems/<slug>/problem.md`.

## 6. Verify and Benchmark

### Check correctness

```bash
python3 run_local.py vector-addition solution.cu
```

### Quick debugging with a small sample test case

```bash
python3 run_local.py vector-addition solution.cu --sample
```

### Run benchmarks (timing + GFLOPs)

```bash
python3 run_local.py vector-addition solution.cu --benchmark
```

### Adjust benchmark parameters

```bash
python3 run_local.py vector-addition solution.cu --benchmark --warmup 5 --iterations 50
```

### Display GPU info

```bash
python3 run_local.py --gpu-info
```

## Example Output

```
GPU: NVIDIA GeForce RTX 3090
CUDA: 12.6

Loading problem: vector-addition
Function name: vector_addition
Parameters: 4

Compiling solution.cu ...
Compilation successful.

Running 7 test cases:

── Verification ────────────────────────────────────────

  ✓ [1/7] n = 2^20: PASS
  ✓ [2/7] n = 2^22: PASS
  ✓ [3/7] n = 2^23: PASS
  ✓ [4/7] n = 2^25: PASS
  ✓ [5/7] n = 2^26: PASS
  ✓ [6/7] n = 2^29: PASS
  ✓ [7/7] n = 2^30: PASS

Verification: 7/7 passed

── Benchmark ───────────────────────────────────────────

  Warmup: 3 iterations
  Timed:  10 iterations

  Test Case                                Avg (ms)   Min (ms)   Max (ms)     GFLOPs
  ------------------------------------------------------------------------------------
  n = 2^20                                    0.035      0.032      0.041      30.01
  n = 2^22                                    0.085      0.082      0.091      49.34
  ...
```

## Troubleshooting

### `nvcc` not found

Make sure the CUDA Toolkit is installed and `/usr/local/cuda/bin` is in your `PATH`.

### `nvidia-smi` shows no GPU

Update your **Windows** NVIDIA driver to the latest version. WSL2 GPU support requires driver version 510+ (for CUDA 11.6+).

### PyTorch says CUDA is not available

- Verify `nvidia-smi` works inside WSL2 first.
- Make sure you installed the CUDA-enabled build of PyTorch (not the CPU-only version).
- Ensure the PyTorch CUDA version matches your toolkit version (e.g. `cu126` for CUDA 12.6).

### Compilation errors about missing headers

Install the full CUDA Toolkit (`cuda-toolkit` package), not just the driver.

### "Function not found in compiled library"

Make sure your CUDA file declares the entry point with `extern "C"` and uses the correct function name (slug with underscores, e.g. `vector_addition`).
