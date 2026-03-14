#!/usr/bin/env python3
"""
Local runner for Tensara problems on WSL2.

Compiles CUDA solutions, verifies correctness against reference
implementations, and benchmarks performance (runtime and GFLOPs).

Usage:
    python run_local.py <problem-slug> <solution.cu> [options]

Examples:
    python run_local.py vector-addition solution.cu
    python run_local.py vector-addition solution.cu --benchmark
    python run_local.py matrix-multiplication matmul.cu --benchmark --iterations 50
    python run_local.py vector-addition solution.cu --sample

Options:
    --benchmark      Run performance benchmarks after verification
    --warmup N       Number of warmup iterations for benchmarking (default: 3)
    --iterations N   Number of benchmark iterations (default: 10)
    --sample         Use a single small sample test case (for debugging)
    --gpu-info       Display GPU information and exit
"""

import argparse
import ctypes
import importlib.util
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import torch

# ── Type mappings ────────────────────────────────────────────────────────────

CTYPE_MAP = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int": ctypes.c_int,
    "size_t": ctypes.c_size_t,
    "uint8_t": ctypes.c_uint8,
    "uint32_t": ctypes.c_uint32,
    "uint64_t": ctypes.c_uint64,
    "float16": ctypes.c_uint16,
}

INTEGER_TYPES = {"int", "size_t", "uint8_t", "uint32_t", "uint64_t"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def slug_to_func_name(slug: str) -> str:
    """Convert a problem slug to a valid C function name."""
    return slug.replace("-", "_")


def load_problem(slug: str):
    """Dynamically load a problem definition from problems/<slug>/def.py."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    def_path = os.path.join(repo_root, "problems", slug, "def.py")

    if not os.path.isfile(def_path):
        print(f"Error: Problem definition not found at {def_path}")
        sys.exit(1)

    # Add the repo root to sys.path so `from problem import Problem` works
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location("problem_def", def_path)
    module = importlib.util.module_from_spec(spec)

    # Inject repo root into the module's search path
    module.__path__ = [os.path.dirname(def_path)]
    sys.modules["problem_def"] = module
    spec.loader.exec_module(module)

    # Find the Problem subclass in the module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr.__module__ == "problem_def"
            and hasattr(attr, "parameters")
            and hasattr(attr, "reference_solution")
        ):
            return attr()

    print(f"Error: No problem class found in {def_path}")
    sys.exit(1)


def compile_cuda(source_path: str, output_path: str) -> bool:
    """Compile a CUDA source file to a shared library."""
    print(f"Compiling {source_path} ...")

    cmd = [
        "nvcc",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-o", output_path,
        source_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        print("Error: 'nvcc' not found. Please install the CUDA Toolkit.")
        print("See WSL2_SETUP.md for installation instructions.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Error: Compilation timed out after 120 seconds.")
        sys.exit(1)

    if result.returncode != 0:
        print("Compilation failed:\n")
        print(result.stderr)
        return False

    if result.stderr:
        print(f"Compiler warnings:\n{result.stderr}")

    print("Compilation successful.\n")
    return True


def build_cuda_args(
    problem,
    test_case: Dict[str, Any],
    inputs: Tuple,
    expected_output,
) -> Tuple[List, List[torch.Tensor]]:
    """
    Build the ctypes argument list for the CUDA function call.

    The argument order follows the problem's `parameters` list exactly:
      - const pointer params  → input tensor data pointers
      - non-const pointer params → freshly allocated output tensor pointers
      - scalar params → values from create_inputs scalars then get_extra_params
    """
    params = problem.parameters
    extra_params = problem.get_extra_params(test_case)

    # Separate inputs into tensors and scalars
    input_tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    input_scalars = [x for x in inputs if not isinstance(x, torch.Tensor)]

    # Separate extra_params into tensors and scalars
    extra_tensors = [x for x in extra_params if isinstance(x, torch.Tensor)]
    extra_scalars = [x for x in extra_params if not isinstance(x, torch.Tensor)]

    # Prepare output tensors (matching expected output shapes)
    if isinstance(expected_output, (tuple, list)):
        output_templates = list(expected_output)
    else:
        output_templates = [expected_output]
    output_tensors = [torch.zeros_like(t) for t in output_templates]

    # Build queues consumed in parameter order
    const_ptr_q = list(input_tensors) + list(extra_tensors)
    nonconst_ptr_q = list(output_tensors)
    scalar_q = list(input_scalars) + list(extra_scalars)

    ci, ni, si = 0, 0, 0
    args = []

    for param in params:
        if param["pointer"]:
            if param.get("const", False):
                tensor = const_ptr_q[ci]
                ci += 1
            else:
                tensor = nonconst_ptr_q[ni]
                ni += 1
            args.append(ctypes.c_void_p(tensor.data_ptr()))
        else:
            value = scalar_q[si]
            si += 1
            ptype = param["type"]
            ctype = CTYPE_MAP[ptype]
            if ptype in INTEGER_TYPES:
                args.append(ctype(int(value)))
            else:
                args.append(ctype(float(value)))

    return args, output_tensors


def print_gpu_info():
    """Print information about available CUDA GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU detected.")
        return

    count = torch.cuda.device_count()
    print(f"CUDA GPUs detected: {count}\n")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / (1024**3)
        print(f"  [{i}] {props.name}")
        print(f"      Compute capability: {props.major}.{props.minor}")
        print(f"      Memory:             {mem_gb:.1f} GB")
        print(f"      SM count:           {props.multi_processor_count}")
    print()


# ── Main logic ───────────────────────────────────────────────────────────────

def run_verification(
    problem,
    lib: ctypes.CDLL,
    func_name: str,
    test_cases: List[Dict],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run correctness verification for all test cases."""
    results = []
    total = len(test_cases)
    passed = 0

    for idx, tc in enumerate(test_cases, 1):
        name = tc["name"]
        inputs = tc["create_inputs"]()
        expected = problem.reference_solution(*inputs)

        args, output_tensors = build_cuda_args(problem, tc, inputs, expected)

        # Call the CUDA function
        func = getattr(lib, func_name)
        torch.cuda.synchronize()
        func(*args)
        torch.cuda.synchronize()

        # Extract actual output
        actual = output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)

        is_correct, debug_info = problem.verify_result(expected, actual)
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed += 1

        if verbose:
            icon = "\033[92m✓\033[0m" if is_correct else "\033[91m✗\033[0m"
            print(f"  {icon} [{idx}/{total}] {name}: {status}")
            if not is_correct and debug_info:
                for key, val in debug_info.items():
                    print(f"        {key}: {val}")

        results.append({"name": name, "correct": is_correct, "debug": debug_info})

    print(f"\nVerification: {passed}/{total} passed\n")
    return results


def run_benchmark(
    problem,
    lib: ctypes.CDLL,
    func_name: str,
    test_cases: List[Dict],
    warmup: int = 3,
    iterations: int = 10,
) -> List[Dict[str, Any]]:
    """Run performance benchmarks for all test cases."""
    results = []
    func = getattr(lib, func_name)

    print(f"{'Test Case':<40} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'GFLOPs':>10}")
    print("-" * 84)

    for tc in test_cases:
        name = tc["name"]
        inputs = tc["create_inputs"]()
        expected = problem.reference_solution(*inputs)
        args, _ = build_cuda_args(problem, tc, inputs, expected)

        # Warmup
        for _ in range(warmup):
            torch.cuda.synchronize()
            func(*args)
            torch.cuda.synchronize()

        # Timed runs using CUDA events
        times_ms = []
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            func(*args)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)

        try:
            flops = problem.get_flops(tc)
            gflops = (flops / (avg_ms / 1000)) / 1e9
        except Exception:
            flops = None
            gflops = None

        gflops_str = f"{gflops:>10.2f}" if gflops is not None else "       N/A"
        print(f"  {name:<38} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {gflops_str}")

        results.append({
            "name": name,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "gflops": gflops,
        })

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compile, verify, and benchmark CUDA solutions for Tensara problems."
    )
    parser.add_argument("slug", nargs="?", help="Problem slug (e.g. vector-addition)")
    parser.add_argument("solution", nargs="?", help="Path to CUDA solution file (.cu)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations (default: 10)")
    parser.add_argument("--sample", action="store_true", help="Use sample test case only (smaller, for debugging)")
    parser.add_argument("--gpu-info", action="store_true", help="Display GPU information and exit")

    args = parser.parse_args()

    if args.gpu_info:
        print_gpu_info()
        return

    if not args.slug or not args.solution:
        parser.print_help()
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Error: No CUDA-capable GPU detected.")
        print("Make sure you have NVIDIA drivers installed and CUDA is available in WSL2.")
        print("See WSL2_SETUP.md for setup instructions.")
        sys.exit(1)

    # Display GPU info
    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}\n")

    # Load problem
    print(f"Loading problem: {args.slug}")
    problem = load_problem(args.slug)
    func_name = slug_to_func_name(args.slug)
    print(f"Function name: {func_name}")
    print(f"Parameters: {len(problem.parameters)}\n")

    # Compile
    solution_path = os.path.abspath(args.solution)
    if not os.path.isfile(solution_path):
        print(f"Error: Solution file not found: {solution_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        so_path = os.path.join(tmpdir, "solution.so")
        if not compile_cuda(solution_path, so_path):
            sys.exit(1)

        lib = ctypes.CDLL(so_path)

        # Verify the expected function exists
        if not hasattr(lib, func_name):
            print(f"Error: Function '{func_name}' not found in compiled library.")
            print(f"Make sure your CUDA file defines: extern \"C\" void {func_name}(...)")
            sys.exit(1)

        # Generate test cases
        if args.sample:
            sample = problem.generate_sample()
            test_cases = [sample] if isinstance(sample, dict) else sample
            print("Using sample test case:\n")
        else:
            test_cases = problem.generate_test_cases()
            print(f"Running {len(test_cases)} test cases:\n")

        # Verify correctness
        print("── Verification ────────────────────────────────────────\n")
        verification = run_verification(problem, lib, func_name, test_cases)

        all_passed = all(r["correct"] for r in verification)

        # Benchmark (only if all tests pass)
        if args.benchmark:
            if not all_passed:
                print("Skipping benchmark — not all tests passed.\n")
            else:
                print("── Benchmark ───────────────────────────────────────────\n")
                print(f"  Warmup: {args.warmup} iterations")
                print(f"  Timed:  {args.iterations} iterations\n")
                run_benchmark(
                    problem, lib, func_name, test_cases,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
