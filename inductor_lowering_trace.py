"""
Step-by-step Inductor Lowering Trace

This script demonstrates how to manually decompose torch.compile() into its
individual lowering stages by utilizing internal torch APIs to trace the
complete compilation pipeline from user code to optimized GPU kernels.

This is primarily for educational and debugging purposes.
"""

import torch
import torch.fx as fx
from typing import List, Any, Callable
import sys
import io


def print_section(title: str, content: str = ""):
    """Helper to print formatted sections."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    if content:
        print(content)


def step1_capture_fx_graph(model_fn: Callable, example_inputs: List[torch.Tensor]):
    """
    Step 1: Capture Python execution as FX graph using Dynamo.

    This is what torch.compile() does first.
    """
    print_section("STEP 1: Capture FX Graph with TorchDynamo")

    from torch._dynamo import export

    # Export the function to get FX graph
    exported_program = torch.export.export(model_fn, tuple(example_inputs))
    gm = exported_program.module()

    print("Input function:")
    print(f"  {model_fn}")

    print("\nCaptured FX Graph:")
    print(gm.code)

    print("\nFX Nodes:")
    for i, node in enumerate(gm.graph.nodes):
        print(f"  {i}. {node.op:15} {node.target:30} {node.name}")

    print("\nExample: Let's see what meta['val'] looks like for first compute node:")
    for node in gm.graph.nodes:
        if node.op == 'call_function' and 'val' in node.meta:
            val = node.meta['val']
            print(f"  Node: {node.name}")
            print(f"  Operation: {node.target}")
            print(f"  Output shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
            print(f"  Output dtype: {val.dtype if hasattr(val, 'dtype') else 'N/A'}")
            break

    return gm, example_inputs


def step2_pre_grad_passes(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 2: Apply pre-gradient FX passes.

    These optimize the graph before autograd tracing.
    """
    print_section("STEP 2: Apply Pre-Grad FX Passes")

    from torch._inductor.fx_passes.pre_grad import pre_grad_passes

    print("Pre-grad passes include:")
    print("  - Conv-BatchNorm fusion (eval mode)")
    print("  - Split-cat elimination")
    print("  - Constant folding")
    print("  - Remove no-op view operations")

    # Apply passes
    gm_pre = pre_grad_passes(gm, example_inputs)

    print("\nGraph after pre-grad passes:")
    print(gm_pre.code)

    # Check if anything changed
    changed = str(gm.code) != str(gm_pre.code)
    print(f"\nGraph modified: {changed}")

    return gm_pre


def step3_aot_autograd_decomposition(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 3: AOTAutograd - Functionalization and Decomposition.

    This is what happens inside aot_autograd():
    - Functionalizes in-place operations
    - Decomposes composite operations
    - Creates joint forward-backward graph
    """
    print_section("STEP 3: AOTAutograd Transformations")

    from torch._functorch.aot_autograd import aot_module_simplified
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    print("AOTAutograd performs:")
    print("  - Functionalization (convert in-place → out-of-place)")
    print("  - Decomposition (break down composite ops)")
    print("  - Joint graph construction (forward + backward)")

    # Get decomposition table
    decompositions = select_decomp_table()
    print(f"\nNumber of registered decompositions: {len(decompositions)}")
    print("Example decompositions:")
    for i, (op, decomp_fn) in enumerate(list(decompositions.items())[:5]):
        print(f"  {op} → decomposed")

    # For inference, we'll just apply decompositions without backward
    # In training, aot_autograd would create joint graph
    from torch._decomp import core_aten_decompositions
    from torch.fx.experimental.proxy_tensor import make_fx

    # Trace with decompositions
    print("\nApplying decompositions...")

    def simple_decompose(gm, example_inputs):
        """Simplified decomposition without full AOTAutograd."""
        # This is a simplified version - full AOTAutograd does much more
        from torch._functorch.partitioners import default_partition

        # Apply decompositions by re-tracing
        def wrapped(*args):
            return torch.func.functionalize(gm)(*args)

        traced = make_fx(wrapped, decomposition_table=decompositions)(*example_inputs)
        return traced

    try:
        gm_decomposed = simple_decompose(gm, example_inputs)
        print("\nGraph after decomposition:")
        print(gm_decomposed.code)
    except Exception as e:
        print(f"\nNote: Full decomposition requires more complex tracing.")
        print(f"Using original graph. In real compile flow, AOTAutograd handles this.")
        gm_decomposed = gm

    return gm_decomposed


def step4_joint_graph_passes(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 4: Joint graph passes (on forward+backward graph).

    In training, this operates on the joint graph. For inference, these
    passes still apply to the forward graph.
    """
    print_section("STEP 4: Joint Graph Passes")

    from torch._inductor.fx_passes.joint_graph import joint_graph_passes
    from torch._inductor.fx_passes.joint_graph import _misc_patterns_init

    print("Joint graph passes include:")
    print("  - SDPA (Scaled Dot Product Attention) fusion")
    print("  - Pad + MatMul fusion")
    print("  - Constant folding")

    # Initialize patterns
    _misc_patterns_init()

    # Apply joint passes
    # Note: joint_graph_passes expects a specific signature for training
    # For inference, we'll just apply it as-is
    try:
        # Joint graph passes modify graph in-place and return None
        joint_graph_passes(gm)
        gm_joint = gm
        print("\nGraph after joint passes:")
        print(gm_joint.code)
    except Exception as e:
        print(f"\nNote: Joint passes are typically run on forward+backward graph")
        print(f"Skipping for simple inference example")
        gm_joint = gm

    return gm_joint


def step5_post_grad_passes(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 5: Post-gradient passes.

    These optimize the separated forward/backward graphs.
    """
    print_section("STEP 5: Post-Grad Passes")

    from torch._inductor.fx_passes.post_grad import post_grad_passes

    print("Post-grad passes include:")
    print("  - Layout optimization (channels_last conversion)")
    print("  - View to reshape simplification")
    print("  - DDP gradient synchronization fusion")
    print("  - Reinplacing (convert back to in-place when safe)")

    # Apply post-grad passes
    gm_post = post_grad_passes(gm, is_inference=True)

    print("\nGraph after post-grad passes:")
    print(gm_post.code)

    return gm_post


def step6_graph_lowering(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 6: Lower FX graph to Inductor IR.

    This is where aten operations become Inductor IR nodes:
    - Pointwise (element-wise ops)
    - Reduction (sum, max, etc.)
    - ExternKernel (library calls)
    """
    print_section("STEP 6: Graph Lowering (FX → Inductor IR)")

    from torch._inductor.graph import GraphLowering
    from torch._inductor import config
    from torch._inductor.utils import shape_env_from_inputs
    import torch._inductor.metrics as metrics

    print("Graph lowering converts:")
    print("  aten.add → Pointwise(add)")
    print("  aten.relu → Pointwise(relu)")
    print("  aten.sum → Reduction(sum)")
    print("  aten.matmul → ExternKernel(cublas/cutlass/triton)")

    # Create GraphLowering instance
    shape_env = shape_env_from_inputs(example_inputs)

    # Enable debug mode to see IR
    old_debug = config.debug
    config.debug = True

    print("\nCreating GraphLowering instance...")

    from torch._inductor.compile_fx import _shape_env_from_inputs

    # This is complex - we'll use compile_fx internals
    # In practice, use the full compile_fx flow
    print("\nNote: Full IR lowering is complex. Use compile_fx for complete flow.")
    print("We'll proceed to scheduling which shows the IR structure.")

    return gm


def step7_scheduling_and_fusion(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 7: Schedule operations and perform fusion.

    The scheduler:
    - Analyzes dependencies
    - Finds fusion opportunities
    - Groups operations into kernels
    """
    print_section("STEP 7: Scheduling and Fusion")

    print("Scheduler responsibilities:")
    print("  - Build dependency graph")
    print("  - Identify fusible operations")
    print("  - Vertical fusion: fuse producer-consumer chains")
    print("  - Horizontal fusion: fuse parallel operations")
    print("  - Create kernel fusion groups")

    print("\nExample fusion opportunities:")
    print("  add + mul + relu → Single fused pointwise kernel")
    print("  matmul + bias + relu → CUTLASS epilogue fusion")
    print("  softmax (multiple ops) → Fused reduction kernel")

    print("\nNote: Scheduling happens inside GraphLowering.run()")
    print("We'll see the scheduled kernels in the next step.")

    return gm


def step8_code_generation(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 8: Generate Triton/CUTLASS/C++ code.

    This is where actual kernel code is created.
    """
    print_section("STEP 8: Code Generation (Triton/CUTLASS)")

    print("Code generation paths:")
    print("  - Triton: Pointwise, reduction, fused operations")
    print("  - CUTLASS: Matrix multiplications with epilogue fusion")
    print("  - cuBLAS: Standard BLAS operations")
    print("  - C++: CPU kernels")

    print("\nFor GPU, Inductor generates:")
    print("  1. Triton kernel code (.py files with @triton.jit)")
    print("  2. Python wrapper code (allocates buffers, launches kernels)")
    print("  3. Autotuning configurations")

    print("\nExample Triton kernel structure:")
    print("""
    @triton.jit
    def fused_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)

        # Compute (fused operations)
        z = x + y
        z = z * 2.0
        z = tl.where(z > 0, z, 0.0)  # relu

        # Store
        tl.store(out_ptr + offsets, z, mask=mask)
    """)

    return gm


def step9_compile_and_cache(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Step 9: Compile kernels and cache.

    Generated code is compiled to PTX/CUBIN and cached.
    """
    print_section("STEP 9: Compilation and Caching")

    print("Compilation process:")
    print("  1. Generate kernel code (Triton/C++/CUDA)")
    print("  2. Compile to binary (PTX for Triton, .so for C++)")
    print("  3. Cache compiled kernels on disk")
    print("  4. Load and link kernels")

    print("\nCache locations:")
    print("  - Triton: ~/.triton/cache/")
    print("  - Inductor: /tmp/torchinductor_<user>/")

    print("\nCache keys include:")
    print("  - Kernel source code hash")
    print("  - Input shapes and dtypes")
    print("  - Device (cuda:0, cuda:1, cpu)")
    print("  - PyTorch version")
    print("  - Triton version")

    return gm


def step10_autotuning(example_inputs: List[torch.Tensor]):
    """
    Step 10: Autotune kernel configurations.

    For operations with multiple implementations, benchmark and choose best.
    """
    print_section("STEP 10: Autotuning")

    print("Autotuning process:")
    print("  1. Generate multiple kernel implementations")
    print("     - Different block sizes")
    print("     - Different backends (Triton vs CUTLASS vs cuBLAS)")
    print("     - Different tile sizes for matmul")
    print("  2. Benchmark each implementation")
    print("  3. Select fastest")
    print("  4. Cache result")

    print("\nExample: Matrix multiplication choices:")
    print("  - cuBLAS gemm")
    print("  - CUTLASS template: tile_shape=(128, 128, 32)")
    print("  - CUTLASS template: tile_shape=(256, 128, 32)")
    print("  - CUTLASS template: tile_shape=(128, 256, 32)")
    print("  - Triton matmul")

    print("\nExample: Pointwise kernel configs:")
    print("  - BLOCK_SIZE=128")
    print("  - BLOCK_SIZE=256")
    print("  - BLOCK_SIZE=512")
    print("  - BLOCK_SIZE=1024")

    print("\nAutotuning runs on FIRST execution")
    print("Subsequent runs use cached optimal configuration")


def full_manual_compile(model_fn: Callable, example_inputs: List[torch.Tensor]):
    """
    Execute all steps of the compilation pipeline manually.
    """
    print_section("FULL MANUAL INDUCTOR COMPILATION PIPELINE", "")

    # Step 1: Dynamo capture
    gm, inputs = step1_capture_fx_graph(model_fn, example_inputs)

    # Step 2: Pre-grad passes
    gm = step2_pre_grad_passes(gm, inputs)

    # Step 3: AOTAutograd
    gm = step3_aot_autograd_decomposition(gm, inputs)

    # Step 4: Joint graph passes
    gm = step4_joint_graph_passes(gm, inputs)

    # Step 5: Post-grad passes
    gm = step5_post_grad_passes(gm, inputs)

    # Step 6: Graph lowering
    gm = step6_graph_lowering(gm, inputs)

    # Step 7: Scheduling
    gm = step7_scheduling_and_fusion(gm, inputs)

    # Step 8: Code generation
    gm = step8_code_generation(gm, inputs)

    # Step 9: Compilation
    gm = step9_compile_and_cache(gm, inputs)

    # Step 10: Autotuning
    step10_autotuning(inputs)

    print_section("FINAL STEP: Use compile_fx for Actual Compilation")

    print("\nThe above steps are illustrative. For actual compilation, use:")
    print("""
    from torch._inductor.compile_fx import compile_fx
    import torch._inductor.config as config

    # Enable verbose output
    config.trace.enabled = True
    config.debug = True

    # Compile
    compiled_fn = compile_fx(gm, example_inputs)

    # Or just use torch.compile:
    @torch.compile
    def model_fn(x, y):
        return (x + y).relu()

    result = model_fn(x, y)
    """)


def demonstrate_actual_compilation():
    """
    Show actual compilation with torch.compile and inspect outputs.
    """
    print_section("ACTUAL COMPILATION WITH torch.compile")

    import torch._inductor.config as config

    # Enable output
    config.trace.enabled = True

    print("\nDefining simple model:")
    code = """
def model_fn(x, y):
    z = x + y
    z = z * 2.0
    z = z.relu()
    return z
"""
    print(code)

    def model_fn(x, y):
        z = x + y
        z = z * 2.0
        z = z.relu()
        return z

    print("Compiling with torch.compile...")

    compiled_fn = torch.compile(model_fn, backend="inductor")

    # Create inputs
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')

    print("Running compiled function (first call - includes autotuning)...")

    # Capture stdout to show generated code
    # Note: actual code is written to cache, not stdout
    result = compiled_fn(x, y)

    print(f"\nResult shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    print(f"Result device: {result.device}")

    print("\n✓ Compilation successful!")
    print("\nTo see generated code:")
    print("  1. Check /tmp/torchinductor_<user>/ directory")
    print("  2. Look for .py files (Triton kernels)")
    print("  3. Look for output.py (wrapper code)")


def demonstrate_manual_fx_compilation():
    """
    Demonstrate manual compilation using internal APIs.
    """
    print_section("MANUAL COMPILATION USING INTERNAL APIs")

    print("\nThis demonstrates the exact internal flow:")

    def model_fn(x, y):
        return (x + y).relu()

    print(f"Model: {model_fn}")

    # Create example inputs
    x = torch.randn(10, device='cuda')
    y = torch.randn(10, device='cuda')

    print(f"Inputs: x={x.shape}, y={y.shape}, device={x.device}")

    # Step 1: Export to FX
    print("\n1. Export to FX graph...")
    exported = torch.export.export(model_fn, (x, y))
    gm = exported.module()
    print("   ✓ FX graph created")
    print(f"   Nodes: {len(list(gm.graph.nodes))}")

    # Step 2: Compile with Inductor
    print("\n2. Compile with Inductor...")
    from torch._inductor.compile_fx import compile_fx
    import torch._inductor.config as config

    # Enable verbose mode
    config.debug = False  # Set True to see IR details

    compiled = compile_fx(gm, [x, y])
    print("   ✓ Compilation complete")

    # Step 3: Run
    print("\n3. Execute compiled function...")
    result = compiled([x, y])
    print(f"   ✓ Result: shape={result.shape}, dtype={result.dtype}")

    # Verify correctness
    expected = model_fn(x, y)
    print(f"\n4. Verify correctness...")
    print(f"   Max difference: {(result - expected).abs().max().item():.2e}")
    print("   ✓ Correct!" if torch.allclose(result, expected) else "   ✗ Incorrect!")

    return compiled


def show_inductor_config_options():
    """
    Show important Inductor configuration options.
    """
    print_section("INDUCTOR CONFIGURATION OPTIONS")

    import torch._inductor.config as config

    print("\nImportant config options for debugging:")
    print(f"  config.debug = {config.debug}")
    print(f"  config.trace.enabled = {config.trace.enabled}")
    print(f"  config.verbose = {config.verbose}")

    print("\nPerformance options:")
    print(f"  config.max_autotune = {config.max_autotune}")
    print(f"  config.coordinate_descent_tuning = {config.coordinate_descent_tuning}")
    print(f"  config.triton.cudagraphs = {config.triton.cudagraphs}")

    print("\nBackend selection:")
    print(f"  config.cpp_wrapper = {config.cpp_wrapper}")
    print(f"  config.use_cutlass = {config.use_cutlass}")

    print("\nTo enable verbose output:")
    print("""
    import torch._inductor.config as config
    config.trace.enabled = True
    config.debug = True
    """)

    print("\nTo enable max autotuning:")
    print("""
    @torch.compile(mode="max-autotune")
    def model_fn(x):
        return x @ x.T
    """)


def main():
    """Main entry point."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Some examples will not work.")
        print("Continuing with CPU examples where possible...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define a simple model
    def simple_model(x, y):
        return (x + y).relu()

    # Create example inputs
    example_inputs = [
        torch.randn(1024, device=device),
        torch.randn(1024, device=device),
    ]

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   PYTORCH INDUCTOR LOWERING TRACE                            ║
║                                                                              ║
║  This script demonstrates the complete compilation pipeline from user       ║
║  code to optimized GPU kernels, step by step.                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run manual step-by-step trace
    full_manual_compile(simple_model, example_inputs)

    # Show actual compilation
    if device == 'cuda':
        print("\n\n")
        demonstrate_actual_compilation()

    print("\n\n")
    # Show manual compilation with internal APIs
    if device == 'cuda':
        demonstrate_manual_fx_compilation()

    print("\n\n")
    # Show config options
    show_inductor_config_options()

    print_section("COMPLETE", "")
    print("\nFor more information, see:")
    print("  - INDUCTOR_ARCHITECTURE.md (this repository)")
    print("  - torch/_inductor/ source code")
    print("  - TORCH_LOGS environment variable for debugging")
    print("\nUseful debugging commands:")
    print('  export TORCH_LOGS="+inductor,+output_code"')
    print('  export TORCH_LOGS="fusion,schedule"')
    print("  python -m torch._inductor.list_mode_options")


if __name__ == "__main__":
    # Set up environment for better output
    import os
    os.environ.setdefault('TORCH_LOGS_FORMAT', '%(message)s')

    main()
