# torch._inductor Architecture Guide

This document provides a comprehensive architectural overview of PyTorch's Inductor compiler, focusing on GPU/CUDA code generation using Triton and CUTLASS backends.

## What is Inductor?

Inductor is PyTorch's native compiler backend that generates optimized kernel code for CPUs and GPUs. It sits downstream of TorchDynamo (the graph capture mechanism) and is responsible for:
- Taking FX graphs and lowering them to device-specific operations
- Performing graph-level optimizations
- Fusing operations for better performance
- Generating optimized kernels (Triton for GPU, C++ for CPU)
- Autotuning kernel configurations

## High-Level Flow

```
User Code
   ↓
torch.compile()
   ↓
TorchDynamo (graph capture)
   ↓
FX Graph (high-level ops like aten.add, aten.matmul)
   ↓
AOTAutograd (functionalization, decomposition, forward/backward split)
   ↓
Inductor Entry Point
   ↓
FX Passes (pre-grad, joint, post-grad)
   ↓
Graph Lowering (FX → Inductor IR)
   ↓
Scheduling & Fusion
   ↓
Code Generation (Triton/CUTLASS/C++)
   ↓
Autotuning & Compilation
   ↓
Optimized Callable
```

## Entry Point: Where Dynamo Hands Off to Inductor

### Main Entry Point
**File**: [torch/_inductor/__init__.py](torch/_inductor/__init__.py)

```python
def compile(gm: torch.fx.GraphModule,
            example_inputs: list[InputType],
            options: Optional[dict[str, Any]] = None):
    from .compile_fx import compile_fx
    return compile_fx(gm, example_inputs, config_patches=options)
```

### Compilation Pipeline Entry
**File**: [torch/_inductor/compile_fx.py](torch/_inductor/compile_fx.py)

The main compilation flow is:
1. `compile_fx()` - Public API, validates inputs
2. `_compile_fx_inner()` - Sets up compilation context
3. `_compile_fx_main()` - Core compilation orchestration
4. Calls `aot_autograd()` with custom compilers for forward/backward

**Key Function**: `_compile_fx_main()`

This function:
- Applies pre-gradient FX passes
- Invokes AOTAutograd with `fw_compiler` and `bw_compiler`
- Handles post-compilation wrapping

### Dynamo → Inductor Interface
**File**: [torch/_dynamo/backends/common.py](torch/_dynamo/backends/common.py)

When you call `torch.compile(model, backend="inductor")`, Dynamo:
1. Captures the model execution as an FX graph
2. Calls the backend (inductor) with the graph and example inputs
3. Inductor returns a compiled callable

The interface is simply:
```python
compiled_fn = backend(gm: fx.GraphModule, example_inputs: List[Tensor]) -> Callable
```

## AOTAutograd Integration

### Role of AOTAutograd
**File**: [torch/_functorch/aot_autograd.py](torch/_functorch/aot_autograd.py) (functorch)

AOTAutograd sits between Dynamo and Inductor, performing critical transformations:

1. **Functionalization**: Converts in-place operations to out-of-place
2. **Decomposition**: Breaks down composite operations into primitives
3. **Joint Graph Construction**: Creates a combined forward+backward graph
4. **Partitioning**: Separates forward and backward passes
5. **Tracing**: Ensures all operations are traceable

### AOTAutograd Flow in Inductor

```python
# In _compile_fx_main()
aot_autograd(
    fw_compiler=fw_compiler_base,  # Compiles forward pass
    bw_compiler=bw_compiler_base,  # Compiles backward pass
    inference_compiler=inference_compiler,
    decompositions=decompositions,
    partition_fn=min_cut_rematerialization_partition,
    ...
)(gm, example_inputs)
```

**What gets passed to fw_compiler**:
- Functionalized, decomposed forward graph
- All tensor operations are now in primitive form
- Ready for graph passes and lowering

## IR Transformation Pipeline

### Stage 1: Pre-Gradient Passes (FX Graph → FX Graph)

**File**: [torch/_inductor/fx_passes/pre_grad.py](torch/_inductor/fx_passes/pre_grad.py)

**Input**: Original FX graph from Dynamo
**Output**: Optimized FX graph before autograd

Key optimizations:
- **Conv-BatchNorm fusion**: Folds BN into conv weights when in eval mode
- **Remove redundant ops**: Eliminates no-ops like `x.view(x.shape)`
- **Constant folding**: Pre-computes constant expressions
- **Split-cat elimination**: Removes unnecessary split→cat patterns
- **Normalization decomposition**: Breaks down LayerNorm, GroupNorm, etc.

**Example Pattern**:
```python
# Before: Conv2d → BatchNorm2d (eval mode)
y = F.batch_norm(F.conv2d(x, w, b), bn_weight, bn_bias, mean, var)

# After: Fused Conv2d with modified weights
y = F.conv2d(x, w_fused, b_fused)
```

### Stage 2: Joint Graph Passes (Joint FX Graph → Joint FX Graph)

**File**: [torch/_inductor/fx_passes/joint_graph.py](torch/_inductor/fx_passes/joint_graph.py)

**Input**: Joint forward+backward graph from AOTAutograd
**Output**: Optimized joint graph

Key optimizations:
- **SDPA (Scaled Dot Product Attention) fusion**: Pattern matches and fuses attention
- **Pad + MatMul fusion**: Combines padding with matrix multiplication
- **Efficient attention**: Replaces manual attention with `torch.nn.functional.scaled_dot_product_attention`
- **Constant folding**: Removes redundant computations

**Example Pattern**:
```python
# Before: Manual attention implementation
q, k, v = x.chunk(3, dim=-1)
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
attn = F.softmax(scores, dim=-1)
out = attn @ v

# After: Fused SDPA
out = F.scaled_dot_product_attention(q, k, v)
```

### Stage 3: Post-Gradient Passes (FX Graph → FX Graph)

**File**: [torch/_inductor/fx_passes/post_grad.py](torch/_inductor/fx_passes/post_grad.py)

**Input**: Separate forward and backward FX graphs
**Output**: Optimized graphs ready for lowering

Key optimizations:
- **Layout optimization**: Converts tensors to channels_last when beneficial
- **View to reshape**: Simplifies view operations
- **DDP (Distributed Data Parallel) fusion**: Optimizes gradient synchronization
- **Reinplacing**: Converts out-of-place ops back to in-place when safe
- **Padding removal**: Eliminates unnecessary padding operations

**File**: [torch/_inductor/fx_passes/reinplace.py](torch/_inductor/fx_passes/reinplace.py)
- Analyzes aliasing and mutation patterns
- Safely converts functional ops back to in-place variants for memory efficiency

### Stage 4: Graph Lowering (FX Graph → Inductor IR)

**File**: [torch/_inductor/graph.py](torch/_inductor/graph.py)

**Class**: `GraphLowering`

**Input**: Optimized FX graph
**Output**: Inductor IR (list of IR nodes representing computations)

This is where the graph transitions from FX representation to Inductor's internal IR:

```python
class GraphLowering:
    def __init__(self, gm: fx.GraphModule, ...):
        self.graph = gm.graph
        self.buffers = {}  # Intermediate tensors
        self.operations = []  # IR operations

    def run(self, *example_inputs):
        # Iterate through FX nodes
        for node in self.graph.nodes:
            if node.op == 'call_function':
                # Lower FX operation to IR
                ir_node = self.lower_node(node)
                self.operations.append(ir_node)

        # Schedule and fuse operations
        return self.compile_to_fn()
```

**Lowering Process**:
1. **Walks the FX graph** node by node
2. **Looks up lowering rules** in [torch/_inductor/lowering.py](torch/_inductor/lowering.py)
3. **Creates IR nodes** representing the computation
4. **Tracks buffers** (intermediate tensors)
5. **Builds dependency graph** for scheduling

**File**: [torch/_inductor/ir.py](torch/_inductor/ir.py)

Inductor IR types:
- `Pointwise`: Element-wise operations (add, mul, relu, etc.)
- `Reduction`: Reduce operations (sum, max, softmax, etc.)
- `ExternKernel`: Calls to external libraries (cuBLAS, cuDNN, CUTLASS)
- `TemplateKernel`: Pre-written kernel templates
- `ComputedBuffer`: Intermediate storage

**File**: [torch/_inductor/lowering.py](torch/_inductor/lowering.py)

Contains the lowering registry mapping `torch.ops.aten.*` to IR:

```python
@lowerings.register(torch.ops.aten.add.Tensor)
def add(x, y):
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda idx: ops.add(x.load(idx), y.load(idx)),
        ranges=x.get_size(),
    )
```

## Scheduling and Fusion

**File**: [torch/_inductor/scheduler.py](torch/_inductor/scheduler.py)

**Class**: `Scheduler`

After lowering to IR, the scheduler:

### 1. Dependency Analysis
- Builds a dependency graph between IR nodes
- Identifies which operations can be fused
- Determines execution order

### 2. Fusion Opportunities
- **Horizontal fusion**: Combines operations at the same level
- **Vertical fusion**: Fuses producer-consumer chains
- **Reduction fusion**: Combines multiple reductions

**Example**:
```python
# Three separate pointwise ops
y1 = x + 1      # Pointwise op 1
y2 = y1 * 2     # Pointwise op 2
y3 = y2.relu()  # Pointwise op 3

# Scheduler fuses into single kernel:
y3 = ((x + 1) * 2).relu()  # Single fused kernel
```

### 3. Kernel Selection
**File**: [torch/_inductor/select_algorithm.py](torch/_inductor/select_algorithm.py)

For operations like matmul, the scheduler creates multiple implementation choices:
- cuBLAS
- CUTLASS templates (different tile sizes, epilogue fusions)
- Triton implementations

These are stored as `AlgorithmChoice` objects for later autotuning.

### 4. Node Scheduling
- Topologically sorts operations
- Groups fusible operations
- Respects memory and compute dependencies

## GPU Code Generation: Triton Path

### Triton Kernel Generation

**File**: [torch/_inductor/codegen/triton.py](torch/_inductor/codegen/triton.py)

**Classes**:
- `TritonKernel`: Represents a single Triton kernel
- `TritonScheduling`: Manages kernel generation and fusion

**Flow**:
1. Scheduler groups fusible IR nodes
2. `TritonScheduling.codegen_node()` is called for each group
3. Creates a `TritonKernel` instance
4. Generates Triton code through `TritonKernel.codegen_body()`

### Triton Code Structure

A generated Triton kernel looks like:

```python
import triton
import triton.language as tl

@triton.jit
def triton_kernel_name(
    in_ptr0,  # Input pointer
    out_ptr0, # Output pointer
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(in_ptr0 + offsets, mask=mask)

    # Compute (fused operations)
    y = x + 1.0
    y = y * 2.0
    y = tl.where(y > 0, y, 0.0)  # relu

    # Store output
    tl.store(out_ptr0 + offsets, y, mask=mask)
```

### Triton Kernel Types

**Pointwise Kernels**:
- Element-wise operations
- Broadcasting
- Simple memory access patterns

**Reduction Kernels**:
- Sum, max, min, etc.
- Softmax
- Layer normalization
- Multiple reduction dimensions

**Persistent Reduction Kernels**:
- Keeps data in registers across iterations
- Better for large reductions

### Key Triton Generation Methods

**In `TritonKernel` class**:

- `codegen_body()`: Main code generation entry point
- `codegen_iteration()`: Generates loop bodies
- `load()`: Generates load instructions
- `store()`: Generates store instructions
- `compute()`: Generates computation code

### Triton Autotuning

**File**: [torch/_inductor/runtime/triton_heuristics.py](torch/_inductor/runtime/triton_heuristics.py)

Generated kernels include autotuning configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

## GPU Code Generation: CUTLASS Path

### CUTLASS Template System

**File**: [torch/_inductor/codegen/cuda/gemm_template.py](torch/_inductor/codegen/cuda/gemm_template.py)

**File**: [torch/_inductor/codegen/cuda/cutlass_utils.py](torch/_inductor/codegen/cuda/cutlass_utils.py)

CUTLASS is NVIDIA's library for high-performance GEMMs (General Matrix Multiply). Inductor uses CUTLASS for:
- Matrix multiplications (matmul, bmm)
- Fused operations (matmul + bias + activation)
- Mixed-precision compute

### CUTLASS Code Generation Flow

1. **Pattern Matching**: Scheduler identifies matmul operations
2. **Template Selection**: Chooses appropriate CUTLASS template
3. **Configuration**: Sets tile sizes, epilogue operations
4. **Code Generation**: Instantiates C++ template code
5. **Compilation**: Compiles via nvcc

### CUTLASS Template Structure

**File**: [torch/_inductor/kernel/mm.py](torch/_inductor/kernel/mm.py)

```cpp
// Generated CUTLASS kernel (simplified)
#include "cutlass/gemm/device/gemm.h"

using GemmKernel = cutlass::gemm::device::Gemm<
    ElementA,              // Matrix A element type
    LayoutA,               // Matrix A layout
    ElementB,              // Matrix B element type
    LayoutB,               // Matrix B layout
    ElementC,              // Output element type
    LayoutC,               // Output layout
    ElementAccumulator,    // Accumulator type
    OperatorClass,         // Tensor core or SIMT
    ArchTag,               // SM version (80, 90, etc.)
    ThreadblockShape,      // 128x128x32
    WarpShape,             // 64x64x32
    InstructionShape,      // 16x8x16
    EpilogueOp             // Bias, activation, etc.
>;
```

### CUTLASS Epilogue Fusion

CUTLASS supports fusing operations after the GEMM:

```python
# Pattern: matmul + bias + relu
z = torch.relu(x @ w.T + bias)

# Gets compiled to single CUTLASS kernel with:
# - GEMM: x @ w.T
# - Epilogue: + bias, then relu
```

**Supported Epilogue Operations**:
- Bias addition
- Activation functions (ReLU, GELU, SiLU)
- Scaling
- Residual connections

## Kernel Selection and Autotuning

### Algorithm Selection

**File**: [torch/_inductor/select_algorithm.py](torch/_inductor/select_algorithm.py)

**Class**: `AlgorithmSelector`

For operations like matmul, Inductor generates multiple implementations:

```python
choices = [
    # cuBLAS
    ExternKernelChoice(
        kernel_name="cublas_gemm",
        ...
    ),

    # CUTLASS template 1: 128x128x32 tile
    CUTLASSGemmTemplate(
        tile_shape=(128, 128, 32),
        ...
    ),

    # CUTLASS template 2: 256x128x32 tile
    CUTLASSGemmTemplate(
        tile_shape=(256, 128, 32),
        ...
    ),

    # Triton implementation
    TritonTemplate(
        ...
    ),
]
```

### Autotuning Process

**File**: [torch/_inductor/runtime/autotune_cache.py](torch/_inductor/runtime/autotune_cache.py)

**Flow**:
1. **First run**: Benchmark all choices
2. **Measure time**: Run each kernel multiple times
3. **Select best**: Choose fastest implementation
4. **Cache result**: Save to disk indexed by (op, shapes, dtypes, device)
5. **Subsequent runs**: Load cached choice

**Autotune Cache Location**:
```
~/.triton/autotune_cache_v2/
```

**Cache Key**:
```python
cache_key = (
    operation_name,
    input_shapes,
    input_dtypes,
    device,
    torch_version,
)
```

### Autotuning for Triton Kernels

Triton uses `@triton.autotune` decorator:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
        # ... more configs
    ],
    key=['M', 'N', 'K'],
)
```

Triton's autotuner:
- Compiles all configurations
- Benchmarks each on first call
- Caches result per input shapes

## Wrapper Code Generation

**File**: [torch/_inductor/codegen/wrapper.py](torch/_inductor/codegen/wrapper.py)

After kernel generation, Inductor creates a Python wrapper that:
1. Allocates output tensors
2. Manages temporary buffers
3. Launches kernels in correct order
4. Handles dynamic shapes

**Example Generated Wrapper**:

```python
def inductor_wrapper(input0):
    # Allocate buffers
    buf0 = torch.empty((1024,), device='cuda', dtype=torch.float32)
    buf1 = torch.empty((1024,), device='cuda', dtype=torch.float32)

    # Launch kernel 1
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    triton_kernel_0[grid](input0, buf0, 1024, BLOCK_SIZE=256)

    # Launch kernel 2
    triton_kernel_1[grid](buf0, buf1, 1024, BLOCK_SIZE=256)

    return buf1
```

### GPU-Specific Wrapper

**File**: [torch/_inductor/codegen/cpp_wrapper_gpu.py](torch/_inductor/codegen/cpp_wrapper_gpu.py)

For AOTI (Ahead-Of-Time Inductor), generates C++ wrapper code:

```cpp
torch::Tensor inductor_entry_point(torch::Tensor input0) {
    // Allocate output
    auto buf0 = torch::empty({1024},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));

    // Launch CUDA kernel
    triton_kernel_0<<<grid, block>>>(
        input0.data_ptr<float>(),
        buf0.data_ptr<float>(),
        1024
    );

    return buf0;
}
```

## Key Files and Their Roles

### Entry and Orchestration

| File | Purpose |
|------|---------|
| [torch/_inductor/__init__.py](torch/_inductor/__init__.py) | Public API entry point |
| [torch/_inductor/compile_fx.py](torch/_inductor/compile_fx.py) | Main compilation orchestration |
| [torch/_inductor/graph.py](torch/_inductor/graph.py) | Graph lowering (FX → IR) |
| [torch/_inductor/scheduler.py](torch/_inductor/scheduler.py) | Operation scheduling and fusion |

### FX Graph Passes

| File | Stage | Purpose |
|------|-------|---------|
| [torch/_inductor/fx_passes/pre_grad.py](torch/_inductor/fx_passes/pre_grad.py) | Pre-grad | Conv-BN fusion, split-cat elimination |
| [torch/_inductor/fx_passes/joint_graph.py](torch/_inductor/fx_passes/joint_graph.py) | Joint | SDPA fusion, pad+MM fusion |
| [torch/_inductor/fx_passes/post_grad.py](torch/_inductor/fx_passes/post_grad.py) | Post-grad | Layout optimization, DDP fusion |
| [torch/_inductor/fx_passes/reinplace.py](torch/_inductor/fx_passes/reinplace.py) | Post-grad | Convert ops back to in-place |
| [torch/_inductor/fx_passes/fuse_attention.py](torch/_inductor/fx_passes/fuse_attention.py) | Joint | Attention pattern matching |
| [torch/_inductor/fx_passes/group_batch_fusion.py](torch/_inductor/fx_passes/group_batch_fusion.py) | Post-grad | Batch dimension fusion |

### IR and Lowering

| File | Purpose |
|------|---------|
| [torch/_inductor/ir.py](torch/_inductor/ir.py) | IR node definitions (Pointwise, Reduction, etc.) |
| [torch/_inductor/lowering.py](torch/_inductor/lowering.py) | Lowering registry (aten ops → IR) |
| [torch/_inductor/decomposition.py](torch/_inductor/decomposition.py) | Operation decompositions |

### Code Generation - Common

| File | Purpose |
|------|---------|
| [torch/_inductor/codegen/common.py](torch/_inductor/codegen/common.py) | Shared codegen utilities |
| [torch/_inductor/codegen/wrapper.py](torch/_inductor/codegen/wrapper.py) | Python wrapper generation |
| [torch/_inductor/codegen/simd.py](torch/_inductor/codegen/simd.py) | SIMD kernel base classes |

### Code Generation - Triton

| File | Purpose |
|------|---------|
| [torch/_inductor/codegen/triton.py](torch/_inductor/codegen/triton.py) | Triton kernel generation |
| [torch/_inductor/codegen/triton_utils.py](torch/_inductor/codegen/triton_utils.py) | Triton helper utilities |
| [torch/_inductor/codegen/triton_split_scan.py](torch/_inductor/codegen/triton_split_scan.py) | Split-scan reduction pattern |

### Code Generation - CUDA/CUTLASS

| File | Purpose |
|------|---------|
| [torch/_inductor/codegen/cuda/gemm_template.py](torch/_inductor/codegen/cuda/gemm_template.py) | CUTLASS GEMM templates |
| [torch/_inductor/codegen/cuda/cutlass_utils.py](torch/_inductor/codegen/cuda/cutlass_utils.py) | CUTLASS helper functions |
| [torch/_inductor/codegen/cuda/cuda_kernel.py](torch/_inductor/codegen/cuda/cuda_kernel.py) | CUDA kernel infrastructure |
| [torch/_inductor/codegen/cuda/cuda_template.py](torch/_inductor/codegen/cuda/cuda_template.py) | CUDA template system |
| [torch/_inductor/codegen/cpp_wrapper_gpu.py](torch/_inductor/codegen/cpp_wrapper_gpu.py) | C++ GPU wrapper for AOTI |

### Code Generation - C++

| File | Purpose |
|------|---------|
| [torch/_inductor/codegen/cpp.py](torch/_inductor/codegen/cpp.py) | C++ CPU kernel generation |
| [torch/_inductor/codegen/cpp_gemm_template.py](torch/_inductor/codegen/cpp_gemm_template.py) | CPU GEMM templates |
| [torch/_inductor/codegen/cpp_wrapper_cpu.py](torch/_inductor/codegen/cpp_wrapper_cpu.py) | C++ CPU wrapper |

### Kernel Selection and Autotuning

| File | Purpose |
|------|---------|
| [torch/_inductor/select_algorithm.py](torch/_inductor/select_algorithm.py) | Algorithm selection and benchmarking |
| [torch/_inductor/kernel/mm.py](torch/_inductor/kernel/mm.py) | Matrix multiply kernel choices |
| [torch/_inductor/kernel/conv.py](torch/_inductor/kernel/conv.py) | Convolution kernel choices |
| [torch/_inductor/runtime/autotune_cache.py](torch/_inductor/runtime/autotune_cache.py) | Autotune result caching |
| [torch/_inductor/runtime/triton_heuristics.py](torch/_inductor/runtime/triton_heuristics.py) | Triton autotuning configs |

### Utilities

| File | Purpose |
|------|---------|
| [torch/_inductor/config.py](torch/_inductor/config.py) | Configuration options |
| [torch/_inductor/codecache.py](torch/_inductor/codecache.py) | Kernel compilation and caching |
| [torch/_inductor/utils.py](torch/_inductor/utils.py) | General utilities |
| [torch/_inductor/debug.py](torch/_inductor/debug.py) | Debug utilities and logging |

## Complete Data Flow Example

Let's trace a simple example: `z = torch.relu(x + y)`

### 1. User Code
```python
@torch.compile
def foo(x, y):
    return torch.relu(x + y)
```

### 2. Dynamo Capture
- Captures execution as FX graph
- Nodes: `[placeholder:x, placeholder:y, call_function:add, call_function:relu, output]`

### 3. Entry to Inductor
- Dynamo calls `inductor.compile(gm, example_inputs)`
- Enters [compile_fx.py:compile_fx()](torch/_inductor/compile_fx.py)

### 4. Pre-Grad Passes
- [fx_passes/pre_grad.py:pre_grad_passes()](torch/_inductor/fx_passes/pre_grad.py)
- Graph unchanged (no patterns match)

### 5. AOTAutograd
- Functionalizes graph (already functional)
- Decomposes if needed
- Since no backward, just forward pass

### 6. Post-Grad Passes
- [fx_passes/post_grad.py:post_grad_passes()](torch/_inductor/fx_passes/post_grad.py)
- Checks for layout optimizations (none needed)

### 7. Graph Lowering
- [graph.py:GraphLowering.run()](torch/_inductor/graph.py)
- Lowers `aten.add` → `Pointwise(add)`
- Lowers `aten.relu` → `Pointwise(relu)`
- Result: Two `Pointwise` IR nodes

### 8. Scheduling
- [scheduler.py:Scheduler](torch/_inductor/scheduler.py)
- Analyzes dependencies: `relu` depends on `add`
- Fusion opportunity: Fuse `add + relu` into single kernel
- Creates `FusedSchedulerNode([add, relu])`

### 9. Triton Code Generation
- [codegen/triton.py:TritonScheduling](torch/_inductor/codegen/triton.py)
- Generates single Triton kernel:

```python
@triton.jit
def fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)

    # Fused computation
    z = x + y
    z = tl.where(z > 0, z, 0.0)  # relu

    tl.store(out_ptr0 + offsets, z, mask=mask)
```

### 10. Wrapper Generation
- [codegen/wrapper.py](torch/_inductor/codegen/wrapper.py)
- Generates Python wrapper:

```python
def inductor_wrapper(x, y):
    buf0 = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    fused_add_relu_0[grid](x, y, buf0, x.numel(), BLOCK_SIZE=256)
    return buf0
```

### 11. Compilation
- [codecache.py:PyCodeCache.load()](torch/_inductor/codecache.py)
- Compiles Triton kernel to PTX/CUBIN
- Compiles wrapper to Python bytecode
- Caches both on disk

### 12. Return Compiled Function
- Returns optimized callable
- First call: Runs kernel
- Subsequent calls: Uses cached version

## Debugging and Introspection

### Environment Variables

**Comprehensive Logging**:
```bash
export TORCH_LOGS="+dynamo,+aot,+inductor,+output_code"
```

**Individual Systems**:
```bash
export TORCH_LOGS="+dynamo"         # TorchDynamo graph capture
export TORCH_LOGS="+aot"            # AOTAutograd transformations
export TORCH_LOGS="+inductor"       # Inductor compilation
export TORCH_LOGS="+output_code"    # Generated kernel code
export TORCH_LOGS="+graph_code"     # FX graph code
```

**Granular Inductor Logging**:
```bash
export TORCH_LOGS="graph_breaks,graph_code,graph_sizes,guards,recompiles,+inductor"
export TORCH_LOGS_FORMAT="%(levelname)s: %(message)s"
```

**Specific Inductor Artifacts**:
```bash
export TORCH_LOGS="fusion"          # Fusion decisions
export TORCH_LOGS="schedule"        # Scheduling decisions
export TORCH_LOGS="perf_hints"      # Performance hints
export TORCH_LOGS="post_grad_graphs" # Post-grad FX graphs
```

### Config-Based Debugging

**Enable Tracing**:
```python
import torch._inductor.config as config

# Print generated code
config.trace.enabled = True

# Save intermediate graphs
config.debug = True

# Print FX graph transformations
config.trace.log_autotuning = True

# Print fusion decisions
config.trace.log_fusion = True

# Output directory for artifacts
config.trace.upload_tar = "/tmp/inductor_debug"
```

### Print Generated Code

**Method 1: TORCH_COMPILE_DEBUG**
```python
import torch

torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

@torch.compile
def foo(x):
    return x + 1

foo(torch.randn(10, device='cuda'))
# Prints all generated code to stdout
```

**Method 2: Save to File**
```python
import torch._inductor.config as config

config.trace.enabled = True
config.trace.upload_tar = "/tmp/inductor_output"

@torch.compile
def foo(x):
    return x + 1

foo(torch.randn(10, device='cuda'))
# Saves all artifacts to /tmp/inductor_output/
```

**Method 3: Access Directly**
```python
import torch
from torch._inductor import compile_fx

gm = torch.export.export(lambda x: x + 1, (torch.randn(10),)).module()
compiled = compile_fx(gm, [torch.randn(10, device='cuda')])

# compiled.output_code contains generated Python/Triton code
```

### Inspect Intermediate Stages

**View FX Graph**:
```python
@torch.compile(backend="eager")
def foo(x):
    return x + 1

# Or manually:
from torch._dynamo import optimize
gm, guards = optimize(torch._dynamo.eval_frame.innermost_fn)(foo, torch.randn(10))
print(gm.code)
```

**View Pre-Grad Graph**:
```python
from torch._inductor.fx_passes.pre_grad import pre_grad_passes

# After capturing FX graph
gm_pre = pre_grad_passes(gm, example_inputs)
print(gm_pre.code)
```

**View Post-Grad Graph**:
```python
from torch._inductor.fx_passes.post_grad import post_grad_passes

# After AOTAutograd
gm_post = post_grad_passes(gm, example_inputs)
print(gm_post.code)
```

**View Inductor IR**:
```python
import torch._inductor.config as config

config.debug = True  # Prints IR nodes

@torch.compile
def foo(x):
    return x.relu()

foo(torch.randn(10, device='cuda'))
# Prints: Pointwise IR nodes
```

**View Scheduling Decisions**:
```bash
export TORCH_LOGS="schedule"

# Or in Python:
import torch._logging
torch._logging.set_logs(schedule=True)
```

**View Fusion Decisions**:
```bash
export TORCH_LOGS="fusion"

# Or:
import torch._logging
torch._logging.set_logs(fusion=True)
```

### Inductor-Specific APIs

**Dump Generated Code**:
```python
from torch._inductor.compile_fx import compile_fx
import torch._inductor.config as config

config.trace.enabled = True

compiled = compile_fx(gm, example_inputs)
# Access generated code:
print(compiled.output_code.getvalue())
```

**Access Inductor Metrics**:
```python
from torch._inductor import metrics

@torch.compile
def foo(x):
    return x.relu()

result = foo(torch.randn(10, device='cuda'))

# View compilation metrics
print(metrics.get_metric_table())
# Shows: kernel count, fusion opportunities, etc.
```

**Disable Specific Optimizations**:
```python
import torch._inductor.config as config

# Disable fusion
config.aggressive_fusion = False

# Disable CUDA graphs
config.triton.cudagraphs = False

# Disable autotuning
config.max_autotune = False

# Use only cuBLAS (no CUTLASS/Triton)
config.use_cutlass = False
```

### Step Through Compilation

**Manual Compilation Pipeline**:
```python
import torch
from torch._dynamo import optimize
from torch._inductor.compile_fx import compile_fx
from torch._inductor.fx_passes.pre_grad import pre_grad_passes
from torch._inductor.fx_passes.post_grad import post_grad_passes

def manual_compile(fn, *example_inputs):
    # Step 1: Capture with Dynamo
    from torch._dynamo import export
    gm, _ = export(fn)(*example_inputs)
    print("=== FX Graph ===")
    print(gm.code)

    # Step 2: Pre-grad passes
    gm = pre_grad_passes(gm, example_inputs)
    print("\n=== After Pre-Grad ===")
    print(gm.code)

    # Step 3: Post-grad passes (simplified, normally AOTAutograd does this)
    gm = post_grad_passes(gm, is_inference=True)
    print("\n=== After Post-Grad ===")
    print(gm.code)

    # Step 4: Compile with Inductor
    compiled = compile_fx(gm, example_inputs)
    print("\n=== Generated Code ===")
    # Access via codecache

    return compiled

# Usage:
compiled = manual_compile(lambda x: x.relu(), torch.randn(10, device='cuda'))
```

## GPU-Specific Configuration

### Triton Settings

```python
import torch._inductor.config as config

# CUDA graphs
config.triton.cudagraphs = True

# Max autotuning
config.max_autotune = True
config.max_autotune_gemm_backends = "TRITON,CUTLASS,CUBLAS"

# Triton cache directory
config.triton.cache_dir = "/tmp/triton_cache"
```

### CUTLASS Settings

```python
# Enable CUTLASS
config.use_cutlass = True

# CUTLASS template preferences
config.cuda.cutlass_dir = "/path/to/cutlass"
config.cuda.cutlass_max_profiling_configs = 4
```

### Memory Optimization

```python
# Inplace operations
config.inplace_buffers = True

# Memory planning
config.memory_planning = True

# Benchmark memory-optimal configs
config.max_autotune_pointwise = True
```

## Related Systems

### TorchDynamo (Graph Capture)
**Package**: `torch._dynamo`
**Role**: Captures Python execution as FX graphs
**Handoff**: Passes FX graph to Inductor backend

### FX (Graph Representation)
**Package**: `torch.fx`
**Role**: Intermediate representation of PyTorch programs
**Used by**: Both Dynamo (output) and Inductor (input)

### AOTAutograd (Ahead-of-Time Autograd)
**Package**: `torch._functorch.aot_autograd`
**Role**: Tracing autograd logic before execution
**Used by**: Inductor for backward pass compilation

## Performance Tips

### 1. Enable Max Autotune
```python
@torch.compile(mode="max-autotune")
def foo(x):
    return x @ x.T
```

### 2. Enable CUDA Graphs
```python
@torch.compile(mode="reduce-overhead")
def foo(x):
    return x.relu()
```

### 3. Profile and Iterate
```bash
# See what's being compiled
export TORCH_LOGS="+dynamo,+inductor"

# Profile GPU kernels
nsys profile -o output python script.py
```

### 4. Check for Fusion Opportunities
```bash
export TORCH_LOGS="fusion"
# Look for "not fusing" messages
```

### 5. Warm Up for Autotuning
```python
# First run is slow (autotuning)
compiled_fn(x)

# Subsequent runs use cached best config
for _ in range(100):
    compiled_fn(x)  # Fast
```

## Common Issues and Solutions

### Issue: Recompilations
**Symptom**: Slow first call on every input shape
**Solution**: Use dynamic shapes or bucketing
```python
@torch.compile(dynamic=True)
def foo(x):
    return x.relu()
```

### Issue: Graph Breaks
**Symptom**: Python print in TORCH_LOGS: "graph break"
**Solution**: Refactor to avoid unsupported operations
```bash
export TORCH_LOGS="graph_breaks"
# Shows why Dynamo couldn't capture full graph
```

### Issue: Slow Compilation
**Symptom**: Long first-call latency
**Solution**: Reduce autotuning search space
```python
config.max_autotune_gemm_backends = "CUBLAS"  # Skip CUTLASS/Triton
config.autotune_fallback_to_aten = False
```

### Issue: Correctness Problems
**Symptom**: NaN or wrong results
**Solution**: Enable numerical checks
```python
config.debug_check_inf_and_nan = True
```

## Conclusion

This architecture document provides a comprehensive map of TorchInductor's design and implementation. Key takeaways:

1. **Entry Point**: `torch._inductor.compile()` called by Dynamo
2. **Pipeline**: FX → AOTAutograd → Passes → Lowering → Scheduling → Codegen
3. **IR Stages**: FX Graph → Inductor IR → Triton/CUTLASS/C++
4. **GPU Paths**: Triton for pointwise/reduction, CUTLASS for GEMM
5. **Optimization**: Fusion, layout optimization, autotuning

For more details, refer to the source files linked throughout this document.
