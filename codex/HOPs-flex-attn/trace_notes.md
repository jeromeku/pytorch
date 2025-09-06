How To Trace And Inspect

- Run the example:
  - `python codex/HOPs-flex-attn/minimal_example.py`

- Recommended env for maximum visibility:
  - `TORCH_COMPILE_DEBUG=1` — dumps FX and Inductor IR under a timestamped directory; see `torch/_dynamo/config.py:497` for `DEBUG_DIR_VAR_NAME`.
  - `TORCH_LOGS=+dynamo,inductor,aot` — verbose logs across phases.
  - Optional:
    - `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1` — dumps launch parameters for generated Triton kernels.
    - `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` — stable kernel names for easier grepping.

What you should see

- In the captured FX graph, a `call_function` node with target `torch.ops.higher_order.flex_attention` and two registered submodules on the root GraphModule named `sdpa_score` and `sdpa_mask` (inserted by the HOP trace_impl).
- In Inductor debug/dumps: the FlexAttention lowering constructs two subgraph buffers via `build_subgraph_buffer(...)` and inlines them inside Triton templates via the `modification(...)` macro (see templates common.py.jinja around the `post_mod_scores` / `mask_mod_output` sites).
- On CPU runs (device==cpu), a C++ template is selected instead (see `torch/_inductor/codegen/cpp_flex_attention_template.py`).

Pointers back to source

- HOP entry/impls: `torch/_higher_order_ops/flex_attention.py` (ProxyTorchDispatchMode, CompositeExplicitAutograd, functionalization, fake)
- Dynamo helper: `torch/_dynamo/_trace_wrapped_higher_order_op.py:123` (TransformGetItemToIndex)
- Inductor forward/backward lowerings: `torch/_inductor/kernel/flex/flex_attention.py`
- Common subgraph inlining/scatter: `torch/_inductor/kernel/flex/common.py`
- Templates: `torch/_inductor/kernel/flex/templates/*.jinja`

