User → HOP → Dynamo/FX → Inductor → Kernel

This is the annotated, end-to-end trace of how FlexAttention is implemented with Higher‑Order Ops (HOPs) and compiled by Inductor. Each step includes clickable file:line references and short code excerpts.

1) User API entry (torch.nn.attention.flex_attention)

- Path: torch/nn/attention/flex_attention.py:1664
- Source (excerpt):

  - torch/nn/attention/flex_attention.py:1664
    out, lse, max_scores = flex_attention_hop(
      query, key, value, score_mod, block_mask.as_tuple(), scale, kernel_options,
    )

- Context: When already inside a Dynamo compile (torch.compiler.is_dynamo_compiling()), the user API calls the HOP directly. Otherwise it wraps the HOP in torch.compile to trigger compilation.

  - torch/nn/attention/flex_attention.py:1658
    if torch.compiler.is_dynamo_compiling():
        ... call HOP directly ...
    else:
        ... wrap _flex_attention_hop_wrapper with torch.compile(..., fullgraph=True) ...

  - torch/nn/attention/flex_attention.py:1715
    flex_fn = torch.compile(_flex_attention_hop_wrapper, backend=backend, fullgraph=True)

2) The FlexAttention HOP definition and Python impls

- HOP object and __call__:
  - torch/_higher_order_ops/flex_attention.py:80
    class FlexAttentionHOP(HigherOrderOperator):
      def __init__(self): super().__init__("flex_attention", cacheable=True)
  - torch/_higher_order_ops/flex_attention.py:110
    flex_attention = FlexAttentionHOP()

- CompositeExplicitAutograd (math fallback, eager):
  - torch/_higher_order_ops/flex_attention.py:270
    @flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
    def sdpa_dense(...):
        out, lse, max_scores = math_attention(...)
        out = _permute_strides(out, query.stride()); return out, lse, max_scores

- ProxyTorchDispatchMode (FX tracing for Dynamo/make_fx):
  - torch/_higher_order_ops/flex_attention.py:366
    @flex_attention.py_impl(ProxyTorchDispatchMode)
    def flex_attention_proxy_torch_dispatch_mode(mode, query, key, value, score_mod, block_mask, ...):
        return trace_flex_attention(mode, query, key, value, score_mod, block_mask, ...)

- HOP tracing hook (captures score_mod and mask_mod subgraphs):
  - torch/_higher_order_ops/flex_attention.py:297
    def trace_flex_attention(proxy_mode, query, key, value, score_mod, block_mask, scale, kernel_options, ...):
      from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
      example_out = flex_attention(...)
      example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] + [query.new_zeros((), dtype=torch.int) for _ in range(4)]
      with TransformGetItemToIndex():
          score_graph = reenter_make_fx(score_mod)(*example_vals, *score_mod_other_buffers)
          mask_graph = reenter_make_fx(mask_mod)(*mask_example_vals, *mask_mod_other_buffers)
      qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score"); root.register_module(qualname, score_graph)
      mask_qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_mask"); root.register_module(mask_qualname, mask_graph)
      out_proxy = tracer.create_proxy("call_function", flex_attention, proxy_args, {})
      return track_tensor_tree(example_out, out_proxy, ...)

- Functionalization and FakeTensor support:
  - torch/_higher_order_ops/flex_attention.py:394
    @flex_attention.py_functionalize_impl ... def flex_attention_functionalize(...)
  - torch/_higher_order_ops/flex_attention.py:480
    @register_fake(flex_attention) def flex_attention_fake_impl(...)

3) Small Dynamo helper used during tracing

- Path: torch/_dynamo/_trace_wrapped_higher_order_op.py:123
- Purpose: TransformGetItemToIndex changes Tensor.__getitem__ behavior so score_mod(mask) indexing with tensor indices emits ops kept in the trace.
- Excerpt:
  class TransformGetItemToIndex(TorchFunctionMode):
    def __torch_function__(..., func, ...):
      if func == torch.Tensor.__getitem__:
        ... return mod_index(args[0], index_args)

4) Inductor lowering entry points for FlexAttention

- Forward op lowering registration:
  - torch/_inductor/kernel/flex/flex_attention.py:78
    @register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
    def flex_attention(query, key, value, subgraph, block_mask, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers):
      if device == "cpu": return lower_cpu(...)
      ... build subgraph buffers from captured FX graphs ...
      subgraph_buffer = build_subgraph_buffer(placeholder_inps + list(score_mod_other_buffers), subgraph)
      mask_graph_buffer = build_subgraph_buffer(mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph)
      ... choose template; possibly flex_decode; else build Triton template ...

- Backward op lowering registration:
  - torch/_inductor/kernel/flex/flex_attention.py:522
    @register_lowering(torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None)
    def flex_attention_backward(...):
      ... validate and process joint_graph ... create Triton template for bwd ...

- Common flex utilities for subgraph inlining and scatter grads:
  - torch/_inductor/kernel/flex/common.py:103
    def build_subgraph_module_buffer(args, graph_module):
      from ...subgraph_lowering import PointwiseSubgraphLowering
      pw_subgraph = PointwiseSubgraphLowering(..., additional_lowerings={zeros_and_scatter: zeros_and_scatter_lowering})
      pw_subgraph.run(*args)
      ... convert graph outputs into ComputedBuffer(s) to inline into templates ...

  - torch/_inductor/kernel/flex/common.py:44
    def zeros_and_scatter_lowering(shape, indices, values):
      grad = _full(0, ...);
      scatter = Scatter(..., scatter_mode="atomic_add")
      buffer = ComputedBuffer(..., layout=MutationLayoutSHOULDREMOVE(...), data=scatter)

5) Inductor IR and template glue

- IR allowance for HOPs mutating input(s) via template buffers:
  - torch/_inductor/ir.py:4896
    allowed_set = (torch.ops.higher_order.flex_attention, torch.ops.higher_order.flex_attention_backward)

- Template selection heuristics (BLOCK sizes, etc.):
  - torch/_inductor/choices.py:89
    def get_flex_attention_fwd_configs(self, head_dim, dtype, device_type="cuda") -> list[Any]
  - torch/_inductor/choices.py:95
    def get_flex_attention_bwd_configs(self, head_dim, dtype, device_type="cuda") -> list[Any]

6) Kernel templates and where score/mask subgraphs are inlined

- Triton forward template entry:
  - torch/_inductor/kernel/flex/flex_attention.py:69
    flex_attention_template = TritonTemplate(name="flex_attention", grid=flex_attention_grid, source=load_template("flex_attention") + load_template("utilities") + load_template("common"))

- The template calls forward_inner, which applies score_mod and mask_mod via generated code from captured subgraphs:
  - torch/_inductor/kernel/flex/templates/common.py.jinja:52
    {{ modification(subgraph_number=0, output_name="post_mod_scores", score="qk", b="off_z", h="off_h", m="m", n="n", out="qk") }}
  - torch/_inductor/kernel/flex/templates/common.py.jinja:68
    {{ modification(subgraph_number=1, output_name="mask_mod_output", score="qk", b="off_z", h="off_h", m="m", n="n") }}

- CPU fallback path uses a C++ codegen template:
  - torch/_inductor/codegen/cpp_flex_attention_template.py:211
    FLEX_ATTENTION_TEMPLATE = r""" ... generates vectorized CPU kernels and brgemm helpers ... """

7) Backprop graph capture and kernel

- The HOP defines a FlexAttentionBackwardHOP similarly, with ProxyTorchDispatchMode impl that records fw_graph and joint_graph used by Inductor bwd lowering:
  - torch/_higher_order_ops/flex_attention.py:113
    class FlexAttentionBackwardHOP(HigherOrderOperator): ...
  - torch/_higher_order_ops/flex_attention.py:1086
    @flex_attention_backward.py_impl(ProxyTorchDispatchMode) def flex_attention_backward_proxy_torch_dispatch_mode(...)
  - torch/_higher_order_ops/flex_attention.py:990
    def trace_flex_attention_backward(...): ... build graphs and return a call_function to flex_attention_backward ...

8) Where the HOP gets into your FX graph

- During Dynamo capture, the ProxyTorchDispatchMode impl inserts a call_function node targeting `torch.ops.higher_order.flex_attention` and registers `sdpa_score`/`sdpa_mask` GraphModules on the root tracer. This is what Inductor later consumes via `build_subgraph_buffer` to inline your score/mask logic into the kernel.

9) Minimal mental model of data/shape flow

- Inputs: (query, key, value) of shapes [Bq, Hq, M, Dqk] / [Bkv, Hkv, N, Dqk] / [Bkv, Hkv, N, Dv].
- HOP (ProxyTorchDispatchMode): captures two subgraphs:
  - score_mod: f(score, b, h, m, n, ...buffers) → modified score
  - mask_mod:  f(b, h, m, n, ...buffers) → boolean mask
- Inductor lowering:
  - Lifts both subgraphs into Triton template symbols via `build_subgraph_buffer(...)` and template `modification(...)` macro calls.
  - Selects kernel configs via Inductor choices; emits forward Triton kernel and optional CPU template.
  - Backward: captures fw_graph and joint_graph and emits a Triton kernel with captured-grad plumbing.

10) Why HOPs help with compiling FlexAttention

- Clear boundary: HOPs isolate the high-level op boundary (`flex_attention`) and let the dispatcher install specialized behaviors by mode: eager math, FX tracing, functionalization, autograd, fake tensor.
- Safe capture: The ProxyTorchDispatchMode impl runs `make_fx` on user score/mask functions under a `TransformGetItemToIndex` TorchFunctionMode to preserve tensor indexing semantics in the trace.
- Inductor‑friendly: The HOP surface produces a single FX `call_function` to `torch.ops.higher_order.flex_attention` with attached subgraph modules; Inductor can lower this atomically into a template and inline those subgraphs directly into the Triton kernel.
- Backward graphing: The HOP’s backward op similarly captures the necessary subgraphs (`fw_graph` and `joint_graph`) so Inductor can generate fused backprop kernels including gradients for captured buffers.

