# Dynamo Front-end Passes

## 1. Frame Conversion
* **Source:** [`torch/_dynamo/convert_frame.py`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_dynamo/convert_frame.py#L1584-L1619)
* **IR:** Python bytecode → FX graph
* **Purpose:** intercepts Python execution to trace operators into an FX `GraphModule`.
* **Snippet:**
  ```python
  import torch
  def fn(x): return (x + x).relu()
  gm, _ = torch._dynamo.export(fn, torch.randn(2,2))
  print("captured graph:\n", gm.graph)
  ```

## 2. Graph Deduplication
* **Source:** [`torch/_dynamo/graph_deduplication.py`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_dynamo/graph_deduplication.py#L37-L63)
* **IR:** FX `GraphModule`
* **Purpose:** finds structurally identical regions and replaces them with a shared subgraph via `apply_graph_deduplication`.
* **Snippet:**
  ```python
  from torch._dynamo.graph_deduplication import apply_graph_deduplication
  gm, _ = torch._dynamo.export(fn, torch.randn(2,2))
  print("before:\n", gm.graph)
  apply_graph_deduplication(gm)
  print("after:\n", gm.graph)
  ```

## 3. Runtime Asserts & Dead Code Elimination
* **Source:** [`torch/fx/passes/runtime_assert.py`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/fx/passes/runtime_assert.py#L55-L91)
* **IR:** FX `GraphModule`
* **Purpose:** inserts shape/runtime guards and enables dead argument pruning.
* **Snippet:**
  ```python
  from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
  gm, _ = torch._dynamo.export(fn, torch.randn(2,2))
  print("before:\n", gm.graph)
  insert_deferred_runtime_asserts(gm, gm.shape_env, "demo")
  gm.graph.eliminate_dead_code(); gm.recompile()
  print("after:\n", gm.graph)
  ```

**Related Issues/PRs**
- [#122379](https://github.com/pytorch/pytorch/issues/122379) – shape bug in `split_cat_norm`.
- [#117957](https://github.com/pytorch/pytorch/issues/117957) – SDPA accuracy and attention fusion.
- [pytorch/pytorch#154460](https://github.com/pytorch/pytorch/pull/154460) – fix `remove_noop_ops` type mismatch.
