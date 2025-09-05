# Inductor Joint Graph Passes

* **Entry point:** [`joint_graph_passes`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L561-L619)
* **IR:** joint forward+backward FX graph

## Pass sequence
| Step | Pass | Source | Purpose |
|---|---|---|---|
|1|`canonicalize_aten_ir_passes`|[`joint_graph.py#L573-L575`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L573-L575)|normalize ATen IR before other passes|
|2|`joint_custom_pre_pass`|[`joint_graph.py#L576-L580`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L576-L580)|optional user-specified pre-pass|
|3|`remove_noop_ops`|[`joint_graph.py#L582-L585`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L582-L585)|strip view/alias ops|
|4|`constant_fold_uniform_value`|[`joint_graph.py#L586-L589`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L586-L589)|fold uniform constants|
|5|`joint_custom_pre_pass` (again)|[`joint_graph.py#L591-L595`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L591-L595)|allow second pre-pass after folding|
|6|`pass_patterns`|[`joint_graph.py#L597-L603`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L597-L603)|pattern-based canonicalization|
|7|`replace_random_passes`|[`joint_graph.py#L604-L607`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L604-L607)|swap `aten.rand*` for backend RNG|
|8|`joint_custom_post_pass`|[`joint_graph.py#L609-L612`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L609-L612)|optional user-specified post-pass|
|9|`stable_topological_sort`|[`joint_graph.py#L615-L618`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/joint_graph.py#L615-L618)|final graph cleanup|

## Example
```python
import copy, torch
from torch._dynamo import export
from torch._inductor.fx_passes.pre_grad import pre_grad_passes
from torch._inductor.fx_passes.joint_graph import joint_graph_passes

def f(x):
    return x + torch.rand_like(x)

gm, _ = export(f, torch.randn(2))
pre_grad_passes(gm, [torch.randn(2)])
gm2 = copy.deepcopy(gm)
print("before:\n", gm2.graph)
joint_graph_passes(gm2)
print("after:\n", gm2.graph)
```

**Related Issue**
- [#117957](https://github.com/pytorch/pytorch/issues/117957) – SDPA accuracy and attention fusion updates.
