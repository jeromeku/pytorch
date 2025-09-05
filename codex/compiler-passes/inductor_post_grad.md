# Inductor Post-Grad Passes

* **Entry point:** [`post_grad_passes`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L83-L210)
* **IR:** normalized, functional FX graph per direction

## Pass sequence
| Step | Pass | Source | Purpose |
|---|---|---|---|
|1|`remove_fsdp2_unsharded_param_graph_input_usage`|[`post_grad.py#L95-L96`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L95-L96)|drop redundant FSDP hooks|
|2|`gm.graph.eliminate_dead_code`|[`post_grad.py#L98-L100`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L98-L100)|remove dead nodes|
|3|`reorder_for_locality`|[`post_grad.py#L102-L105`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L102-L105)|improve memory locality (inference)|
|4|`post_grad_custom_pre_pass`|[`post_grad.py#L109-L112`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L109-L112)|optional user pre-pass|
|5|`grouped_gemm_pass` / `concat_linear_woq_int4`|[`post_grad.py#L114-L129`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L114-L129)|MKLDNN-specific fusions|
|6|`group_batch_fusion_passes`|[`post_grad.py#L131-L134`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L131-L134)|batch-level fusions|
|7|`remove_noop_ops`|[`post_grad.py#L135`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L135)|strip clones/aliases|
|8|`remove_assert_ops`|[`post_grad.py#L136-L137`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L136-L137)|remove tensor metadata asserts|
|9|`pass_patterns` loop|[`post_grad.py#L139-L142`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L139-L142)|pattern-based rewrites|
|10|`POST_GRAD_FUSIONS` loop|[`post_grad.py#L143-L166`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L143-L166)|optional fusions (e.g., attention)|
|11|`B2B_GEMM_PASS`|[`post_grad.py#L165-L166`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L165-L166)|back-to-back GEMM fusion|
|12|`micro_pipeline_tp_pass`|[`post_grad.py#L168-L170`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L168-L170)|tensor parallel pipeline|
|13|`fuse_ddp_communication`|[`post_grad.py#L171-L178`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L171-L178)|fuse distributed ops|
|14|`post_grad_custom_post_pass`|[`post_grad.py#L180-L183`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L180-L183)|optional user post-pass|
|15|`stable_topological_sort`|[`post_grad.py#L185`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L185)|final ordering|
|16|`move_constructors_to_gpu`|[`post_grad.py#L187-L188`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/post_grad.py#L187-L188)|place tensor constructors on device|

Additional backend-specific passes (custom backends, bucketing) follow afterward.

## Example
```python
import copy, torch
from torch._dynamo import export
from torch._inductor.fx_passes.post_grad import post_grad_passes

def f(x):
    y = x.clone()
    return y + 1

gm, _ = export(f, torch.randn(2))
gm2 = copy.deepcopy(gm)
print("before:\n", gm2.graph)
post_grad_passes(gm2, is_inference=False)
print("after:\n", gm2.graph)
```

**Related PR**
- [pytorch/pytorch#154460](https://github.com/pytorch/pytorch/pull/154460) – resolves `remove_noop_ops` type mismatch.
