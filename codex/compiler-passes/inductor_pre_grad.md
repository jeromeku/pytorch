# Inductor Pre-Grad Passes

* **Entry point:** [`pre_grad_passes`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L268-L360)
* **IR:** raw FX graph before autodiff

## Default pass order
| Step | Pass | Source | Purpose |
|---|---|---|---|
|1|`normalization_pass_aten`|[`pre_grad.py#L52`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L52)|standardize pre-dispatch ATen ops|
|2|`normalize_node_kwargs_pass`|[`pre_grad.py#L87-L88`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L87-L88)|canonicalize kwargs|
|3|`remove_noop_pass`|[`pre_grad.py#L111-L112`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L111-L112)|prune no-op view/alias ops|
|4|`relu_nan_to_num`|[`pre_grad.py#L123-L124`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L123-L124)|stabilize relu by replacing NaNs|
|5|`fuse_chunk_reshape_concat_pass`|[`pre_grad.py#L107-L108`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L107-L108)|fold chunk→reshape→concat sequences|
|6|`group_batch_fusion_passes`|[`group_batch_fusion.py#L1394-L1405`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/group_batch_fusion.py#L1394-L1405)|apply batch-level fusion heuristics|
|7|`normalize_node_kwargs_pass`|[`pre_grad.py#L87-L88`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L87-L88)|second canonicalization pass|
|8|`fuse_chunk_squeeze_cat_pass`|[`pre_grad.py#L44-L45`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L44-L45)|fuse chunk→squeeze→cat|
|9|`merge_concats_pass`|[`pre_grad.py#L119-L120`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L119-L120)|merge adjacent concats|
|10|`fuse_split_linear_add_pass`|[`pre_grad.py#L41-L43`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L41-L43)|fuse split→linear→add|
|11|`remove_reshape_pass`|[`pre_grad.py#L47-L49`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L47-L49)|eliminate redundant reshapes|
|12|`fuse_parallel_linear_pass`|[`pre_grad.py#L91-L92`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L91-L92)|combine parallel linear ops|
|13|`remove_split_ops_pass`|[`pre_grad.py#L99-L100`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L99-L100)|drop unused splits|
|14|`stack_to_unsqueeze_pass`|[`pre_grad.py#L115-L116`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L115-L116)|turn stack into unsqueeze|
|15|`fuse_chunk_reshape_unsqueeze_concat_pass`|[`pre_grad.py#L103-L104`](https://github.com/pytorch/pytorch/blob/3f3215be754e8c923c0f7adf5c0356efe074ff5c/torch/_inductor/fx_passes/pre_grad.py#L103-L104)|fuse chunk→reshape→unsqueeze→concat|

Optional passes such as `fuse_split_getitem_squeeze_cat` and Triton LCE passes can be enabled via the `add_passes` argument.

## Example
```python
import copy, torch
from torch._dynamo import export
from torch._inductor.fx_passes.pre_grad import pre_grad_passes

def toy(x):
    return torch.nn.functional.conv2d(x, torch.randn(1,1,3,3))

gm, _ = export(toy, torch.randn(1,1,5,5))
gm_opt = copy.deepcopy(gm)
print("before:\n", gm_opt.graph)
pre_grad_passes(gm_opt, [torch.randn(1,1,5,5)])
print("after:\n", gm_opt.graph)
```

**Related Issue**
- [#122379](https://github.com/pytorch/pytorch/issues/122379) – split/cat normalization bug.
