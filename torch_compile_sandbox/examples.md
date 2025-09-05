Relevant In-Tree Examples (tests)

Inductor FX passes
- post_grad_passes used explicitly:
  - test/inductor/test_provenance_tracing.py:498–504 — runs post_grad_passes on a graph and checks provenance propagation.
- joint_graph_passes coverage with simple patterns and counters:
  - test/inductor/test_pattern_matcher.py:784–787, 811–817, 821–824, 833–836, 843–846, 855–867, 871–900 — calls joint_graph.joint_graph_passes(gm) then validates simplified graphs.
- Custom pre/post passes examples:
  - test/inductor/test_custom_post_grad_passes.py:171–219 — patches config to inject custom passes and validates match counts.

Dynamo bytecode / eval frame
- Bytecode/eval frame hooks and accounting:
  - test/dynamo/test_utils.py:295–312 — exercises convert_frame.log_dynamo_start and related logging.
  - test/dynamo/test_logging.py:936–964 — structured log categories include "bytecode" and tooling helpers.
- Various tests also validate `set_eval_frame`, bytecode cache lifecycle, and package export paths, e.g.:
  - test/dynamo/test_frame_init.py:5–130
  - test/dynamo/test_misc.py:8566–8605 (DynamoOutput), 11424–11435 (bytecode cache freed), 12434–12462 (cellvar update in reconstructed bytecode)

End-to-end torch.compile usage in Inductor tests
- Numerous tests compile small functions/models with torch.compile to exercise the backend:
  - test/inductor/test_binary_folding.py:81, 192–199, 261
  - test/inductor/test_fused_attention.py:89, 249, 259
  - test/inductor/test_cuda_repro.py:154, 186, 211, 243, 294, 313, 334, 408, 416, 438, 453, 482, 505, 536, 562, 579, 661, 685, 702

Tip: open these files alongside pipeline_*.md to see pass effects in realistic contexts.

