#!/usr/bin/env python
import os, subprocess, sys
HERE=os.path.dirname(__file__)
for d in ["demo_max_autotune.py","demo_gemm_backends.py","demo_triton_cudagraphs.py","demo_trace_enabled.py","demo_fx_graph_cache.py","demo_worker_threads.py"]:
 print('===',d); subprocess.run([sys.executable, os.path.join(HERE,d)], check=False)
print('Done.')
