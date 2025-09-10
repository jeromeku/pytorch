
import contextlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config
from torch._inductor.debug import (
    create_kernel_information_json,
    create_mapping_pre_post_grad_nodes,
    create_node_mapping_kernel_to_post_grad,
    reset_inductor_kernel_provenance_debug_handle,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from test_aot_inductor_utils import AOTIRunnerUtil
from torch._logging import set_logs
import logging

set_logs(inductor=logging.DEBUG)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        x = a * 3.14
        y = torch.addmm(c, x, b)
        z = torch.nn.functional.gelu(y)
        return z


class Model2(torch.nn.Module):
    # this test model is used for combo kernel provenance tracing info
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        a1 = torch.nn.functional.relu(a)
        b1 = torch.nn.functional.sigmoid(b)
        c1 = torch.nn.functional.tanh(c)
        return a1, b1, c1

class Tester(TestCase):
    def _check_provenance_tracing_kernel_to_post_grad(self, filepath, expected_data):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        print(actual_data)
        actual_data = actual_data["cppCodeToPost"]
        # check that the generated provenance tracing artifact is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_data.items()))

    def _check_provenance_tracking_node_mappings(self, filepath, expected_mapping):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        # check that the generated provenance tracing node mapping is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_mapping))


def _test_pt_tracing_combo_kernel(self, backend):
    """This test checks that generated provenance tracing artifact from triton combo kernel to post grad nodes"""
    a = torch.randn(10, 10, device="cuda")
    b = torch.randn(20, 20, device="cuda")
    c = torch.randn(10, 10, device="cuda")
    example_inputs = (a, b, c)

    model = Model2()
    reset_inductor_kernel_provenance_debug_handle()

    with config.patch(
        {
            "trace.enabled": True,
            "trace.debug_dir": "./inductor_debug_no_fusion",
            "force_disable_caches": True,
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
        }
    ):
        # with self.assertLogs(
        #     logging.getLogger("torch._inductor.debug"),
        #     level=logging.WARNING,
        # ) as cm:
        if backend == "aot_inductor":
            AOTIRunnerUtil.run(model, example_inputs)
        elif backend == "export":
            ep = torch.export._trace._export(model, example_inputs)
            compiled = torch.compile(ep.module(), backend=backend)
            compiled(*example_inputs)
        else:
            compiled = torch.compile(model, fullgraph=True)
            compiled(*example_inputs)

        # self.assertEqual(len(cm.output), 1)
        # m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
        # self.assertTrue(m)
        # filepath = Path(m.group(1)).resolve()
        # expected_data = {"triton_poi_fused_0:1": ["relu", "sigmoid", "tanh"]}
        # self._check_provenance_tracing_kernel_to_post_grad(filepath, expected_data)

self = Tester()

_test_pt_tracing_combo_kernel(self, backend="compile")