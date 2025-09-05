import torch
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.virtualized import V
from torch._dynamo.utils import detect_fake_mode


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x * 3


if __name__ == "__main__":
    x = torch.randn(8, 10)
    ep = torch.export.export(Model().eval(), (x,)).run_decompositions()
    gm = ep.module()

    fake_inputs = [node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"]
    fake_mode = detect_fake_mode(fake_inputs)
    V.set_fake_mode(fake_mode)

    print("Before:")
    print(gm.print_readable(print_output=False))
    post_grad_passes(gm, is_inference=True)
    print("\nAfter post_grad_passes:")
    print(gm.print_readable(print_output=False))

