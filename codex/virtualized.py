from torch._inductor.virtualized import V, ops

"""
Minimal, correct examples showing how to use torch._inductor.virtualized.

Key idea: write your computation against V.ops (exposed here as `ops`).
Different handlers can then interpret the same body as a trace, a string
for codegen, or perform local simplifications.
"""


def loop_body(x):
    """Build a tiny expression using virtualized ops only.

    We purposely include a few identities so a simplifying handler can fold:
      - t + 0  -> t
      - 2 * 3  -> 6
      - (t + 0) + 6 -> t + 6
    """
    t = ops.arg(x)
    return ops.add(ops.add(t, 0), ops.mul(2, 3))


class TraceOps:
    """Toy handler that records calls and returns a simple tuple form."""

    def __init__(self):
        self.calls = []

    def arg(self, x):
        self.calls.append(("arg", x))
        return ("arg", x)

    def add(self, a, b):
        self.calls.append(("add", a, b))
        return ("add", a, b)

    def mul(self, a, b):
        self.calls.append(("mul", a, b))
        return ("mul", a, b)


# Example 1: Trace the body to a tuple form
tracer = TraceOps()
with V.set_ops_handler(tracer):
    traced = loop_body("x")
print("Trace calls:", tracer.calls)
print("Traced expr:", traced)


class SimplifyOps:
    """Wrapper handler that performs a few local simplifications, then delegates."""

    def __init__(self, inner):
        self.inner = inner

    def arg(self, x):
        return self.inner.arg(x)

    def add(self, a, b):
        # constant folding and x + 0 → x
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        if a == 0:
            return b
        if b == 0:
            return a
        return self.inner.add(a, b)

    def mul(self, a, b):
        # x * 1 → x, 1 * x → x, constant folding
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        if a == 1:
            return b
        if b == 1:
            return a
        return self.inner.mul(a, b)


# Example 2: Simplify on the fly, then trace what remains
base = TraceOps()
simp = SimplifyOps(base)
with V.set_ops_handler(simp):
    simplified = loop_body("x")
print("Simplified expr:", simplified)  # expected: ('add', ('arg', 'x'), 6)


# class CLikeOps:
#     """Turn the same body into a C-ish expression string."""

#     def arg(self, name):
#         return str(name)

#     def add(self, a, b):
#         return f"({a} + {b})"

#     def mul(self, a, b):
#         return f"({a} * {b})"


# # Example 3: Direct codegen to a string
# with V.set_ops_handler(CLikeOps()):
#     src = loop_body("x[i]")
# print("Emitted expr:", src)  # -> "((x[i] + 0) + (2 * 3))"

# If you want “post-simplified” codegen, just wrap it:
# with V.set_ops_handler(SimplifyOps(CLikeOps())):
#     src = loop_body("x[i]")
# print("Emitted, simplified:", src)  # -> "(x[i] + 6)"


# Example 4: Minimal “Inductor-like” IR building
# ---------------------------------------------
# Real Inductor lowerings write elementwise math in terms of V.ops.
# A handler converts those ops into an IR with basic CSE, which later
# stages fuse and codegen. Below is a tiny sketch of that idea.


class IRNode:
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __repr__(self):
        if self.op == "arg":
            return f"{self.args[0]}"
        sym = {"add": "+", "mul": "*"}.get(self.op, self.op)
        a, b = self.args
        return f"({a} {sym} {b})"


class IRGraph:
    def __init__(self):
        self.nodes = []
        self.intern = {}

    def node(self, op, args):
        # Basic CSE: reuse identical node if seen
        key = (op, args)
        if key in self.intern:
            return self.intern[key]
        n = IRNode(op, args)
        self.nodes.append(n)
        self.intern[key] = n
        return n


class InductorLikeOps:
    def __init__(self, g: IRGraph):
        self.g = g

    def arg(self, name):
        return self.g.node("arg", (str(name),))

    def add(self, a, b):
        return self.g.node("add", (a, b))

    def mul(self, a, b):
        return self.g.node("mul", (a, b))


g = IRGraph()
with V.set_ops_handler(InductorLikeOps(g)):
    ir = loop_body("x")

print("IR nodes (CSE-applied order):")
for i, n in enumerate(g.nodes):
    print(f"  n{i}: {n}")
print("Final expression:", ir)
