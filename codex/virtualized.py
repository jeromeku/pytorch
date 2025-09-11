from torch._inductor.virtualized import V, ops
# https://chatgpt.com/c/68c2cd78-9680-8321-a535-66e3207d9ec1

def loop_body(x):
    # Wrap the raw input so arithmetic like (t + 1) dispatches through V.ops
    t = ops.arg(x)
    # Deliberately includes no-op math to make simplification visible later
    return ((t + 0) * 1) + (2 * 3)   # should simplify to (t + 6)
class TraceOps:
    def __init__(self):
        self.calls = []
    def arg(self, x):              self.calls.append(("arg", x)); return x
    def add(self, a, b):           self.calls.append(("add", a, b)); return ("add", a, b)
    def mul(self, a, b):           self.calls.append(("mul", a, b)); return ("mul", a, b)

tracer = TraceOps()
with V.set_ops_handler(tracer):
    out = loop_body("x")

print("Trace calls:", tracer.calls)
print("Result placeholder:", out)

# -------------- #

class SimplifyOps:
    def __init__(self, inner):
        self.inner = inner
    def arg(self, x): return self.inner.arg(x)
    def add(self, a, b):
        # constant folding and x + 0 → x
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        if a == 0: return b
        if b == 0: return a
        return self.inner.add(a, b)
    def mul(self, a, b):
        # x * 1 → x, 1 * x → x, constant folding
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        if a == 1: return b
        if b == 1: return a
        return self.inner.mul(a, b)

base = TraceOps()
simp = SimplifyOps(base)
with V.set_ops_handler(simp):
    out = loop_body("x")
print("Simplified result:", out)   # expected: ('add','x',6) or plain math result if both sides folded

# ---------------- #

class CLikeOps:
    # Turn the same body into a C-ish expression string
    def arg(self, name):           return str(name)
    def add(self, a, b):           return f"({a} + {b})"
    def mul(self, a, b):           return f"({a} * {b})"

with V.set_ops_handler(CLikeOps()):
    src = loop_body("x[i]")
print("Emitted expr:", src)         # -> "((x[i] + 0) * 1) + (2 * 3)"
# If you want “post-simplified” codegen, just wrap it:
with V.set_ops_handler(SimplifyOps(CLikeOps())):
    src = loop_body("x[i]")
print("Emitted, simplified:", src)  # -> "(x[i] + 6)"

# # Fresh process only. If you’ve already run code above, a handler may still be installed.
# # The default for V.ops is a Mock/Null-style handler; touching it meaningfully will error.
# try:
#     print(ops.add(1, 2))  # will fail when no real handler is installed
# except Exception as e:
#     print("As expected, no handler installed ->", type(e).__name__, e)
