"""Rigorous gradient checks — run the engine in float64 so FD noise is negligible."""
import numpy as np
import mini_ai
mini_ai.DTYPE = np.float64            # crank precision for the check only
from mini_ai import (Tensor, softmax, cross_entropy, gelu, layernorm, embedding,
                     MiniGPT)

np.random.seed(0)
EPS = 1e-6


def rand(*s):
    return Tensor(np.random.randn(*s), requires_grad=True)


def check(name, tensors, loss_fn, tol=1e-6):
    for t in tensors:
        t.grad = None
    loss_fn().backward()
    ok = True
    for t in tensors:
        num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = t.data[idx]
            t.data[idx] = old + EPS; lp = loss_fn().data.item()
            t.data[idx] = old - EPS; lm = loss_fn().data.item()
            t.data[idx] = old
            num[idx] = (lp - lm) / (2 * EPS)
            it.iternext()
        ana = t.grad
        denom = max(np.max(np.abs(num)), np.max(np.abs(ana)), 1e-12)
        err = np.max(np.abs(ana - num)) / denom
        ok &= err < tol
        print(f"  [{'ok ' if err < tol else 'FAIL'}] {name:26s} rel err {err:.2e}")
    return ok


allok = True
a, b = rand(4, 5), rand(5, 3)
allok &= check("matmul", [a, b], lambda: (a @ b).sum())

a, b = rand(4, 1), rand(4, 5)
allok &= check("broadcast mul", [a, b], lambda: (a * b).sum())

a = rand(3, 6); w = np.random.randn(3, 6)
allok &= check("softmax weighted", [a], lambda: (softmax(a, -1) * w).sum())

logits = rand(5, 7); targets = np.random.randint(0, 7, size=5)
allok &= check("cross_entropy", [logits], lambda: cross_entropy(logits, targets))

a = rand(4, 6); w2 = np.random.randn(4, 6)
allok &= check("gelu weighted", [a], lambda: (gelu(a) * w2).sum())

x, g, be = rand(4, 8), rand(8), rand(8); w3 = np.random.randn(4, 8)
allok &= check("layernorm", [x, g, be], lambda: (layernorm(x, g, be) * w3).sum())

W = rand(10, 4); idx = np.random.randint(0, 10, size=(3, 5)); w4 = np.random.randn(3, 5, 4)
allok &= check("embedding", [W], lambda: (embedding(W, idx) * w4).sum())

m = MiniGPT(vocab=12, block_size=8, n_embd=16, n_head=2, n_layer=1, seed=1)
X = np.random.randint(0, 12, size=(2, 8))
Y = np.random.randint(0, 12, size=(2, 8))
allok &= check("model loss (all params)", list(m.params.values()),
               lambda: cross_entropy(m.forward(X).reshape(-1, 12), Y.reshape(-1)), tol=1e-5)

print("\nALL PASS" if allok else "\nSOME CHECKS FAILED")
