"""
mini_ai.py — a tiny GPT ("mini AI") written from scratch in NumPy.

No PyTorch, no TensorFlow. Just NumPy and ~600 lines you can actually read.

It contains:
  1. A small autograd engine (Tensor + backward ops) — like micrograd, but array-based.
  2. A tiny GPT: token+position embeddings -> [attention -> MLP] x N -> logits.
  3. An AdamW optimizer.
  4. A resumable training loop (checkpoint save/load) so training can continue
     across many short runs.
  5. Text generation (temperature + top-k sampling).

Usage:
  python mini_ai.py info
  python mini_ai.py train  --steps 2000 --batch 16 --lr 3e-3 --resume
  python mini_ai.py sample --tokens 500 --prompt "ROMEO: "
"""

import argparse
import json
import math
import os
import time

import numpy as np

DTYPE = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.npz")
LOSS_LOG = os.path.join(HERE, "loss_log.json")
CORPUS = os.path.join(HERE, "shakespeare.txt")


# ---------------------------------------------------------------------------
# 1. Autograd engine
# ---------------------------------------------------------------------------

def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after numpy broadcasting."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _noop():
    return None


def _acc(t, g):
    """Accumulate a gradient into tensor t."""
    g = np.asarray(g, dtype=DTYPE)
    if t.grad is None:
        t.grad = g
    else:
        t.grad = t.grad + g


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """A minimal reverse-mode autodiff tensor backed by a NumPy array."""

    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=DTYPE)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = _children
        self._op = _op
        self._backward = lambda: None

    # --- elementwise / arithmetic ---
    def __add__(self, other):
        other = _t(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = _t(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_t(other))

    def __rsub__(self, other):
        return _t(other) + (-self)

    def __truediv__(self, other):
        other = _t(other)
        out = Tensor(self.data / other.data,
                     self.requires_grad or other.requires_grad, (self, other), "/")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad / other.data, self.data.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(-out.grad * self.data / (other.data ** 2),
                                         other.data.shape))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = _t(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "@")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad @ other.data.swapaxes(-1, -2),
                                        self.data.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(self.data.swapaxes(-1, -2) @ out.grad,
                                         other.data.shape))
        out._backward = _backward
        return out

    def pow(self, p):
        out = Tensor(self.data ** p, self.requires_grad, (self,), "pow")

        def _backward():
            if self.requires_grad:
                _acc(self, out.grad * p * (self.data ** (p - 1)))
        out._backward = _backward
        return out

    # --- reductions ---
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _acc(self, np.broadcast_to(g, self.data.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "mean")
        n = self.data.size if axis is None else self.data.shape[axis]

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _acc(self, np.broadcast_to(g, self.data.shape).copy() / n)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                _acc(self, out.grad * out.data)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                _acc(self, out.grad / self.data)
        out._backward = _backward
        return out

    # --- shape ops ---
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                _acc(self, out.grad.reshape(self.data.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")
        inv = tuple(np.argsort(axes))

        def _backward():
            if self.requires_grad:
                _acc(self, out.grad.transpose(inv))
        out._backward = _backward
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], self.requires_grad, (self,), "getitem")

        def _backward():
            if not self.requires_grad:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad[key] += out.grad
        out._backward = _backward
        return out

    # --- reverse pass ---
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
            # release this node's closure + edges now that it is consumed, so the
            # graph is acyclic and gets freed immediately (keeps memory flat)
            v._prev = ()
            v._backward = _noop


# --- primitive ops that are cleaner (and more stable) fused ---

def softmax(x, axis=-1):
    xd = x.data
    xd = xd - np.max(xd, axis=axis, keepdims=True)
    e = np.exp(xd)
    s = e / np.sum(e, axis=axis, keepdims=True)
    out = Tensor(s, x.requires_grad, (x,), "softmax")

    def _backward():
        if x.requires_grad:
            g = out.grad
            _acc(x, s * (g - np.sum(g * s, axis=axis, keepdims=True)))
    out._backward = _backward
    return out


def cross_entropy(logits, targets):
    """Mean token cross-entropy. logits [N,V], targets int [N]."""
    x = logits.data
    n = targets.size
    z = x - np.max(x, axis=-1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
    flat = logp.reshape(-1, logp.shape[-1])
    tgt = targets.reshape(-1)
    loss = -np.mean(flat[np.arange(n), tgt])
    out = Tensor(loss, logits.requires_grad, (logits,), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            p = np.exp(flat)
            p[np.arange(n), tgt] -= 1.0
            p /= n
            _acc(logits, p.reshape(x.shape))
    out._backward = _backward
    return out


def gelu(x):
    """GPT-2 style tanh-approximation GELU."""
    c = math.sqrt(2.0 / math.pi)
    xd = x.data
    u = c * (xd + 0.044715 * xd ** 3)
    t = np.tanh(u)
    out = Tensor(0.5 * xd * (1.0 + t), x.requires_grad, (x,), "gelu")

    def _backward():
        if x.requires_grad:
            du = c * (1.0 + 3.0 * 0.044715 * xd ** 2)
            d = 0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t * t) * du
            _acc(x, out.grad * d)
    out._backward = _backward
    return out


def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(-1, keepdims=True)
    inv = (var + eps).pow(-0.5)
    return (xc * inv) * gamma + beta


def embedding(W, idx):
    idx = np.asarray(idx)
    out = Tensor(W.data[idx], W.requires_grad, (W,), "embedding")

    def _backward():
        if not W.requires_grad:
            return
        if W.grad is None:
            W.grad = np.zeros_like(W.data)
        np.add.at(W.grad, idx.reshape(-1), out.grad.reshape(-1, W.data.shape[-1]))
    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------

class MiniGPT:
    def __init__(self, vocab, block_size=64, n_embd=64, n_head=4, n_layer=2, seed=1337):
        assert n_embd % n_head == 0
        self.vocab, self.block_size = vocab, block_size
        self.n_embd, self.n_head, self.n_layer = n_embd, n_head, n_layer
        self.hd = n_embd // n_head
        rng = np.random.default_rng(seed)
        self.params = {}

        def param(name, *shape, scale=0.02):
            self.params[name] = Tensor(
                rng.normal(0.0, scale, size=shape).astype(DTYPE),
                requires_grad=True)

        param("tok_emb", vocab, n_embd, scale=0.02)
        param("pos_emb", block_size, n_embd, scale=0.01)
        for i in range(n_layer):
            param(f"ln1g{i}", n_embd, scale=1.0); param(f"ln1b{i}", n_embd, scale=0.0)
            param(f"ln2g{i}", n_embd, scale=1.0); param(f"ln2b{i}", n_embd, scale=0.0)
            param(f"Wq{i}", n_embd, n_embd); param(f"bq{i}", n_embd, scale=0.0)
            param(f"Wk{i}", n_embd, n_embd); param(f"bk{i}", n_embd, scale=0.0)
            param(f"Wv{i}", n_embd, n_embd); param(f"bv{i}", n_embd, scale=0.0)
            param(f"Wo{i}", n_embd, n_embd); param(f"bo{i}", n_embd, scale=0.0)
            param(f"W1{i}", n_embd, 4 * n_embd); param(f"b1{i}", 4 * n_embd, scale=0.0)
            param(f"W2{i}", 4 * n_embd, n_embd); param(f"b2{i}", n_embd, scale=0.0)
        param("lnfg", n_embd, scale=1.0); param("lnfb", n_embd, scale=0.0)
        param("Wlm", n_embd, vocab); param("blm", vocab, scale=0.0)

        # causal mask (constant, no grad): 0 on/below diagonal, -inf above
        m = np.zeros((block_size, block_size), dtype=np.float32)
        m[np.triu_indices(block_size, k=1)] = -1e9
        self.mask = m

    def n_params(self):
        return sum(p.data.size for p in self.params.values())

    def _block(self, x, i):
        P, B, T = self.params, x.data.shape[0], x.data.shape[1]
        H, hd = self.n_head, self.hd

        h = layernorm(x, P[f"ln1g{i}"], P[f"ln1b{i}"])
        q = h @ P[f"Wq{i}"] + P[f"bq{i}"]
        k = h @ P[f"Wk{i}"] + P[f"bk{i}"]
        v = h @ P[f"Wv{i}"] + P[f"bv{i}"]

        def heads(t):  # [B,T,d] -> [B*H,T,hd]
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3).reshape(B * H, T, hd)

        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(0, 2, 1)) * (1.0 / math.sqrt(hd))  # [B*H,T,T]
        att = att + self.mask[:T, :T]
        att = softmax(att, axis=-1)
        y = (att @ v).reshape(B, H, T, hd).transpose(0, 2, 1, 3).reshape(B, T, self.n_embd)
        x = x + (y @ P[f"Wo{i}"] + P[f"bo{i}"])

        h2 = layernorm(x, P[f"ln2g{i}"], P[f"ln2b{i}"])
        h2 = gelu(h2 @ P[f"W1{i}"] + P[f"b1{i}"])
        x = x + (h2 @ P[f"W2{i}"] + P[f"b2{i}"])
        return x

    def forward(self, idx):
        P, B, T = self.params, idx.shape[0], idx.shape[1]
        x = embedding(P["tok_emb"], idx) + P["pos_emb"][:T].reshape(1, T, self.n_embd)
        for i in range(self.n_layer):
            x = self._block(x, i)
        x = layernorm(x, P["lnfg"], P["lnfb"])
        return x @ P["Wlm"] + P["blm"]


# ---------------------------------------------------------------------------
# 3. Optimizer
# ---------------------------------------------------------------------------

class AdamW:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.01):
        self.params = params          # dict name -> Tensor
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for name, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            mhat = self.m[name] / bc1
            vhat = self.v[name] / bc2
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * p.data)


# ---------------------------------------------------------------------------
# 4. Data, training, sampling, checkpointing
# ---------------------------------------------------------------------------

def load_corpus():
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return text, chars, stoi, itos, data


def get_batch(data, block_size, batch_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def save_ckpt(model, opt, step, chars, losses, rng):
    arrays = {f"param::{k}": p.data for k, p in model.params.items()}
    arrays.update({f"m::{k}": v for k, v in opt.m.items()})
    arrays.update({f"v::{k}": v for k, v in opt.v.items()})
    arrays["meta::step"] = np.array([step])
    arrays["meta::t"] = np.array([opt.t])
    arrays["meta::chars"] = np.array(list("".join(chars)))
    np.savez(CKPT, **arrays)
    with open(LOSS_LOG, "w") as f:
        json.dump(losses, f)


def try_load(model, opt, chars):
    if not os.path.exists(CKPT):
        return 0
    z = np.load(CKPT, allow_pickle=False)
    ck_chars = "".join(z["meta::chars"].tolist())
    if ck_chars != "".join(chars):
        print("[warn] vocab differs from checkpoint; starting fresh")
        return 0
    for k, p in model.params.items():
        p.data = z[f"param::{k}"].astype(DTYPE)
    for k in opt.m:
        opt.m[k] = z[f"m::{k}"].astype(DTYPE)
        opt.v[k] = z[f"v::{k}"].astype(DTYPE)
    opt.t = int(z["meta::t"][0])
    return int(z["meta::step"][0])


def train(args):
    text, chars, stoi, itos, data = load_corpus()
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    rng = np.random.default_rng(args.seed + 1)

    model = MiniGPT(len(chars), args.block, args.embd, args.head, args.layer, seed=args.seed)
    opt = AdamW(model.params, lr=args.lr, wd=args.wd)
    step = try_load(model, opt, chars) if args.resume else 0
    losses = []
    if os.path.exists(LOSS_LOG):
        try:
            losses = json.load(open(LOSS_LOG))
        except Exception:
            losses = []

    print(f"mini-gpt: {model.n_params():,} params | vocab {len(chars)} | "
          f"block {args.block} | resume step {step}")
    t0 = time.time()
    running = None
    last_val = None
    for s in range(args.steps):
        if time.time() - t0 > args.max_seconds:
            print(f"[time budget] stopping at step {step} to stay within the run limit")
            break
        x, y = get_batch(train_data, args.block, args.batch, rng)
        logits = model.forward(x)
        loss = cross_entropy(logits.reshape(-1, logits.data.shape[-1]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        running = loss.data.item() if running is None else 0.9 * running + 0.1 * loss.data.item()

        if step % args.log_every == 0 or s == 0:
            vx, vy = get_batch(val_data, args.block, args.batch, rng)
            vl = cross_entropy(model.forward(vx).reshape(-1, len(chars)), vy)
            last_val = vl.data.item()
            losses.append({"step": step, "train": round(float(running), 4),
                           "val": round(float(last_val), 4)})
            print(f"step {step:6d} | train {running:.3f} | val {last_val:.3f} "
                  f"| {(time.time()-t0):.0f}s")

        if step % args.save_every == 0:
            save_ckpt(model, opt, step, chars, losses, rng)

    save_ckpt(model, opt, step, chars, losses, rng)
    print(f"done: step {step} | saved checkpoint -> {os.path.basename(CKPT)}")
    return step


def sample(args):
    text, chars, stoi, itos, data = load_corpus()
    model = MiniGPT(len(chars), args.block, args.embd, args.head, args.layer, seed=args.seed)
    opt = AdamW(model.params)
    step = try_load(model, opt, chars)
    print(f"# loaded checkpoint at step {step}\n")

    prompt = args.prompt or "\n"
    idx = np.array([[stoi.get(c, 0) for c in prompt]], dtype=np.int64)
    rng = np.random.default_rng(args.seed + 7)
    V = len(chars)

    for _ in range(args.tokens):
        cond = idx[:, -args.block:]
        logits = model.forward(cond).data[0, -1] / max(args.temperature, 1e-6)
        if args.top_k:
            kth = np.sort(logits)[-args.top_k]
            logits = np.where(logits < kth, -1e9, logits)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        nxt = rng.choice(V, p=p)
        idx = np.concatenate([idx, np.array([[nxt]])], axis=1)

    out = prompt + "".join(itos[i] for i in idx[0, len(prompt):])
    print(out)


def info(args):
    text, chars, stoi, itos, data = load_corpus()
    model = MiniGPT(len(chars), args.block, args.embd, args.head, args.layer)
    print(f"corpus     : {len(text):,} chars, vocab {len(chars)}")
    print(f"params     : {model.n_params():,}")
    print(f"config     : block={args.block} embd={args.embd} head={args.head} layer={args.layer}")
    if os.path.exists(CKPT):
        z = np.load(CKPT, allow_pickle=False)
        print(f"checkpoint : step {int(z['meta::step'][0])}")
    else:
        print("checkpoint : none yet")


def main():
    p = argparse.ArgumentParser(description="Tiny GPT from scratch in NumPy")
    p.add_argument("cmd", choices=["train", "sample", "info"])
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--embd", type=int, default=64)
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--max_seconds", type=float, default=240.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()
    {"train": train, "sample": sample, "info": info}[args.cmd](args)


if __name__ == "__main__":
    main()
