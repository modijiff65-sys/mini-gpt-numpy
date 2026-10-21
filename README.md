# mini-gpt-numpy

A tiny character-level GPT — **112,577 parameters** — written from scratch in NumPy.
No PyTorch, no TensorFlow. Just a hand-written autograd engine, a small transformer, an
AdamW optimizer, and a resumable training loop.

It learns on tiny Shakespeare and starts writing Shakespeare-ish text.

```
ROMEO: fair good worn.

Second Servictorn:
And amentage, but this great great us miserable us
The being of than the ads, and for time
Not moore so thou lest the earther'd;
```

## What's inside `mini_ai.py`

| Piece | What it does |
|---|---|
| `Tensor` | the autograd engine — `+ * @ sum mean exp log reshape transpose`, each with its own backward pass |
| `softmax / cross_entropy / gelu / layernorm / embedding` | fused primitives, more numerically stable than composing them |
| `MiniGPT` | embeddings → N × [self-attention + MLP] → logits; 30 parameter tensors |
| `AdamW` | decoupled weight decay, bias-corrected moments |
| `train` / `sample` | resumable training (checkpoint save/load) + temperature & top-k sampling |

Config: `block_size=64`, `n_embd=64`, `n_head=4`, `n_layer=2`.

## Run it locally

```bash
pip install numpy
python mini_ai.py info
python mini_ai.py train --resume --steps 100000 --max_seconds 260
python mini_ai.py sample --tokens 400 --prompt "ROMEO: " --temperature 0.8 --top_k 40
```

## The gradients are correct

Before any training, the engine is checked against numeric (finite-difference) gradients
in float64 — every op and the whole model pass at ~1e-9 relative error.

```bash
python test_grad.py     # ends with "ALL PASS"
```

## Free cloud training (the jugaad)

A normal sandbox gives you 2 CPU cores and ~2 GB RAM. GitHub gives every repo a free CI
machine: on a **public** repo, standard runners are **free and unlimited**, and each job
gets **4 vCPU / 16 GB RAM** for up to **6 hours**.

`.github/workflows/train.yml` uses that. On a schedule (and on manual dispatch) it:

1. checks out the repo and installs NumPy,
2. runs the gradient sanity check,
3. trains one resumable chunk,
4. **commits the updated `ckpt.npz` back to the repo** — so the next run continues where
   this one stopped and no progress is ever lost.

Because the checkpoint lives in git, the loop is self-sustaining: every run picks up the
latest weights, trains, and pushes the result. Nothing to keep running on your machine.

Trigger a run by hand from the **Actions** tab → *mini-gpt training loop* → *Run workflow*.

### Notes and limits

- GitHub's own cron supports a 5-minute interval (`*/5 * * * *`), but scheduled workflows
  are disabled after 60 days of repository inactivity — any commit resets that.
- Each run commits a new `ckpt.npz` (~1.3 MB), so the repo grows over time. Lower the
  schedule frequency, or move the checkpoint to Git LFS / artifacts, if that matters.
- No GPU on the free tier. For GPU training, register a **self-hosted runner** on a
  machine with a GPU — GitHub Actions is free for self-hosted runners too.

## Files

| File | What it is |
|---|---|
| `mini_ai.py` | the whole model — autograd, GPT, optimizer, training, sampling |
| `test_grad.py` | float64 numeric gradient checks |
| `.github/workflows/train.yml` | the free cloud training loop |
| `ckpt.npz` | trained weights + optimizer state + step count (appears after the first run) |
| `loss_log.json` | train/val loss history |
| `shakespeare.txt` | the corpus (downloaded by the workflow if missing) |
