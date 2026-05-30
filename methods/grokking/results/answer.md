# Grokking: generalization far beyond overfitting

## Problem

Why and when does an overparameterized network generalize rather than merely memorize? Natural
datasets make this nearly impossible to study cleanly — slow, confounded, mild effects. The goal is a
controllable testbed where memorization and true generalization are cleanly separable, the underlying
structure is exactly known, experiments run on one GPU, and the generalization dynamics are pronounced.

## Setup

Train a small transformer to fill in the blanks of a **binary operation table** a∘b=c. Every element
a, b, c is an **abstract symbol with no internal structure** (no decimal digits, no permutation
notation), so the only way to predict a held-out cell is to infer the operation from symbol
interactions. Each equation is the token sequence ⟨a⟩ ⟨∘⟩ ⟨b⟩ ⟨=⟩ ⟨c⟩; a random training fraction of
the full table is the train set, the rest is validation (disjoint cells of the *same* table, so
generalization is fully decoupled from training performance).

- **Operations** (prime p = 97): x+y, x−y, x/y, x²+y², x²+xy+y², x²+xy+y²+x, x³+xy, x³+xy²+y (mod 97);
  a mixed op; and S₅ products x·y, x·y·x⁻¹, x·y·x.
- **Model:** decoder-only transformer, causal masking, **2 layers, width 128, 4 heads** (~4×10⁵
  non-embedding params); loss/accuracy computed **only on the answer token** (the c after =).
- **Optimizer:** AdamW, lr 1e-3, **weight decay 1**, β=(0.9, 0.98), linear LR warmup over 10 updates,
  minibatch 512 (or half the train set), budget **10⁵ updates** (up to 10⁶ to expose the most delayed
  cases; the dramatic modular-division figure uses Adam with no weight decay and a 10⁶ budget).

## The phenomenon

**Grokking — delayed generalization.** Training accuracy reaches ~100% in <10³ steps (fast
memorization) while validation accuracy stays at chance. If training continues far past this — roughly
**1000× longer**, around 10⁵ steps — validation accuracy *suddenly* climbs from chance to perfect
generalization, long after the point of overfitting. Validation *loss* rises (overfitting) and then
*second-descends* much later. This is distinct from capacity-axis double descent: it is purely along
the training-time axis, occurs tens of thousands of epochs past first interpolation, and the accuracy
is monotone (no peak).

## Key findings

- **Data efficiency / learning time.** Within a range of training fractions the converged accuracy
  stays at 100%, but the optimization time to reach 99% validation accuracy grows ~exponentially as the
  dataset shrinks (near 25–30% data for S₅, a 1% drop in data raises median steps-to-generalize by
  40–50%), while steps to memorize the train set stay at 10³–10⁴. Grokking is most dramatic near the
  minimal generalizing fraction. A long optimization budget is essential — short budgets miss grokking.
- **Structure.** Symmetric operations (x+y, x·y, x²+y²) need less data than non-symmetric ones (a
  transformer can express a symmetric function by ignoring positional embeddings). x−y and x/y need
  about the same data, since mod-p multiplication and mod-(p−1) addition are isomorphic up to
  relabeling (primitive root) and the symbols are abstract. Some operations (x³+xy²+y mod 97) never
  generalize within budget — pure memorization.

## What drives it

- **Weight decay** is the standout intervention — more than halves the data needed to generalize.
  Decay toward the origin beats decay toward the initialization, so a small-norm prior explains part
  (not all) of the benefit. Gradient/weight noise also help (consistent with biasing optimization
  toward flat minima), and learning rate must sit within ~1 order of magnitude. Sharpness of the
  found minimum is strongly anti-correlated with validation accuracy (Spearman ≈ −0.8), suggesting
  grokking happens once parameters reach flatter regions of the loss landscape.
- **Capacity is not the bottleneck:** injecting random-label outliers, the network still reaches 100%
  train accuracy regardless of outlier count; small numbers barely hurt generalization. So generalizing
  at all requires a non-trivial explanation (a slow drift to a structured low-norm solution).
- **Recovered structure:** t-SNE of the output-layer embeddings shows modular addition laid out on a
  circle (a "number line" closing into a ring under +8) and S₅ clustered into subgroup cosets —
  clearer with weight decay.

## Implementation

Grounded in the canonical setup (binary-op datasets of abstract-symbol equations, decoder-only
transformer scored on the answer token, AdamW + warmup + long budget).

```python
import torch, torch.nn as nn, torch.nn.functional as F
EQ, OP = "=", "∘"

def all_equations(operation, modulus):
    eqs = []
    for a in range(modulus):
        for b in range(modulus):
            if operation == "+":   c = (a + b) % modulus
            elif operation == "-": c = (a - b) % modulus
            elif operation == "/":
                if b == 0: continue
                c = (a * pow(b, -1, modulus)) % modulus          # x / y mod p
            else: raise ValueError(operation)
            eqs.append((a, OP, b, EQ, c))                        # <a> <op> <b> <=> <c>
    return eqs

def make_dataset(operation, modulus, train_fraction, rng):
    eqs = all_equations(operation, modulus)
    vocab = {tok: i for i, tok in enumerate([EQ, OP] + list(range(modulus)))}  # abstract symbols
    data = torch.stack([torch.tensor([vocab[t] for t in eq]) for eq in eqs])
    perm = torch.randperm(len(data), generator=rng)
    n = int(round(train_fraction * len(data)))
    return data[perm[:n]], data[perm[n:]], len(vocab)

class GrokTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           activation="gelu", batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, tokens):
        T = tokens.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        h = self.embed(tokens) + self.pos[:T]
        return self.unembed(self.blocks(h, mask=mask))

def answer_loss_acc(logits, tokens):
    pred, target = logits[:, -2, :], tokens[:, -1]              # score only the answer token
    return F.cross_entropy(pred, target), (pred.argmax(-1) == target).float().mean()

def train(model, train_data, val_data, num_steps=10**5, lr=1e-3, weight_decay=1.0,
          batch_size=512, warmup=10):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    bs = min(batch_size, len(train_data)); history = []
    for step in range(num_steps):                               # train far past memorization
        batch = train_data[torch.randint(0, len(train_data), (bs,))]
        loss, train_acc = answer_loss_acc(model(batch), batch)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0:
            with torch.no_grad():
                _, val_acc = answer_loss_acc(model(val_data), val_data)
            history.append((step, train_acc.item(), val_acc.item()))
    return history
```
