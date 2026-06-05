# Efficient Neural Architecture Search (ENAS)

## Problem

Reinforcement-learning neural architecture search finds excellent architectures but is
ruinously expensive: it trains every candidate child network from scratch to convergence,
reads one validation number, and discards the trained weights — hundreds of GPU-days per
search. ENAS removes that cost.

## Key idea

All candidate architectures in a search space are **subgraphs of a single directed acyclic
graph (DAG)**. Put the parameters on the edges of that DAG and let **every child model
share one pool of weights ω**: selecting an architecture = selecting which edges (and hence
which shared parameters) are active. No candidate is ever trained from scratch; each reuses
weights already trained by every other candidate that touched the same edges. Search drops
from ~tens of thousands of GPU-hours to under 16 hours on a single GPU.

## Algorithm

Two parameter sets — controller θ (an LSTM policy π(m; θ) that autoregressively samples an
architecture) and shared child weights ω — trained in alternating phases:

1. **Train ω (fix θ).** SGD to minimize the expected child loss
   `E_{m~π(m;θ)}[ L(m; ω) ]`. With a sampled m fixed, L is differentiable in ω, so use the
   Monte-Carlo gradient `(1/M) Σ ∇_ω L(m_i; ω)` — and `M = 1` (one architecture per
   minibatch) suffices, since a full epoch averages over many subgraphs. Run over one pass
   of the training data.
2. **Train θ (fix ω).** Maximize the expected reward `E_{m~π}[ R(m, ω) ]`. R is sampled and
   non-differentiable, so use REINFORCE with a moving-average baseline b:
   `∇_θ E[R] ≈ (R − b)·∇_θ log π(m; θ)`. R is measured on the **validation** split
   (accuracy for vision; `c/valid_ppl` for language) to select for generalization. Add the
   sample entropy to the reward and apply a temperature + scaled-tanh to the logits to keep
   exploration alive and prevent premature collapse. Optimize θ with Adam for ~2000 steps.

Alternate (1) and (2). After training, sample several architectures, keep the best by
validation reward, and **retrain that single architecture from scratch** with full settings.

**Search spaces.** Recurrent cell: N-node DAG, each node samples a previous index (which
selects a shared edge matrix `W_{ℓ,j}`) and an activation; loose ends averaged; gated
(highway) transitions. Conv macro: per layer, sample which earlier layers to skip-connect
(outputs concatenated) and one of six ops (3×3/5×5 conv, 3×3/5×5 separable conv, 3×3
max/avg pool); a KL penalty toward a skip-density prior ρ≈0.4. Conv micro: search a conv
cell + reduction cell (each node picks two inputs and two of five ops, results added) and
stack them.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F


class Controller(nn.Module):
    """Autoregressive LSTM policy pi(architecture; theta)."""
    def __init__(self, num_ops, num_nodes, hidden=100,
                 temperature=5.0, tanh_constant=2.5):
        super().__init__()
        self.hidden, self.num_ops, self.num_nodes = hidden, num_ops, num_nodes
        self.temperature, self.tanh_constant = temperature, tanh_constant
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.g_emb = nn.Parameter(torch.zeros(1, hidden))
        self.op_emb = nn.Embedding(num_ops, hidden)
        self.op_soft = nn.Linear(hidden, num_ops)
        self.w_prev = nn.Linear(hidden, hidden, bias=False)
        self.w_curr = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def _sample(self, logit):
        logit = self.tanh_constant * torch.tanh(logit / self.temperature)
        dist = torch.distributions.Categorical(F.softmax(logit, dim=-1))
        c = dist.sample()
        return c, dist.log_prob(c), dist.entropy()

    def sample(self):
        h = torch.zeros(1, self.hidden); c = torch.zeros(1, self.hidden)
        x = self.g_emb
        arc, lps, ents, anchors = [], [], [], []
        for node in range(self.num_nodes):
            h, c = self.lstm(x, (h, c))
            if anchors:
                q = torch.tanh(torch.cat(anchors, 0) + self.w_curr(h))
                prev, lp, e = self._sample(self.v(q).transpose(0, 1))
                arc.append(int(prev)); lps.append(lp); ents.append(e)
                x = anchors[int(prev)]
            anchors.append(self.w_prev(h))
            h, c = self.lstm(x, (h, c))
            op, lp, e = self._sample(self.op_soft(h))
            arc.append(int(op)); lps.append(lp); ents.append(e)
            x = self.op_emb(op)
        return arc, torch.stack(lps).sum(), torch.stack(ents).sum()


class SharedChild(nn.Module):
    """Single DAG holding all children in superposition; one parameter set per
    (node, predecessor, op) edge, shared by every architecture that uses it."""
    def __init__(self, num_ops, num_nodes, channels):
        super().__init__()
        self.edges = nn.ModuleDict({
            f"{n}_{p}_{o}": make_op(o, channels)
            for n in range(num_nodes) for p in range(n) for o in range(num_ops)})

    def forward(self, x, arc):
        h = [x]
        for node, (prev, op) in enumerate(parse(arc), start=1):
            h.append(self.edges[f"{node}_{prev}_{op}"](h[prev]))
        loose = loose_ends(arc, len(h))
        return sum(h[i] for i in loose) / len(loose)


def train_shared_weights(controller, child, loader, opt_omega):     # phase 1
    for x, y in loader:
        with torch.no_grad():
            arc, _, _ = controller.sample()
        opt_omega.zero_grad()
        F.cross_entropy(child(x, arc), y).backward()   # L(m; omega), M = 1
        opt_omega.step()


def train_controller(controller, child, valid, opt_theta, baseline,    # phase 2
                     entropy_weight=1e-4, bl_dec=0.99):
    for _ in range(2000):
        arc, log_prob, entropy = controller.sample()
        x, y = next(iter(valid))
        with torch.no_grad():
            reward = accuracy(child(x, arc), y)         # R on VALIDATION split
        baseline = baseline - (1 - bl_dec) * (baseline - reward)
        loss = -log_prob * (reward - baseline) - entropy_weight * entropy
        opt_theta.zero_grad(); loss.backward(); opt_theta.step()
    return baseline


def search(controller, child, train, valid, opt_omega, opt_theta, epochs):
    baseline = 0.0
    for _ in range(epochs):
        train_shared_weights(controller, child, train, opt_omega)
        baseline = train_controller(controller, child, valid, opt_theta, baseline)
    cands = [controller.sample()[0] for _ in range(10)]
    best = max(cands, key=lambda a: reward_on_minibatch(child, a, valid))
    return best   # retrain this single architecture from scratch with full settings
```

Controller θ trained with Adam (lr 3.5e-4); shared ω with SGD/Nesterov. The whole search
runs on one GPU in well under a day — roughly a 1000× reduction in search cost — while
discovering architectures competitive with from-scratch RL search.
