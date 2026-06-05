# NASNet: Learning Transferable Architectures

## Problem

Architecture search on ImageNet directly is computationally prohibitive. NASNet searches on a small proxy dataset (CIFAR-10) and transfers the result to ImageNet by designing a **search space whose unit is independent of network depth and input scale** — so the discovered structure can be re-stacked for any target.

## Key idea

Search a **convolutional cell**, not a whole network, and stack copies of it. Two cell types:
- **Normal Cell** — returns a feature map of the same spatial size.
- **Reduction Cell** — strided initial operations halve height and width.

The outer skeleton is fixed by hand (a pattern of Normal Cells with Reduction Cells inserted; filters doubled at each resolution drop). The number of Normal-Cell repeats N and the initial filter count are free knobs set per target (small for CIFAR/search, large for ImageNet) — this is the transfer/scaling mechanism.

## Cell construction (the search space)

Each cell takes two inputs h_i, h_{i-1} (outputs of the previous two cells) and is built from **B blocks** (B=5). Each block is **5 decisions** by 5 softmax classifiers:

```
1. select a hidden state (from h_i, h_{i-1}, or states made by earlier blocks)
2. select a second hidden state from the same pool
3. select an operation for the state from step 1
4. select an operation for the state from step 2
5. select a combine method: elementwise add OR concatenate -> new hidden state
```

The new hidden state joins the pool. After B blocks, all unused hidden states are concatenated in depth as the cell output. Operations: identity, sep-conv 3×3/5×5/7×7, 1×7-then-7×1 conv, 1×3-then-3×1 conv, dilated 3×3, 1×1 conv, 3×3 conv, avg-pool 3×3, max-pool 3×3/5×5/7×7. To emit both cells the controller makes **2×5B = 10B** softmax predictions (first 5B Normal, next 5B Reduction).

## Search

A one-layer LSTM controller (100 hidden units) samples a cell pair; a child network built by stacking it is trained on CIFAR-10 for a short fixed schedule (~20 epochs, cosine LR decay, small N=2 during search); held-out validation accuracy R is the reward. The joint probability of an architecture is the product of the 10B softmax probabilities; the controller is updated by policy gradient — **Proximal Policy Optimization** (more stable/faster than REINFORCE) with an entropy penalty (~1e-5) and an exponential-moving-average reward baseline (decay 0.95). Random search over the same space is the baseline. Distributed via a workqueue with hundreds of child workers.

## Regularization: ScheduledDropPath

The multi-branch cells are regularized by dropping whole **paths** (op edges) stochastically with surviving-path rescaling. Fixed-probability DropPath barely helps; **linearly increasing the drop probability over the course of training** (ScheduledDropPath) significantly improves final performance.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['identity','conv_1x3_3x1','conv_1x7_7x1','dil_conv_3x3',
       'avg_pool_3x3','max_pool_3x3','max_pool_5x5','max_pool_7x7',
       'conv_1x1','conv_3x3','sep_conv_3x3','sep_conv_5x5','sep_conv_7x7']
COMBINE = ['add', 'concat']

class ControllerRNN(nn.Module):
    def __init__(self, B=5, hidden=100):
        super().__init__()
        self.B, self.lstm = B, nn.LSTMCell(hidden, hidden)
        self.h_in = nn.Linear(hidden, hidden)        # select-hidden-state softmax
        self.h_op = nn.Linear(hidden, len(OPS))      # select-operation softmax
        self.h_cb = nn.Linear(hidden, len(COMBINE))  # select-combine softmax

    def sample_cell(self, n_inputs=2):
        decisions, logp, states = [], 0.0, n_inputs
        h = c = torch.zeros(1, self.lstm.hidden_size)
        x = torch.zeros(1, self.lstm.hidden_size)
        for _ in range(self.B):
            picks = []
            for head, n in [(self.h_in, states),(self.h_in, states),
                            (self.h_op, len(OPS)),(self.h_op, len(OPS)),
                            (self.h_cb, len(COMBINE))]:
                h, c = self.lstm(x, (h, c))
                p = F.softmax(head(h)[:, :n], dim=-1)
                a = torch.multinomial(p, 1)
                logp = logp + torch.log(p[0, a]); picks.append(int(a)); x = h
            decisions.append(picks); states += 1
        return decisions, logp

    def sample(self):                                 # Normal + Reduction cells
        nd, lpn = self.sample_cell()
        rd, lpr = self.sample_cell()
        return (nd, rd), lpn + lpr

def scheduled_drop_path(paths, step, total_steps, training, max_drop=0.2):
    p = max_drop * step / total_steps                 # drop prob ramps linearly
    out = []
    for path in paths:
        if training and torch.rand(1).item() < p:
            out.append(torch.zeros_like(path))
        else:
            out.append(path / (1 - p) if training else path)
    return sum(out)

def search(controller, proxy_loader, n_iters, decay=0.95, ent_w=1e-5):
    opt = torch.optim.Adam(controller.parameters(), lr=3.5e-4)   # PPO update underneath
    baseline = None
    for _ in range(n_iters):
        cells, logp = controller.sample()
        child = build_network(cells, N=2, init_filters=32, n_reductions=2)
        R = train_and_eval_child(child, proxy_loader, epochs=20)
        baseline = R if baseline is None else decay*baseline + (1-decay)*R
        loss = -(R - baseline) * logp
        opt.zero_grad(); loss.backward(); opt.step()
```
