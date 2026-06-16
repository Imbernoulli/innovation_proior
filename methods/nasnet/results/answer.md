# NASNet

## Problem

Architecture search on ImageNet directly is computationally prohibitive. NASNet searches on a small proxy dataset (CIFAR-10) and transfers the result to ImageNet by designing a **search space whose unit is independent of network depth and input scale**, so the discovered structure can be re-stacked for any target.

## Key idea

Search a **convolutional cell**, not a whole network, and stack copies of it. Two cell types:
- **Normal Cell** -- returns a feature map of the same spatial size.
- **Reduction Cell** -- operations applied to the two cell inputs use stride 2, halving height and width.

The outer skeleton is fixed by hand (a pattern of Normal Cells with Reduction Cells inserted; filters doubled at each resolution drop). The number of Normal-Cell repeats N and the initial filter count are free knobs set per target (small for CIFAR/search, large for ImageNet), which is the transfer/scaling mechanism.

## Cell construction (the search space)

Each cell takes two inputs h_i, h_{i-1} (outputs of the previous two cells) and is built from **B blocks** (B=5). Each block is **5 decisions** by 5 softmax classifiers:

```
1. select a hidden state (from h_i, h_{i-1}, or states made by earlier blocks)
2. select a second hidden state from the same pool
3. select an operation for the state from step 1
4. select an operation for the state from step 2
5. select a combine method: elementwise add or concatenate -> new hidden state
```

The new hidden state joins the pool. After B blocks, unused hidden states generated inside the cell are concatenated in depth as the cell output. The 13 operation choices, in table order, are: identity; 1x3 then 3x1 convolution; 1x7 then 7x1 convolution; 3x3 dilated convolution; 3x3 average pooling; 3x3 max pooling; 5x5 max pooling; 7x7 max pooling; 1x1 convolution; 3x3 convolution; 3x3 depthwise-separable convolution; 5x5 depthwise-separable convolution; and 7x7 depthwise-separable convolution. All operations have a strided option, used in a Reduction Cell when the selected hidden state is one of the two original cell inputs. Convolutions use ReLU -> convolution -> BatchNorm; selected separable convolutions are applied twice without BN/ReLU inserted between depthwise and pointwise pieces; and 1x1 convolutions are inserted as needed to make shapes match. To emit both cells the controller makes **2x5B = 10B** softmax predictions (first 5B Normal, next 5B Reduction).

## Search

A one-layer LSTM controller (100 hidden units, weights initialized uniformly between -0.1 and 0.1) samples a cell pair; a child network built by stacking it is trained on CIFAR-10 for 20 epochs with N=2, a momentum optimizer (momentum 0.9), L2 weight decay, and cosine LR decay during search; held-out validation accuracy R is the reward. The joint probability of an architecture is the product of the 10B softmax probabilities, and the controller gradient is scaled by validation accuracy. The controller is updated with **Proximal Policy Optimization rather than REINFORCE** using learning rate 0.00035, entropy penalty weight 0.00001, and an exponential-moving-average reward baseline with weight 0.95. The distributed search uses a global workqueue, PPO minibatches of 20 completed architectures, 20,000 sampled child models, then the top 250 architectures are trained to convergence on CIFAR-10. Random search over the same space is the baseline.

## Regularization: ScheduledDropPath

The multi-branch cells are regularized by dropping whole **paths** (op edges) stochastically with path scaling. Fixed-probability DropPath alone does not help much; **linearly increasing the drop probability over the course of training** is the ScheduledDropPath change.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = [
    'identity',
    'conv_1x3_3x1',
    'conv_1x7_7x1',
    'dil_conv_3x3',
    'avg_pool_3x3',
    'max_pool_3x3',
    'max_pool_5x5',
    'max_pool_7x7',
    'conv_1x1',
    'conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
]
COMBINE = ['add', 'concat']

class ControllerRNN(nn.Module):
    def __init__(self, B=5, hidden=100):
        super().__init__()
        self.B = B
        self.max_states = 2 + B
        self.lstm = nn.LSTMCell(hidden, hidden)
        def cell_heads():
            return nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(hidden, self.max_states),
                    nn.Linear(hidden, self.max_states),
                    nn.Linear(hidden, len(OPS)),
                    nn.Linear(hidden, len(OPS)),
                    nn.Linear(hidden, len(COMBINE)),
                ])
                for _ in range(B)
            ])
        self.heads = nn.ModuleList([cell_heads(), cell_heads()])
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def _step(self, head, h, c, x, n_choices, choice=None):
        h, c = self.lstm(x, (h, c))
        probs = F.softmax(head(h)[:, :n_choices], dim=-1)
        if choice is None:
            choice = int(torch.multinomial(probs, 1).item())
        logp = torch.log(probs[0, choice].clamp_min(1e-8))
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return choice, logp, entropy, h, c, h

    def _cell(self, cell_id, h, c, x, choices=None, n_inputs=2):
        decisions, logps, entropies, states = [], [], [], n_inputs
        for block_id in range(self.B):
            block_choices = None if choices is None else choices[block_id]
            n_choices = [states, states, len(OPS), len(OPS), len(COMBINE)]
            picked = []
            for step_id, head in enumerate(self.heads[cell_id][block_id]):
                fixed = None if block_choices is None else int(block_choices[step_id])
                a, lp, ent, h, c, x = self._step(head, h, c, x, n_choices[step_id], fixed)
                picked.append(a); logps.append(lp); entropies.append(ent)
            decisions.append(picked); states += 1
        logp = torch.stack(logps).sum()
        entropy = torch.stack(entropies).sum()
        return decisions, logp, entropy, h, c, x

    def sample(self):
        h = c = torch.zeros(1, self.lstm.hidden_size)
        x = torch.zeros(1, self.lstm.hidden_size)
        normal, lp1, ent1, h, c, x = self._cell(0, h, c, x)
        reduction, lp2, ent2, h, c, x = self._cell(1, h, c, x)
        return {'cells': (normal, reduction), 'old_logp': (lp1 + lp2).detach(), 'entropy': ent1 + ent2}

    def log_prob(self, cells):
        h = c = torch.zeros(1, self.lstm.hidden_size)
        x = torch.zeros(1, self.lstm.hidden_size)
        _, lp1, ent1, h, c, x = self._cell(0, h, c, x, choices=cells[0])
        _, lp2, ent2, h, c, x = self._cell(1, h, c, x, choices=cells[1])
        return lp1 + lp2, ent1 + ent2

def apply_cell_op(name, x, stride):
    # Convolutional choices use ReLU -> convolution -> BatchNorm.
    # Separable convolutions are applied twice, without BN/ReLU between depthwise and pointwise pieces.
    # 1x1 projections are inserted around cell edges whenever shapes differ.
    ...

def align_for_combine(a, b, combine):
    # Use 1x1 projections as needed so addition or depth concatenation is legal.
    ...

def run_cell(cell_decisions, h_i, h_i_minus_1, reduction):
    states = [h_i, h_i_minus_1]
    used = set()
    for first, second, op_first, op_second, combine in cell_decisions:
        stride_first = 2 if reduction and first < 2 else 1
        stride_second = 2 if reduction and second < 2 else 1
        y1 = apply_cell_op(OPS[op_first], states[first], stride_first)
        y2 = apply_cell_op(OPS[op_second], states[second], stride_second)
        y1, y2 = align_for_combine(y1, y2, COMBINE[combine])
        used.update([first, second])
        if COMBINE[combine] == 'add':
            states.append(y1 + y2)
        else:
            states.append(torch.cat([y1, y2], dim=1))
    return torch.cat([state for i, state in enumerate(states) if i >= 2 and i not in used], dim=1)

def scheduled_drop_path(paths, step, total_steps, training, max_drop):
    drop = max_drop * min(float(step), float(total_steps)) / float(total_steps)
    keep = 1.0 - drop
    out = []
    for path in paths:
        if training:
            keep_mask = (torch.rand((), device=path.device) < keep).to(path.dtype)
            out.append(path * keep_mask)
        else:
            out.append(path * keep)
    return out

def build_network(cells, N, init_filters, skeleton):
    # Stack N Normal Cells between Reduction Cells; stride 2 is used on Reduction Cell input edges.
    # Double the filter count whenever spatial size is reduced.
    ...

def ppo_update(controller, batch, optimizer, baseline, clip_eps, entropy_w=0.00001):
    new_logp, entropy, reward, old_logp = [], [], [], []
    for item in batch:
        lp, ent = controller.log_prob(item['cells'])
        new_logp.append(lp); entropy.append(ent)
        old_logp.append(item['old_logp']); reward.append(item['reward'])
    new_logp = torch.stack(new_logp)
    old_logp = torch.stack(old_logp).to(new_logp.device)
    reward = torch.tensor(reward, dtype=new_logp.dtype, device=new_logp.device)
    advantage = reward - baseline
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.min(unclipped, clipped).mean() - entropy_w * torch.stack(entropy).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

def fixed_skeleton(N):
    # Return the manually predetermined placement of Normal and Reduction Cells.
    ...

def search(controller, proxy_loaders, clip_eps, optimizer, init_filters,
           n_architectures=20000, baseline_weight=0.95):
    # The controller update is PPO with learning rate 0.00035 and entropy weight 0.00001.
    baseline, batch = None, []
    for _ in range(n_architectures):
        sample = controller.sample()
        child = build_network(sample['cells'], N=2, init_filters=init_filters, skeleton=fixed_skeleton(N=2))
        reward = train_and_eval_child(
            child,
            proxy_loaders,
            epochs=20,
            optimizer='momentum',
            momentum=0.9,
            weight_decay='l2',
            lr_schedule='cosine',
        )
        baseline = reward if baseline is None else baseline_weight * baseline + (1 - baseline_weight) * reward
        batch.append({'cells': sample['cells'], 'old_logp': sample['old_logp'], 'reward': reward})
        if len(batch) == 20:
            ppo_update(controller, batch, optimizer, baseline, clip_eps)
            batch.clear()
```
