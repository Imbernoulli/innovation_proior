Let me start from the cost wall, because it dictates the design. I'd love to run architecture search straight on ImageNet — that's the target, that's where accuracy matters. But each candidate has to be trained on over a million high-resolution images to get a single reward, and the search needs thousands of candidates. That's months. So searching directly on the big dataset is off the table. The only way out is to search on something small and cheap — CIFAR-10 as a proxy — and have the result *transfer* to ImageNet. But a network found on 32×32 CIFAR images won't even fit 299×299 ImageNet inputs, and an architecture tuned to one depth/scale has no reason to be good at another. So transfer isn't automatic; I have to *engineer the search space* so that whatever I search for is intrinsically scale- and depth-independent.

What would a scale-independent search variable even look like? Look at how good architectures are actually built. Inception, ResNet — they're not bespoke top to bottom; they're a small *motif* (an Inception module, a residual block) repeated many times, with weights differing but structure identical. The depth of the net is just "how many times you stack the motif," and the input scale is handled by the same motif at different resolutions. So the *structure* of a network factorizes into (a) a small repeatable unit and (b) a stacking pattern. If I search only the *unit* — a convolutional *cell* — and fix the stacking pattern by hand, then the thing I searched is decoupled from depth and from input size. To handle a bigger dataset I just stack more copies and use more filters. That's the lever for transfer: search a cell on CIFAR, then stack it deeper for ImageNet. And searching a small cell is far cheaper than searching a whole network, and a cell is more likely to generalize across tasks. So the cell becomes the search target.

Now, a network has to do two things to images: process them at constant resolution, and reduce resolution. One cell type can't cleanly do both, because a resolution-reducing cell needs strided operations on its inputs. So I'll define *two* cell types: a **Normal Cell** that returns a feature map of the same spatial size, and a **Reduction Cell** whose initial operations on the cell inputs use stride two, halving height and width. The outer skeleton is fixed by hand — a chosen pattern of normal cells with reduction cells inserted at points where I want to downsample (and the standard heuristic: double the filter count whenever spatial size halves, to keep per-layer compute roughly constant). The number of normal-cell repeats N between reductions and the initial filter count are free knobs I set per target: small N and few filters for CIFAR / fast search, large N and more filters for ImageNet. Same two cells, different stacking — that's how one search serves many scales.

So the only thing left to *search* is the internal structure of those two cells. How do I let a controller emit a cell? A cell takes two inputs — the outputs of the previous two cells (h_i and h_{i-1}) — analogous to how a residual/Inception design feeds forward from earlier layers. Inside, I build up a set of hidden states. I'll have the controller construct the cell as a sequence of **blocks**, each block producing one new hidden state from two existing ones. Concretely, each block is five decisions made by five softmax classifiers:

```
Step 1: pick a hidden state (from h_i, h_{i-1}, or any state created in previous blocks)
Step 2: pick a second hidden state from the same pool
Step 3: pick an operation to apply to the Step-1 state
Step 4: pick an operation to apply to the Step-2 state
Step 5: pick how to combine the two results -> a new hidden state
```

The combine in Step 5 is either elementwise addition or concatenation along the filter axis. The newly created hidden state is appended to the pool, so later blocks can build on it — this is what lets the cell form arbitrary DAGs of branches, not just a chain. Repeat for B blocks; B=5 is a good size. At the end, every hidden state that no block consumed gets concatenated along the depth axis to form the cell's output — nothing is wasted.

The operations in Steps 3–4 are drawn from a fixed menu pulled from what's proven useful in CNNs: identity, a range of separable convolutions (3×3, 5×5, 7×7), the asymmetric 1×7-then-7×1 and 1×3-then-3×1 convolutions, dilated 3×3, plain 1×1 and 3×3 convolutions, and average/max pooling at several sizes (3×3, 5×5, 7×7). Each operation has a strided variant so it can serve in a Reduction Cell.

Could the Normal and Reduction cells share one structure? I could force them equal, but reduction and identity-preserving transforms are genuinely different jobs, so it's better to let the controller learn them separately. That just doubles the controller's output: to emit both cells, the controller makes 2×5B predictions — the first 5B for the Normal Cell, the next 5B for the Reduction Cell. Since each block is 5 softmaxes and there are B blocks per cell and 2 cells, the controller is 5B + 5B = 10B softmax decisions, one LSTM unrolled over those steps (a one-layer LSTM, 100 hidden units works).

Now the optimization of the controller. This is reinforcement learning over discrete choices, same skeleton as searching a full network: the controller samples a cell pair, I build a child network by stacking it, train the child on CIFAR for a short fixed schedule, measure validation accuracy R, and use R to update the controller so that good cells get higher probability. The joint probability of a sampled architecture is the product of the 10B softmax probabilities; scaling the gradient of that log-probability by R (the policy-gradient idea) pushes probability mass toward high-reward cells and away from bad ones, with a baseline (an exponential moving average of past rewards, decay ~0.95) to reduce variance.

For the policy update itself I'll use Proximal Policy Optimization rather than plain REINFORCE. REINFORCE works but is high-variance and the updates can be erratic; PPO's clipped surrogate keeps each update from moving the policy too far, which makes controller training faster and more stable — exactly what I want when each reward costs a child-training run. Add a small entropy penalty (weight ~1e-5) to keep the controller exploring rather than collapsing onto one cell too early. The whole thing is run on a big distributed worker pool: a workqueue of sampled architectures, hundreds of child workers each training one child on a GPU and reporting its accuracy back; the controller updates on minibatches of completed children. To keep search cheap I use a small stack during search (N=2 normal cells between reductions) and a short child schedule (~20 epochs) with cosine LR decay — short, but enough to *rank* cells; cosine decay actually helps differentiate good from bad in the short run.

There's also a strong random-search baseline lurking here. If I instead sample each of the 10B decisions uniformly rather than from the controller's softmax, I get random architectures from the *same* search space. If a well-designed search space already concentrates probability on good architectures, random search will be decent — so it's the honest baseline to beat, and it tells me how much the RL controller is really adding versus how much credit goes to the search space design.

One thing breaks when I stack these cells deeply: regularization. These cells are heavily multi-branch, and naively dropping out individual convolutional activations degrades them. The better tool for multi-branch nets is to drop whole *paths* — each path (an op edge in the cell) is stochastically zeroed during training and the surviving paths rescaled, like stochastic depth but at the path granularity. But plain DropPath with a fixed drop probability barely helps NASNet. What does help is making the drop probability *increase linearly over the course of training* — start near zero (let the net learn freely) and ramp the regularization up as training proceeds. Call it ScheduledDropPath. The scheduling is the part that matters; fixed-rate DropPath doesn't move the needle, the linear ramp does, on both CIFAR and ImageNet.

Let me write the skeleton.

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['identity','conv_1x3_3x1','conv_1x7_7x1','dil_conv_3x3',
       'avg_pool_3x3','max_pool_3x3','max_pool_5x5','max_pool_7x7',
       'conv_1x1','conv_3x3','sep_conv_3x3','sep_conv_5x5','sep_conv_7x7']
COMBINE = ['add', 'concat']

class ControllerRNN(nn.Module):
    # one LSTM emitting 2*5B softmax decisions: a Normal Cell then a Reduction Cell
    def __init__(self, B=5, hidden=100):
        super().__init__()
        self.B, self.lstm = B, nn.LSTMCell(hidden, hidden)
        self.h_in = nn.Linear(hidden, hidden)      # "select hidden state" softmax (size grows per block)
        self.h_op = nn.Linear(hidden, len(OPS))    # "select operation" softmax
        self.h_cb = nn.Linear(hidden, len(COMBINE))# "select combine" softmax

    def sample_cell(self, n_inputs=2):
        decisions, logp = [], 0.0
        states = n_inputs
        h = c = torch.zeros(1, self.lstm.hidden_size)
        x = torch.zeros(1, self.lstm.hidden_size)
        for _ in range(self.B):                    # B blocks, 5 softmaxes each
            picks = []
            for head, n_choices in [(self.h_in, states), (self.h_in, states),
                                    (self.h_op, len(OPS)), (self.h_op, len(OPS)),
                                    (self.h_cb, len(COMBINE))]:
                h, c = self.lstm(x, (h, c))
                p = F.softmax(head(h)[:, :n_choices], dim=-1)
                a = torch.multinomial(p, 1)        # sample a discrete decision
                logp = logp + torch.log(p[0, a])   # accumulate log-prob (product over softmaxes)
                picks.append(int(a)); x = h
            decisions.append(picks); states += 1   # new hidden state joins the pool
        return decisions, logp

    def sample(self):                              # Normal Cell + Reduction Cell
        nd, lpn = self.sample_cell()
        rd, lpr = self.sample_cell()
        return (nd, rd), lpn + lpr                 # joint logp = sum over all 10B softmaxes

def scheduled_drop_path(x, paths, step, total_steps, max_drop=0.2):
    # drop whole paths with probability that LINEARLY increases over training
    p = max_drop * step / total_steps
    out = []
    for path in paths:
        if self.training and torch.rand(1) < p:
            out.append(torch.zeros_like(path))     # drop this path
        else:
            out.append(path / (1 - p) if self.training else path)  # rescale survivors
    return sum(out)

def build_network(decisions, N, init_filters, n_reductions):
    # fixed skeleton: stack N Normal Cells, then a Reduction Cell, repeated; double filters at each reduction
    ...

def search(controller, proxy_loader, baseline_decay=0.95, entropy_w=1e-5):
    opt = torch.optim.Adam(controller.parameters(), lr=3.5e-4)   # PPO update underneath
    baseline = None
    for _ in range(n_iters):
        (cells, logp), entropy = controller.sample(), 0.0
        child = build_network(cells, N=2, init_filters=32, n_reductions=2)
        R = train_and_eval_child(child, proxy_loader, epochs=20)  # reward = held-out accuracy
        baseline = R if baseline is None else baseline_decay*baseline + (1-baseline_decay)*R
        # policy gradient (PPO clipped surrogate in practice): scale logp grad by advantage
        loss = -(R - baseline) * logp - entropy_w * entropy
        opt.zero_grad(); loss.backward(); opt.step()
```

Causal chain: searching on ImageNet is unaffordable, so I search on CIFAR and need transfer, which forces a search space whose unit is scale- and depth-independent; good nets are stacked repeated motifs, so I search a *cell* (two of them — Normal preserving resolution and Reduction halving it via strided inputs) inside a hand-fixed skeleton with N and filter count as free scaling knobs; each cell is built by a controller LSTM as B blocks of five softmax decisions (two inputs, two ops, one combine), emitting 2×5B decisions for both cells; the controller is trained by policy gradient with PPO (more stable than REINFORCE) and an entropy bonus, rewarded by short-schedule child accuracy on the proxy, with random search as the honest baseline; and because the stacked multi-branch cells need regularization that fixed DropPath doesn't provide, I drop whole paths with a probability that ramps up linearly through training.
