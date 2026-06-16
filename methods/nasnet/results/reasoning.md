Let me start from the cost wall, because it dictates the design. I'd love to run architecture search straight on ImageNet -- that's the target, that's where accuracy matters. But each candidate has to be trained on over a million high-resolution images to get a single reward, and the search needs thousands of candidates. Searching directly on the big dataset is too expensive. The only way out is to search on something small and cheap -- CIFAR-10 as a proxy -- and have the result *transfer* to ImageNet. But a network found on 32x32 CIFAR images won't even fit 299x299 ImageNet inputs, and an architecture tuned to one depth/scale has no reason to be good at another. So transfer isn't automatic; I have to *engineer the search space* so that whatever I search for is intrinsically scale- and depth-independent.

What would a scale-independent search variable even look like? Look at how good architectures are actually built. Inception, ResNet -- they're not bespoke top to bottom; they're a small *motif* (an Inception module, a residual block) repeated many times, with weights differing but structure identical. The depth of the net is just "how many times you stack the motif," and the input scale is handled by the same motif at different resolutions. So the *structure* of a network factorizes into (a) a small repeatable unit and (b) a stacking pattern. If I search only the *unit* -- a convolutional *cell* -- and fix the stacking pattern by hand, then the thing I searched is decoupled from depth and from input size. To handle a bigger dataset I just stack more copies and use more filters. That's the lever for transfer: search a cell on CIFAR, then stack it deeper for ImageNet. And searching a small cell is far cheaper than searching a whole network, and a cell is more likely to generalize across tasks. So the cell becomes the search target.

Now, a network has to do two things to images: process them at constant resolution, and reduce resolution. One cell type can't cleanly do both, because a resolution-reducing cell needs strided operations on its inputs. So I'll define two cell types: a Normal Cell that returns a feature map of the same spatial size, and a Reduction Cell where operations applied to the two cell inputs use stride two, halving height and width. The outer skeleton is fixed by hand -- a chosen pattern of normal cells with reduction cells inserted at points where I want to downsample (and the standard heuristic: double the filter count whenever spatial size halves, to keep per-layer compute roughly constant). The number of normal-cell repeats N between reductions and the initial filter count are free knobs I set per target: small N and few filters for CIFAR / fast search, large N and more filters for ImageNet. Same two cells, different stacking -- that's how one search serves many scales.

So the only thing left to search is the internal structure of those two cells. How do I let a controller emit a cell? A cell takes two inputs -- the outputs of the previous two cells (h_i and h_{i-1}) -- analogous to how a residual/Inception design feeds forward from earlier layers. Inside, I build up a set of hidden states. I have the controller construct the cell as a sequence of blocks, each block producing one new hidden state from two existing ones. One block is five decisions made by five softmax classifiers: select a hidden state from h_i, h_{i-1}, or any state created in previous blocks; select a second hidden state from the same pool; select an operation for the first selected state; select an operation for the second selected state; then select how to combine the two results into a new hidden state.

That combine choice is either elementwise addition or concatenation along the filter axis. The newly created hidden state is appended to the pool, so later blocks can build on it -- this is what lets the cell form arbitrary DAGs of branches, not just a chain. Repeat for B blocks; I use B=5. At the end, the hidden states generated inside the cell that no later block consumed get concatenated along the depth axis to form the cell's output.

The operations in Steps 3 and 4 need to be a fixed CNN primitive menu, and the menu has to be concrete enough that the controller only picks among discrete choices: identity; 1x3 then 3x1 convolution; 1x7 then 7x1 convolution; 3x3 dilated convolution; 3x3 average pooling; 3x3 max pooling; 5x5 max pooling; 7x7 max pooling; 1x1 convolution; 3x3 convolution; 3x3 depthwise-separable convolution; 5x5 depthwise-separable convolution; and 7x7 depthwise-separable convolution. Each operation has a strided option, and in a Reduction Cell I use stride two when the selected hidden state is one of the two original cell inputs. To keep shapes compatible after arbitrary choices, I insert 1x1 convolutions when needed. Convolutional operations follow ReLU, convolution, BatchNorm. If the selected operation is depthwise separable, I apply that separable operation twice, and I do not insert BatchNorm or ReLU between its depthwise and pointwise pieces.

Could the Normal and Reduction cells share one structure? I could force them equal, but reduction and identity-preserving transforms are different jobs, so it is better to let the controller learn them separately. That doubles the controller's output: to emit both cells, the controller makes 2x5B predictions -- the first 5B for the Normal Cell, the next 5B for the Reduction Cell. Since each block is 5 softmaxes and there are B blocks per cell and 2 cells, the controller emits 5B + 5B = 10B softmax decisions from one one-layer LSTM with 100 hidden units, with its weights initialized uniformly between -0.1 and 0.1.

Once the cell is a sequence of discrete choices, the controller can be optimized the same way a full-network NAS controller is optimized: it samples a cell pair, I build a child network by stacking it, train the child on CIFAR for a short fixed schedule, measure held-out validation accuracy R, and use R to update the controller so that good cells get higher probability. The joint probability of a sampled architecture is the product of the 10B softmax probabilities; that probability term gives me the controller gradient, and scaling the update by validation accuracy pushes probability mass toward high-reward cells and away from bad ones. I keep an exponential-moving-average reward baseline with weight 0.95 so the update can use an advantage instead of raw accuracy alone.

For the policy update itself I use Proximal Policy Optimization rather than plain REINFORCE, with controller learning rate 0.00035. The point is not to change the architecture objective; it is to make controller training faster and more stable when every reward costs a child-training run. I add an entropy penalty with weight 0.00001 to keep the controller exploring rather than collapsing onto one cell too early. The whole thing runs through a global workqueue: sampled architectures go into the queue, child workers train them on GPUs and report held-out accuracy, and the controller updates with PPO on minibatches of 20 completed architectures. I stop after 20,000 sampled child models, then take the top 250 architectures and train them to convergence on CIFAR-10. To keep search cheap I use a small stack during search (N=2 normal cells between reductions) and a 20-epoch child schedule with a momentum optimizer, momentum 0.9, L2 weight decay, and cosine learning-rate decay; cosine decay helps differentiate good architectures during that short run.

There's also a strong random-search baseline lurking here. If I instead sample each of the 10B decisions uniformly rather than from the controller's softmax, I get random architectures from the *same* search space. If a well-designed search space already concentrates probability on useful architectures, random search will be decent, so it is the honest baseline to beat, and it tells me how much the RL controller is really adding versus how much credit goes to the search space design.

One thing breaks when I stack these cells deeply: regularization. These cells are heavily multi-branch, and naively dropping out individual convolutional filters degrades them. The better tool for multi-branch nets is to drop whole *paths* -- each path, meaning an operation edge in the cell, is stochastically removed during training and scaled at test time like DropPath. Fixed-probability DropPath alone does not help much. What does help is making the drop probability *increase linearly over the course of training*: start near zero, let the net learn freely, and ramp the regularization up as training proceeds. Call it ScheduledDropPath. The scheduling is the part that matters.

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
        return {'cells': (normal, reduction),
                'old_logp': (lp1 + lp2).detach(),
                'entropy': ent1 + ent2}

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

def fixed_skeleton(N):
    # Return the manually predetermined placement of Normal and Reduction Cells.
    ...

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

Searching on ImageNet is unaffordable, so I search on CIFAR and need transfer, which forces a search space whose unit is scale- and depth-independent; good nets are stacked repeated motifs, so I search a cell pair (Normal preserving resolution and Reduction halving it through stride-two operations on the two cell inputs) inside a hand-fixed skeleton with N and filter count as free scaling knobs; each cell is built by a controller LSTM as B blocks of five softmax decisions (two inputs, two ops, one combine), emitting 2x5B decisions for both cells; the controller is trained with PPO, an entropy bonus, and an EMA reward baseline, rewarded by 20-epoch child accuracy on the proxy, with random search as the honest baseline; and because fixed-probability DropPath alone does not help much for the stacked multi-branch cells, I drop whole paths with a probability that ramps up linearly through training.
