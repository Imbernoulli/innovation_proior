The target I care about is ImageNet, but that is exactly where I cannot afford to search. Architecture search means sampling thousands of candidate networks and training each one far enough to read off a reward, and on ImageNet a single candidate has to chew through more than a million high-resolution images before it tells me anything. Hand-designed architectures — VGG, Inception, ResNet, DenseNet — sidestep this by spending expert effort instead of compute, but that is the very effort I want to automate, and they cannot be re-tailored to a new compute budget without a human redesigning them. The reinforcement-learning NAS of Zoph & Le searches an entire network description with an RNN controller and REINFORCE, which works but is fatally coupled to the target dataset's scale: the search cost grows with how big the images and the network are, and the network it finds is specific to one input resolution and one depth, with no reason to remain good at another. So the only economical path is to search on a small proxy — CIFAR-10, at 32×32 — and have the result transfer up to ImageNet. The catch is that transfer is not free: a network discovered on 32×32 inputs will not even accept 299×299 ones, and an architecture tuned to one depth has no built-in reason to be good at another. The need, then, is not just a cheaper search but a search whose *unit of discovery* is intrinsically independent of input scale and network depth.

I propose NASNet. The pivot is to stop searching whole networks and instead search a small, repeatable convolutional building block — a *cell* — while fixing the way cells are stacked by hand. This is taken straight from how good architectures are actually built: Inception and ResNet are not bespoke top to bottom, they are a single motif repeated many times with structure held identical and only the weights differing. That observation factorizes a network into a repeatable unit and a stacking pattern, and the stacking pattern is precisely what carries depth and scale. If I search only the unit and fix the pattern, the discovered object is decoupled from both. To serve a bigger dataset I just stack more copies and widen the filters — no re-search. Concretely I define two cell types because a network must do two distinct jobs to a feature map: a Normal Cell returns a feature map of the same spatial size, and a Reduction Cell halves height and width by using stride two on the operations applied to its two cell inputs. I do not force the two cells to share one structure: downsampling and resolution-preserving transformation are genuinely different jobs, so it is better to let the controller learn each separately even though that doubles what it must emit. Around these two cells sits a hand-fixed skeleton: Normal Cells repeated, Reduction Cells inserted where I want to downsample, and the standard heuristic of doubling the filter count whenever spatial resolution halves so that per-layer compute stays roughly constant. The number of Normal-Cell repeats $N$ between reductions and the initial filter count are free scaling knobs — small for the cheap CIFAR search ($N=2$), large for ImageNet — and that pair of knobs *is* the transfer mechanism.

The only thing left to discover is the internal wiring of those two cells, and I express that wiring as a sequence of discrete decisions so a controller LSTM can emit it. A cell takes two inputs, $h_i$ and $h_{i-1}$, the outputs of the previous two cells, analogous to how a residual block feeds forward from an earlier layer. Inside, it is built from $B$ blocks, and I use $B=5$. Each block produces one new hidden state from two existing ones and is exactly five softmax decisions: pick a first hidden state from the pool of $\{h_i, h_{i-1}\}$ plus all states created by earlier blocks; pick a second hidden state from the same pool; pick an operation to apply to the first; pick an operation to apply to the second; and pick how to combine the two results. The combine is either elementwise addition or concatenation along the filter axis, and the resulting hidden state is appended to the pool so later blocks can build on it. That append-to-pool step is what lets a cell form an arbitrary directed acyclic graph of branches rather than a straight chain, which is the whole point of giving the search expressive power. After all $B$ blocks, every hidden state generated inside the cell that no later block consumed is concatenated in depth to form the cell's output, so nothing computed is wasted. The operation menu is a fixed set of thirteen CNN primitives, in table order: identity; $1{\times}3$ then $3{\times}1$ convolution; $1{\times}7$ then $7{\times}1$ convolution; $3{\times}3$ dilated convolution; $3{\times}3$ average pooling; $3{\times}3$, $5{\times}5$, and $7{\times}7$ max pooling; $1{\times}1$ convolution; $3{\times}3$ convolution; and $3{\times}3$, $5{\times}5$, $7{\times}7$ depthwise-separable convolutions. Every operation has a strided variant, used in a Reduction Cell precisely when the selected hidden state is one of the two original cell inputs (i.e. its index is below 2). Convolutional choices follow the order ReLU $\to$ convolution $\to$ BatchNorm; a selected separable convolution is applied twice with no BN or ReLU inserted between its depthwise and pointwise pieces; and $1{\times}1$ convolutions are inserted wherever shapes would otherwise not line up for the combine. Because the controller must emit both a Normal and a Reduction cell, it makes $2 \times 5B = 10B$ softmax predictions from a single one-layer LSTM with 100 hidden units, weights initialized uniformly in $[-0.1, 0.1]$ — the first $5B$ describing the Normal Cell, the next $5B$ the Reduction Cell.

With the cell reduced to discrete choices, the controller trains the same way a full-network NAS controller does, just over a far cheaper object. It samples a cell pair; I stack it into a child network, train that child on CIFAR-10 for a short fixed schedule, and read its held-out validation accuracy $R$ as the reward. The joint probability of a sampled architecture is the product of its $10B$ softmax probabilities, so the controller gradient is the gradient of that log-probability, and scaling the update by $R$ pushes probability mass toward high-reward cells and away from bad ones. I subtract an exponential-moving-average reward baseline with weight $0.95$ so the controller learns from an advantage $R - b$ rather than raw accuracy. For the policy update I use Proximal Policy Optimization rather than plain REINFORCE — not to change the objective but because every reward costs a full child-training run, so I want each batch of expensive samples to update the controller as stably and efficiently as possible. The PPO loss clips the importance ratio $r = \exp(\text{new\_logp} - \text{old\_logp})$ against the advantage,
$$\mathcal{L} = -\,\mathbb{E}\big[\min(r\,A,\ \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\,A)\big] - \lambda_H\,\mathbb{E}[H],$$
with controller learning rate $0.00035$ and an entropy bonus of weight $\lambda_H = 0.00001$ added so the controller keeps exploring instead of collapsing onto a single cell too early. The whole thing runs through a global workqueue: sampled architectures go into the queue, child workers train them on GPUs and report held-out accuracy, and the controller updates with PPO on minibatches of 20 completed architectures. The search runs for 20,000 sampled children, then I take the top 250 cells and train them to convergence on CIFAR-10. To keep search cheap, children use $N=2$ and a 20-epoch schedule with a momentum optimizer (momentum $0.9$), L2 weight decay, and cosine learning-rate decay — cosine decay matters here because it cleanly separates good architectures from bad ones within a short run. I also keep an honest baseline in view: sampling each of the $10B$ decisions uniformly instead of from the learned softmax yields random architectures from the *same* search space, and if the space is well designed that random search is already decent, so beating it is the real test of how much the RL controller adds beyond the space design itself.

One thing breaks when these cells are stacked deeply: regularization. The cells are heavily multi-branch, and dropping out individual convolutional filters degrades them. The right tool for multi-branch nets is to drop whole *paths* — each path being an operation edge in the cell — stochastically during training, with test-time rescaling, as in DropPath. But a fixed drop probability alone barely helps. What works is to make the drop probability increase linearly over the course of training: start near zero so the net learns freely, then ramp the regularization up as training proceeds. I call this ScheduledDropPath, and the scheduling is the load-bearing part — at training step $t$ over $T$ total steps the drop probability is $p_{\text{drop}} = p_{\max}\,\min(t, T)/T$ and the keep probability $1 - p_{\text{drop}}$ is what scales each path at test time.

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
