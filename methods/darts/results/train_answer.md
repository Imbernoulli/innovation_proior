Neural architecture search promises to remove the manual effort of designing networks, but the dominant approaches treat architecture choice as a discrete combinatorial problem. Reinforcement-learning controllers and evolutionary methods sample architectures, train each one to convergence, and use validation accuracy as a scalar reward. This works but costs thousands of GPU-days because every proposal requires a full training run before the search can learn anything. The root cause is that a hard discrete choice carries no gradient with respect to validation performance, so optimization cannot exploit gradient information and must fall back on expensive sample-and-evaluate loops.

The way around this is to relax the discrete architecture into a continuous representation. If every per-edge operation choice can be expressed as a differentiable weighting, then the architecture itself becomes a set of continuous parameters and we can optimize it by gradient descent. We still need to generalize rather than memorize, so the architecture should be trained for validation loss while ordinary network weights are trained for training loss. This leads to a bilevel problem. The challenge is solving it without paying the price of fully training the inner problem at every architecture step.

The method is DARTS, Differentiable Architecture Search. It works inside a cell-based search space: a cell is a small directed acyclic graph of nodes, each node sums transformed predecessors, and the network is built by stacking copies of the discovered cell. On every edge of the cell, instead of choosing a single operation, DARTS places all candidate operations and computes a softmax-weighted sum of their outputs. The weights come from learnable logits, one vector per edge. The architecture is therefore encoded by these continuous logits, which can be optimized with standard gradients.

DARTS optimizes the architecture logits for validation loss while the conv filters and batch-norm statistics are optimized for training loss. Because validation loss depends on the architecture both directly and through the weights that the architecture induces, the architecture gradient must account for the coupling between the two. DARTS approximates the optimal weights with a single unrolled training step and differentiates through that step. This yields a direct gradient term and a second-order correction involving a Hessian-vector product. The Hessian-vector product is approximated cheaply by central finite differences, so the architecture update remains linear in the number of parameters rather than quadratic. A first-order variant ignores the second-order term entirely and simply uses the gradient of validation loss with respect to the architecture logits, which is faster but less faithful.

Once the search converges, the soft mixture is converted back into a discrete cell. For each intermediate node, DARTS keeps the strongest incoming non-zero operations from distinct predecessors, excluding the zero operation from selection. This gives the final architecture, which is then used to build a larger network and trained from scratch for final evaluation.

```python
import torch, torch.nn as nn, torch.nn.functional as F

PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
NONE = PRIMITIVES.index('none')

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in PRIMITIVES:
            op = make_op(name, C, stride, affine=False)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C, reduction):
        super().__init__()
        self.steps = steps
        self.multiplier = multiplier
        self.reduction = reduction
        self.ops = nn.ModuleList()
        for i in range(steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(MixedOp(C, stride))

    def forward(self, s0, s1, weights):
        states, offset = [s0, s1], 0
        for _ in range(self.steps):
            s = sum(self.ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

class SearchNetwork(nn.Module):
    def _initialize_alphas(self, steps=4):
        k = sum(1 for i in range(steps) for _ in range(2 + i))
        self.alphas_normal = nn.Parameter(torch.zeros(k, len(PRIMITIVES)))
        self.alphas_reduce = nn.Parameter(torch.zeros(k, len(PRIMITIVES)))

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def named_weight_parameters(self):
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) not in arch_ids]

    def weight_parameters(self):
        return [p for _, p in self.named_weight_parameters()]

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            alpha = self.alphas_reduce if cell.reduction else self.alphas_normal
            s0, s1 = s1, cell(s0, s1, F.softmax(alpha, dim=-1))
        return self.classifier(self.global_pooling(s1).view(s1.size(0), -1))

    def genotype(self):
        return {
            'normal': parse_cell(F.softmax(self.alphas_normal, dim=-1).detach().cpu()),
            'reduce': parse_cell(F.softmax(self.alphas_reduce, dim=-1).detach().cpu()),
            'concat': range(2 + self.steps - self.multiplier, self.steps + 2),
        }

def _concat(xs):
    return torch.cat([x.contiguous().view(-1) for x in xs])

def construct_model_from_theta(model, theta):
    model_new = model.new()
    model_dict = model.state_dict()
    offset = 0
    for name, p in model.named_weight_parameters():
        n = p.numel()
        model_dict[name] = theta[offset:offset + n].view_as(p)
        offset += n
    assert offset == theta.numel()
    model_new.load_state_dict(model_dict)
    return model_new

def compute_unrolled_model(model, train_batch, eta, weight_opt, momentum, weight_decay):
    weights = list(model.weight_parameters())
    loss = L_train(model, train_batch)
    theta = _concat([p.detach() for p in weights])
    moment = _concat([weight_opt.state[p].get('momentum_buffer', torch.zeros_like(p))
                      for p in weights]) * momentum
    grads = torch.autograd.grad(loss, weights)
    dtheta = _concat([g.detach() for g in grads]) + weight_decay * theta
    return construct_model_from_theta(model, theta - eta * (moment + dtheta))

def hessian_vector_product(model, train_batch, vector, r=1e-2):
    weights = list(model.weight_parameters())
    R = r / _concat(vector).norm()
    for p, v in zip(weights, vector):
        p.data.add_(R, v)
    grads_p = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector):
        p.data.sub_(2 * R, v)
    grads_n = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector):
        p.data.add_(R, v)
    return [(gp - gn) / (2 * R) for gp, gn in zip(grads_p, grads_n)]

def architect_step(model, train_batch, val_batch, eta, weight_opt, arch_opt, unrolled=True):
    arch_opt.zero_grad()
    if not unrolled or eta == 0:
        L_val(model, val_batch).backward()
    else:
        unrolled_model = compute_unrolled_model(
            model, train_batch, eta, weight_opt, momentum=0.9, weight_decay=3e-4)
        L_val(unrolled_model, val_batch).backward()
        dalpha = [a.grad.detach().clone() for a in unrolled_model.arch_parameters()]
        vector = [p.grad.detach().clone() for p in unrolled_model.weight_parameters()]
        implicit = hessian_vector_product(model, train_batch, vector)
        for alpha, da, h in zip(model.arch_parameters(), dalpha, implicit):
            alpha.grad = (da - eta * h).detach().clone()
    arch_opt.step()

def search(model, train_loader, val_loader):
    weight_opt = torch.optim.SGD(model.weight_parameters(), lr=0.025,
                                 momentum=0.9, weight_decay=3e-4)
    arch_opt = torch.optim.Adam(model.arch_parameters(), lr=3e-4,
                                betas=(0.5, 0.999), weight_decay=1e-3)
    for train_batch in train_loader:
        val_batch = next(iter(val_loader))
        eta = weight_opt.param_groups[0]['lr']
        architect_step(model, train_batch, val_batch, eta, weight_opt, arch_opt, unrolled=True)
        weight_opt.zero_grad()
        L_train(model, train_batch).backward()
        nn.utils.clip_grad_norm_(model.weight_parameters(), 5)
        weight_opt.step()

def parse_cell(weights, steps=4, k=2):
    gene, start, n = [], 0, 2
    for _ in range(steps):
        node_edges = weights[start:start + n]
        keep = sorted(range(n),
                      key=lambda j: -max(node_edges[j][op].item()
                                         for op in range(len(PRIMITIVES)) if op != NONE))[:k]
        for j in keep:
            op = max((op for op in range(len(PRIMITIVES)) if op != NONE),
                     key=lambda op: node_edges[j][op].item())
            gene.append((PRIMITIVES[op], j))
        start += n
        n += 1
    return gene
```
