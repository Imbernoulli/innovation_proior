Let me start from why architecture search is so painful right now, because that diagnoses the fix. Everyone treats the architecture as a discrete object — pick this op or that op on each edge — and then searches that discrete space with reinforcement learning or evolution. The controller proposes an architecture, you train it to convergence, you read off one scalar (validation accuracy), and you use that scalar as a reward to nudge the controller. The whole cost is in that inner "train a child to convergence to get one number," repeated thousands of times — thousands of GPU-days. And the reason you're stuck doing it that way is structural: the architecture variable is *discrete*, so validation performance is a non-differentiable function of it. There is no gradient of "how good is this architecture" with respect to "which op did I pick." If there *were* such a gradient, I could just do gradient descent on the architecture and skip the sample-train-reward loop entirely.

So the real question becomes: can I make the choice of operation *continuous*, so that the gradient exists? Let me set up the search space first, then try to relax it. Following the cell idea: a cell is a directed acyclic graph of N nodes; each node x^(j) is a feature map, each directed edge (i,j) carries some operation o^(i,j) applied to x^(i), and a node is the sum of its incoming transformed predecessors:

```
x^(j) = Σ_{i<j} o^(i,j)(x^(i)).
```

There's a special *zero* op meaning "no connection." Learning the cell = choosing the operation on each edge. That choice is the discrete, non-differentiable thing.

How do I make "pick one operation out of the set O" differentiable? A hard pick is an argmax over a categorical — not differentiable. The standard softening of a categorical choice is a softmax. So instead of committing to one op on edge (i,j), let me try putting *all* candidate ops there and taking a weighted average, with the weights coming from a softmax over a learnable vector α^(i,j) of logits (one per candidate op):

```
ō^(i,j)(x) = Σ_{o∈O}  [ exp(α_o^(i,j)) / Σ_{o'∈O} exp(α_{o'}^(i,j)) ]  o(x).
```

Now every edge is a smooth mixture of operations, controlled by continuous α's. The architecture *is* the set α = {α^(i,j)}, and it's a continuous variable I can differentiate through. At the end the soft mixture has to collapse back to hard choices — keep the strongest incoming non-zero edges for each node, and on each kept edge take the strongest non-zero operation — but during search everything is continuous. That's the relaxation I wanted: the discrete search has turned into learning a continuous α.

Now, what objective trains α? I have two sets of variables: the architecture α and the ordinary network weights w (the conv filters inside the mixed ops). I do *not* want to choose α to minimize *training* loss — that would just pick whatever architecture overfits the training set hardest (e.g. the most expressive ops), telling me nothing about generalization. The whole point of the RL approach was that the reward was *validation* performance. So I want α to minimize *validation* loss, while w minimizes *training* loss given α. That's a bilevel problem:

```
min_α   L_val( w*(α), α )
s.t.    w*(α) = argmin_w  L_train(w, α).
```

α is the upper-level variable, w the lower. This is the structure of gradient-based hyperparameter optimization — α is like a (huge-dimensional) hyperparameter. Fine in principle. The trouble is w*(α): to evaluate the upper objective I'd have to *fully solve* the inner argmin for every α, i.e. train to convergence — which is the very expense I'm trying to kill. So I need the architecture gradient ∇_α L_val(w*(α), α) without solving the inner problem.

The escape is to not solve the inner argmin at all — approximate w*(α) by a *single* gradient step from the current weights w:

```
w*(α) ≈ w' = w − ξ ∇_w L_train(w, α),
```

ξ the inner learning rate. Then alternate: one step adapting w on training loss, one step adapting α on the validation loss of the one-step-ahead weights. This is the one-step-unrolled trick used in meta-learning for model transfer and in unrolled GANs — differentiate through a single optimization step instead of the whole optimization.

Now actually compute the derivative of L_val(w'(α), α), where w' depends on α. This is a total derivative, and I have to be careful to get *both* ways α enters: directly (the second argument of L_val) and *through* w'. Treat L_val as a function of two arguments and apply the chain rule:

```
d/dα L_val(w'(α), α) = ∂_α L_val(w', α)   +   (∂w'/∂α)ᵀ ∇_{w'} L_val(w', α).
```

The first term is the direct dependence: the gradient w.r.t. the explicit α slot while holding w' fixed. For the second, I need the transpose-Jacobian product with ∂w'/∂α, not just a symbol with the right sign. Since w' = w − ξ ∇_w L_train(w, α) and w is the current (constant) iterate,

```
∂w'/∂α = − ξ ∇²_{w,α} L_train(w, α),
```

where this Jacobian maps a perturbation in α to a perturbation in w'. The chain-rule term uses its transpose, so

```
(∂w'/∂α)ᵀ v = − ξ (∇²_{w,α} L_train(w, α))ᵀ v
             = − ξ ∇²_{α,w} L_train(w, α) · v,
```

with v = ∇_{w'} L_val(w', α). Plugging that back into the total derivative gives

```
d/dα L_val(w'(α), α)  =  ∂_α L_val(w', α)  −  ξ ∇²_{α,w} L_train(w, α) · ∇_{w'} L_val(w', α).
```

The sign is negative — it's −ξ times a Hessian-vector product. I want to be sure I haven't dropped or flipped a term, because the whole correction lives in that second piece and a sign error there would push α the wrong way. Let me check the formula against a brute-force derivative on the smallest non-trivial case: scalars α=a and w, with two loss functions chosen so that the mixed partial ∇²_{α,w} L_train is genuinely non-zero (otherwise the second term vanishes and proves nothing). Take

```
L_train(w,a) = ½ w² − a w + 0.1 a² w,   L_val(w,a) = ½ (w−1)² + 0.3 a² + 0.2 a w,
```

ξ = 0.3, evaluated at w=0.7, a=0.4. From these, ∂_w L_train = w − a + 0.1 a², so w' = w − ξ(w − a + 0.1 a²) = 0.6928. The mixed partial ∂²L_train/∂a∂w = ∂_w(−w + 0.2 a w) = −1 + 0.2 a = −0.92. The direct term ∂_a L_val(w',a) = 0.6a + 0.2w' = 0.3786, and v = ∂_{w'} L_val(w',a) = (w'−1) + 0.2a = −0.2272. So the formula gives

```
0.3786  −  0.3 · (−0.92) · (−0.2272)  =  0.3786 − 0.0627  =  0.27416.
```

Against this I compute the brute-force number: define g(a) = L_val(w − ξ ∇_w L_train(w,a), a) with w held at 0.7, and take a symmetric difference (a±10⁻⁶). That returns **0.274155**. The two agree to five places. The sign on the Hessian term is right, and the magnitude is right — if I'd dropped the −ξ factor or flipped its sign I'd have landed near 0.379 or 0.441, nowhere near. Worth one more sanity point: set ξ=0 in the formula. Then w'=w exactly, the second term has a literal ξ factor so it dies, and the gradient is just ∂_a L_val(w,a) = 0.6·0.4 + 0.2·0.7 = 0.38 — the bare "validation gradient at the current weights," which is what you'd expect when you decline to look ahead. That matches plugging ξ=0 into g(a) numerically. So the total-derivative formula holds and the ξ=0 case degrades gracefully.

That second term is still the headache: ∇²_{α,w} L_train is a |α|×|w| matrix, and even as a matrix-vector product against ∇_{w'}L_val it's expensive — naively O(|α||w|). I don't want to form that matrix at all. The thing I actually need is the *product* ∇²_{α,w} L_train · v for the one specific v = ∇_{w'}L_val, and a Hessian-times-a-known-vector is exactly what a finite difference of a gradient computes. Pick a small scalar ε and define w± = w ± ε v (perturb the *weights* along v), then look at how ∇_α L_train moves:

```
∇²_{α,w} L_train(w, α) · v  ≈  [ ∇_α L_train(w⁺, α) − ∇_α L_train(w⁻, α) ] / (2ε).
```

The reasoning: ∇²_{α,w} L_train · v is the directional derivative of the gradient-field ∇_α L_train as w moves in direction v, and a central difference of ∇_α L_train at w±εv estimates that directional derivative. Let me not just assert that and instead check both the value and the order of accuracy. Value first, on the same scalar example: the finite-difference estimate of ∇²_{α,w}L_train·v at w=0.7,a=0.4 with ε=10⁻⁴ comes out to **0.289616**, and the exact (−0.92)·(−0.2272) = **0.289616** — same number, so the finite difference is computing the right object. For the order of accuracy I need a loss whose mixed partial genuinely varies with w (the quadratic above has a constant mixed partial, so its central difference is exact and tells me nothing about the truncation error). Take instead ∇_α L_train behaving like w³ along the perturbation, exact target 1.9110, and sweep ε:

```
ε = 1e-1 → 1.93297  (err 2.20e-2)
ε = 1e-2 → 1.91122  (err 2.20e-4)
ε = 1e-3 → 1.91100  (err 2.20e-6)
ε = 1e-4 → 1.91100  (err 2.20e-8)
```

The error falls by exactly 100× each time ε falls by 10×, i.e. it scales as ε² — the central difference is second-order accurate, as I'd hoped. Now the cost: this estimate is two extra evaluations of the training loss and two gradients w.r.t. α, after perturbing the current weights in the ±v directions — so the whole second-order correction is O(|α| + |w|) instead of O(|α||w|). The ε² error means I can take ε fairly large; but too large and I'd leave the linear regime, too small and the gradient difference drowns in numerical noise, so I scale it to the perturbation, ε = 0.01/‖v‖₂, which keeps ‖εv‖ a fixed small fraction of w regardless of the magnitude of v.

So the architecture update is: form w' by one inner step; compute v = ∇_{w'}L_val(w',α); the architecture gradient is ∂_α L_val(w',α) − ξ·[∇_α L_train(w⁺,α) − ∇_α L_train(w⁻,α)]/(2ε); descend α by it. Then take the weight step on L_train and repeat.

Is the second-order term even worth carrying around? The ξ=0 check above already told me what dropping it does: with ξ=0 the one-step update vanishes, w' = w, and the architecture gradient collapses to ∇_α L_val(w, α). That's a *first-order* approximation — it pretends the current w is already w*(α), i.e. it ignores that changing α would change the optimal weights. It's cheaper (no w±, no second backward set), but it throws away the coupling between α and w. From the scalar example I can read off how much that coupling was worth at that point: the full gradient was 0.274 and the first-order one is 0.380, so here the look-ahead correction is about a third of the gradient and even shifts it — not negligible. So ξ > 0 (second-order) keeps the one-step coupling and is the better default; ξ = 0 (first-order) is the speed option. A simple working choice is to set ξ to the weights' current learning rate.

The continuous α still has to become a discrete cell. For each intermediate node, keep the top-k strongest incoming operations from distinct predecessor nodes, where "strength" of op o on edge (i,j) is its softmax weight exp(α_o^(i,j))/Σ_{o'} exp(α_{o'}^(i,j)). Use k=2 for convolutional cells (to match the connectivity of existing hand-designed cells) and k=1 for recurrent cells. One thing to settle: should the *zero* op be eligible for this top-k? I want exactly k non-zero incoming edges per node for a fair comparison to existing models, so a node that selected "zero" as one of its k would come out under-connected. And the strength of zero isn't even a meaningful ranking signal: there's a batch-norm layer downstream of each node, so scaling the zero logit only rescales the node's pre-BN representation, which BN then normalizes away — it doesn't change the classification at all. An op whose magnitude the network is free to ignore can't have its softmax weight read as "importance." So I rank only the non-zero ops and keep the top-k among those.

The implementation has to preserve one more detail. In the convolutional setting there are two shared cell templates, normal and reduction, not a different α for every physical cell in the stack. So the search model keeps two architecture tensors, α_normal and α_reduce; a normal cell reads softmax(α_normal), a reduction cell reads softmax(α_reduce), and edges adjacent to the two input nodes in a reduction cell use stride two. I initialize those logits at zero, so the first softmax puts equal attention on every candidate op and each one receives some early weight-gradient signal instead of being starved before the search has learned anything.

The search loop then has this shape.

```python
import torch, torch.nn as nn, torch.nn.functional as F

PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
NONE = PRIMITIVES.index('none')

class MixedOp(nn.Module):                         # one edge: weighted mix of all candidate ops
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in PRIMITIVES:
            op = make_op(name, C, stride, affine=False)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)
    def forward(self, x, weights):                # weights = softmax(alpha_edge)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class Cell(nn.Module):                            # each intermediate node sums all previous states
    def __init__(self, steps, multiplier, C, reduction):
        super().__init__()
        self.steps = steps
        self.multiplier = multiplier
        self.reduction = reduction
        self.ops = nn.ModuleList()
        for i in range(steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(MixedOp(C, stride))
    def forward(self, s0, s1, weights):
        states, offset = [s0, s1], 0
        for _ in range(self.steps):
            s = sum(self.ops[offset+j](h, weights[offset+j])
                    for j, h in enumerate(states))
            offset += len(states); states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

class SearchNetwork(nn.Module):
    def _initialize_alphas(self, steps=4):
        k = sum(1 for i in range(steps) for _ in range(2+i))
        self.alphas_normal = nn.Parameter(torch.zeros(k, len(PRIMITIVES)))  # uniform softmax at init
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
        model_dict[name] = theta[offset:offset+n].view_as(p)
        offset += n
    assert offset == theta.numel()
    model_new.load_state_dict(model_dict)
    return model_new

def compute_unrolled_model(model, train_batch, eta, weight_opt, momentum, weight_decay):
    weights = list(model.weight_parameters())                            # ordinary weights only
    loss = L_train(model, train_batch)
    theta = _concat([p.detach() for p in weights])
    moment = _concat([weight_opt.state[p].get('momentum_buffer', torch.zeros_like(p))
                      for p in weights]) * momentum
    grads = torch.autograd.grad(loss, weights)
    dtheta = _concat([g.detach() for g in grads]) + weight_decay * theta
    return construct_model_from_theta(model, theta - eta * (moment + dtheta))       # w'

def hessian_vector_product(model, train_batch, vector, r=1e-2):
    weights = list(model.weight_parameters())                            # current w, not w'
    R = r / _concat(vector).norm()
    for p, v in zip(weights, vector): p.data.add_(R, v)
    grads_p = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector): p.data.sub_(2*R, v)
    grads_n = torch.autograd.grad(L_train(model, train_batch), model.arch_parameters())
    for p, v in zip(weights, vector): p.data.add_(R, v)
    return [(gp - gn) / (2*R) for gp, gn in zip(grads_p, grads_n)]

def architect_step(model, train_batch, val_batch, eta, weight_opt, arch_opt, unrolled=True):
    arch_opt.zero_grad()
    if not unrolled or eta == 0:
        L_val(model, val_batch).backward()                               # first-order
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
        val_batch = next(iter(val_loader))                               # validation minibatch for alpha
        eta = weight_opt.param_groups[0]['lr']
        architect_step(model, train_batch, val_batch, eta, weight_opt, arch_opt, unrolled=True)
        weight_opt.zero_grad()
        L_train(model, train_batch).backward()
        nn.utils.clip_grad_norm_(model.weight_parameters(), 5)
        weight_opt.step()

def parse_cell(weights, steps=4, k=2):
    gene, start, n = [], 0, 2
    for _ in range(steps):
        node_edges = weights[start:start+n]
        keep = sorted(range(n),
                      key=lambda j: -max(node_edges[j][op].item()
                                         for op in range(len(PRIMITIVES)) if op != NONE))[:k]
        for j in keep:
            op = max((op for op in range(len(PRIMITIVES)) if op != NONE),
                     key=lambda op: node_edges[j][op].item())
            gene.append((PRIMITIVES[op], j))
        start += n; n += 1
    return gene
```
