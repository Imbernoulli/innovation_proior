Let me start from why architecture search is so painful right now, because that diagnoses the fix. Everyone treats the architecture as a discrete object — pick this op or that op on each edge — and then searches that discrete space with reinforcement learning or evolution. The controller proposes an architecture, you train it to convergence, you read off one scalar (validation accuracy), and you use that scalar as a reward to nudge the controller. The whole cost is in that inner "train a child to convergence to get one number," repeated thousands of times — thousands of GPU-days. And the reason you're stuck doing it that way is structural: the architecture variable is *discrete*, so validation performance is a non-differentiable function of it. There is no gradient of "how good is this architecture" with respect to "which op did I pick." If there *were* such a gradient, I could just do gradient descent on the architecture and skip the sample-train-reward loop entirely.

So the real question becomes: can I make the choice of operation *continuous*, so that the gradient exists? Let me set up the search space first, then relax it. Following the cell idea: a cell is a directed acyclic graph of N nodes; each node x^(j) is a feature map, each directed edge (i,j) carries some operation o^(i,j) applied to x^(i), and a node is the sum of its incoming transformed predecessors:

```
x^(j) = Σ_{i<j} o^(i,j)(x^(i)).
```

There's a special *zero* op meaning "no connection." Learning the cell = choosing the operation on each edge. That choice is the discrete, non-differentiable thing.

How do I make "pick one operation out of the set O" differentiable? A hard pick is an argmax over a categorical — not differentiable. The standard softening of a categorical choice is a softmax. So instead of committing to one op on edge (i,j), put *all* candidate ops there and take a weighted average, with the weights coming from a softmax over a learnable vector α^(i,j) of logits (one per candidate op):

```
ō^(i,j)(x) = Σ_{o∈O}  [ exp(α_o^(i,j)) / Σ_{o'∈O} exp(α_{o'}^(i,j)) ]  o(x).
```

Now every edge is a smooth mixture of operations, controlled by continuous α's. The architecture *is* the set α = {α^(i,j)}, and it's a continuous variable I can differentiate through. At the end I'll recover a discrete cell by taking the most likely op per edge, argmax_o α_o^(i,j) — but during search everything is continuous. Good. The discrete search collapsed into learning a continuous α.

Now, what objective trains α? I have two sets of variables: the architecture α and the ordinary network weights w (the conv filters inside the mixed ops). I do *not* want to choose α to minimize *training* loss — that would just pick whatever architecture overfits the training set hardest (e.g. the most expressive ops), telling me nothing about generalization. The whole point of the RL approach was that the reward was *validation* performance. So I want α to minimize *validation* loss, while w minimizes *training* loss given α. That's a bilevel problem:

```
min_α   L_val( w*(α), α )
s.t.    w*(α) = argmin_w  L_train(w, α).
```

α is the upper-level variable, w the lower. This is exactly the structure of gradient-based hyperparameter optimization — α is like a (huge-dimensional) hyperparameter. Fine in principle. The trouble is w*(α): to evaluate the upper objective I'd have to *fully solve* the inner argmin for every α, i.e. train to convergence — which is the very expense I'm trying to kill. So I need the architecture gradient ∇_α L_val(w*(α), α) without solving the inner problem.

The escape is to not solve the inner argmin at all — approximate w*(α) by a *single* gradient step from the current weights w:

```
w*(α) ≈ w' = w − ξ ∇_w L_train(w, α),
```

ξ the inner learning rate. Then alternate: one step adapting w on training loss, one step adapting α on the validation loss of the one-step-ahead weights. This is the same one-step-unrolled trick used in meta-learning for model transfer and in unrolled GANs — differentiate through a single optimization step instead of the whole optimization. Note a sanity check: if w were already a local optimum for the inner problem, ∇_w L_train = 0, the step does nothing, w' = w, and the architecture gradient reduces to ∇_α L_val(w, α) — exactly what you'd want at convergence.

Now actually compute ∇_α L_val(w', α), where w' depends on α. This is a total derivative, and I have to be careful to get *both* ways α enters: directly (the second argument of L_val) and *through* w'. Treat L_val as a function of two arguments and apply the chain rule:

```
∇_α L_val(w', α) = [∂L_val/∂α]   +   (∂w'/∂α)ᵀ ∇_{w'} L_val(w', α).
```

The first term is the direct dependence — write it ∇_α L_val(w', α), the gradient w.r.t. the explicit α slot. For the second, I need ∂w'/∂α. Since w' = w − ξ ∇_w L_train(w, α) and w is the current (constant) iterate,

```
∂w'/∂α = − ξ  ∂/∂α [ ∇_w L_train(w, α) ] = − ξ ∇²_{α,w} L_train(w, α),
```

the matrix of mixed second derivatives (rows α, columns w). Plugging in:

```
∇_α L_val(w', α)  =  ∇_α L_val(w', α)  −  ξ ∇²_{α,w} L_train(w, α) · ∇_{w'} L_val(w', α).
```

(Same symbol on both sides for the direct term; the point is the *second* term, the correction from w' depending on α.) The sign is negative — it's −ξ times a Hessian-vector product — and that minus sign matters: it says "move α so that the one-step weight update it induces lowers validation loss." Lose the sign and you'd push α the wrong way.

That second term is the headache: ∇²_{α,w} L_train is a |α|×|w| matrix, and even as a matrix-vector product against ∇_{w'}L_val it's expensive — naively O(|α||w|). I'm not forming that matrix. Use a finite-difference approximation of the Hessian-vector product. Let v = ∇_{w'} L_val(w', α) be the vector I'm multiplying by, pick a small scalar ε, and define

```
w± = w ± ε v.
```

A central finite difference of the gradient ∇_α L_train along the direction v in *weight* space gives exactly the mixed-partial-times-v:

```
∇²_{α,w} L_train(w, α) · v  ≈  [ ∇_α L_train(w⁺, α) − ∇_α L_train(w⁻, α) ] / (2ε).
```

Why this is the right object: differentiating L_train first w.r.t. α and then taking a directional derivative w.r.t. w in direction v is the same mixed second derivative, and a central difference of ∇_α L_train at w±εv estimates that directional derivative to O(ε²). Now the cost: two extra forward/backward passes on the *weights* to build w±, and two backward passes for ∇_α L_train — so the whole second-order correction is O(|α| + |w|) instead of O(|α||w|). A good ε is something like 0.01/‖v‖₂ so the perturbation is small but not lost to numerical noise.

So the architecture update is: form w' by one inner step; compute v = ∇_{w'}L_val(w',α); the architecture gradient is ∇_α L_val(w',α) − ξ·[∇_α L_train(w⁺,α) − ∇_α L_train(w⁻,α)]/(2ε); descend α by it. Then take the weight step on L_train and repeat.

Now — is the second-order term even worth it? Consider ξ = 0: the one-step update vanishes, w' = w, and the whole second term (the ξ-multiplied Hessian piece) disappears. The architecture gradient is just ∇_α L_val(w, α). That's the *first-order* approximation — pretend the current w is already w*(α), i.e. ignore that changing α would change the optimal weights. It's cheaper (no w±, no second backward set), but it's blind to the coupling between α and w, and empirically it does worse. So ξ > 0 (second-order) is the better default; ξ = 0 (first-order) is the speed option. A reasonable ξ is the weights' own learning rate.

Last piece: turning the continuous α into a discrete cell. For each intermediate node, keep the top-k strongest incoming operations from distinct predecessor nodes, where "strength" of op o on edge (i,j) is its softmax weight exp(α_o^(i,j))/Σ_{o'} exp(α_{o'}^(i,j)). Use k=2 for convolutional cells (to match the connectivity of existing hand-designed cells) and k=1 for recurrent cells. Crucially, *exclude the zero op* from this top-k selection — for two reasons. First, I need exactly k non-zero incoming edges per node for a fair comparison to existing models. Second, the strength of zero is underdetermined: with batch normalization downstream, inflating the zero logit only rescales the node's representation and doesn't change the classification, so its softmax weight isn't a meaningful "importance." So rank only the non-zero ops, keep the top-k.

Let me write the search loop.

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5',
       'max_pool_3x3','avg_pool_3x3','skip_connect','none']

class MixedEdge(nn.Module):                      # one edge: weighted mix of all candidate ops
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([make_op(n, C) for n in OPS])
    def forward(self, x, alpha_edge):            # alpha_edge: logits, one per op
        w = F.softmax(alpha_edge, dim=-1)        # ō = Σ softmax(α)_o · o(x)
        return sum(wi * op(x) for wi, op in zip(w, self.ops))

class Cell(nn.Module):                            # DAG of N nodes
    def __init__(self, N, C):
        super().__init__()
        self.N = N
        self.edges = nn.ModuleList([MixedEdge(C) for _ in range(num_edges(N))])
    def forward(self, s0, s1, alpha):
        states = [s0, s1]
        offset = 0
        for j in range(self.N):
            s = sum(self.edges[offset+i](h, alpha[offset+i]) for i, h in enumerate(states))
            offset += len(states); states.append(s)
        return torch.cat(states[2:], dim=1)       # output = concat of intermediate nodes

def hessian_vector(model, alpha, w_loader, v, eps):
    # central difference: ∇²_{α,w} L_train · v  ≈  (∇_α L_train(w+εv) − ∇_α L_train(w−εv)) / 2ε
    R = eps / (v.norm() + 1e-12)
    for p, vi in zip(model.weights(), v): p.data.add_(R, vi)      # w+ = w + εv
    g_plus = torch.autograd.grad(L_train(model, next(w_loader), alpha), alpha)
    for p, vi in zip(model.weights(), v): p.data.sub_(2*R, vi)    # w- = w - εv
    g_minus = torch.autograd.grad(L_train(model, next(w_loader), alpha), alpha)
    for p, vi in zip(model.weights(), v): p.data.add_(R, vi)      # restore w
    return [(gp - gm) / (2*R) for gp, gm in zip(g_plus, g_minus)]

def arch_step(model, alpha, train_batch, val_batch, xi, eps, opt_alpha):
    opt_alpha.zero_grad()
    if xi == 0:                                   # first-order: ∇_α L_val(w, α)
        L_val(model, val_batch, alpha).backward()
    else:                                         # second-order (one-step unrolled)
        w0 = [p.detach().clone() for p in model.weights()]
        gw = torch.autograd.grad(L_train(model, train_batch, alpha), model.weights())
        for p, g in zip(model.weights(), gw): p.data.sub_(xi, g)  # w' = w − ξ∇_w L_train
        dalpha = torch.autograd.grad(L_val(model, val_batch, alpha), alpha, retain_graph=True)
        v       = torch.autograd.grad(L_val(model, val_batch, alpha), model.weights())
        hv = hessian_vector(model, alpha, train_iter, v, eps)
        for p, w in zip(model.weights(), w0): p.data.copy_(w)     # restore w
        for a, da, h in zip(alpha, dalpha, hv):
            a.grad = da - xi * h                  # ∇_α L_val(w',α) − ξ·HVP
    opt_alpha.step()

def search(model, alpha, train_loader, val_loader, xi, eps):
    opt_w = torch.optim.SGD(model.weights(), lr=xi, momentum=0.9, weight_decay=3e-4)
    opt_a = torch.optim.Adam(alpha, lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    for (xt, yt), (xv, yv) in zip(train_loader, val_loader):
        arch_step(model, alpha, (xt, yt), (xv, yv), xi, eps, opt_a)   # update α on val loss
        opt_w.zero_grad(); L_train(model, (xt, yt), alpha).backward(); opt_w.step()  # update w

def derive_discrete_cell(alpha):                   # keep top-2 non-zero ops per node, by softmax strength
    cell = []
    for node_edges in group_by_node(alpha):
        strengths = [(F.softmax(e, -1)[:-1].max(), i) for i, e in enumerate(node_edges)]  # drop 'none'
        keep = sorted(strengths, reverse=True)[:2]
        cell.append([(i, F.softmax(node_edges[i], -1)[:-1].argmax()) for _, i in keep])
    return cell
```

Causal chain: NAS is expensive because the architecture is discrete, so validation performance has no gradient w.r.t. it and you must sample-and-train; so I relax the per-edge categorical op-choice into a softmax mixture controlled by continuous logits α, making the architecture differentiable; I want α chosen for generalization, so it's a bilevel problem — α minimizes validation loss subject to w minimizing training loss; to avoid solving the inner argmin I approximate w*(α) by one gradient step w' and differentiate through it, which by the chain rule yields a direct term plus a −ξ Hessian-vector correction that I compute by central finite difference in O(|α|+|w|); dropping the correction (ξ=0) gives the cheaper first-order variant; finally I discretize by keeping each node's top-2 strongest non-zero ops by softmax weight.
