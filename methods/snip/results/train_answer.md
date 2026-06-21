What I want is small networks. Modern convolutional and fully-connected models carry millions of weights, and the folklore — backed by compression-based generalization arguments — is that most of them are dead weight: a much smaller subnetwork should match the dense reference's accuracy. The object is therefore clean. Given a dataset $D$ and a budget $\kappa$ on the number of surviving weights, minimize the loss subject to a hard sparsity constraint,
$$\min_{w}\; L(w; D) = \frac{1}{n}\sum_i \ell\big(w; (x_i, y_i)\big), \qquad w \in \mathbb{R}^m,\; \|w\|_0 \le \kappa.$$
The $L_0$ constraint is combinatorial and there is no hope of solving it directly for millions of weights, so the entire problem reduces to two choices: by what *criterion* do we decide which connections to keep, and at what *time* in the pipeline do we apply it. A good criterion should be cheap, free of hand-tuned prune/retrain schedules, and robust across convolutional, residual, and recurrent architectures.

The two criteria that dominate the field both fail this test for the same underlying reason. Magnitude pruning scores connection $j$ by $s_j = |w_j|$ and throws away the small weights. Curvature pruning, in the Optimal Brain Damage / Optimal Brain Surgeon lineage, expands the loss around a trained minimum, $\delta L = (\partial L/\partial w)^\top \delta w + \tfrac12 \delta w^\top H \delta w + O(\|\delta w\|^3)$, drops the first-order term because the gradient vanishes at convergence, and reads off $s_j = w_j^2 H_{jj}/2$ (or $w_j^2/(2[H^{-1}]_{jj})$ with the full Hessian). The Hessian variant is more principled but $H$ is huge, dense, not positive-definite, and intractable for large nets — OBS even re-inverts $H$ per removed weight. The deeper problem, common to *both* families, is that the scores depend on the **scale of the weights**: $|w_j|$ is literally the weight, and the OBD score is $w_j^2$ times curvature. A weight-scaled score is only meaningful once the weights have organized, i.e. once the network is trained; on a fresh random initialization $|w_j|$ is just noise from the init distribution and says nothing about whether connection $j$ matters. That is exactly why pruning has classically been a post-training step dragging an expensive prune–retrain cycle behind it — train fully, score, cut, retrain to heal, repeat — with the retraining loop being the cost and its schedule a pile of heuristics. Worse, riding on weight scale makes the scores sensitive to learning rate and to architecture, since a normalization layer silently rescales weights and shifts who gets pruned. Penalty/projection methods that attack the constrained problem directly tend to achieve worse sparsity and need heavily tuned hyperparameters, so they are no escape either.

I propose SNIP — single-shot network pruning based on connection sensitivity — which prunes once, at initialization, before any training, in a single forward–backward pass, and then trains only the survivors. The move that makes this possible is to ask a different question than "is this weight big?" Instead I ask "does this *connection* matter to the loss for this task?", in a way that is robust to the arbitrary scale of the random weights. The trouble with the raw weights is that they conflate the *strength* of a connection with *whether the connection is even there*; I want to isolate the second. So I introduce a second variable per connection, an indicator gate $c_j \in \{0,1\}$ that says connection $j$ is present (1) or removed (0), and write the effective weights as the Hadamard product $c \odot w$. The constrained problem becomes $\min_{c,w} L(c \odot w; D)$ with $c \in \{0,1\}^m$, $\|c\|_0 \le \kappa$. This nested combinatorial problem over $c$ is strictly *harder* than the original, so I have no intention of solving it — that is not why $c$ is there. The gate is there because $c_j$ is now a clean dial that turns connection $j$ on and off, decoupled from its weight $w_j$, and the importance of a connection is simply how much the loss reacts when I flip its dial.

The exact effect of removing connection $j$ is $\Delta L_j = L(\mathbf{1} \odot w; D) - L\big((\mathbf{1} - e_j) \odot w; D\big)$, the loss with everything on minus the loss with only connection $j$ knocked out, where $e_j$ is the $j$-th unit vector. This is intrinsically about the *connection* and can be evaluated at any $w$, including random init — but it is hopeless to compute directly ($m+1$ forward passes, $m$ in the millions) and, since $c$ is binary, $L$ is not even differentiable in $c$. The escape is to relax $c$ off the integers. For an infinitesimal nudge from the all-on network, $\Delta L_j$ is approximated by the derivative of $L$ with respect to $c_j$ evaluated at the all-ones point:
$$g_j(w; D) = \left.\frac{\partial L(c \odot w; D)}{\partial c_j}\right|_{c=\mathbf{1}} = \lim_{\delta \to 0} \frac{L(c \odot w; D) - L\big((c - \delta e_j)\odot w; D\big)}{\delta}\bigg|_{c=\mathbf{1}}.$$
This is the rate of change of $L$ as $c_j$ slides from 1 toward $1-\delta$, and crucially I do not need $m+1$ forward passes: $\partial L/\partial c_j$ for *all* $j$ at once is one forward–backward pass through autodiff, because $c$ enters as a multiplicative gate so the backward pass hands me every $g_j$ simultaneously. The $m+1$ forward passes collapse to one.

What makes this the right score becomes clear on staring at $g_j$. With the effective gated weight $u_j = c_j w_j$, the chain rule gives
$$\frac{\partial L}{\partial c_j} = \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial c_j} = \frac{\partial L}{\partial u_j}\cdot w_j.$$
So the connection-sensitivity gradient is the ordinary effective-weight gradient multiplied by the current weight. Compare it to the bare $\partial L/\partial w_j$ that older sensitivity criteria (Mozer–Smolensky, Karnin) effectively used: the bare gradient measures the change from an *additive* perturbation $\delta$ applied uniformly, in isolation from how big $w_j$ already is. But $\partial L/\partial c_j$ is the derivative with respect to a multiplicative gate, where a small change in $c_j$ changes the effective weight by $\delta\cdot w_j$. That is exactly the pruning operation I care about — removing a connection shrinks that connection's own contribution toward zero, not adding the same small number to every weight. This does not make the initialized weights irrelevant; the score still uses $w$ through the forward pass and through the factor above, which is precisely why initialization has to be chosen carefully. What it removes is the need for $w$ to be *pretrained*: the score is the loss response to a present connection, not a trained magnitude used as a proxy.

Two details finish the criterion. First, sign. If $g_j$ is large and *negative*, connection $j$ still has a large effect on the loss — moving its gate changes the objective a lot, so it is important and must not be discarded. What matters is the *magnitude* of the effect regardless of direction. Since the computational object is the gate vector, the local saliency is $|g \odot c|$, and at the scoring moment $c=\mathbf{1}$ so this is exactly $|\partial L/\partial c|$. To make scores comparable across the network and untied from the absolute scale of the loss or the batch, normalize by the total:
$$a_j = |g_j(w; D)\,c_j|,\quad c=\mathbf{1}, \qquad s_j = \frac{a_j}{\sum_k a_k} = \frac{|g_j(w; D)|}{\sum_k |g_k(w; D)|}.$$
This is the connection-sensitivity score. Normalization does not change the ordering, so given the budget $\kappa$ I keep the top-$\kappa$ entries by $|\partial L/\partial c|$ via the threshold $c_j = \mathbf{1}[\, s_j - \tilde{s}_\kappa \ge 0\,]$, where $\tilde{s}_\kappa$ is the $\kappa$-th largest entry of $s$, ties broken to land exactly $\kappa$. Sort descending, threshold, done.

Second, because the score still touches $w$ through the $(\partial L/\partial u_j)\cdot w_j$ factor and the forward pass, the choice of initial $w$ is not irrelevant; it must be sensible, and two failure modes have to be avoided. If the initial weights are too large, post-nonlinearity activations saturate, local gradients go flat, and $\partial L/\partial u_j$ becomes uninformative — so the weights must sit in a reasonable range, which is what standard initialization schemes are for. And for robustness across architectures, fixed-variance random weights are not enough: with a fixed variance everywhere, the variance of the signal and hence of the gradients drifts from layer to layer, and then the saliency picks up a spurious dependence on depth and width, with whole layers getting systematically smaller or larger $g_j$ because of variance drift rather than because their connections matter less or more. The fix is a variance-scaling initialization, Glorot- or He-style, that holds signal variance roughly constant through the layers, so $g_j$ does not become a proxy for architectural scale. Variance scaling is not decoration here; it is what keeps connection sensitivity from tracking architecture-scale drift. Finally, $s_j$ depends on $D$ and on $L$, i.e. on the *task* — a feature, since the retained connections are the ones important for this task — and one mini-batch of reasonable size suffices for the single-shot ranking, with the option to accumulate $|g|$ over several batches if a steadier estimate is wanted.

The whole method then assembles itself. Variance-scaling init; one mini-batch; one forward–backward pass with the multiplicative gate $c=\mathbf{1}$ to get every $\partial L/\partial c_j$; form $|g \odot c|$, which here is $|g|$, and keep the top-$\kappa$ entries; then throw the gates away as variables, fix the binary mask $c$, and train the sparse network the standard way as $w^\star = \arg\min_w L(c \odot w; D)$. No pretraining, no prune–retrain cycle, no hyperparameters beyond $\kappa$. In an autodiff framework the clean way to compute $g_j$ is the gate trick itself: attach a multiplicative mask parameter initialized to ones in front of each weight, freeze the weights while scoring, run one batch, backprop, and read off the gradient *of the mask* — that is $\partial L/\partial c_j$ by construction. Ranking $|\texttt{mask.grad} * \texttt{mask}|$ is the same as ranking $|\partial L/\partial c|$ at $c=\mathbf{1}$, and after pruning the fixed mask keeps the removed connections at zero during normal training.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GatedConv2d(nn.Conv2d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))  # gate c = 1
        self.weight.requires_grad = False                              # freeze w to get dL/dc
    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class GatedLinear(nn.Linear):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))
        self.weight.requires_grad = False
    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def variance_scaling_init(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

def compute_keep_mask(net, batch, keep_fraction):
    x, y = batch
    net.zero_grad(set_to_none=True)
    F.cross_entropy(net(x), y).backward()                 # gates at c = 1
    gated = [m for m in net.modules() if isinstance(m, (GatedConv2d, GatedLinear))]
    scores = [torch.abs(m.weight_mask.grad * m.weight_mask).detach() for m in gated]  # |g * c|
    flat_scores = torch.cat([s.flatten() for s in scores])
    flat_scores = flat_scores / flat_scores.sum().clamp_min(torch.finfo(flat_scores.dtype).tiny)

    k = int(keep_fraction * flat_scores.numel())
    k = max(1, min(k, flat_scores.numel()))
    keep = torch.zeros_like(flat_scores)
    keep[torch.topk(flat_scores, k, sorted=False).indices] = 1.0

    masks, offset = [], 0
    for score in scores:
        n = score.numel()
        masks.append(keep[offset:offset + n].view_as(score))
        offset += n
    return gated, masks

def prune_and_train(net, gated, masks, loader, epochs, lr):
    for m, msk in zip(gated, masks):
        with torch.no_grad():
            m.weight_mask.copy_(msk)        # fix binary mask c
        m.weight_mask.requires_grad = False
        m.weight.requires_grad = True       # train surviving weights
    net.zero_grad(set_to_none=True)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()                      # pruned weights stay zero (mask fixed at 0)
```
