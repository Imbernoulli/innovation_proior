ANIL's numbers closed one axis and, by closing it, pointed straight at the one I had not touched. The bet paid off as framed: at 5-shot ANIL recovered everything Meta-SGD lost and more — miniImageNet 5-shot back to 0.6366 (Meta-SGD had sagged to 0.6237) and CIFAR-FS 5-shot up to 0.7138, past MAML's 0.7067 and the best 5-shot CIFAR number on the ladder, the seed spread Meta-SGD had widened (std 0.0113) tightening back to 0.0035. At 1-shot it did not just hold Meta-SGD's gain, it improved on it: 0.4815 against 0.4760, the tightest 1-shot spread yet (std 0.0029). So ANIL is the strongest rung on every benchmark — and that is also the problem. I have now optimized two of the three axes of the inner loop, and counting the launch point, three of the design knobs. MAML learned *where to start*; Meta-SGD learned a per-coordinate *rate*; ANIL settled *which parameters* adapt. What no rung has learned is the *direction*: every method so far steps along the raw gradient, and Meta-SGD only rescales it coordinate-by-coordinate, which is a diagonal tilt at best. The inner step is still, geometrically, "follow the gradient" — the axis left on the table, and exactly where ill-conditioning lives.

Look hard at what the surviving inner loop does. ANIL adapts only the head — a single linear classifier mapping the 1600-dim frozen feature to five logits — by plain SGD, ten steps at evaluation. So the entire adaptation at test time is *gradient descent on a five-way logistic regression over a fixed 1600-dimensional feature*, and that is precisely where the *direction* of the step matters most, because the conditioning of that little problem is set by the feature covariance, which I do not control and which is generally far from isotropic: some feature directions are high-variance and correlated across the 1600 inputs, others nearly dead. Plain SGD on an ill-conditioned logistic regression zig-zags — fast along the cheap directions, crawling along the informative ones — and ten steps from a single example per class is not many to waste on a zig-zag. ANIL's gain came from *removing* the body's pointless inner loop; what remains is a head whose inner loop is still geometrically naive. So the next move is to fix *how* that surviving step moves: precondition the gradient so it points down the valley instead of across it.

The classical preconditioners are second-order — Newton's $\theta - \alpha H^{-1}\nabla\mathcal{L}$, natural gradient's $\theta - \alpha F^{-1}\nabla\mathcal{L}$ — and both turn the zig-zag into a straight shot. But I cannot lift them in, for the same few-shot pathology that has shaped every rung: Newton and natural gradient *compute* $H$ or $F$ from the current task's data, and the current task is one or five examples. A curvature estimated from five points is the curvature of *the support set*, and stepping as far as possible along it is stepping as far as possible to fit five points — exactly how I overfit, the same way Meta-SGD's support-tuned capacity meta-overfit at 5-shot. So I must *not* compute the preconditioner from the task's loss. But the meta-learning frame gives the escape it always has: the task *distribution*. If a shared initialization transfers (it does — MAML) and a shared per-coordinate rate transfers (it does — Meta-SGD), then a shared *preconditioner matrix* should transfer too. So I learn the preconditioner $M$ across tasks, jointly with $\theta$, by the same meta-objective — not "fit the support faster" but "transform the inner gradient so the *post-adaptation query* loss generalizes." $M$ becomes a meta-parameter trained for held-out performance, not estimated from data, which is what distinguishes this from naively dropping a Hessian into the head's inner loop.

I propose Meta-Curvature: a learned, Kronecker-factored gradient preconditioner. The affordability problem is where it has to be made concrete. A full $M$ is $\dim(\theta)\times\dim(\theta)$; even restricted to the head that is $(5\cdot1600)^2 \approx 6.4\times10^7$ numbers, and the harness budget caps the optimizer's learnable state at roughly the model size. The structural observation that breaks it open is that a layer's parameters are not a flat vector — they are a *tensor*. A conv weight is $G\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times d}$ (output channels, input channels, filter size $d=h\cdot w$), the head is the $d=1$ case $\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}}$, and the gradient has the same shape. The natural dependencies run along the three modes *separately*: among the $d$ elements within a filter, among the $C_{\text{in}}$ input channels, among the $C_{\text{out}}$ output channels. So instead of one giant matrix mixing all $C_{\text{out}}C_{\text{in}}d$ coordinates, factor the transform into three small matrices per layer — $M_f\in\mathbb{R}^{d\times d}$, $M_i\in\mathbb{R}^{C_{\text{in}}\times C_{\text{in}}}$, $M_o\in\mathbb{R}^{C_{\text{out}}\times C_{\text{out}}}$ — and apply each along its own axis with an $n$-mode product:
$$\text{MC}(G) = G \times_3 M_f \times_2 M_i \times_1 M_o.$$
Read each factor: $\times_3 M_f$ linearly recombines the elements within each filter (a kernel's spatial structure); $\times_2 M_i$ recombines across input channels; $\times_1 M_o$ recombines across output channels (each filter's gradient becomes a combination of all filters'). Composed, the three examine the dependencies of *all* the layer's gradient coordinates but with $d^2 + C_{\text{in}}^2 + C_{\text{out}}^2$ numbers instead of $(C_{\text{out}}C_{\text{in}}d)^2$. As a single never-materialized matrix this is the Kronecker product $M_o \otimes M_i \otimes M_f$ — the same structure K-FAC uses for the Fisher ($A\otimes G$); the decisive difference is that K-FAC computes its factors from training-loss statistics and inverts them, whereas I *learn* mine for generalization and never invert anything.

This is strictly richer than every rung, and the special cases make it precise. If every $M$ is the *identity*, then $\text{MC}(G)=G$ and the inner step is $\theta - \alpha\nabla\mathcal{L}$ — exactly MAML. If every $M$ is *diagonal*, the transform is a per-coordinate rescaling — exactly Meta-SGD. The full Kronecker-factored $M$ keeps the *off-diagonal* coordinate dependencies a diagonal throws away: it can combine the gradient on one input channel with another's, precisely the off-valley correction plain SGD (and Meta-SGD's diagonal) cannot make. So I initialize all factors to identity, starting the method as MAML, and let meta-training deform them away as it discovers which coordinate recombinations help the post-adaptation query loss — the same "start from behavior I trust" principle that set Meta-SGD's uniform rates. And I expect it to *generalize* rather than just fit faster, for a reason I can read off the meta-gradient. With inner step $\theta^\tau = \theta - \alpha M\nabla\mathcal{L}_{\text{train}}$, the gradient of the meta-loss with respect to a flattened $M$ is
$$-\alpha\,\nabla_\theta\mathcal{L}_{\text{val}}(\theta^\tau)\cdot\nabla_\theta\mathcal{L}_{\text{train}}(\theta)^\top,$$
an outer product that *resembles* the empirical Fisher but differs in three features that all favor generalization — it uses *separate* train and validation sets (the validation set never enters a within-task Fisher), it evaluates the validation gradient at the *post-adaptation* parameters, and it does not force $M$ positive-definite (I let it be indefinite, trading the descent guarantee for the freedom to transform the gradient however most helps held-out points).

The harness forces the implementation and the budget forces the one principled deviation. In `__init__` I attach, per model parameter, the three factor matrices sized to its modes, all identity-initialized: a 4D conv weight $[C_{\text{out}}, C_{\text{in}}, k_h, k_w]$ folds its spatial dims into $d = k_h k_w$; the 2D head is the $d=1$ case; 1D bias and BatchNorm parameters carry only the output-mode $M_o$. The inner step computes the support cross-entropy, takes `torch.autograd.grad` over the parameters with the graph kept, transforms each gradient by its factors via the three $n$-mode products (plain `einsum` matrix multiplies along each axis), and applies the differentiable update $p \leftarrow p - \alpha\,\text{MC}(g)$ by re-routing parameter tensors so the outer loop can backprop into *both* $\theta$ and the factors; `meta_parameters()` returns all the factors. This is the literal generalization of the earlier edits: MAML used a scalar in `maml_update`, Meta-SGD a per-parameter vector in `update_module`, and here a per-parameter *transform* with the same differentiable re-route. The budget deviation: the factors are tiny everywhere the input dimension is small — every conv layer has $C_{\text{in}}\le 64$, so $M_i$ is at most $64\times64$ and the full Kronecker curvature is essentially free. The single oversized factor is the head's input mode: $C_{\text{in}}=1600$ makes a dense $M_i$ a $1600\times1600 = 2.56$M-parameter matrix, twenty times the whole model — and that single large $C_{\text{in}}\cdot d$ block is exactly the cost the factorization exists to avoid. So for any mode larger than a threshold (`MAX_DENSE = 256`) I degrade its factor to a *diagonal* — a per-element rate over that mode, Meta-SGD's footprint — while keeping dense factors on every affordable mode. Concretely the head keeps a dense $5\times5$ output-mode $M_o$ (it can freely recombine the five class-logit gradients, the cheap and meaningful part for a five-way head) and a diagonal input-mode rate over the 1600 features. The total optimizer state then lands at $\approx$ 80K against a model of $\approx$ 121K, comfortably inside the $\approx$ 291K budget — full structured curvature where affordable, a diagonal fallback only on the one mode that cannot be. The factors still initialize to identity (diagonal factors to ones), so the method still begins as MAML.

The delta from ANIL is the axis ANIL left untouched. ANIL fixed *which* parameters adapt and left the surviving head's step as plain gradient descent; meta-curvature fixes *how* that step moves by preconditioning the gradient with a learned, Kronecker-factored, generalization-trained curvature — applied across all layers, not only the head, so the body's outer-loop-learned features are themselves shaped by a better-conditioned meta-objective. The bar is ANIL's real numbers and the falsifiable claims point at where the *direction* should matter most. At **1-shot** — the hardest, most ill-conditioned setting, where ten steps from one example per class are most wasted on a zig-zag — meta-curvature should clear ANIL's 0.4815; a preconditioned step that points down the valley should extract more from those ten steps than plain SGD, so I expect the largest gain there and would treat *failure* to beat 0.4815 as the central claim falsified. At **5-shot** I expect to at least match ANIL's 0.6366 (miniImageNet) and 0.7138 (CIFAR-FS) and likely edge past, since the structured curvature is a strict superset of the identity (MAML) and diagonal (Meta-SGD) steps and is meta-trained for generalization rather than support-fitting — the risk I will watch being meta-overfitting from the larger factor count, the failure that bit Meta-SGD at 5-shot, contained here by the identity initialization and the diagonal fallback on the oversized head mode. Having exhausted *where to start*, *how far per coordinate*, and *which parameters*, the trajectory ends on the construction that finally learns *which direction* — the structured, generalization-oriented preconditioner of the inner gradient that captures the coordinate dependencies every earlier rung's diagonal-or-identity step could not.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — finale: Meta-Curvature
class _MetaCurvatureFactor(nn.Module):
    """Per-parameter Kronecker-factored meta-curvature.

    Conv/linear weight G in R^{C_out x C_in x d} (d = kh*kw for conv, 1 for linear):
    MC(G) = G x3 M_f x2 M_i x1 M_o, with M_o, M_i, M_f identity-initialized (so the
    transform starts as the identity and the inner step starts as MAML). A mode
    larger than MAX_DENSE (the classifier head's C_in=1600) degrades its factor to
    a diagonal per-element rate to stay inside the parameter budget; 1D params
    (bias / BatchNorm) carry only the output-mode factor.
    """
    MAX_DENSE = 256  # dense factor only for modes up to this size

    def __init__(self, param: Tensor):
        super().__init__()
        # build factors on the parameter's device/dtype (the model is on CUDA;
        # the harness never .to()'s the optimizer, so factors must inherit here).
        kw = {"device": param.device, "dtype": param.dtype}
        if param.dim() >= 2:                          # conv [Co,Ci,kh,kw] or linear [Co,Ci]
            self.c_out, self.c_in = param.shape[0], param.shape[1]
            self.d = int(param[0, 0].numel()) if param.dim() > 2 else 1
            self.kind = "tensor"
            self.dense_o = self.c_out <= self.MAX_DENSE
            self.dense_i = self.c_in <= self.MAX_DENSE
            self.M_o = nn.Parameter(torch.eye(self.c_out, **kw) if self.dense_o
                                    else torch.ones(self.c_out, **kw))
            self.M_i = nn.Parameter(torch.eye(self.c_in, **kw) if self.dense_i
                                    else torch.ones(self.c_in, **kw))
            self.M_f = nn.Parameter(torch.eye(self.d, **kw))
        else:                                         # bias / BatchNorm: output mode only
            self.kind = "vector"
            self.c_out = param.shape[0]
            self.dense_o = self.c_out <= self.MAX_DENSE
            self.M_o = nn.Parameter(torch.eye(self.c_out, **kw) if self.dense_o
                                    else torch.ones(self.c_out, **kw))

    def transform(self, g: Tensor) -> Tensor:
        if self.kind == "vector":
            return (self.M_o @ g) if self.dense_o else (self.M_o * g)
        G = g.reshape(self.c_out, self.c_in, self.d)
        G = torch.einsum("oid,fd->oif", G, self.M_f)                       # x3 M_f (filter)
        G = (torch.einsum("oid,ji->ojd", G, self.M_i) if self.dense_i      # x2 M_i (input)
             else G * self.M_i.view(1, self.c_in, 1))
        G = (torch.einsum("oid,jo->jid", G, self.M_o) if self.dense_o      # x1 M_o (output)
             else G * self.M_o.view(self.c_out, 1, 1))
        return G.reshape(g.shape)


class InnerLoopOptimizer(nn.Module):
    """MAML inner loop with a learned Kronecker-factored gradient preconditioner
    (Meta-Curvature, Park & Oliva 2019). Identity init => MAML; diagonal => Meta-SGD."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr
        self.mc = nn.ModuleList([_MetaCurvatureFactor(p) for p in model.parameters()])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            updates = [-self.inner_lr * mc.transform(g)
                       for mc, g in zip(self.mc, grads)]
            model = self._apply_update(model, updates)
        return model

    def _apply_update(self, model: nn.Module, updates: List[Tensor]) -> nn.Module:
        # differentiable p <- p + u; re-route parameter tensors so the outer loop
        # backprops into BOTH the initialization theta AND the meta-curvature factors.
        for p, u in zip(model.parameters(), updates):
            p.update = u

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and getattr(p, "update", None) is not None:
                    module._parameters[name] = p + p.update
                    p.update = None
            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        return list(self.mc.parameters())
```
