I am filling exactly one function — `run_attack` — in a frozen white-box harness, and the scaffold default returns the clean images, which by construction flips nothing: its ASR is zero. So the first rung's job is clean: produce the cheapest attack that beats that floor, the primitive every later rung will be measured against and built from. The only generator I already trust is the existence-proof method, minimizing $c\,\lVert r\rVert_2 + \mathrm{loss}_f(x+r, l)$ over the box with box-constrained L-BFGS and a line search over $c$ until the label flips. It works, but it runs an iterative constrained optimizer *with a line search per single image* — across 1000 samples and six scenarios that is dead on arrival, and it hands me no cheap closed-form move to build a ladder on. The interface gives me only a forward pass and one input-gradient via `torch.autograd.grad`; I want an attack that costs about exactly that.

So I ask the prior question, because I suspect the cheap method and the *reason these examples exist* are the same insight. The reflexive story blames extreme nonlinearity — adversarial examples as needles hidden in high-curvature pockets — but that fights the evidence: idiosyncratic per-model pockets would not line up across different architectures, yet a perturbation crafted for one model fools another and they even agree on the wrong class, and a plain linear softmax on raw pixels is vulnerable too, where there is no nonlinearity to blame. So I take the opposite hypothesis and work out the simplest model exactly. A linear unit computes $w^\top x$; feed it $x + \eta$ and the activation moves by $w^\top\eta$. Maximizing that swing under $\lVert\eta\rVert_\infty \le \varepsilon$ decouples coordinate by coordinate — each $w_i\eta_i$ is maximized by pushing $\eta_i$ to the corner of its allowed interval in the direction of $w_i$ — giving $\eta = \varepsilon\,\mathrm{sign}(w)$ and an output swing $w^\top\eta = \varepsilon\lVert w\rVert_1$. Stare at the scaling: with $n$ weights of average magnitude $m$, the swing is $\varepsilon\, m\, n$ — the perturbation size $\lVert\eta\rVert_\infty = \varepsilon$ does not grow with $n$, but the output swing grows *linearly* in dimension. Adversarial examples need *dimension and linearity*, not curvature; that alone explains the vulnerable linear softmax. And the modern building blocks — ReLU, BatchNorm, convolution — are deliberately near-linear because that is what makes nets trainable, so my working hypothesis is that ResNet20, VGG11-BN and MobileNetV2 are *too linear to resist a linear perturbation*.

I propose the **fast gradient sign method (FGSM)**: a single closed-form signed step along the input-gradient of the training loss. The thing I attack is the scalar loss $J(\theta, x, y)$ the model was trained with, and raising it pushes the logits away from the true label. I have no global linear form for $J$, but I have its first-order behavior around the current input, so I Taylor-expand:
$$J(\theta, x + \eta, y) \approx J(\theta, x, y) + \eta^\top \nabla_x J(\theta, x, y),$$
and maximize the part I control, $\eta^\top g$ with $g = \nabla_x J$, under the per-pixel budget. This is the linear-model problem again with $g$ playing the role of $w$. The clean general statement makes the norm's role explicit: by Hölder's inequality with conjugate exponents $\infty$ and $1$,
$$\eta^\top g \le \lVert\eta\rVert_\infty\,\lVert g\rVert_1 \le \varepsilon\,\lVert g\rVert_1,$$
with equality when every nonzero-gradient coordinate has the same sign as $g_i$ and is pushed to the boundary $\lvert\eta_i\rvert = \varepsilon$. So a canonical maximizer is forced:
$$\eta = \varepsilon\,\mathrm{sign}\!\big(\nabla_x J(\theta, x, y)\big),$$
achieving $\eta^\top g = \varepsilon\lVert g\rVert_1$ — one forward pass, one backward pass, an elementwise sign, scale by $\varepsilon$.

The load-bearing choice is *the sign, not the gradient itself*, and the geometry says why. The feasible set $\{\eta : \lVert\eta\rVert_\infty \le \varepsilon\}$ is an axis-aligned cube and the linearized objective is linear, so the maximum sits at a *corner* — a vector all of whose coordinates are $\pm\varepsilon$, i.e. $\varepsilon\,\mathrm{sign}(g)$. If the budget had instead been $L_2$ the feasible set would be a ball, the maximizer $\varepsilon\, g/\lVert g\rVert_2$, and large-magnitude coordinates would soak up the budget; under $L_\infty$ each coordinate has its own independent allowance $\varepsilon$, so the right move spends every one fully in its gradient's direction and only the *sign* of each partial derivative matters, never its magnitude. And $L_\infty$ is the honest budget here, not merely the convenient one: an 8-bit image discards everything below $1/255$ of its range, so the meaningful notion of imperceptible is per-coordinate — no single feature moves past the precision floor — which is exactly a max-norm constraint, and exactly the constraint the harness scores me against.

The one real worry is that this is a first-order method: I linearized $J$ and then took $\eta$ as large as the *whole* budget allows, a full-$\varepsilon$ jump in every coordinate at once, so $x + \varepsilon\,\mathrm{sign}(g)$ maximizes the *approximation*, not the true curved loss inside the box. Two things reassure me. Where the model genuinely is linear in $x$ — logistic regression — there is no approximation and the sign step is the exact worst case in the box. And my whole hypothesis is that the nonlinear models are too linear to resist, so if the step reliably raises the loss in practice, that success is itself evidence for the linearity explanation. But the linearization will waste budget wherever the loss bends fast away from its tangent plane — I expect the single step to flip a large majority on the bottlenecked ResNet and depthwise MobileNetV2, and to leave the most headroom on the wider, shallower VGG11-BN whose low-resolution feature maps absorb small per-pixel perturbations more robustly. That architecture-dependent shortfall is the gap the next rung closes by iterating. So FGSM is the right *primitive* but a deliberately weak *adversary*: the floor, not the finish.

Beyond the bare math the harness forces a little scaffold discipline. It rejects any sample that leaves $[0,1]$ or violates the budget, counting it an attack failure, so I project the *perturbation* to $[-\varepsilon, \varepsilon]$ first, then add it and clamp the image to $[0,1]$. With a single signed step of size exactly $\varepsilon$ the perturbation already lies in the box, so this delta-clamp is a no-op here — but it makes the budget projection explicit and identical in form to the iterative rungs that follow, where it stops being a no-op. The gradient bookkeeping is the rest: detach and clone the inputs with `requires_grad_`, build the loss with `F.cross_entropy`, take exactly one `torch.autograd.grad`, and do the pixel arithmetic under `torch.no_grad`. `n_classes` and `device` go unused — the gradient already knows the class count through the loss — so I bind them to a throwaway to keep the signature honest.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    _ = (device, n_classes)
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x)[0]

    with torch.no_grad():
        x_adv = x + eps * grad.sign()
        delta = torch.clamp(x_adv - images, min=-eps, max=eps)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)

    return x_adv.detach()
```
