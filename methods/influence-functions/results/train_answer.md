The best models we have in vision, speech, and language are large opaque function approximators, and when one of them makes a particular call on a particular test input the consequential question is *why*. Almost all of the interpretability work answers this by poking at the trained model — fitting a local linear surrogate around the test point, or perturbing the test pixels and watching the output wiggle — but that only ever explains the prediction in terms of the model and its input. The model, though, is not a god-given object: it is the minimizer of a loss over a training set. The prediction therefore flowed from the *data*, through the optimizer, into the parameters, out to the test point. The "why" I want to answer is the one nobody has a practical handle on: which training examples are responsible for this prediction, and how would the prediction change if a particular training point were absent or slightly different?

The conceptually cleanest answer is a counterfactual. Take a training point $z$, delete it, refit the model to get $\hat\theta_{-z}$, and report how the test loss moved, $L(z_{\text{test}}, \hat\theta_{-z}) - L(z_{\text{test}}, \hat\theta)$. That number *is* how much $z$ mattered for the prediction on $z_{\text{test}}$ — no approximation, no philosophy. Its only flaw is cost: with $n$ training points this leave-one-out (LOO) experiment demands $n$ full retrainings, which is hopeless for a deep net with millions of parameters. The cheap stand-in people reach for instead — rank training points by Euclidean closeness to the test point, i.e. by the inner product $x \cdot x_{\text{test}}$ when norms are equal — ignores the learning dynamics entirely; with non-negative features (pixels) it has $x \cdot x_{\text{test}} \ge 0$ always, so it brands *every* same-label point "helpful" and misses the same-label points that genuinely hurt the prediction. The classical robust-statistics diagnostics (Hampel's influence curve, Cook's distance and relatives) are exactly about quantifying a single observation's effect on an estimator, but they were built and validated on small, convex, twice-differentiable, fully-converged linear models with exact matrix inverses, and they never crossed into modern ML because they need the empirical-risk Hessian — a $p \times p$ matrix one cannot form, let alone invert, at $p \approx 10^6$ — and because real models are non-convex, stopped early, and partly non-differentiable. So the definition I want is right and the *computation* is the whole enemy.

I propose **influence functions**: keep the counterfactual definition of responsibility, but kill the retraining by computing the derivative of the test loss with respect to the *presence* of a training point. The trick that makes "presence" differentiable is to make it continuous. Instead of all-or-nothing membership, give $z$ a weight and define $\hat\theta_{\varepsilon,z} = \arg\min_\theta \frac1n\sum_i L(z_i,\theta) + \varepsilon\, L(z,\theta)$. At $\varepsilon=0$ this is the ordinary model $\hat\theta$; cranking $\varepsilon$ up counts $z$ extra, cranking it down counts $z$ less. The bookkeeping that makes deletion fall out is that removing $z$ entirely means dropping its weight from $1/n$ to $0$, i.e. taking $\varepsilon = -1/n$. So if I can get the derivative $d\hat\theta_{\varepsilon,z}/d\varepsilon$ at $\varepsilon=0$, I can linearly extrapolate the deletion effect as roughly $-(1/n)$ times that derivative — and I never retrain, I only evaluate a derivative at the model I already have.

To get that derivative in closed form I differentiate the stationarity condition of the perturbed minimizer rather than the $\arg\min$ itself. Write $R(\theta) = \frac1n\sum_i L(z_i,\theta)$ for the empirical risk, and (for now) assume it is twice-differentiable and strongly convex, so $\hat\theta$ is the unique minimizer and the Hessian $H_{\hat\theta} = \nabla^2 R(\hat\theta) = \frac1n\sum_i \nabla^2_\theta L(z_i,\hat\theta)$ is positive definite and invertible. The perturbed estimator satisfies $0 = \nabla R(\hat\theta_{\varepsilon,z}) + \varepsilon\,\nabla L(z,\hat\theta_{\varepsilon,z})$ for every $\varepsilon$ near $0$, with $\hat\theta_{\varepsilon,z}\to\hat\theta$ as $\varepsilon\to0$. Setting $\Delta_\varepsilon = \hat\theta_{\varepsilon,z}-\hat\theta$ and Taylor-expanding the optimality condition about $\hat\theta$, dropping $o(\|\Delta_\varepsilon\|)$,
$$0 \approx \big[\nabla R(\hat\theta) + \varepsilon\nabla L(z,\hat\theta)\big] + \big[\nabla^2 R(\hat\theta) + \varepsilon\nabla^2 L(z,\hat\theta)\big]\Delta_\varepsilon.$$
Two simplifications collapse this: $\hat\theta$ minimizes $R$, so $\nabla R(\hat\theta)=0$ kills the first bracket down to $\varepsilon\nabla L(z,\hat\theta)$; and keeping only the leading order in $\varepsilon$ lets me drop the $\varepsilon\nabla^2 L$ term inside the inverse (it is order $\varepsilon^2$ once multiplied by the already-order-$\varepsilon$ right-hand side). What remains, after dividing by $\varepsilon$ and taking $\varepsilon\to0$, is the influence of upweighting $z$ on the parameters,
$$\mathcal I_{\text{up,params}}(z) = \left.\frac{d\hat\theta_{\varepsilon,z}}{d\varepsilon}\right|_{\varepsilon=0} = -\,H_{\hat\theta}^{-1}\,\nabla_\theta L(z,\hat\theta).$$
The reading is one Newton step: I formed the quadratic model of the risk around $\hat\theta$ and asked where the minimizer moves when I add a sliver of $L(z,\cdot)$; a quadratic's minimizer shifts by $-(\text{curvature})^{-1}\cdot(\text{gradient of the perturbation})$, so $\mathcal I_{\text{up,params}}$ is exactly that — upweighting $z$ pulls the parameters toward reducing $L(z,\cdot)$, and $H^{-1}$ translates "pull" into "displacement," accounting for how stiff the landscape is in each direction. This is the classic M-estimation influence-function object; what is new is the use I put it to.

Because I asked about a *test prediction*, not parameters, I chain-rule through. The influence of upweighting $z$ on the loss at $z_{\text{test}}$ is
$$\mathcal I_{\text{up,loss}}(z,z_{\text{test}}) = \left.\frac{d L(z_{\text{test}},\hat\theta_{\varepsilon,z})}{d\varepsilon}\right|_{\varepsilon=0} = -\,\nabla_\theta L(z_{\text{test}},\hat\theta)^\top\, H_{\hat\theta}^{-1}\, \nabla_\theta L(z,\hat\theta),$$
a symmetric bilinear form in the train gradient and the test gradient glued by the inverse Hessian: read right to left, $\nabla L(z,\hat\theta)$ is how upweighting $z$ pushes the parameters, $H^{-1}$ turns the push into a displacement, and $\nabla L(z_{\text{test}},\hat\theta)^\top$ reads off the resulting change in test loss. The estimated effect of *removing* $z$ on the test loss is $\approx -(1/n)\,\mathcal I_{\text{up,loss}}(z,z_{\text{test}})$.

It is worth seeing concretely why this beats the bare inner product. Instantiate it for logistic regression with $p(y\mid x)=\sigma(y\theta^\top x)$, $y\in\{-1,1\}$, $\sigma(t)=1/(1+e^{-t})$. Then $\nabla_\theta L = -\sigma(-y\theta^\top x)\,y\,x$, where the factor $\sigma(-y\theta^\top x)$ is exactly the model's probability of being *wrong* on $z$; the Hessian is the data-weighted covariance $H=\frac1n\sum_i \sigma(\theta^\top x_i)\sigma(-\theta^\top x_i)\,x_i x_i^\top$; and
$$\mathcal I_{\text{up,loss}}(z,z_{\text{test}}) = -\,y_{\text{test}}\,y\,\cdot\,\sigma(-y_{\text{test}}\theta^\top x_{\text{test}})\,\cdot\,\sigma(-y\theta^\top x)\,\cdot\,x_{\text{test}}^\top H_{\hat\theta}^{-1} x.$$
Two things appear that $x\cdot x_{\text{test}}$ lacks. The $\sigma(-y\theta^\top x)$ factor up-weights training points the model gets wrong — high-loss outliers, which we know dominate an estimator's parameters. And $H^{-1}$ sits between the inputs instead of the identity: $H$ is the covariance of the other points' gradients, so $H^{-1}$ measures *resistance*. If $z$'s gradient points where many other training points also have curvature, moving the parameters that way is expensive — it would spike everyone else's loss — so $z$'s influence is damped; if it points in a direction of little variation, the parameters swing there cheaply and $z$'s influence is amplified. Crucially $H^{-1}$ can make even a same-label point's influence *negative*, which is precisely what catches the same-label points that drag the boundary the wrong way; the inner product, blind to all of this, treats every direction as equally stiff and so makes systematic sign errors. The inverse Hessian and the loss factor are not decoration — they are the difference between a heuristic and the actual effect of training.

The whole obstacle is $H^{-1}$ at scale, and the resolution is to notice that the formula never needs the *matrix*, only its action on a vector. Define, once per test point, $s_{\text{test}} = H_{\hat\theta}^{-1}\nabla_\theta L(z_{\text{test}},\hat\theta)$; then for every one of the $n$ training points $\mathcal I_{\text{up,loss}}(z,z_{\text{test}}) = -\,s_{\text{test}}\cdot \nabla_\theta L(z,\hat\theta)$ is a cheap dot product, and the $n$ inversions I dreaded collapse into one. Computing $s_{\text{test}}$ itself rests on two ingredients. The first is the fast Hessian-vector product: $H$ appears in the gradient's expansion, $\nabla(\theta+rv)=\nabla(\theta)+r\,Hv+O(r^2)$, and rather than finite-difference (which bleeds precision as $r\to0$) I take the exact limit, $Hv = \nabla_\theta(v\cdot\nabla_\theta L)$ — form the gradient $g=\nabla_\theta L$ in one reverse pass, dot it with a constant $v$, and differentiate that scalar again in a second reverse pass. The result is exactly $Hv$ in about two gradient evaluations, $O(p)$, with no explicit matrix and no finite-difference noise (this is Pearlmutter's trick). The second ingredient is to solve $H s = \nabla L(z_{\text{test}})$ using only those matrix-vector products, and there are two routes. Because $H\succ0$, solving $Hs=v$ equals minimizing the strictly convex quadratic $\frac12 t^\top H t - v^\top t$, which conjugate gradients (Newton-CG) does using only $Ht$. Alternatively, the stochastic LiSSA route exploits the Neumann series $H^{-1}=\sum_{i\ge0}(I-H)^i$, valid when $0\prec H\preceq I$: the recursion $\tilde H_0^{-1}v=v$, $\tilde H_j^{-1}v = v + (I-\nabla^2_\theta L(z_{s_j},\hat\theta))\,\tilde H_{j-1}^{-1}v$ with each $z_{s_j}$ drawn uniformly is unbiased because a single point's Hessian is an unbiased estimate of $H$ and the recursion is linear in $H$; each step costs one single-point HVP, no full pass. I enforce $\nabla^2 L\preceq I$ by scaling the loss down (which leaves $\arg\min$ untouched and is undone at the end), add a damping $\lambda I$ to keep the curvature PD, run to depth $t$, and average $r$ independent runs to cut variance. The total cost to score all training points for one test point is $O(np + rtp)$ — linear in both $n$ and $p$, with $rt=O(n)$ enough in practice — which is the regime I needed.

A finer attribution than "delete the whole point" comes from moving mass onto a perturbed copy. To ask how the prediction would change if a training input were modified, $z=(x,y)\mapsto(x+\delta,y)$, I take $\hat\theta_{\varepsilon,z_\delta,-z}=\arg\min \frac1n\sum_i L(z_i,\theta)+\varepsilon L(z_\delta,\theta)-\varepsilon L(z,\theta)$; the same stationarity-Taylor calculation gives $d\hat\theta/d\varepsilon|_0 = -H^{-1}(\nabla_\theta L(z_\delta,\hat\theta)-\nabla_\theta L(z,\hat\theta))$, with the replacement effect $\hat\theta_{z_\delta,-z}-\hat\theta\approx \frac1n(\mathcal I_{\text{up,params}}(z_\delta)-\mathcal I_{\text{up,params}}(z))$ — and this never assumed $\delta$ small or even continuous, so it covers discrete data and label flips too. For small continuous $\delta$ I differentiate with respect to the perturbation direction, using $\nabla_\theta L(z_\delta)-\nabla_\theta L(z)\approx[\nabla_x\nabla_\theta L(z)]\delta$, to get the per-feature influence
$$\mathcal I_{\text{pert,loss}}(z,z_{\text{test}}) = -\,\nabla_\theta L(z_{\text{test}},\hat\theta)^\top H_{\hat\theta}^{-1}\,\nabla_x\nabla_\theta L(z,\hat\theta) \in \mathbb R^d,$$
so $\mathcal I_{\text{pert,loss}}\cdot\delta$ is the first-order effect on the test loss and stepping $\delta$ along $\mathcal I_{\text{pert,loss}}^\top$ maximally raises it. Computationally this reuses the same $s_{\text{test}}$ and is again an HVP-flavored object. It hands me a training-set analogue of adversarial examples: pick a target $z_{\text{test}}$ and iterate a training image $\tilde z_i \leftarrow \Pi(\tilde z_i + \alpha\,\mathrm{sign}(\mathcal I_{\text{pert,loss}}(\tilde z_i, z_{\text{test}})))$, projecting back onto images that share the same 8-bit representation so the change is invisible after quantization, retraining after each step — and the magnitude of $\mathcal I_{\text{pert,loss}}$ is itself a measure of how vulnerable the model is to tampering with its training data.

The derivation assumed an exact minimizer of a smooth, strongly convex risk, which is false for the models I care about, so I must show the estimate still tracks real LOO retraining under two violations. For non-convex, non-converged $\tilde\theta$ — where the risk gradient $g=\frac1n\sum\nabla L(z_i,\tilde\theta)$ is nonzero and $H_{\tilde\theta}$ can be indefinite, making $H^{-1}$ meaningless — I form a *convex* quadratic model with $H_{\tilde\theta}+\lambda I$, adding the damping $\lambda I$ exactly when negative eigenvalues appear; this is precisely L2 regularization, so influence is computed under a slightly regularized model, and the same $\lambda I$ helps the LiSSA series converge. That this still means something follows from expanding a Newton step from $\tilde\theta$ after upweighting $z$: it decomposes as $-H_{\tilde\theta}^{-1}g - \varepsilon H_{\tilde\theta}^{-1}\nabla L(z,\tilde\theta)$, a $z$-independent drift (common to all points, so it cancels in any ranking or comparison) plus $\varepsilon\cdot\mathcal I_{\text{up,params}}(z)$, so the *relative* influence is still the right quantity. For a non-differentiable loss — a hinge $\mathrm{Hinge}(s)=\max(0,1-s)$, whose second derivative is zero everywhere it exists — the failure is structural rather than random: influence rests on a quadratic model, but a piecewise-linear loss has identically zero curvature, so the Hessian carries no information about how close a support vector is to the margin, which is exactly what governs its influence, and the estimate overestimates support vectors. The cure follows from naming the failure: swap in, for the influence computation only, the smoothed surrogate $\mathrm{SmoothHinge}(s,t)=t\log(1+\exp((1-s)/t))\to\mathrm{Hinge}(s)$ as $t\to0$, a softplus hinge with genuine $t$-controlled curvature that is largest exactly where points sit near the margin. The same move generalizes — a ReLU is a one-sided hinge — so wherever a model has a non-differentiable piece, a smoothed stand-in restores the curvature signal and the machinery keeps working.

So it all hangs together as one idea: responsibility is the derivative of the test loss with respect to upweighting a training point; that derivative is $-\nabla L(z_{\text{test}})^\top H^{-1}\nabla L(z)$; and the only obstacle, $H^{-1}$ at scale, dissolves once I see I need only its action on a vector — compute $Hv$ exactly by differentiating $v\cdot\nabla L$, fold the $n$ inversions into one precomputed $s_{\text{test}}=H^{-1}\nabla L(z_{\text{test}})$, solve that single system with CG or with the scaled, damped stochastic Neumann recursion, and reduce influence to a dot product per training point. The implementation is pure autodiff: the user specifies only the loss; gradients and HVPs come for free.

```python
import torch

def grad_params(scalar_loss, params, create_graph=False):
    """Gradient of a scalar loss w.r.t. params (one reverse pass)."""
    return list(torch.autograd.grad(scalar_loss, params, create_graph=create_graph))

def hvp(loss, params, v):
    """Hessian-vector product H v, exact and O(p): H v = grad( v . grad L ). Never forms H."""
    g = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((gi * vi.detach()).sum() for gi, vi in zip(g, v))
    return list(torch.autograd.grad(dot, params, retain_graph=True))

def _hvp_over_data(model, train_data, loss_fn, params, v, batch_size=None):
    """Empirical-risk HVP H v = (1/n) sum_i grad^2 L(z_i) v, accumulated over the dataset."""
    total, nb = [torch.zeros_like(p) for p in params], 0
    for xb, yb in train_data.batches(batch_size):
        part = hvp(loss_fn(model, (xb, yb)), params, v)
        total = [t + p for t, p in zip(total, part)]; nb += 1
    return [t / nb for t in total]

def inverse_hvp_cg(model, train_data, loss_fn, params, v, damping=0.0):
    """s = H^{-1} v via Newton-CG on  1/2 t^T H t - v^T t  (H > 0). Uses only H t."""
    import numpy as np
    from scipy.optimize import fmin_ncg
    shapes = [p.shape for p in params]; sizes = [p.numel() for p in params]
    def to_list(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(torch.tensor(x[i:i+n], dtype=params[0].dtype).reshape(s)); i += n
        return out
    def flat(ts): return np.concatenate([t.detach().cpu().numpy().ravel() for t in ts])
    def Hx(x):
        hv = _hvp_over_data(model, train_data, loss_fn, params, to_list(x))
        return flat([h + damping * xi for h, xi in zip(hv, to_list(x))])
    f      = lambda x: 0.5 * np.dot(Hx(x), x) - np.dot(flat(v), x)
    fprime = lambda x: Hx(x) - flat(v)
    res = fmin_ncg(f=f, x0=flat(v), fprime=fprime,
                   fhess_p=lambda x, p: Hx(p), avextol=1e-8, maxiter=100)
    return to_list(res)

def inverse_hvp_lissa(model, train_data, loss_fn, params, v,
                      scale=10.0, damping=0.0, num_samples=1, recursion_depth=5000,
                      batch_size=1):
    """s = H^{-1} v via the stochastic Neumann recursion (LiSSA).
       e_0 = v;  e_j = v + (I - H_sample/scale - damping I) e_{j-1};  return e_t / scale."""
    result = None
    for _ in range(num_samples):
        cur = [vi.clone() for vi in v]
        for _ in range(recursion_depth):
            xb, yb = train_data.sample_batch(batch_size)      # unbiased H sample
            Hcur = hvp(loss_fn(model, (xb, yb)), params, cur)
            cur = [vi + (1 - damping) * ci - hi / scale
                   for vi, ci, hi in zip(v, cur, Hcur)]
        contrib = [ci / scale for ci in cur]
        result = contrib if result is None else [r + c for r, c in zip(result, contrib)]
    return [r / num_samples for r in result]

def influence_on_test_loss(model, train_data, test_point, loss_fn, params,
                           approx="cg", **kw):
    """I_up,loss(z, z_test) = - s_test . grad L(z) for every training z. Removal effect ~ -(1/n) I."""
    v = grad_params(loss_fn(model, test_point), params)        # grad L(z_test)
    s_test = (inverse_hvp_cg if approx == "cg" else inverse_hvp_lissa)(
        model, train_data, loss_fn, params, v, **kw)
    out = []
    for z in train_data.each_example():
        gz = grad_params(loss_fn(model, z), params)
        out.append(-sum((si * gi).sum() for si, gi in zip(s_test, gz)).item())
    return out

def grad_influence_wrt_input(model, x, y, s_test, loss_fn, params):
    """Returns d/dx ( s_test . grad_theta L(z) ) = s_test^T grad_x grad_theta L(z) = - I_pert,loss(z, z_test).
       To raise the test loss, step x along + I_pert,loss^T, i.e. x <- Pi(x - a*sign(returned))."""
    x = x.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(model, (x, y)), params, create_graph=True)
    scalar = sum((gi * si.detach()).sum() for gi, si in zip(g, s_test))
    return torch.autograd.grad(scalar, x)[0]
```
