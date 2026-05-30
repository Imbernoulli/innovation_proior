OK, let me think this through from scratch. I'm training a big overparameterized network — enough capacity that I can drive the training loss `L_S(w)` essentially to zero. And here's the thing that keeps nagging me: when I do that, the training loss value tells me almost nothing about test performance. I can take two runs, two different random seeds, both land at `L_S(w) ≈ 0`, and one generalizes noticeably better than the other. The loss surface is non-convex with a huge number of minima, lots of them at basically the same (near-zero) training loss, and they spread out across a wide range of test errors. So "minimize the training loss" is underdetermined in exactly the way I care about: it picks *a* minimum, not the *right* minimum.

So what distinguishes the good minima from the bad ones? It's not the loss value at the point — those are equal. It has to be something about the *neighborhood* of the point. And there's an old intuition here I keep coming back to: flat minima generalize, sharp minima don't. Hochreiter and Schmidhuber argued this back in the 90s through a description-length lens — a minimum where the loss stays low over a big, low-precision box of weights needs fewer bits to pin down, and by an Occam/MDL argument that should generalize. Keskar and collaborators saw the same thing empirically for large-batch training: large batches converge to sharp minima — points where the Hessian has big eigenvalues and the loss shoots up if you nudge the weights — and those generalize worse than the flat minima small-batch noise tends to find. And Jiang et al. did the careful large-scale thing: they took forty different candidate complexity measures and asked which one actually tracks the generalization gap across many trained models, and the sharpness-based measures came out on top, ahead of norm- and margin-based ones.

So the empirical signal is pretty strong: sharpness around `w` is the thing correlated with bad generalization. But all of this is *diagnostic*. Hochreiter-Schmidhuber gave a flat-minimum search, but it leans on second-derivative quantities that are miserable to compute and differentiate through at the scale of a real network. Keskar's sharpness is a *measurement*, computed after the fact. Jiang's study *measures* trained models. Nobody here is handing me a cheap thing I can stick into SGD and actually train with. That's the gap.

Let me try to state what I actually want, mathematically. The standard thing minimizes the loss *value* at a point. I want to instead make sure the *whole neighborhood* of `w` has low loss — equivalently, low loss *and* low curvature. What if I don't minimize `L_S(w)` but the worst case of the loss in a small ball around `w`? That is,

  minimize over `w` of  `max_{‖ε‖ ≤ ρ} L_S(w + ε)`.

If I can make the *highest* the loss gets anywhere within radius `ρ` of `w` be small, then I've found a `w` whose entire neighborhood is low — a flat basin, by construction. A sharp minimum would have a huge `max` inside its ball even if `L_S(w)` itself is zero, so it gets penalized automatically. This feels right. But "feels right" isn't enough; let me see if I can ground it in something quantitative instead of just the flatness folklore, because I want to know I'm optimizing a real upper bound on test loss, not a vibe.

PAC-Bayes is the natural place to look. The McAllester bound controls a *stochastic* predictor: if I draw my weights from a posterior `Q`, then with high probability over the training set,

  `E_{w∼Q}[L_D(w)] ≤ E_{w∼Q}[L_S(w)] + sqrt( (KL(Q‖P) + log(n/δ)) / (2(n−1)) )`,

for any data-independent prior `P`. Now follow Dziugaite-Roy and Neyshabur: take `P = N(0, σ_P² I)` centered at the origin and `Q = N(w, σ_Q² I)` centered at my trained weights, with `σ_Q = ρ` say. Then `E_{w∼Q}[L_S(w)] = E_{ε∼N(0,ρ²I)}[L_S(w+ε)]` — the loss averaged over Gaussian weight perturbations. And the KL between two isotropic Gaussians is `(1/2)[ (k σ_Q² + ‖μ_P − μ_Q‖²)/σ_P² − k + k log(σ_P²/σ_Q²) ]`, which with `μ_P = 0`, `μ_Q = w` is governed by `‖w‖²/σ²`. So the bound has the shape

  `E_{ε∼N(0,ρ)}[L_D(w+ε)] ≤ E_{ε∼N(0,ρ)}[L_S(w+ε)] + sqrt( (k log(1 + ‖w‖²/(kσ²)) + …) / (n−1) )`.

Let me make sure I can actually get from the *expected* perturbed loss to the *worst-case* perturbed loss I wrote down, because those aren't the same and I don't want to hand-wave the gap. The bound has `E_ε[L_S(w+ε)]` on the right with `ε` Gaussian, `σ_Q = σ`. I want to replace that expectation by the max over an `ℓ₂`-ball, since the max is what measures sharpness and what I can later attack with a first-order solve. For a Gaussian `ε` with per-coordinate variance `σ²` in `k` dimensions, `‖ε‖₂²` is `σ²` times a chi-square with `k` degrees of freedom, which concentrates: by the Laurent-Massart tail, `P(‖ε‖₂² − kσ² ≥ 2σ²√(kt) + 2σ²t) ≤ e^{-t}`. Set `t = (1/2)log n` so the failure probability is `1/√n`. Then with probability `1 − 1/√n`,

  `‖ε‖₂² ≤ σ²( k + 2√(k · (1/2)log n) + 2·(1/2)log n ) = σ²( k + √(2k log n) + log n ) ≤ σ² k (1 + √(log(n)/k))²`,

since `√(2k log n) ≤ 2√(k log n)` and `k(1+√(log(n)/k))² = k + 2√(k log n) + log n`.

So if I *choose* `σ` such that `σ² k (1 + √(log(n)/k))² = ρ²`, then with high probability the Gaussian draw lands inside the ball of radius `ρ`. On that event, `E_ε[L_S(w+ε)] ≤ max_{‖ε‖₂≤ρ} L_S(w+ε)` (the Gaussian mass is mostly inside the ball, and the rare outside-event I pay for with a tiny `1/√n` additive term, which folds into the constants). And on the left, the theorem's working assumption is that adding Gaussian noise doesn't *decrease* the test loss at the final solution, `L_D(w) ≤ E_ε[L_D(w+ε)]` — reasonable at a real minimum, since you're sitting near the bottom and perturbing can only push you up. Plugging the chosen `σ` (which makes `σ² = ρ²/(k(1+√(log n /k))²)`, i.e. `1/(kσ²) = (1+√(log n /k))²/ρ²`) into the `log(1+‖w‖²/(kσ²))` complexity term, I land at

  `L_D(w) ≤ max_{‖ε‖₂ ≤ ρ} L_S(w + ε) + sqrt( ( k log(1 + (‖w‖²/ρ²)(1+√(log(n)/k))²) + 4 log(n/δ) + O(1) ) / (n − 1) ).`

That's exactly the object I want: the population loss is upper-bounded by the *worst-case loss in a ρ-ball* plus a term that's increasing in `‖w‖²/ρ²`. Write the complexity term abstractly as `h(‖w‖²/ρ²)`, `h` strictly increasing. So minimizing the right-hand side over `w` means: minimize `max_{‖ε‖₂≤ρ} L_S(w+ε)` and keep `‖w‖²` controlled. Now I'm not just chasing flatness folklore — I'm minimizing a genuine high-probability upper bound on the test loss.

Let me pull the sharpness out explicitly so I can see what I'm really doing. Trivially,

  `max_{‖ε‖≤ρ} L_S(w+ε) = [ max_{‖ε‖≤ρ} L_S(w+ε) − L_S(w) ] + L_S(w)`.

The bracket is precisely a sharpness measure — how much the loss can rise as I move `ρ` away from `w` — and it's the same shape Keskar used. So the right-hand side of my bound is (sharpness) + (training loss) + (a function of `‖w‖²/ρ²`). The bound is literally telling me: simultaneously make the loss low *and* the neighborhood flat *and* the weights not-too-big. The last term `h(‖w‖²/ρ²)` is ugly and its exact form is an artifact of which inequalities I used in the proof; I'm not going to optimize that exact function. Instead I'll substitute the standard stand-in, a plain `λ‖w‖²` weight-decay term, which is monotone in `‖w‖²` just like `h` and is what every training pipeline already has. So the objective becomes

  `min_w   L_S^{SAM}(w) + λ‖w‖²`,  where  `L_S^{SAM}(w) ≜ max_{‖ε‖_p ≤ ρ} L_S(w + ε)`.

I wrote `ℓ_p` instead of `ℓ₂` — the bound came out for `p = 2`, but nothing about the *procedure* forces `p = 2`, so let me keep `p ∈ [1,∞]` general for now and pick later. `ρ ≥ 0` is the neighborhood radius, the one real hyperparameter.

Now the hard part. How do I minimize this by gradient descent? I need `∇_w L_S^{SAM}(w)`, and `L_S^{SAM}` has an inner `max` over `ε` baked into it. The inner max is itself an optimization over a `ρ`-ball, and the loss is non-convex in `ε`, so I can't solve it exactly. But I don't need to — `ρ` is small, so I only need the behavior of `L_S` in a tiny neighborhood of `w`. Linearize. First-order Taylor of `L_S(w+ε)` in `ε` around `0`:

  `L_S(w + ε) ≈ L_S(w) + ε^T ∇_w L_S(w)`.

The constant `L_S(w)` doesn't depend on `ε`, so the inner maximizer is

  `ε̂(w) = argmax_{‖ε‖_p ≤ ρ}  ε^T ∇_w L_S(w)`.

This is now a *linear* objective maximized over a `ρ`-ball in the `ℓ_p` norm — a classical dual-norm problem, exactly the same shape that shows up when you build a first-order adversarial perturbation in input space. Let me actually solve it rather than quote it; the algebra is where it's easy to slip a norm. Write `g = ∇_w L_S(w)`. By Hölder's inequality, for conjugate exponents `1/p + 1/q = 1`,

  `ε^T g ≤ ‖ε‖_p ‖g‖_q ≤ ρ ‖g‖_q`,

so the maximum possible value is `ρ‖g‖_q`. When is Hölder tight? Equality needs `|ε_i|^p ∝ |g_i|^q` componentwise, and `ε^T g = Σ ε_i g_i` is maximized (not just `|·|`) when each `ε_i` carries the *sign* of `g_i`. So set

  `ε̂_i = c · sign(g_i) · |g_i|^{q-1}`

for some constant `c > 0` fixed by the constraint `‖ε̂‖_p = ρ`. Check: `‖ε̂‖_p^p = c^p Σ_i |g_i|^{(q-1)p}`. Here's the spot to be careful — `(q-1)p`. From `1/p + 1/q = 1` I get `p + q = pq`, hence `q = pq − p = p(q − 1)`, so `(q-1)p = q`. So `‖ε̂‖_p^p = c^p Σ_i |g_i|^q = c^p ‖g‖_q^q`. Setting that equal to `ρ^p` gives `c = ρ / (‖g‖_q^q)^{1/p}`. So

  `ε̂(w) = ρ · sign(∇_w L_S(w)) · |∇_w L_S(w)|^{q-1} / ( ‖∇_w L_S(w)‖_q^q )^{1/p}`,  with `1/p + 1/q = 1`,

where `sign` and `|·|^{q-1}` are elementwise. Let me sanity-check the two endpoints I actually care about. For `p = 2`: `q = 2`, so `ε̂ = ρ · sign(g) · |g|^1 / (‖g‖_2^2)^{1/2} = ρ · g / ‖g‖_2`. Clean — it's just the gradient rescaled to have norm exactly `ρ`. (One has to be careful writing this: the denominator is `‖g‖₂`, not `‖g‖₂²`; the `^{1/p} = ^{1/2}` is what kills the square.) For `p = ∞`: `q = 1`, so `ε̂ = ρ · sign(g) · |g|^{0} / (‖g‖_1^1)^{0} = ρ · sign(g)` — the pure sign vector, the FGSM-style step. And `p = 1` would put `q = ∞` and concentrate the whole perturbation onto the single largest-gradient coordinate, which is a degenerate way to probe sharpness, so I'll set it aside.

Now substitute `ε̂(w)` back and differentiate the objective. The SAM loss is `L_S^{SAM}(w) = L_S(w + ε̂(w))` (using the maximizer for the inner max), so by the chain rule, treating `w + ε̂(w)` as the argument,

  `∇_w L_S^{SAM}(w) ≈ ∇_w [ L_S(w + ε̂(w)) ] = (d(w + ε̂(w))/dw)^T · ∇L_S |_{w + ε̂(w)}`
            `= ∇L_S|_{w+ε̂(w)} + (dε̂(w)/dw)^T · ∇L_S|_{w+ε̂(w)}`.

The first term is just: evaluate the ordinary gradient of `L_S`, but at the *perturbed* point `w + ε̂(w)` instead of at `w`. The second term is the annoying one: `dε̂(w)/dw`. Since `ε̂` is built from `∇_w L_S(w) = g(w)`, its derivative in `w` pulls down a `dg/dw`, which is the Hessian of `L_S`. So this term implicitly involves the Hessian.

Do I have to compute it? Not the full Hessian — `dε̂/dw` only ever multiplies a vector here, so it shows up as a Hessian-vector product, and HVPs are cheap (one extra backward, no `k×k` matrix ever materialized). So I *could* keep it. But let me count the cost of keeping it versus dropping it, and ask whether it even helps. If I drop the second term entirely, the gradient approximation collapses to

  `∇_w L_S^{SAM}(w) ≈ ∇_w L_S(w) |_{w + ε̂(w)}`,

which is beautifully simple: it's exactly the *ordinary* batch gradient, just computed at the ascended point `w + ε̂(w)`. No second derivatives at all. The whole procedure becomes: figure out the worst-case direction `ε̂` at `w` (needs one gradient at `w`), step there, take the ordinary gradient at the perturbed point, and use *that* to update the original `w`.

Is dropping the Hessian term legitimate, or am I throwing away the signal? Two reasons it's fine, maybe even better. First, intuitively, the dominant effect of being in a sharp region is already captured: I evaluated the gradient where the loss is *highest* in the neighborhood, so I'm descending the worst case, and the curvature-correction term is a second-order refinement of *where exactly* the max sits. Second — and this is the one that actually settles it for me — if I imagine running it both ways and comparing, keeping the second-order term doesn't buy generalization; if anything it can hurt, and the first-order updates track the full updates closely (their directions stay nearly aligned through most of training, only drifting near convergence). So the more expensive version isn't the more accurate-where-it-matters version. Drop it. The reason that the cheap thing generalizes at least as well as the expensive thing is genuinely a little mysterious, and worth probing later, but as an engineering decision it's clear: drop the second-order term.

So now the full step, concretely, as a thing SGD can drive. One SAM update of `w`:

  1. Compute `g = ∇_w L_B(w)` on the current batch `B` — one forward, one backward at `w`.
  2. Form `ε̂(w) = ρ · g / ‖g‖₂` (taking `p = 2`).
  3. Ascend: move to `w + ε̂(w)`.
  4. Compute `g_SAM = ∇_w L_B(w) |_{w + ε̂(w)}` — a second forward and backward, at the perturbed point.
  5. Restore `w`, and apply the descent there with the base optimizer: `w ← w − η · g_SAM`.

Two backward passes per step. That's the price: SAM costs `2×` a normal step, so to compare fairly against plain SGD I should let plain SGD run twice as many epochs. The `‖g‖₂` in step 2 needs a tiny `+1e-12` so I never divide by zero when the gradient happens to vanish.

Which `p`? The bound was derived at `p = 2`, and `p = 2` gives the cleanest `ε̂ = ρ g/‖g‖₂`. If I think about `p = ∞` (the sign step) versus `p = 2`, the sign step throws away all the *magnitude* structure of the gradient and treats every coordinate as equally worth perturbing, which seems wasteful when some directions are far steeper than others. And against a *random* perturbation of the same Euclidean norm — `ε̂ = ρ z/‖z‖₂`, `z ∼ N(0,I)` — the adversarial (gradient-aligned) choice should win, because random noise mostly probes flat directions and only occasionally hits a steep one, whereas `ε̂ = ρ g/‖g‖₂` aims straight up the steepest local direction, which is exactly where sharpness lives. So `p = 2`, adversarial, is the default; `ρ` I'll tune over a small grid like `{0.01, 0.02, 0.05, 0.1, 0.2, 0.5}`, and `0.05` looks like a robust default I can fall back on without tuning.

There's one more thing I glossed. My derivation defined the inner `max` over the *whole* training set `L_S`. But in practice I compute `ε̂` on a *batch* — and if I'm on multiple accelerators, I compute it independently on each shard of size `m` and then average the resulting SAM gradients, rather than first averaging gradients across shards and then doing a single `ε̂`. Is that the same objective? No, and the difference is interesting. Averaging per-shard SAM updates is equivalent to changing the objective to a *sum of independent `ε`-maximizations*, each over a disjoint subset of `m` points, instead of one `ε`-maximization over the global sum. Call the sharpness this induces `m`-sharpness. As `m` shrinks, each maximization is over fewer points, so each `ε̂` is sharper-tailored to its little subset, and empirically smaller `m` generalizes *better* — and, intriguingly, the `m`-sharpness measure with `m < n` correlates with the actual generalization gap *better* than the full-batch sharpness my theorem started from. That's a happy accident: the thing I'm forced to do for parallelism (not syncing the perturbations across accelerators) is also the thing that generalizes best. So I won't sync `ε̂`; I'll compute it per-shard.

Let me also double check the *sign* in the ascend step, because it's the easiest place to flip something and silently get plain SGD with extra steps. The inner problem is a `max` of the loss. So `ε̂` points *uphill*: `ε̂ = +ρ g/‖g‖₂`, and I *add* it to `w` to climb to the local worst case. Then I compute the gradient there and *descend* on `w` (subtract `η g_SAM`). Add to climb, subtract to descend — both signs are as they should be. If I'd subtracted in the ascend step I'd be evaluating the gradient at a *lower*-loss nearby point, which is the opposite of probing sharpness.

Now to code. The clean way is to make this an optimizer wrapper around a base optimizer (SGD, Adam, whatever), with two methods: a `first_step` that does the ascend, and a `second_step` that restores and does the descent through the base optimizer. The training loop hands the optimizer a closure that does a forward-backward, so the optimizer can recompute the gradient at the perturbed point.

```python
import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        # SAM wraps an ordinary optimizer; the descent in step 5 is delegated to it.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # steps 1-3: with g already on .grad (from a backward at w),
        # build e_hat = rho * g / ||g||_2 and ASCEND to w + e_hat.
        grad_norm = self._grad_norm()                 # ||g||_2 over all params
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) # rho / ||g||  (avoid /0)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()           # remember w
                # adaptive variant rescales per-weight by w^2 (scale-invariant); off by default
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                                        # w <- w + e_hat  (climb)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # step 5: with .grad now holding g_SAM (a backward at w + e_hat),
        # restore w and let the base optimizer take the descent step.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]   # back to w from w + e_hat
        self.base_optimizer.step()                # w <- w - eta * g_SAM  (the real, sharpness-aware step)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM needs a closure that does a full forward-backward"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)   # ascend, then clear g
        closure()                         # step 4: recompute gradient g_SAM at w + e_hat
        self.second_step()                # restore w, descend

    def _grad_norm(self):
        # ||g||_2 over the whole parameter vector (one scalar). Per-shard => m-sharpness.
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
        return norm
```

And the training loop, where the closure is what lets the optimizer get a second gradient at the perturbed weights:

```python
def train_step(model, x, y, optimizer, loss_fn):
    # first forward-backward: gradient g at w (for building e_hat)
    enable_running_stats(model)                 # batchnorm: count this pass normally
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)        # ascend to w + e_hat

    # second forward-backward: gradient g_SAM at w + e_hat
    disable_running_stats(model)                # don't let the 2nd pass pollute BN stats
    loss_fn(model(x), y).backward()
    optimizer.second_step()                     # restore w, descend with base optimizer
    return loss
```

So the whole causal chain: training loss alone is underdetermined among many equal-loss minima that generalize differently → flatness of the basin is what tracks generalization, and PAC-Bayes turns that into a real upper bound whose key term is the worst-case loss in a `ρ`-ball → so minimize `max_{‖ε‖≤ρ} L_S(w+ε)` (plus weight decay standing in for the bound's `‖w‖²` term) → the inner max is intractable but linearizes, and the linear-objective-over-a-`ρ`-ball is a dual-norm problem with closed form `ε̂ = ρ·sign(g)|g|^{q-1}/(‖g‖_q^q)^{1/p}`, which for `p=2` is just `ρ g/‖g‖₂` → differentiating the composite gives the gradient at the ascended point plus a Hessian-vector term that I drop, both for speed and because the cheap version generalizes at least as well → leaving a two-pass update (ascend to `w+ε̂`, take the ordinary gradient there, descend at `w`) that any base optimizer can drive, costing `2×` per step, with the per-shard `m`-sharpness variant being a free generalization bonus from parallelism.
