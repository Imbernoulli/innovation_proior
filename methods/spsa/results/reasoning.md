Let me start from what actually goes wrong when I try to certify a model as robust. I have a classifier and I want to know whether, inside the `L_inf` ball of radius `eps` around a clean image, there exists a perturbation that flips the prediction. My only honest answer is to go look for one — to attack. And the attack I reach for first, PGD, takes a gradient step on the margin loss and projects back into the ball. It works beautifully when the gradient is real. But I keep running into models where it isn't. Some of them put a non-differentiable transform in front of the network — JPEG compression, a discrete purification loop — and backprop through the whole pipeline simply does not exist; PGD has nothing to step on. Others are differentiable but the gradient is a lie: it's tiny, or noisy, or points the wrong way, so PGD wanders, finds nothing, and I dutifully report a high robust accuracy. The thing that should scare me is that I cannot tell those two situations apart from the outside. "PGD failed" might mean "this model is robust" or it might mean "this model hid its gradient from me." As long as my attacker depends on the gradient, a defense can defeat the *attacker* without defeating the *attack surface*, and I'd never know.

So the constraint I want to impose on myself is brutal but clarifying: build an attack that never touches the gradient or the weights. It may only feed inputs forward and read the logits back. If a model survives an attack that doesn't use gradients at all, then "it survived" can't be blamed on missing or masked gradients — the evidence finally means something. The objective is the same one PGD uses: minimize the correct-class advantage `J(x) = m(x)_{y0} - max_{j != y0} m(x)_j` over `x` in the `eps`-ball, where `m(x)_j` is the logit of class `j`. `J < 0` exactly when the image is misclassified. I'm keeping the margin form on purpose — cross-entropy saturates once the model is confident, the softmax pins at 0 or 1 and the loss goes flat, so the value barely moves as I perturb the input; the logits stay roughly linear out there, so the margin keeps a usable slope long after cross-entropy has died. I want that slope, because all I'm going to have is differences in the objective.

Now, how do I descend an objective I can only *measure*, never differentiate? This is the classical setting of stochastic approximation. If I had a noisy gradient I'd just run Robbins–Monro, `x_{k+1} = x_k - a_k * ghat_k`, with the step `a_k` decaying neither too fast nor too slow, and I'd converge. But I don't have `ghat`; I have `J`. The textbook move when you only have function values is to rebuild the gradient out of finite differences, the way Kiefer and Wolfowitz did: for each coordinate `i`, probe a little to each side and difference,

  ghat_i = ( J(x + c e_i) - J(x - c e_i) ) / (2c),

with `e_i` the i-th unit vector and `c` a small probe radius. That's just the definition of the partial derivative, made numerical, and it's unbiased to leading order: Taylor-expand and the linear terms give `ghat_i = ∂_i J + O(c^2)`. I'll use the two-sided form rather than one-sided on purpose — the symmetric difference cancels the even-order term, so the bias is `O(c^2)` instead of `O(c)`, and it keeps the noise centered.

And immediately I hit the wall. To assemble one full gradient this way I need to probe every coordinate, and `x` is the entire image: `D = C·H·W`, which is 3072 for a CIFAR image and over 150,000 for ImageNet. Each coordinate costs two forward queries, so one gradient costs `2D` queries — six thousand for one CIFAR step, three hundred thousand for one ImageNet step — and I need many steps. The query cost scales with the dimension of the input, and the input is enormous. Under any realistic budget this is hopeless. Coordinatewise finite differencing is the right idea and the wrong economics. So the real question sharpens: can I get an estimate of the gradient whose *cost does not grow with `D`*?

Let me stare at why FDSA is so expensive. It's because I insist on isolating each coordinate — I perturb along `e_i` alone so that the difference `J(x+c e_i) - J(x-c e_i)` reports purely on `∂_i J`. The isolation is what costs me `D` separate probes. What if I don't isolate? What if I perturb *all* coordinates at once, with a single random vector, and then somehow read each coordinate's derivative out of that one shared measurement? Let me just try it and see what survives. Pick a random direction `v ∈ R^D` and take the two-sided difference along it:

  df = J(x + c v) - J(x - c v).

Taylor-expanding each side around `x`:

  J(x ± c v) = J(x) ± c (v · ∇J) + (c^2/2) v^T H v ± O(c^3),

so the symmetric difference is

  df = 2c (v · ∇J) + O(c^3),

the quadratic terms cancel by symmetry. So `df / (2c) ≈ v · ∇J = Σ_j v_j ∂_j J`. That's a single scalar — the directional derivative along `v`. It mixes all `D` partials together. How do I get the i-th partial back out of a scalar that mixes everything? Here's the gamble: divide by `v_i`.

  ghat_i := df / (2c v_i) = (1/v_i) Σ_j v_j ∂_j J + O(c^2)
         = ∂_i J  +  Σ_{j ≠ i} (v_j / v_i) ∂_j J  +  O(c^2).

Look at that. The `j = i` term gives me exactly `∂_i J` (the `v_i/v_i` cancels). And every other partial `∂_j J` shows up too, but carried by a *random* coefficient `v_j / v_i`. The wanted signal is clean; the contamination is the cross-talk `Σ_{j≠i} (v_j/v_i) ∂_j J`. So this is only useful if that cross-talk vanishes when I average over the randomness in `v`. Let me check. Take `v` to have independent, mean-zero components. Then for `j ≠ i`, by independence,

  E[ v_j / v_i ] = E[v_j] · E[1/v_i] = 0 · E[1/v_i] = 0,

provided `E[1/v_i]` is *finite* — that's the load-bearing caveat, and I'll come back to it because it's about to decide everything. Granting it for a second: each cross term has expectation zero, so

  E[ ghat_i ] = ∂_i J + O(c^2).

The estimator is almost unbiased — bias only `O(c^2)` from the Taylor remainder — and I got it from **exactly two function evaluations**, `J(x + c v)` and `J(x - c v)`, *regardless of `D`*. The whole gradient vector `ghat`, all `D` components, comes out of those same two scalars: I compute `df` once and then divide it by `2c v_i` coordinate by coordinate. Two queries, any dimension. That's the economics I needed. The reason it works is that I stopped demanding each measurement be *clean*; I let each measurement be a noisy, biased-per-sample estimate whose *errors cancel in expectation*, and I buy back accuracy with averaging instead of with isolation. One random simultaneous perturbation carries, on average, as much directional information as a full sweep of one-at-a-time perturbations.

Now I have to settle the caveat I waved past, because it's not a technicality — it's the difference between this working and detonating. I needed `E[1/v_i]` to be finite, and in the usual regularity conditions I need finite inverse moments such as `E[|1/v_i|]` and `E[1/v_i^2]`. What distribution should `v` come from? My instinct, and the instinct of basically everyone who's done random-direction gradient estimates for neural nets, is to sample `v` Gaussian: `v ~ N(0, I)`. Smooth, isotropic, the natural choice. Let me check the condition. For `v_i ~ N(0,1)`, the density is bounded away from zero near `t = 0` while `1/|t|` and `1/t^2` blow up there, so the required inverse moments diverge; the signed principal value is not a finite expectation I can use in the estimator. The cross-talk terms `(v_j/v_i)∂_j J` no longer have a well-defined finite mean under the condition I need. In practice this shows up as occasional gigantic, wild estimates whenever some `v_i` lands near zero and I divide by it. Wall. The obvious distribution is forbidden by the very algebra that makes the trick work. Same fate for a uniform distribution centered at zero — it also puts mass near `v_i = 0` and violates the finite-inverse-moment condition.

So I need a mean-zero distribution whose components stay *away from zero*, so that `1/v_i` is well-behaved and has finite first and second moments. The cleanest such object is the symmetric Bernoulli: `v_i ∈ {+1, -1}`, each with probability `1/2`. Mean zero. And `1/v_i`? Since `v_i` is `±1`, `1/v_i = v_i` — it's `±1` too, trivially bounded, finite moments of every order. `E[1/v_i] = E[v_i] = 0`, so the cross-talk really does cancel. There's even a small beauty here that simplifies the code: because `v_i = ±1`, *dividing* by `v_i` is the same as *multiplying* by `v_i`. The estimator

  ghat = (df / (2c)) · v

needs no division by the perturbation at all — one scalar `df/(2c)` times the same `±1` vector I already drew. The Rademacher choice isn't an aesthetic preference over Gaussian; it's *forced* by the finite-inverse-moment condition, and as a bonus it collapses the per-coordinate division into a single broadcast multiply. (If I really wanted a near-Gaussian feel I'd have to gouge a hole out of the distribution around zero to keep the inverse moment finite — a symmetric two-part uniform with the middle removed — but that's a contortion to rescue a distribution the condition already rejected; the symmetric `±1` is the natural answer.)

Let me also make sure I believe the efficiency claim and not just assert it. FDSA spends `2D` evaluations to get a gradient with some accuracy; this simultaneous-perturbation estimate spends 2. Is the per-step accuracy really comparable, or am I just trading queries for garbage? The asymptotic story from stochastic-approximation theory is the reassuring one: with a decaying probe `c_k` and step `a_k`, the simultaneous-perturbation iterate converges almost surely to a stationary point under the same smoothness conditions as FDSA, and — the key result — it reaches the *same asymptotic mean-squared error* per iteration as FDSA while using a factor of `D` fewer function evaluations. The intuition is exactly the "errors cancel in directions" picture: each single random-direction estimate is wrong (it points partly sideways because of the cross-talk), but over many iterations the sideways errors average out, the way independent noise averages out of a sample mean, so the *path* tracks the true descent path on average even though no single step follows it. So I'm not trading queries for garbage; I'm trading "exact direction now, `D` queries" for "noisy direction now, 2 queries, correct on average." For high `D` that trade is overwhelmingly in my favor.

Still, "correct on average" with high per-sample variance is going to make a single step jittery, and a jittery descent on a noisy objective — remember the defended model may itself be stochastic, random resizing or dropout making each query noisy — could thrash. The cure is the same averaging that the asymptotics rely on, but I can do it *within* a step instead of only across steps: draw `n` independent Rademacher vectors `v^{(1)}, …, v^{(n)}`, form `n` independent two-query estimates, and average them,

  ghat_bar = (1/n) Σ_{i=1}^n  ( J(x + c v^{(i)}) - J(x - c v^{(i)}) ) / (2c) · v^{(i)}.

Each estimate is independent, so the variance of the average falls like `1/n`, and the cross-talk that survives in any one estimate gets knocked down toward its zero mean. This costs `2n` queries per step, and it's the natural knob for trading queries against estimate quality. It also happens to be exactly what a GPU wants: the `2n` forward passes are a batch, evaluated at once. So a step costs `2n` queries, and `n` is mine to set — large `n` for a clean, reliable direction when I care about attack strength, small `n` when I'm query-starved. The number of cross-talk-cancelling samples per step is the dial between "weak but cheap" and "strong but expensive."

Now I have a noisy gradient estimate and I'm back in Robbins–Monro territory — descend with it and project. Let me think about the descent step itself. The plain stochastic-approximation update is `x' = x - a · ghat_bar`. That's fine and it's what the basic recursion prescribes. But `ghat_bar` is *noisy* and its coordinates are unevenly scaled — some pixels have large derivatives, some tiny, and the estimate's noise floor is the same across them. A raw global step size `a` is the wrong tool for that: it's the same per-coordinate-scaling problem that plagues training noisy networks. The standard fix there is Adam — keep per-coordinate exponential moving averages of the estimate and of its square, and step by their ratio `m̂/(√v̂ + ε)`. The first moment smooths the noise across steps (a second layer of averaging on top of the within-step averaging), and the `1/√v̂` rescaling gives every pixel its own effective step so a few large-derivative pixels don't dominate. So I'll run Adam on the perturbation, feeding it `ghat_bar` as if it were a real gradient — the estimator is unbiased enough that Adam can't tell the difference, and Adam's robustness to noisy, unevenly-scaled gradients is exactly what this noisy directional estimate needs. It converges faster here than a raw step; that's the whole reason to prefer it.

I keep the variable as the *perturbation* `dx` rather than the image `x` itself, because the constraint lives on `dx`. After each Adam step I have to return to the feasible set: the perturbation must satisfy `||dx||_inf ≤ eps`, and the resulting image `x0 + dx` must stay a valid image in `[0,1]`. Both are simple clamps. Clamp `dx` componentwise to `[-eps, eps]` — that's the Euclidean projection onto the `L_inf` ball, since the ball is a box and box-projection is per-coordinate clipping. Then clamp `x0 + dx` to `[0,1]` and fold the result back into `dx`. That's the projection step of the recursion, `x_{t+1} = argmin_{x ∈ N_eps(x0)} ||x' - x||`, made concrete. For the compact implementation I can start `dx` at zero and let the repeated noisy steps move it; if I later wrap this in a multi-start evaluation harness, random starts are an outer-loop choice, not a change to the estimator itself.

Let me also be honest about the probe radius `c` (I've been calling the attack's version `delta`). It controls the bias–noise trade in the finite difference. Smaller `c` means smaller Taylor bias (`O(c^2)`), good. But the *signal* in `df = J(x+c v) - J(x-c v)` scales with `c` — it's `2c(v·∇J)` — while the measurement noise from the model's own stochasticity is a fixed floor independent of `c`. So as `c → 0` the signal-to-noise ratio of each probe *degrades* like `c`. There's a sweet spot: small enough that the directional-derivative approximation is faithful, large enough that the difference rises above the noise floor. A value around `0.01` on `[0,1]`-scaled pixels sits in that window — well inside the `eps`-ball for typical `eps`, and big enough to produce a measurable `df`. The Adam learning rate, the step on `dx`, I'll likewise keep modest (around `0.01`) so the noisy estimate doesn't yank the iterate out of the productive region in a single step.

At this point the loop is concrete. I draw Rademacher perturbations (one sign field per spatial pixel in the implementation, shared across channels as it is expanded across the image tensor), evaluate the model at `x + c v` and `x - c v` in a batch, form the two-sided difference, multiply by the same `v` (division-equals-multiplication for `±1`), average over the `n` samples to kill cross-talk and noise, hand that to Adam as the gradient on `dx`, and project back into the box. The logit margin I compute in code is the negative of the correct-class advantage, `max_wrong - true`; it is positive once the attack succeeds, so for an untargeted attack the quantity I give Adam to minimize is its negative, which is exactly `J`.

```python
import torch
from torch.nn.modules.loss import _Loss


class MarginalLoss(_Loss):
    """Carlini-Wagner logit margin: max nontarget logit - target logit.
    Positive iff misclassified; its negative is the objective minimized for untargeted attack."""

    def forward(self, logits, targets):
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets, top_logits[..., 1], top_logits[..., 0],
        )
        loss = max_nontarget_logits - target_logits   # > 0 means misclassified
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError("unknown reduction: '%s'" % (self.reduction,))


class Attack:
    """Forward-only black-box attack: it may call the model to get logits, nothing else."""

    def __init__(self, model, eps, delta, lr, nb_iter, nb_sample, max_batch_size):
        self.model = model
        self.eps = eps                 # L_inf radius of the feasible ball
        self.delta = delta             # c: finite-difference probe radius (~0.01)
        self.lr = lr                   # Adam step on the perturbation (~0.01)
        self.nb_iter = nb_iter         # number of descent steps
        self.nb_sample = nb_sample     # n: simultaneous-perturbation samples per step
        self.max_batch_size = max_batch_size
        self.loss_fn = MarginalLoss(reduction="none")
        self.targeted = False

    def loss(self, logits, y):
        # untargeted: minimize correct-class advantage by maximizing this margin
        m = self.loss_fn(logits, y)
        return m if self.targeted else -m

    def linf_clamp_(self, dx, x, eps):
        # project the perturbation onto the L_inf box, keep x+dx a valid image in [0,1]
        dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        x_adv = torch.clamp(x + dx_clamped, min=0.0, max=1.0)
        dx += x_adv - x - dx                          # write the feasible value in-place
        return dx

    def _batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size] * (n // max_batch_size)
        if n % max_batch_size:
            batches.append(n % max_batch_size)
        return batches

    @torch.no_grad()
    def spsa_grad(self, images, labels, delta, nb_sample, max_batch_size):
        # estimate the gradient of the attack objective using only forward queries:
        #   for each of nb_sample Rademacher v, two-sided difference along v,
        #   then sum (df / (2*delta)) * v and average.  Two queries per sample, any D.
        grad = torch.zeros_like(images)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)

        def f(xv, yv):
            return self.loss(self.model(xv), yv)

        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()

        v = torch.empty_like(images[:, :1, ...])      # one perturbation per pixel,
        for batch_size in self._batch_sizes(nb_sample, max_batch_size):
            x_ = images[:batch_size]
            y_ = labels[:batch_size]
            vb = v[:batch_size].bernoulli_().mul_(2.0).sub_(1.0)   # Rademacher +-1
            v_ = vb.expand_as(x_).contiguous()                     # shared across channels
            x_ = x_.reshape(-1, *images.shape[2:])
            y_ = y_.reshape(-1, *labels.shape[2:])
            v_ = v_.reshape(-1, *v.shape[2:])
            x_shape = images[:batch_size].shape
            df = f(x_ + delta * v_, y_) - f(x_ - delta * v_, y_)   # two-sided difference
            df = df.view(-1, *([1] * (v_.dim() - 1)))
            grad_ = df / (2.0 * delta * v_)            # = (df/2delta)*v since v in {+-1}
            grad_ = grad_.view(x_shape).sum(dim=0)     # sum the per-sample estimates
            grad += grad_
        grad /= nb_sample                              # average over samples: var ~ 1/n
        return grad

    def perturb(self, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)   # Adam on the noisy estimate
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(x + dx, y, self.delta,
                                     self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps)       # project back into the ball
        return x + dx
```

The causal chain: I wanted to certify robustness, but every gradient-based attack can be defeated by a model that simply withholds a usable gradient, so a passing attack told me nothing. I forced myself to use only forward queries. Rebuilding the gradient coordinate-by-coordinate (Kiefer–Wolfowitz) is honest but costs `2D` queries, and `D` is the whole image — fatal. Perturbing all coordinates at once with a single random vector and reading each partial back out by dividing the shared directional difference by that coordinate's perturbation gives, in expectation, the true gradient from just two queries — the cross-coordinate contamination cancels because the perturbation components are independent and mean-zero. That cancellation needs finite inverse moments, which rules out Gaussian and uniform and selects the symmetric `±1` (Rademacher) distribution, which as a bonus turns division into multiplication. Averaging `n` such estimates per step drives the variance down like `1/n` and batches cleanly on a GPU; feeding the average to Adam handles the residual noise and the uneven per-pixel scaling; clamping to the `L_inf` box and to `[0,1]` keeps every iterate a valid in-budget image. The probe radius trades Taylor bias against the model's noise floor, and a small Adam step keeps the noisy direction from overshooting. The result is a gradient-free `L_inf` attack whose cost is independent of the input dimension — exactly the trustworthy adversary I needed to tell real robustness from hidden vulnerability.
