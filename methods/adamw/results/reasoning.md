OK, let me think this through from scratch. The thing nagging at me is a mismatch I keep seeing in practice. Everyone reaches for Adam — it's fast, it barely needs tuning, it just works on a fresh problem. And yet when I look at who actually holds the top numbers on CIFAR-10, CIFAR-100, ImageNet, it's plain SGD with momentum, every time. Wilson and colleagues hammered this point across a whole spread of tasks — image classification, character language modeling, parsing — adaptive methods generalize worse than a well-tuned SGD with momentum, and the unsettling part is that this happens even when Adam reaches a *lower* training loss. So it's not that Adam fails to optimize. It optimizes fine. It just lands somewhere that tests worse.

People have floated explanations — sharp minima versus flat minima, some intrinsic defect of adaptation — but none of them is a thing I can put my finger on and fix. I want a concrete, mechanical cause. So let me stop theorizing about minima and go look at what I actually *do* differently between SGD and Adam when I train these image models. The one ingredient that's always present in the SGD recipes that win, and that I also switch on for Adam without thinking, is weight decay. So let me stare at weight decay itself.

What is weight decay, really? The original idea is dead simple: at every step, before you do anything else, shrink every weight a little toward zero, then take your gradient step.

  θ_{t+1} = (1 − λ) θ_t − α ∇f_t(θ_t).

That's it — a multiplicative pull toward the origin, rate λ per step. Now, that is *not* how I actually implement it. Nobody writes that line. What I write — what every library writes — is an L2 penalty on the loss:

  f^reg(θ) = f(θ) + (λ'/2)‖θ‖²,   so   ∇f^reg = ∇f + λ'θ,

and I hand that augmented gradient to the optimizer. And we all call this "weight decay" interchangeably. Why do we get away with that? Because for plain SGD the two are literally the same update. Let me just check it so I'm sure. SGD on the L2 objective:

  θ_{t+1} = θ_t − α∇f^reg = θ_t − α∇f − αλ'θ_t.

And the genuine weight-decay rule, expanded:

  θ_{t+1} = (1 − λ)θ_t − α∇f = θ_t − λθ_t − α∇f.

Line them up. The gradient terms match. The shrink terms match iff αλ' = λ, i.e. λ' = λ/α. Fine — they're identical, *provided* I set the L2 coefficient to λ/α.

Hold on. Let me not breeze past that "provided." The equivalence isn't free; it carries a coupling. If there's some true best per-step shrink λ for my problem, then the L2 coefficient I should dial in is λ/α — which *moves the moment I change the learning rate*. So if I'm tuning by sweeping the L2 coefficient λ' (which is what the knob in my code actually is), the best λ' is not a property of the problem alone; it slides around as I retune α. The two hyperparameters are tangled. That already smells like part of why SGD has a reputation for being finicky about its learning rate — half of what I'm "retuning" when I move α is silently the regularization strength. Mental note. But for SGD at least the *updates* are genuinely equal, so this is a tuning annoyance, not a correctness problem.

Now Adam. Here's where I should be careful, because Adam doesn't take a raw gradient step — it preconditions. Let me write the whole thing out, because the place the regularizer enters is going to matter. With L2 the way I'd code it, the regularizer rides inside the gradient:

  g_t = ∇f_t(θ_{t−1}) + λ'θ_{t−1},
  m_t = β₁ m_{t−1} + (1−β₁) g_t,
  v_t = β₂ v_{t−1} + (1−β₂) g_t²,
  m̂_t = m_t/(1−β₁ᵗ),  v̂_t = v_t/(1−β₂ᵗ),
  θ_t = θ_{t−1} − α m̂_t/(√v̂_t + ε).

And now I see the gradient of my regularizer, λ'θ, has been swept up into both moving averages and — this is the part that should bother me — it gets divided by √v̂_t along with everything else. The shrink on each coordinate is no longer a clean λθ. It's λ'θ filtered through the per-coordinate adaptive scaling.

Let me make this precise instead of hand-wavy. Abstract the optimizer as a preconditioned step: whatever Adam is doing, one step has the form θ_{t+1} = θ_t − α M_t ∇f_t, where M_t is the diagonal preconditioner — for Adam, roughly M_t = diag(1/(√v̂_t + ε)), the per-coordinate inverse RMS gradient magnitude. The defining feature of an *adaptive* method is exactly that M_t is not a scalar multiple of the identity; if it were, it would just be SGD with a rescaled learning rate. So M_t ≠ k I.

Now ask the question I asked for SGD: is there an L2 coefficient λ' that reproduces genuine weight decay? Genuine decay, run through this optimizer, is

  θ_{t+1} = (1 − λ)θ_t − α M_t ∇f_t.

L2 with coefficient λ', where the penalty's gradient λ'θ goes *through the preconditioner* (because it's part of the gradient now):

  θ_{t+1} = θ_t − α M_t(∇f_t + λ'θ_t) = θ_t − αλ' M_t θ_t − α M_t ∇f_t.

For these to be the same update for every θ_t, the shrink parts must match: λθ_t = αλ' M_t θ_t, for all θ_t. That forces M_t = (λ/(αλ')) I — a scalar times the identity. But we just said M_t ≠ k I for an adaptive method. So there is *no* L2 coefficient λ' that makes L2 regularization equal to weight decay for Adam. None. The thing we've been calling the same is not the same the instant the preconditioner stops being trivial.

So the universal habit — "weight decay := add λ'θ to the gradient" — is silently doing something *other* than weight decay for every adaptive optimizer in wide use. That's the concrete, mechanical discrepancy I was hunting for. Now I want to know which way it bends, because "different" only matters if it's different in a harmful direction.

Look again at where λ'θ ends up: it's added to g, then v_t accumulates g², then the update divides by √v_t. So a weight whose loss gradients have historically been large has a large √v̂, and its decay term — riding in the numerator over that same large denominator — gets shrunk. The weights with the biggest gradients get regularized the *least*. And the ones with tiny gradients, small √v̂, get the decay applied at nearly full strength. That's backwards from what I'd want. If anything I'd like to lean *harder* on the high-gradient directions, since those are the brittle ones — a small wiggle there moves the loss a lot. With L2-in-Adam I'm doing the opposite, and there's no single λ' that fixes it: crank λ' up to get real pressure on the big-gradient weights and I'll annihilate the small ones; keep it gentle to protect the small ones and the big-gradient weights are essentially unregularized. That is exactly the symptom people report — Adam barely benefits from L2 at all, its best run with nonzero λ' is about as good as its best run with λ'=0. Of course it is: L2 can't deliver effective uniform pressure here. And on the very datasets where regularization is what separates good from great (the image-classification benchmarks where SGD's L2 is doing real work), Adam shows up under-regularized and loses. There's the link to the generalization gap, and it's mechanical, not mystical.

So what's the fix? It's almost embarrassing — go back to the literal definition. Don't route the decay through the gradient. In the algebra, let λ mean the per-step shrink amount and multiply the weights by (1 − λ) directly, as its own step, sitting *outside* the adaptive machinery:

  θ_t = θ_{t−1} − α m̂_t/(√v̂_t + ε) − λ θ_{t−1},

with g_t now the *raw* loss gradient, no λ'θ added. The √v̂ normalization never touches the decay term. Every weight shrinks by the same factor (1 − λ) regardless of its gradient history. Decoupled. This gives me a variant of Adam where the decay is, finally, actual weight decay — call it AdamW. And the identical move on SGD with momentum — decay the weights directly in the parameter-update line instead of stuffing λ'θ into the gradient that feeds the momentum buffer — gives SGDW. For plain SGD the direct shrink and L2 update are the same when αλ' = λ; with momentum, the point is instead to keep the shrink from being accumulated in the buffer and to make the two knobs cleanly separate.

One detail I shouldn't get sloppy about: scheduling. The reason I want to think about this is that with L2, the decay was *inside* the gradient, so whenever I multiplied the learning rate by some schedule η_t, the decay got multiplied too — it rode along for free. Now that I've pulled the decay out into its own step, it won't see the schedule unless I put it there by hand. If I want the same qualitative behavior under a cosine or step-drop schedule, I should scale the decay step by η_t as well:

  θ_t ← θ_{t−1} − η_t( α m̂_t/(√v̂_t + ε) + λ θ_{t−1} ).

So η_t multiplies both the gradient step and the decay step. Good — that keeps the two moving together under any schedule while keeping λ and α independent as the per-problem knobs. When I translate this into PyTorch-style code, I need to remember that the optimizer argument called `weight_decay` is a coefficient w that the implementation multiplies by the learning rate, so the algebraic shrink is λ = αw for a fixed schedule and η_tαw at step t. That is why the canonical fixed-learning-rate line is `p.mul_(1 - lr * weight_decay)`, and why my explicit-schedule version uses `p.mul_(1 - eta * lr * weight_decay)`.

Now I want to understand *what objective*, if any, this decoupled rule is secretly optimizing — is it minimizing some sensible loss, or have I just bolted on a hack? Let me take the cleanest case I can: freeze the preconditioner. Suppose M_t = diag(s)^{-1} with fixed positive s_i — think of s_i as the typical gradient magnitude of coordinate i, held constant. Run decoupled decay:

  θ_{t+1} = (1 − λ)θ_t − α M_t ∇f_t = θ_t − λθ_t − α ∇f_t/s,

where the division by s is element-wise. Now I'll try to find a static regularized loss whose plain preconditioned gradient step reproduces this. Guess a *scale-adjusted* L2 penalty, one that weights each coordinate by s_i:

  f^sreg(θ) = f(θ) + (λ'/2)‖θ ⊙ √s‖²  = f(θ) + (λ'/2) Σ_i s_i θ_i².

Its gradient is ∇f^sreg = ∇f + λ'(θ ⊙ s) — the √s squares up to a plain s inside the derivative. Take the preconditioned step on it:

  θ_{t+1} = θ_t − α M_t ∇f^sreg = θ_t − α ∇f/s − αλ'(θ ⊙ s)/s = θ_t − α ∇f/s − αλ' θ.

Look at the decay term: the s from the penalty's gradient and the 1/s from the preconditioner cancel exactly, leaving a clean −αλ'θ. Compare with the decoupled-decay step above, −α∇f/s − λθ: identical iff αλ' = λ, i.e. λ' = λ/α. So decoupled weight decay, with a fixed preconditioner, is *equivalent to* L2 regularization on the scale-adjusted penalty ‖θ ⊙ √s‖². And that tells me precisely what decoupling is doing: it scales coordinate i by √s_i in the norm, so the squared penalty weight is s_i — it leans *harder* on the weights with historically large gradients, exactly the brittle directions I wanted to discipline and exactly the ones plain L2-in-Adam was letting off the hook. The intuition and the algebra agree.

The caveat: real Adam doesn't hold M_t fixed, it changes it every step, so this isn't a theorem about the algorithm I'll actually run — it's a fixed-preconditioner snapshot. But it's the right intuition: the equivalent norm scales each coordinate by √s_i, hence the squared penalty weights it by s_i. That's enough to trust that decoupling isn't arbitrary.

Is there an even more principled reason to prefer weight decay over L2 here, beyond "the cancellation is clean"? Let me think about where a shrink-toward-zero term *ought* to live. Picture stochastic optimization as tracking: I'm trying to follow the optimum of one parameter while the others drift, so I carry a distribution over θ, push it forward through a state-transition prior P(θ_{t+1}|θ_t) describing a small data-independent drift, then fold in the mini-batch likelihood. If the prior is Gaussian and the likelihood approximately conjugate, the posterior-mean update comes out as μ_post = μ_prior + Σ_post g — the gradient gets preconditioned by the *posterior covariance*: bigger steps where I'm uncertain, smaller where I'm confident. That's a derivation of the adaptive preconditioner itself, which is reassuring; Adam-like methods are this filter. Now, where does a pull toward zero belong in this picture? It belongs in the *state-transition prior* — the data-independent drift — as P(θ_{t+1}|θ_t) = N((I − A)θ_t, Q), with A a regularizer keeping the weights from wandering off to infinity. Set A = λI and the prior mean becomes (1 − λ)θ_t: the weights are multiplied by (1 − λ) every step. That is decoupled weight decay, dropping straight out of the data-independent part of the model — and crucially it's applied to the *prior*, untouched by the per-coordinate uncertainty Σ. L2 would instead be a term inside the likelihood, and would therefore get scaled by that same uncertainty — which is precisely the entanglement I've been trying to undo. So weight decay isn't just the empirically nicer choice; it's the one that sits in the structurally correct place. Good — that settles *why*, not just *that*.

Once I start running this, I notice the best λ depends on how long I train. That makes sense the moment I write down the cumulative effect: ignoring the gradient, the pure decay multiplies a weight by (1 − λ) once per update, so over a whole run it's (1 − λ)^{#updates}, and the number of updates is (B/b)·T for dataset size B, batch b, and T epochs. Same λ, more updates → far more total shrinkage. So a value tuned on a short run is much too strong on a long one, and a value tuned at one batch size is wrong at another. I'd like a knob that means roughly the same thing across budgets. The cumulative shrink is governed by λ·#updates ∝ λ·BT/b, so the most naive de-coupling would be λ ∝ b/(BT). But let me not just assert the exponent — when I actually look at how the optimal λ moves as I change the budget, a full 1/(BT) correction over-shoots; the optimum slides more gently than that. A square root tracks it well in practice, so I reparameterize

  λ = λ_norm · √( b / (B·T) ),

and tune λ_norm instead. Reading it back: λ_norm is "the decay I'd use if I were allowed only a single batch pass." The payoff is that one λ_norm transfers across very different budgets and even across datasets — e.g. from CIFAR-10 to ImageNet32×32, where an epoch is roughly 24× longer; with the raw λ I'd have been off by something like 5×, badly over-decaying the bigger dataset. I'll flag that the √ is one choice informed by a handful of experiments rather than a law — the durable point is that *some* normalization is needed; the exact exponent is secondary.

The schedule question comes back now. Cosine annealing — η_t = η_min + 0.5(η_max − η_min)(1 + cos(π T_cur/T_i)), which for η_max=1, η_min=0 is just η_t = 0.5 + 0.5cos(π T_cur/T_i) — cools the rate down and, with warm restarts, periodically resets T_cur to 0 so η springs back up while the weights are kept, and the next budget T_i is multiplied by T_mult. This was a clear win for SGD, and I'd tried it on Adam before and been disappointed: it improved Adam's anytime behavior but couldn't catch SGD-with-restarts. Now I understand why it stalled — the restart machinery was sitting on top of a broken regularizer; the underlying L2-in-Adam was leaving the model under-regularized, so no schedule could rescue the final quality. With the decay fixed, the schedule has something solid to stand on and the SGDR recipe carries straight over to Adam — call the combination AdamWR. And because I'm using normalized weight decay, with T set to the epochs in the *current* restart, a single λ_norm holds across restarts of different lengths instead of needing a fresh λ for each. Right before each restart, when η_t hits 0, the weights are at a good resting point — those are the iterates I'd hand back as solutions.

Let me also note what I'd want to check before believing any of this beyond the algebra. I'd look at the 2-D map of test error over (α, λ): for L2-in-SGD I expect the good region to lie along the diagonal — the signature of the α–λ coupling I proved — and for the decoupled versions I expect it to flatten out into a horizontal/vertical band, meaning I can tune the two knobs nearly independently. For Adam I'd expect L2 to barely move the needle versus λ=0, and decoupled decay to open up a genuinely better, broader basin that can finally rival SGD. And for the long runs I'd want to confirm that the win isn't just faster optimization — that at matched training loss, decoupled decay still gives lower test error, i.e. it's really buying generalization. Those are the things the algebra predicts and that I'd want the curves to confirm.

So the whole chain, end to end: weight decay and L2 are the same only for an un-preconditioned step, because the equivalence needs the shrink to commute with the update; the instant the optimizer preconditions with M_t ≠ kI, routing the decay through the gradient makes it inherit the per-coordinate scaling, which under-regularizes exactly the high-gradient weights and leaves Adam effectively unregularized; the fix is to honor the original definition and multiply the weights by (1 − λ) as a separate step outside the adaptive update; a fixed-preconditioner analysis shows this equals a scale-adjusted L2 whose norm scales coordinate i by √s_i and whose squared penalty weight is s_i, and the Bayesian-filtering view puts the decay in the data-independent prior where it belongs; normalizing λ by √(b/BT) makes the knob budget-invariant; and once the regularizer is sound, cosine annealing with warm restarts finally transfers to Adam. Now the code.

```python
import torch

class AdamW:
    """Adam with DECOUPLED weight decay: the (1-lr*wd) shrink is a separate
    step applied to the parameters, never routed through the gradient/preconditioner."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, schedule=None):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                       # raw loss gradient: no wd*p added here
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])  # scales both the step and the decay

            # PyTorch AdamW's core line: direct shrink, outside the adaptive update
            p.mul_(1 - eta * self.lr * self.wd)

            # ordinary Adam moments on the raw gradient
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            denom = (st["v"] / bc2).sqrt().add_(self.eps)
            step_size = eta * self.lr / bc1

            # preconditioned gradient step (decay is NOT inside this)
            p.addcdiv_(st["m"], denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class SGDW:
    """SGD with momentum and decoupled weight decay: decay the weights in the
    parameter update, not by injecting lambda*p into the momentum buffer."""
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4, schedule=None):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p)) for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                       # raw gradient; wd kept out of the momentum buffer
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            # direct shrink of the old weights, separate from the momentum buffer
            p.mul_(1 - eta * self.lr * self.wd)

            st["m"].mul_(self.momentum).add_(g, alpha=eta * self.lr)
            p.add_(st["m"], alpha=-1)        # gradient/momentum step

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def normalized_weight_decay(lambda_norm, batch_size, dataset_size, epochs):
    # algebraic per-step lambda = lambda_norm * sqrt(b / (B*T))
    return lambda_norm * (batch_size / (dataset_size * epochs)) ** 0.5


def pytorch_weight_decay(lambda_norm, batch_size, dataset_size, epochs, lr):
    # PyTorch's AdamW uses param *= (1 - lr*weight_decay), so weight_decay = lambda/lr
    return normalized_weight_decay(lambda_norm, batch_size, dataset_size, epochs) / lr


def cosine_eta(T_cur, T_i):
    # AdamWR schedule multiplier; warm restart resets T_cur->0 and T_i->T_i*T_mult
    import math
    return 0.5 + 0.5 * math.cos(math.pi * T_cur / T_i)
```

To recap the causal chain in one breath: the L2 = weight-decay identity holds only when the update doesn't precondition; Adam preconditions, so L2 puts the decay inside the √v̂ normalization and ends up under-regularizing the high-gradient weights, which is a concrete cause of Adam's weak generalization; pulling the decay out into a direct (1 − λ) multiply restores true weight decay, decouples λ from α, equals a √s-weighted L2 in the fixed-preconditioner limit, sits in the prior under the Bayesian-filtering view, becomes budget-invariant under √(b/BT) normalization, and lets cosine-annealed warm restarts finally work for Adam.
