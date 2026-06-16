Let me start from the thing that keeps nagging at me. I have two optimizers in front of me. SGD with momentum, when I regularize it properly, gives me the best test error I can get on CIFAR-10 and ImageNet — the state-of-the-art residual nets are all trained with it. Adam, which I reach for everywhere else because I never have to babysit per-layer learning rates, just will not match it on those same image-classification problems. And it is not that Adam can't fit the data — Wilson, Roelofs, Stern, Srebro and Recht showed the embarrassing version of this: across image classification, character-level language modeling, and parsing, the adaptive methods reach worse *test* error than SGD with momentum even when their *training* loss is as good or better, and they even built a clean linearly-separable toy where (S)GD gets zero test error while AdaGrad, Adam and RMSProp are driven to near chance. So the adaptive method is finding a solution that is genuinely worse at generalizing, not just one that's harder to fit. There are hypotheses floating around — Adam is drawn to sharp minima (Keskar et al.), though Dinh et al. poked a hole in "sharpness" by showing it's reparameterization-dependent; or there's some intrinsic defect in adaptive preconditioning. None of these tells me what to *change*. I want a mechanism I can put my finger on, and I want the fix to leave Adam's adaptive step alone, because that step is the whole reason I like Adam.

So where could the gap live? On these tasks the thing that makes SGD shine is regularization — the strong nets use heavy L2 / "weight decay." Adam gets the same regularization in the library, supposedly. Let me actually look at how Adam gets regularized, because I've been treating "L2" and "weight decay" as the same word for years and maybe that's the lazy assumption that's biting me.

Pin down both, precisely, with no hand-waving. The thing Hanson and Pratt called weight decay is: at every step shrink the weights toward zero by a fixed multiplicative factor, as a step *separate* from the loss gradient,

  theta_{t+1} = (1 − lambda) theta_t − alpha grad f_t(theta_t),

with lambda the per-step decay rate. The thing the libraries actually implement and *call* weight decay is L2 regularization: add a quadratic penalty to the loss, f^reg_t(theta) = f_t(theta) + (lambda'/2)||theta||^2, whose gradient just tacks lambda' theta onto the loss gradient before the optimizer ever sees it. Two different definitions wearing the same name. For plain SGD I've always known they coincide, and it's worth re-deriving exactly *why*, because the "why" is going to be the whole story. SGD on the L2-augmented loss steps

  theta_{t+1} = theta_t − alpha grad f^reg_t(theta_t) = theta_t − alpha grad f_t(theta_t) − alpha lambda' theta_t.

SGD doing explicit weight decay steps theta_{t+1} = (1 − lambda) theta_t − alpha grad f_t(theta_t) = theta_t − alpha grad f_t(theta_t) − lambda theta_t. Line these up term by term: the loss-gradient parts are identical, and the shrink parts match iff alpha lambda' = lambda, i.e. lambda' = lambda/alpha. So they are literally the same iterate, with that one reparameterization. Good — that's Proposition-1-grade, a clean equality. But notice the reparameterization already carries a warning I usually ignore: the L2 coefficient that reproduces a given decay strength is lambda' = lambda/alpha, *tied to the learning rate*. If there's an overall-best decay lambda, then the best L2 coefficient lambda' moves whenever I move alpha. That's why the best (alpha, lambda') for SGD sit on a diagonal and people complain SGD is finicky — the two knobs are coupled by construction. File that away.

Now the same exercise for Adam, and here's where I have to be careful instead of carrying the SGD intuition over. The library way to L2-regularize Adam is to add lambda' theta to the batch gradient and then run the *whole* Adam machinery on that augmented gradient. So g_t = grad f_t(theta_{t−1}) + lambda' theta_{t−1}, and then

  m_t = beta_1 m_{t−1} + (1−beta_1) g_t,
  v_t = beta_2 v_{t−1} + (1−beta_2) g_t^2,
  theta_t = theta_{t−1} − alpha m_hat_t / (sqrt(v_hat_t) + eps).

Let me try to find a weight-decay lambda that makes Adam's *iterates* match — the way I did for SGD — and watch where it breaks. Strip away the moment-averaging for a second and write Adam abstractly as a preconditioned step: theta_{t+1} = theta_t − alpha M_t grad f_t(theta_t), where M_t is the diagonal matrix that divides each coordinate by its own sqrt(v_hat) (plus the bias-correction and momentum bookkeeping, which don't change the shape of the argument). Adam with L2 on f^reg_t is then

  theta_{t+1} = theta_t − alpha M_t (grad f_t(theta_t) + lambda' theta_t) = theta_t − alpha M_t grad f_t(theta_t) − alpha lambda' M_t theta_t.

Adam with honest decoupled decay lambda would instead be

  theta_{t+1} = (1 − lambda) theta_t − alpha M_t grad f_t(theta_t) = theta_t − alpha M_t grad f_t(theta_t) − lambda theta_t.

The loss-gradient parts match. For the *shrink* parts to match for every theta_t I'd need

  lambda theta_t = alpha lambda' M_t theta_t  for all theta_t,

i.e. lambda I = alpha lambda' M_t. And that's the wall. For SGD M_t was a scalar multiple of the identity, so I could solve it: pick lambda' = lambda/alpha and done. But for an adaptive method M_t is a *diagonal matrix with different entries on the diagonal* — that's the entire definition of being adaptive, M_t ≠ k I. A scalar lambda I can equal alpha lambda' M_t only if M_t happens to be a scalar matrix, which it is precisely *not*. So there is no L2 coefficient lambda' that makes Adam-with-L2 do the same thing as Adam-with-weight-decay. The two are *not the same method* for adaptive optimizers. They were never the same; the equivalence I'd been assuming was an SGD-only accident of M_t = kI.

That reframes the whole generalization gap as a candidate one-line bug: every deep-learning library "regularizes" Adam with L2, calls it weight decay, and on tasks where what actually helps is honest weight decay, Adam has been getting a *different, weaker* thing than SGD all along. Let me make sure I understand *why* it's weaker, not just *that* it differs, because the "why" tells me whether decoupling will actually help or just be different. Look again at alpha lambda' M_t theta_t — the regularizer in the L2 version. M_t divides coordinate i by sqrt(v_hat_{t,i}), the running magnitude of that coordinate's gradient. So a weight whose gradients have historically been *large* sits on a *large* sqrt(v_hat), hence a *small* M_t entry, hence its L2 shrink alpha lambda' M_{t,i} theta_{t,i} is *scaled down*. The weights with the biggest gradients — exactly the ones a regularizer most wants to rein in, because small changes to them swing the function hardest — are the ones L2-in-Adam regularizes *least*. And the coordinates with tiny gradients get the *largest* relative shrink. To get any real decay onto the large-gradient weights I'd have to crank lambda' way up, which then crushes the small-gradient weights. The adaptive normalization, applied to the regularizer, defeats the regularizer. That matches what the tuning sweeps show — Adam's best result with nonzero L2 is barely better than with no L2 at all; the penalty is mostly spinning its wheels.

The fix writes itself once I see the disease: keep lambda theta *out* of the preconditioner. Don't feed the regularizer through M_t. Decay the weights directly, as a separate step, exactly the way Hanson-and-Pratt defined it in the first place:

  theta_t = theta_{t−1} − alpha m_hat_t / (sqrt(v_hat_t) + eps) − lambda theta_{t−1}.

The Adam adaptive step is untouched; the decay term lambda theta_{t−1} stands outside the sqrt(v_hat) normalization, so every weight decays by the same factor (1 − lambda) per step regardless of its gradient history. That's the whole modification — decouple the decay from the gradient-based update. The large-gradient weights now actually get decayed.

I want to also account for learning-rate scheduling cleanly, because there's a subtlety that, if I miss it, silently re-couples the two knobs I just worked to separate. Under L2, the decay term rode *inside* the gradient, so whenever I scheduled the learning rate — say a global per-step multiplier eta_t — the decay got multiplied by eta_t for free, because everything in the gradient-based step is multiplied by eta_t. Now that I've pulled the decay *out* of the gradient step, it would no longer be scheduled unless I attach eta_t to it by hand. If I forget, then as the learning-rate schedule anneals eta_t down, the gradient step shrinks but the decay stays full-strength, so their *relative* sizes drift over the run — which is exactly the kind of hidden coupling I'm trying to kill. So I schedule both: introduce eta_t from a user-supplied SetScheduleMultiplier(t), and apply it to both the adaptive step and the decay,

  theta_t = theta_{t−1} − eta_t ( alpha m_hat_t / (sqrt(v_hat_t) + eps) + lambda theta_{t−1} ).

The same reasoning gives the SGD analogue for free — pull lambda theta out of the momentum gradient and decay separately, eta_t-scheduled — call it SGDW; it's the sanity-check sibling, and it should be *identical in behavior* to ordinary L2-SGD up to the lambda' = lambda/alpha reparameterization, which is a nice consistency test: my change must not alter SGD (where L2 and decay already agree), only Adam.

Now I should check that decoupling buys the hyperparameter-separation I claimed, not just a different shrink. The diagonal SGD basin lived on lambda' = lambda/alpha — change alpha and you must change lambda' or you fall off. With decoupled decay, lambda multiplies theta directly and never gets divided by alpha or by M_t; alpha governs the gradient step, lambda governs the shrink, and the two enter the update through separate, non-interacting terms. So I'd expect the best-setting basin in the (alpha, lambda) plane to straighten out from a diagonal into something axis-aligned: I can fix a not-yet-perfect alpha, tune lambda alone, and land near-optimal — and vice versa. That's a real practical payoff on top of the generalization fix: tuning two coupled knobs is much harder than two separate ones.

Let me push on the *why does decoupled decay generalize better* question with something more than "it actually regularizes the big weights now," because I'd like an equivalent-loss picture even if it's only exact in a simplified case. Take the cleanest tractable case: an adaptive method with a *fixed* (time-independent) diagonal preconditioner M = diag(s)^{-1}, s_i > 0 — think of s_i as a frozen "typical gradient magnitude" for coordinate i. Run it with decoupled weight decay lambda and base rate alpha. Its iterate is

  theta_{t+1} = (1 − lambda) theta_t − alpha grad f_t(theta_t)/s = theta_t − alpha grad f_t(theta_t)/s − lambda theta_t,

division by s elementwise. Now I ask: is there an L2-style regularized loss the *same* fixed-preconditioner method, run *without* decay, would follow identically? Guess a *scale-adjusted* penalty — not ||theta||^2 but ||theta ⊙ sqrt(s)||^2 — and see what falls out. Let f^sreg_t(theta) = f_t(theta) + (lambda'/2)||theta ⊙ sqrt(s)||_2^2. Its gradient is grad f_t(theta) + lambda' (theta ⊙ s) (because d/dtheta of (lambda'/2) sum_i s_i theta_i^2 is lambda' s_i theta_i). The preconditioned step without decay is

  theta_{t+1} = theta_t − alpha (grad f^sreg_t(theta_t))/s = theta_t − alpha grad f_t(theta_t)/s − alpha lambda' (theta_t ⊙ s)/s = theta_t − alpha grad f_t(theta_t)/s − alpha lambda' theta_t,

where the s in (theta ⊙ s)/s cancels — that cancellation is exactly why I guessed the sqrt(s) weighting, so that one factor of s in the penalty's gradient survives the division by the preconditioner. Compare to the decoupled-decay iterate: identical iff alpha lambda' = lambda, i.e. lambda' = lambda/alpha. So decoupled weight decay on this fixed-preconditioner method is *exactly* L2 regularization on the scale-adjusted loss f_t(theta) + (lambda'/2)||theta ⊙ sqrt(s)||^2, equivalently f_t(theta) + (lambda/(2 alpha))||theta ⊙ sqrt(s)||^2 when I write it directly in terms of the per-step decay lambda. And now I can read the regularization off: it penalizes theta_i in proportion to sqrt(s_i). Coordinates with historically large gradients (large s_i) are penalized *more*, not less — the precise opposite of what L2-in-Adam does. I have to be honest that this is a fixed-preconditioner caricature: real Adam changes M_t every step, so this isn't an exact statement about practical Adam. But it pins down the intuition cleanly — decoupling doesn't just "un-weaken" the regularizer, it tilts it toward shrinking the brittle, large-gradient weights harder, which is plausibly *why* the decoupled solution generalizes better.

The same conclusion shows up from a completely different angle: viewing adaptive optimization as Bayesian filtering. Cast training as tracking, by filtering, a distribution over the optimal value of each theta_i given the others. With a Gaussian state-transition prior and an approximately-conjugate likelihood, the filtering mean updates as mu_post = mu_prior + Sigma_post g, where g is the mini-batch gradient — so the gradient preconditioner *is* the posterior covariance Sigma_post: bigger steps where we're more uncertain. Adam and RMSProp are special cases of exactly this. Now where does regularization enter? In the state-transition prior, P(theta_{t+1} | theta_t) = N((I − A) theta_t, Q): A is the term that keeps weights from drifting to infinity over time. Set A = lambda I and the mean is multiplied by (1 − lambda) every step — that is decoupled weight decay, verbatim. And crucially this (1 − lambda) acts on the *mean / the prior* and does *not* depend on the per-parameter uncertainty. L2 regularization, by contrast, is a Gaussian prior on the weights whose log-prob enters as a gradient term, so it *does* get folded through the uncertainty (the preconditioner) — and a prior gets overwhelmed as data accumulates, so its effective decay would *vanish* over a long run, whereas empirically a positive decay stays useful no matter how long I train. So filtering says: weight decay is what emerges straightforwardly; L2 is the thing that doesn't quite fit. Two different routes — the equivalent-loss algebra and the filtering prior — point at the same conclusion, which makes me trust it.

Now a wrinkle the experiments keep throwing at me: the *best* lambda is not stable across training budgets. The longer I train (more batch passes), the smaller the optimal weight decay. That makes mechanical sense — the *total* shrink accumulated over a run is roughly (number of updates) × (per-step decay), and the number of updates is (B/b)·T for batch size b, training-set size B, and T epochs. If what actually matters for generalization is the *total* decay over the run rather than the per-step rate, then to hold total decay fixed while the budget changes I'd scale per-step lambda inversely with the number of updates, lambda ∝ b/(B T). I don't want to claim that exact inverse is right — it's a hypothesis about what's invariant — so let me reparameterize lambda through a budget-normalized knob and let the data pick the exponent. Write

  lambda = lambda_norm · sqrt( b / (B T) ),

so the user sets lambda_norm — the decay you'd use if you only got a single batch pass — and the actual per-step lambda is computed from the budget. The square root is the compromise that transferred best in practice: with it, the best lambda_norm found on a short CIFAR-10 run stays near-best on much longer runs and even carries over to ImageNet32x32 (where an epoch is ~24× longer), whereas the raw lambda would have been several times too large there. I won't oversell the exponent — the durable lesson is that *some* budget normalization makes the decay setting transfer; sqrt is the one I'd ship.

There's one more piece I can now pick up that I couldn't before. A global learning-rate schedule — cosine annealing with warm restarts, the SGDR recipe: cool eta_t along a cosine, then periodically reset it to the top while keeping the weights — is known to help SGD's final and anytime performance a lot. Folklore says don't bother with it for Adam, since Adam already adapts per-parameter rates. But per-parameter adaptation is not the same as a *global* schedule of the overall step magnitude, and I already have eta_t threaded through the update, so a cosine schedule on eta_t costs nothing to try. Concretely, within run i of length T_i epochs,

  eta_t = eta_min^(i) + 0.5 (eta_max^(i) − eta_min^(i)) (1 + cos(pi T_cur / T_i)),

and for the simple range [0,1] this is eta_t = 0.5 + 0.5 cos(pi T_cur / T_i): it starts at 1, cosine-decays to 0 by the end of the run, then a warm restart sets T_cur back to 0 (jumping eta_t back to 1) while keeping the current theta, and the next run gets a longer budget T_{i+1} = T_i · T_mult. T_cur is updated every batch so it isn't integer-constrained, and the solutions to keep are the ones right before each restart where eta_t = 0. I tried giving Adam warm restarts before and it improved anytime behavior but never caught SGD's warm restarts — and now I understand exactly why: the L2 weight decay was the weak link, so the restart variant inherited the same crippled regularization. With the decay decoupled (and budget-normalized so a single lambda_norm works across the geometrically-growing restart lengths), the SGDR recipe carries straight over. That gives the full practical optimizer: the decoupled-decay Adam step, with eta_t following the cosine-restart schedule and lambda computed per step from lambda_norm.

Let me make sure the defaults are inherited honestly and I'm not silently changing Adam. alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8 — all straight from Adam, because I deliberately left the adaptive step alone; the *only* new degree of freedom is lambda (or lambda_norm). Set lambda = 0 and the update collapses back to ordinary Adam exactly — there's no residual change, which is the cleanest possible evidence that decoupling is a strict, surgical addition rather than a reformulation of Adam's core.

Before I write the loop, let me reconcile two equivalent ways to apply the decoupled decay, because the implementation I ship and the equation I wrote look slightly different and I want to know they agree. I wrote the decay as an additive term *after* the gradient step: theta ← theta − eta_t alpha m_hat/(sqrt(v_hat)+eps) − eta_t lambda theta_old, using the *old* theta in the decay. The common tensor implementation instead does the decay *multiplicatively first*: theta ← (1 − eta_t lambda) theta, then theta ← theta − eta_t alpha m_hat/(sqrt(v_hat)+eps). Expand the multiplicative-first version: after the first line theta = theta_old − eta_t lambda theta_old, then the second line subtracts the Adam step, giving theta_old − eta_t lambda theta_old − eta_t alpha m_hat/(sqrt(v_hat)+eps). That's exactly the additive-after form with the decay taken on theta_old, provided the Adam step is computed from the same already-computed gradient and moment state. The Lua/Torch implementation can do the additive old-parameter version after `addcdiv`; PyTorch's optimizer can do the multiplicative version before the Adam moment/update math. If instead I applied the decay multiplicatively *after* the gradient step it would shrink theta_old−step rather than theta_old, differing by a term of order eta_t lambda × step. So I'll use the multiplicative-first form in code because it is one cheap in-place scale and it is exactly the same decoupled update.

So I fill the slot. Two state vectors per parameter for the moment EMAs — the Adam step, untouched — plus the one new line, the decoupled decay applied directly to the weights before the adaptive update, both scaled by eta_t:

```python
import math
import torch


def set_schedule_multiplier(t):
    """Global per-step multiplier eta_t. Fixed (1.0), or a learning-rate schedule;
    for warm restarts, cosine-decay then reset (see compute below)."""
    return 1.0


class AdamW:
    """Adam with DECOUPLED weight decay. The adaptive step is plain Adam; the
    weight decay acts directly on the weights, OUTSIDE the sqrt(v) normalization,
    with effective per-step shrink lr_t * weight_decay in this PyTorch-style API."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, t):
        beta1, beta2 = self.betas
        eta_t = set_schedule_multiplier(t)             # global schedule multiplier
        lr_t = eta_t * self.lr
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                                 # loss gradient ONLY (no lam*theta added)
            if g.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            state = self.state[id(p)]
            if len(state) == 0:
                state['step'] = 0
                state['m'] = torch.zeros_like(p)       # first-moment EMA
                state['v'] = torch.zeros_like(p)       # second-raw-moment EMA
            state['step'] += 1

            # decoupled weight decay: shrink the weights directly, kept OUT of the
            # adaptive normalization. PyTorch's API uses lr_t * weight_decay here.
            if self.weight_decay != 0:
                p.mul_(1 - lr_t * self.weight_decay)

            m, v = state['m'], state['v']
            m.mul_(beta1).add_(g, alpha=1 - beta1)             # m_t = beta1*m + (1-beta1)*g
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)      # v_t = beta2*v + (1-beta2)*g^2
            denom = v.sqrt().add_(self.eps)                    # sqrt(v_t) + eps
            bc1 = 1 - beta1 ** state['step']                   # 1 - beta1^t (bias corr.)
            bc2 = 1 - beta2 ** state['step']                   # 1 - beta2^t
            step_size = lr_t * math.sqrt(bc2) / bc1            # folded bias correction
            p.addcdiv_(m, denom, value=-step_size)             # plain Adam adaptive step
```

In this PyTorch-style API the effective per-step shrink is `lr_t * weight_decay`; in the derivation above, that product is the scheduled lambda. For the full practical variant — cosine annealing with warm restarts and budget-normalized decay — `set_schedule_multiplier` becomes the cosine `eta_t = 0.5 + 0.5*cos(pi*T_cur/T_i)` with the restart bookkeeping (reset `T_cur=0`, set `T_i *= T_mult`), and the per-step lambda is recomputed as `lambda_norm * sqrt(b/(B*T))`.

Let me retrace the causal chain so I'm sure it holds. I started from a concrete, reproducible gap: tuned Adam generalizes worse than tuned SGD-with-momentum on regularization-sensitive image tasks, even at matched training loss, with no mechanism identified. I looked at how Adam is regularized and re-derived that "L2" and "weight decay" coincide for SGD only because of the lambda' = lambda/alpha reparameterization. Carrying that derivation to an adaptive method, the shrink-term match would require lambda I = alpha lambda' M_t, which is impossible for any nonconstant diagonal preconditioner — so L2 and weight decay are *different methods* for Adam, and L2-in-Adam runs the regularizer through the same per-coordinate sqrt(v_hat) that normalizes the gradient, scaling decay *down* exactly on the large-gradient weights a regularizer most wants to shrink. The fix is to keep lambda theta outside the normalization: decouple the decay back to its original separate-step form, schedule it with eta_t so I don't re-couple it to the learning rate, and the result both regularizes the right weights and straightens the (alpha, lambda) tuning basin from a diagonal into axis-aligned. A fixed-preconditioner equivalent-loss derivation shows decoupled decay = L2 on a scale-adjusted penalty that shrinks large-gradient weights *more*, and the Bayesian-filtering view shows decoupled decay falls out of the state-transition prior while L2 does not — two independent justifications. Budget-normalizing lambda by sqrt(b/(BT)) makes the setting transfer across run lengths and datasets, and once the decay is fixed the SGDR cosine-annealing-with-warm-restarts recipe — which had failed for Adam precisely because L2 was the weak link — carries over to give strong anytime performance. The change is one line on top of unmodified Adam, recovering exact Adam at lambda = 0.
