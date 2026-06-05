Let me start from what actually hurts when I train these models. I have a scalar objective f(theta) I can only ever see through noise. Sometimes the noise is minibatch subsampling — the objective is really a sum over examples, f(theta) = E_x[f_x(theta)], and I compute the gradient on a random handful, so each g_t is a noisy but unbiased draw of the true gradient of E[f(theta)] — and sometimes I inject the noise on purpose, like dropout, which randomly zeroes units so even the same example gives me a different gradient on every forward pass. The models are huge, millions of parameters, so anything that forms a Hessian or any matrix the size of theta-by-theta is dead on arrival: I can't store it, let alone invert it. I'm stuck at first order, and I'd better keep memory close to linear in the number of parameters. So the whole game is: out of a stream of noisy gradient vectors and almost no memory, manufacture a good per-coordinate step.

What do I do today? Plain SGD with a hand-picked learning rate alpha, maybe with momentum. Momentum I like the feel of: keep v_t = mu·v_{t-1} + g_t and step theta -= alpha·v_t. If I squint, v_t is an exponentially decaying running average of the gradient — an estimate of the *mean* gradient, the first moment, smoothed over recent steps. That's why it helps on two counts at once. It averages out the minibatch noise, so a single bad draw doesn't yank me around; and it builds up speed along directions that are consistently downhill while cancelling the back-and-forth oscillation across a narrow ravine, because the sideways components alternate sign and average to near zero while the downhill component accumulates. And Sutskever, Martens, Dahl & Hinton showed that a carefully scheduled momentum — ramp it up, then ease it back down near the end — can rival Hessian-free second-order methods, which tells me two things: momentum is genuinely part of the answer, and the *schedule* near the end of training matters, the coefficient wants to come down as I converge. But here's what momentum does *not* give me: every coordinate shares the one global alpha. If one parameter has gradients of size 100 and another of size 0.001, momentum smooths each but steps them at the same scale. I keep having to babysit alpha, and worse, the "right" alpha differs across layers — the weight-shared convolutional layers want something different from the dense layers, because weight sharing means a conv filter's gradient is summed over every spatial location it touches, so it's systematically larger. One global rate is structurally wrong, not just inconvenient. I want per-parameter step sizes.

So who solves per-parameter scaling? AdaGrad, by Duchi, Hazan and Singer. Its move is to divide each coordinate's step by the square root of the running *sum* of its squared gradients,

  theta_{t+1,i} = theta_{t,i} − alpha · g_{t,i} / sqrt( Σ_{s=1}^{t} g_{s,i}^2 ).

I love what this does to sparse features. A feature that almost never fires has a tiny accumulated denominator, so when it finally does fire it gets a big step — exactly the needle-in-the-haystack behavior I want in high-dimensional bag-of-words problems, where most coordinates are zero on most examples and the rare nonzero gradient carries almost all the information. And it comes with theory: in the online-convex setting AdaGrad gets O(sqrt(T)) regret, and for sparse data the adaptive bound can be O(log d · sqrt(T)) instead of O(sqrt(d·T)). That's real, that's a provable win, not a heuristic. But stare at the denominator: it's a *sum that only grows*. Every g^2 it ever sees gets added and never removed. Run long enough and sqrt(Σ g^2) climbs without bound, the effective per-coordinate learning rate alpha/sqrt(Σ g^2) decays monotonically toward zero, and the optimizer simply grinds to a halt — not because it's near an optimum, but because the denominator ate the step. For a convex problem with a fixed, stationary objective maybe that built-in annealing is even desirable. But my objectives are *non-stationary*: minibatch noise, curvature that drifts as the weights move, dropout reshuffling the effective network every pass. On those, the relevant gradient scale *now* has nothing to do with the gradient scale ten thousand steps ago, yet AdaGrad's sum weights them equally and lets the ancient ones strangle the present. I cannot let the learning rate die while the problem is still moving. Wall.

The fix for that is already on the table: RMSProp, from Tieleman and Hinton's lecture. The disease is "a sum that never forgets," so cure it by forgetting — replace the ever-growing sum with an exponential moving average of the squared gradient,

  v_t = beta_2 · v_{t-1} + (1 − beta_2) · g_t^2,   theta_t = theta_{t-1} − alpha · g_t / ( sqrt(v_t) + eps ).

This is the right structural move, and I want to be precise about *why* it's right and not just "it works." Unroll it: v_t = (1−beta_2)Σ_{i=1}^t beta_2^{t−i} g_i^2, so the weight on a gradient from k steps back is (1−beta_2)·beta_2^{k}, which decays geometrically. The effective window — the number of recent squared gradients that meaningfully contribute — is on the order of 1/(1−beta_2). So instead of "all of history, equally weighted" (the sum, whose effective denominator grows like t), I have "roughly the last 1/(1−beta_2) gradients." The denominator now tracks the *recent* gradient scale and stays O(1) instead of climbing, so it can't strangle itself, and because it forgets, it adapts when the objective is non-stationary. Graves even tried a momentum variant where you put a momentum term on the rescaled gradient. So let me say the obvious thing out loud: I want AdaGrad/RMSProp's per-parameter scaling *and* momentum's smoothed direction at the same time, and the moment view tells me they're not even competing — one is a first-moment estimate, the other a second-moment estimate. Keep both EMAs,

  m_t = beta_1 · m_{t−1} + (1 − beta_1) · g_t,
  v_t = beta_2 · v_{t−1} + (1 − beta_2) · g_t^2,

and step theta_t = theta_{t−1} − alpha · m_t / ( sqrt(v_t) + eps ). The numerator is a smoothed gradient (first moment, the mean direction), the denominator a smoothed magnitude (second raw moment, the uncentered variance, i.e. RMS gradient per coordinate). Notice I wrote both EMAs in the "(1−beta)" normalized form, m_t = beta·m + (1−beta)·g, rather than momentum's un-normalized v = mu·v + g. That's deliberate: with the (1−beta) factor the weights (1−beta)Σbeta^{k} sum to (almost) one, so m_t is a genuine *weighted average* of past gradients with the units of a gradient, not a gradient times some accumulation factor. That matters because I'm about to divide m by sqrt(v) and I want the ratio to be a clean dimensionless-ish quantity, not contaminated by a stray 1/(1−beta) scale.

And there's something lovely about that ratio m_t / sqrt(v_t) I should pin down right now: it's scale-invariant. If I rescale all gradients by a constant c — say I multiply the loss by c, or just change units — then m_t scales by c (it's linear in g) and v_t scales by c^2 (it's quadratic in g), so sqrt(v_t) scales by c, and (c·m_t)/sqrt(c^2·v_t) = m_t/sqrt(v_t). The update doesn't care about the overall magnitude of the gradient, only its direction-to-spread ratio. I'd kill for that property, because it means alpha sets a step size in *parameter* space, completely decoupled from the arbitrary magnitude of the loss. This is also exactly the natural-gradient flavor: dividing by sqrt(v) is preconditioning by (the square root of) a diagonal approximation to the Fisher information, since the expected outer product of the gradient with itself is the Fisher, and v estimates its diagonal. It's more conservative than true natural gradient — I take the square root of the inverse diagonal rather than the full inverse — but that conservatism is welcome when the estimate is this noisy. So the structure is: a diagonal preconditioner I can actually afford, built from two EMA vectors.

But let me actually simulate the first few steps in my head, because initialization has burned me before. I init m_0 = 0 and v_0 = 0 — what else, I have no gradients yet, zero is the only honest prior. Take the very first step, t = 1, with beta_2 = 0.999, which I *want* close to 1 so the second-moment estimate is smooth and reliable. Then v_1 = beta_2·0 + (1−beta_2)·g_1^2 = 0.001·g_1^2. So v_1 is a thousand times smaller than g_1^2, and sqrt(v_1) ≈ 0.0316·|g_1| — way smaller than the true gradient magnitude. The denominator is far too small, so my first step is enormous and points who-knows-where. The numerator has the same disease: m_1 = (1−beta_1)·g_1 = 0.1·g_1 with beta_1 = 0.9, ten times too small. Both are shrunk, but not by the *same* factor — m by 0.1, sqrt(v) by sqrt(0.001) ≈ 0.0316 — so the ratio m_1/sqrt(v_1) ≈ 0.1·g_1 / (0.0316·|g_1|) ≈ 3.16·sign(g_1), about three times the size of a "normal" step, on a single noisy gradient I have no business trusting. This is exactly the divergence I've seen with RMSProp when beta_2 is pushed near 1 with no correction. So bolting them together naively reproduces a known failure. Wall. I need to understand precisely *how* biased these EMAs are toward zero so I can undo it, not just damp it with a smaller alpha — because shrinking alpha to survive the first step would cripple all the later steps.

Let me unroll the second-moment recursion all the way back to the zero initialization. With v_0 = 0,

  v_t = (1 − beta_2) · Σ_{i=1}^{t} beta_2^{t−i} · g_i^2.

Sanity check: v_1 = (1−beta_2)·g_1^2 ✓; v_2 = beta_2·v_1 + (1−beta_2)g_2^2 = (1−beta_2)(beta_2 g_1^2 + g_2^2) ✓. Good, that's the closed form. Now I want to know what this estimator equals *on average*, so I can compare it to the thing it's supposed to estimate, the true second moment E[g_t^2]. Take expectations:

  E[v_t] = E[ (1 − beta_2) Σ_{i=1}^{t} beta_2^{t−i} g_i^2 ].

I can write each E[g_i^2] as E[g_t^2] plus a discrepancy. If the second moment were perfectly stationary, E[g_i^2] = E[g_t^2] for all i and I could factor it cleanly out of the sum; in general it drifts, so I write

  E[v_t] = E[g_t^2] · (1 − beta_2) Σ_{i=1}^{t} beta_2^{t−i} + zeta,

where zeta absorbs the mismatch, zeta = (1−beta_2) Σ_i beta_2^{t−i} (E[g_i^2] − E[g_t^2]). The geometric sum is the whole point: Σ_{i=1}^{t} beta_2^{t−i} = 1 + beta_2 + ... + beta_2^{t−1} = (1 − beta_2^t)/(1 − beta_2). Multiply by (1−beta_2) and it collapses to exactly (1 − beta_2^t). So

  E[v_t] = E[g_t^2] · (1 − beta_2^t) + zeta.

There it is in black and white. My EMA, on average, is the true second moment times a factor (1 − beta_2^t), plus a small zeta. And zeta really can be kept small: if the objective is stationary it's exactly zero, and if it drifts, the term E[g_i^2] − E[g_t^2] is only large for old i where the underlying second moment was different — but those old i carry weight beta_2^{t−i}, which is tiny precisely when beta_2 is chosen so the EMA already forgets the distant past, which is the regime I want anyway. So the dominant, *structural* bias is that multiplicative (1 − beta_2^t), and it's not an artifact of any particular gradient sequence — it's there even in expectation, baked in by starting the average from zero. At t = 1 that factor is (1 − beta_2) = 0.001 — a thousand times too small, matching the disaster I simulated. As t grows, beta_2^t → 0 and the factor → 1, so the bias is purely an early-training transient, which is precisely when I saw the blow-ups.

The correction is now forced on me, not invented: divide it out. Define

  v_hat_t = v_t / (1 − beta_2^t),

so that E[v_hat_t] ≈ E[g_t^2] with no leading-order bias. And the identical argument on m_t — same unrolling, same geometric sum but with beta_1 — gives E[m_t] = E[g_t]·(1 − beta_1^t) + drift, so I set

  m_hat_t = m_t / (1 − beta_1^t).

Now I should resist the urge to wave this off as a small early-iteration nicety, because the algebra is telling me exactly the opposite. Look at when (1 − beta_2^t) hurts most: when beta_2 is close to 1, i.e. when (1−beta_2) is small, because then 1 − beta_2^t ≈ (1−beta_2)·t stays far below 1 for many steps. And small (1−beta_2) is exactly what I'm *forced* into for sparse gradients: to get a reliable second-moment estimate out of a coordinate that fires once in a thousand steps, I have to average over a long window, i.e. push beta_2 toward 1. So the case where I most need a smooth, low-variance estimate — sparse, near-1 beta_2 — is the very case where skipping bias correction gives me a near-zero denominator and a gigantic, divergent first step. That *is* the RMSProp instability, now derived from the algebra rather than observed as a symptom. The bias correction isn't cosmetic; it's the thing that makes the near-1 beta_2 regime usable at all, and it's exactly the term RMSProp and AdaDelta don't have. I can also see why the correction *must* be on both moments and not just the denominator: if I de-biased only v, the m̂/√v̂ ratio at t=1 would be 0.1·g / |g| = 0.1·sign(g) — now ten times too *small* — so I'd be trading one mismatch for another. Correcting both restores the ratio to ≈ sign(g) at t=1, the honest "one noisy sample, take a unit-ish step" behavior I want.

So my update becomes theta_t = theta_{t−1} − alpha · m_hat_t / ( sqrt(v_hat_t) + eps ). Let me look hard at the step it takes, because I want the geometry, not just the formula. Set eps = 0 for a moment. The effective step is

  Delta_t = alpha · m_hat_t / sqrt(v_hat_t).

How big is it? m_hat_t and v_hat_t are built from the same gradient stream, so the ratio self-normalizes. In the common case where the gradients are reasonably consistent, m_hat is roughly the mean gradient E[g] and sqrt(v_hat) roughly the root-mean-square sqrt(E[g^2]), so |m_hat/sqrt(v_hat)| ≈ |E[g]|/sqrt(E[g^2]) ≤ 1 always — the mean magnitude never exceeds the RMS, that's just Jensen / Cauchy–Schwarz — so |Delta_t| ⪅ alpha. At the other extreme, the most severe sparsity, a coordinate whose gradient was zero at every step except this one, the corrected ratio can exceed 1. Let me nail the two bounds. From the closed forms, m_hat_t = (1−beta_1)Σ_{k} beta_1^{t−k} g_k / (1−beta_1^t) and v_hat_t = (1−beta_2)Σ_k beta_2^{t−k} g_k^2 / (1−beta_2^t). In the single-nonzero-gradient case (only g_t ≠ 0) these become m_hat_t = (1−beta_1)g_t/(1−beta_1^t) and sqrt(v_hat_t) = sqrt(1−beta_2)·|g_t|/sqrt(1−beta_2^t), so |Delta_t/alpha| = (1−beta_1)/(1−beta_1^t) · sqrt(1−beta_2^t)/sqrt(1−beta_2) ≤ (1−beta_1)/sqrt(1−beta_2) (the t-factors are each ≤ 1 over their counterparts). So:

  |Delta_t| ≤ alpha · (1−beta_1)/sqrt(1−beta_2)   in the case (1−beta_1) > sqrt(1−beta_2),
  |Delta_t| ≤ alpha                                otherwise.

With defaults beta_1 = 0.9, beta_2 = 0.999: (1−beta_1) = 0.1 and sqrt(1−beta_2) = sqrt(0.001) ≈ 0.0316, so the first case holds and the bound is alpha·0.1/0.0316 ≈ 3.16·alpha — but that ceiling triggers only in the pathological one-nonzero-gradient regime; in normal use the ratio sits below 1 and |Delta_t| ⪅ alpha.

This reframes what alpha *is*, and it's the cleanest payoff of the whole construction. Because |Delta_t| ⪅ alpha regardless of gradient magnitude, alpha is an approximate cap on how far any single parameter moves in one step — a *trust region*. I read it as: I trust my current gradient estimate to guide a move of at most about alpha in parameter space, beyond which the gradient just doesn't tell me enough to commit. That makes alpha far easier to set than a raw gradient-scaled rate, because I often *do* have a prior on how far good optima sit from initialization — I can put a rough prior distribution on the parameters, see how wide it is, and pick alpha so I can traverse that range in a sensible number of iterations. The step size is decoupled from the loss scale and re-coupled to the geometry of parameter space, which is the thing I actually have intuition about.

And there's a second gift hiding in m_hat/sqrt(v_hat): read it as a signal-to-noise ratio, mean gradient (signal) over RMS gradient (signal plus noise). Far from an optimum the gradient points consistently one way, mean ≈ RMS, the ratio is near 1, so I take near-full steps of size alpha. As I approach an optimum the true gradient shrinks toward zero while its *variance* does not — the minibatch and dropout noise floor stays — so the mean-to-RMS ratio drops, the SNR drops, and Delta_t automatically shrinks toward zero. That's automatic annealing, built into the dynamics rather than scheduled by hand. And it's the *right* annealing: a small SNR *should* mean small steps, because that's exactly the situation where I'm least sure the direction of m_hat agrees with the true gradient. The method polices its own caution.

Now the eps in sqrt(v_hat) + eps. Until now I set it to zero to see the structure; let me put it back and ask what it's actually for. The hard job is preventing a divide-by-zero: if a coordinate's recent gradients are all zero, v_hat collapses to zero and the bare step alpha·m_hat/sqrt(v_hat) is 0/0 or blows up. A small additive eps in the denominator floors it. But it does something subtler too: when v_hat is tiny but nonzero, the step would be enormous (the scale-invariant ratio amplifies a tiny denominator without limit); eps caps that, because once sqrt(v_hat) ≪ eps the step becomes ≈ alpha·m_hat/eps, a plain momentum step with a fixed scale rather than an exploding one. So eps quietly trades a sliver of scale-invariance — strictly, the ratio is only scale-invariant while sqrt(v_hat) ≫ eps — for a guarantee that the denominator never detonates. That tells me how to size it: small enough that in the normal regime (sqrt(v_hat) of order the gradient RMS, which for typical losses is well above 1e-8) it's negligible and invariance holds, but nonzero so the degenerate coordinates are tamed. 1e-8 is comfortably below any gradient magnitude I expect to see in practice, so it floors the pathological coordinates without perturbing the healthy ones. I'll keep it as a small constant, default 1e-8.

Now, can I recover the methods I came from as special cases? That's the test of whether I've found the right generalization rather than just a third option. Set beta_1 = 0: the first moment vanishes, m_t = g_t, the numerator is the gradient directly — that's RMSProp territory, except I still carry the bias correction RMSProp lacks. Push further: take beta_2 → 1 from below with (1−beta_2) infinitesimal, and watch v_hat. From v_t = (1−beta_2)Σ beta_2^{t−i} g_i^2 divided by (1−beta_2^t): as beta_2 → 1 the weights beta_2^{t−i} → 1 and the normalizer (1−beta_2^t) ≈ (1−beta_2)·t, so v_hat_t → (1−beta_2)Σ g_i^2 / ((1−beta_2)t) = (1/t)Σ_{i=1}^t g_i^2, the *uniform average* of squared gradients. Then if I also replace alpha by an annealed alpha·t^{−1/2}, the update becomes theta − alpha·t^{−1/2}·g_t / sqrt(t^{−1}Σ g_i^2) = theta − alpha·g_t/sqrt(Σ_{i=1}^t g_i^2). That is exactly AdaGrad. So AdaGrad falls out with beta_1 = 0, infinitesimal (1−beta_2), and an annealed alpha — and crucially this correspondence only holds *with* the bias correction: strip it, and beta_2 → 1 sends the bias factor (1−beta_2^t) → 0 without bound, so the uncorrected v → 0 and the updates would be infinitely large. The bias correction is what makes the whole one-parameter family from AdaGrad to RMSProp coherent. The picture clicks shut: momentum gave me the first moment, RMSProp/AdaGrad the second, and the missing piece nobody had written down was the de-biasing that lets me crank beta_2 to where sparsity needs it.

The default constants have to follow from the structure, not be pulled from a hat. The decay rates set EMA windows of about 1/(1−beta) samples. For the first moment, I want enough smoothing to kill minibatch noise but a short enough window that the direction stays *responsive* as the loss landscape changes — beta_1 = 0.9 averages on the order of ten gradients, which is the same regime well-tuned momentum already lives in, so it's the natural carry-over. For the second moment I need a *longer* window, beta_2 = 0.999 averaging on the order of a thousand: a variance/magnitude estimate is intrinsically noisier than a mean estimate because squaring amplifies the spread, and the denominator is what I'm dividing by, so a jittery denominator directly jitters every step. The asymmetry beta_2 ≫ beta_1 isn't arbitrary; it's "estimate the thing in the denominator more carefully than the thing in the numerator." alpha = 0.001 follows from the trust-region reading: it's a deliberately conservative cap on per-step parameter movement, small enough to be safe across the models I care about while the SNR annealing handles fine-tuning near the optimum. And eps = 1e-8 is the divide-by-zero floor, sized well below any healthy gradient RMS as argued above.

Let me write the algorithm as a loop, because I want it trivially droppable into the same minibatch training code I already use:

  m_0 = 0, v_0 = 0, t = 0
  while not converged:
    t += 1
    g_t = ∇_theta f_t(theta_{t−1})
    m_t = beta_1·m_{t−1} + (1−beta_1)·g_t
    v_t = beta_2·v_{t−1} + (1−beta_2)·g_t^2          # elementwise square
    m_hat = m_t/(1−beta_1^t)
    v_hat = v_t/(1−beta_2^t)
    theta_t = theta_{t−1} − alpha·m_hat/(sqrt(v_hat)+eps)

Two extra vectors the size of theta, a handful of elementwise ops, first-order only — memory and compute linear in the parameters, which clears the GPU constraint that killed the quasi-Newton route. One efficiency note I can fold in almost all the way: instead of materializing m_hat and v_hat explicitly each step, push both corrections into the scalar step size. Define alpha_t = alpha·sqrt(1−beta_2^t)/(1−beta_1^t). If eps is zero, then alpha_t·m_t/sqrt(v_t) = alpha·[sqrt(1−beta_2^t)/(1−beta_1^t)]·m_t/sqrt(v_t) = alpha·[m_t/(1−beta_1^t)] / [sqrt(v_t)/sqrt(1−beta_2^t)] = alpha·m_hat/sqrt(v_hat) exactly. With eps present, exact algebra would put the denominator at sqrt(v_t)+eps·sqrt(1−beta_2^t). The practical tensor update I want keeps the PyTorch-style denominator sqrt(v_t)+eps after the raw square root, so the folded form is exact for the bias-correction part and uses a fixed numerical floor rather than the time-scaled eps. Since eps is deliberately tiny, this is the right implementation tradeoff: one scalar step_size per iteration, two in-place EMA updates, one fused divide-and-add, and a stable denominator floor.

Now I want theory, because I'm claiming this is principled and not just three good ideas stapled together. The natural language is online convex optimization: an adversary hands me convex cost functions f_1,...,f_T one at a time; before seeing f_t I must commit to theta_t; I measure myself by regret against the best fixed point for the whole sequence,

  R(T) = Σ_{t=1}^{T} [ f_t(theta_t) − f_t(theta*) ],  theta* = argmin_theta Σ_t f_t(theta).

The gold standard is O(sqrt(T)) regret, because then average regret R(T)/T → 0 — asymptotically I do as well per-round as the best fixed point, which is the most one can ask in this adversarial setting. The entry point is convexity itself, and I'll state it as a lemma I'll lean on: a differentiable convex f lies above its tangent, i.e. for all x, y, f(y) ≥ f(x) + ∇f(x)^T(y − x). Rearranged with y = theta* and x = theta_t, f_t(theta_t) − f_t(theta*) ≤ ∇f_t(theta_t)^T(theta_t − theta*) = g_t^T(theta_t − theta*) = Σ_i g_{t,i}(theta_{t,i} − theta*_i). That converts the whole regret into a sum I can attack coordinate-by-coordinate using my actual update rule. So the plan: bound g_{t,i}(theta_{t,i} − theta*_i) using the update, sum over t and over coordinates i, and hope telescoping plus geometric series leave something growing no faster than sqrt(T). I'll assume bounded gradients ||∇f_t||_2 ≤ G, ||∇f_t||_∞ ≤ G_∞, bounded distances ||theta_t − theta*||_2 ≤ D and ||theta_m − theta_n||_∞ ≤ D_∞, a decaying learning rate alpha_t = alpha/sqrt(t), and — this is where the deep-momentum folklore re-enters as a hard requirement — a first-moment coefficient that *decays*, beta_{1,t} = beta_1·lambda^{t−1} with lambda just under 1.

I need two summation controls first. The first quantity that appears is

  S_i(T) = Σ_{t=1}^{T} |g_{t,i}|/sqrt(t).

I would like the very sharp control

  S_i(T) ≤ 2·G_∞·||g_{1:T,i}||_2,

because that is the summation shape that keeps the clean adaptive regret constant. Let me try the induction, since it looks tempting. Split off the last term:

  Σ_{t=1}^{T} sqrt(g_{t,i}^2/t) = Σ_{t=1}^{T−1} sqrt(g_{t,i}^2/t) + sqrt(g_{T,i}^2/T)
    ≤ 2G_∞||g_{1:T−1,i}||_2 + sqrt(g_{T,i}^2/T)
    = 2G_∞ sqrt( ||g_{1:T,i}||_2^2 − g_{T,i}^2 ) + sqrt(g_{T,i}^2/T),

using ||g_{1:T−1,i}||_2^2 = ||g_{1:T,i}||_2^2 − g_{T,i}^2. Start from the perfect-square identity

  ||g_{1:T,i}||_2^2 − g_{T,i}^2 + g_{T,i}^4/(4||g_{1:T,i}||_2^2)
    = (||g_{1:T,i}||_2 − g_{T,i}^2/(2||g_{1:T,i}||_2))^2,

so

  sqrt(||g_{1:T,i}||_2^2 − g_{T,i}^2)
    ≤ ||g_{1:T,i}||_2 − g_{T,i}^2/(2||g_{1:T,i}||_2).

If I substitute the bounded-gradient relaxation that would be needed for the neat induction, the leftover terms become

  -g_{T,i}^2/(sqrt(T)G_∞) + |g_{T,i}|/sqrt(T)
    = (|g_{T,i}|/sqrt(T))·(1 − |g_{T,i}|/G_∞).

That is nonnegative whenever 0 < |g_{T,i}| < G_∞, not nonpositive. So boundedness alone does not close this induction. The honest bound I can always prove is Cauchy-Schwarz:

  S_i(T) = Σ_{t=1}^{T} |g_{t,i}|/sqrt(t)
         ≤ (Σ_{t=1}^{T} g_{t,i}^2)^{1/2} · (Σ_{t=1}^{T} 1/t)^{1/2}
         ≤ ||g_{1:T,i}||_2 · sqrt(1 + log T).

That weaker bound is still sublinear once ||g_{1:T,i}||_2 ≤ G_∞sqrt(T), but it carries an extra sqrt(log T). For the clean O(sqrt(T)) constant, I have to carry the sharper summation control S_i(T) ≤ 2G_∞||g_{1:T,i}||_2 as an additional gradient-sequence condition. I will be explicit about which one I use.

The momentum terms need their own control. Define gamma = beta_1^2 / sqrt(beta_2), and check the defaults make gamma < 1: with 0.9, 0.999, gamma = 0.81/sqrt(0.999) ≈ 0.81 < 1 ✓. This will be the condition the whole bound hinges on. Under the sharper summation control, the bound I need is

  Σ_{t=1}^{T} m_hat_{t,i}^2 / sqrt(t · v_hat_{t,i}) ≤ (2G_∞/((1−gamma)^2 sqrt(1−beta_2))) · ||g_{1:T,i}||_2.

This controls the sum of squared *effective* steps. I'll use sqrt(1−beta_2^t)/(1−beta_1^t)^2 ≤ 1/(1−beta_1)^2 (numerator ≤ 1, denominator ≥ (1−beta_1)^2 since 1−beta_1^t ≥ 1−beta_1) and expand the moments in closed form:

  m_hat_{t,i} = (1−beta_1)Σ_{k=1}^t beta_1^{t−k} g_{k,i} / (1−beta_1^t),
  v_hat_{t,i} = (1−beta_2)Σ_{j=1}^t beta_2^{t−j} g_{j,i}^2 / (1−beta_2^t).

When I square the numerator, separate the past-gradient contributions, and lower-bound the denominator by the matching squared-gradient term, each old gradient g_{k,i} contributes with the geometric weight beta_1^{2(t−k)}/beta_2^{(t−k)/2} = gamma^{t−k}. The constants outside the sum are at most 1/sqrt(1−beta_2), so each t-term is bounded by

  m_hat_{t,i}^2/sqrt(t v_hat_{t,i})
    ≤ (1/sqrt(1−beta_2)) · Σ_{k=1}^{t} gamma^{t−k}|g_{k,i}|/sqrt(t).

Now swap the order of summation:

  Σ_{t=1}^{T} m_hat_{t,i}^2/sqrt(t v_hat_{t,i})
    ≤ (1/sqrt(1−beta_2)) · Σ_{k=1}^{T} |g_{k,i}| Σ_{t=k}^{T} gamma^{t−k}/sqrt(t).

Since t ≥ k, 1/sqrt(t) ≤ 1/sqrt(k). Let j = t−k, push the finite tail to infinity, and use the slightly looser arithmetic-geometric constant 1/(1−gamma)^2:

  Σ_{t=1}^{T} m_hat_{t,i}^2/sqrt(t v_hat_{t,i}) ≤ (1/((1−gamma)^2 sqrt(1−beta_2))) Σ_{t=1}^{T} |g_{t,i}|/sqrt(t),

and now the choice of summation control decides the displayed constant. With the sharper control, the whole block lands at (2G_∞/((1−gamma)^2 sqrt(1−beta_2)))||g_{1:T,i}||_2, the constant that feeds the clean regret bound. With only the unconditional Cauchy bound, the same line becomes (sqrt(1+log T)/((1−gamma)^2 sqrt(1−beta_2)))||g_{1:T,i}||_2. Either way the useful squeeze is the same: momentum could have made the effective steps pile up dangerously, but because beta_1^2/sqrt(beta_2) < 1 the contributions of old gradients decay geometrically. The sharper version gives the neat O(sqrt(T)) theorem; the unconditional version pays a logarithmic factor.

The regret sum now starts from convexity: R(T) ≤ Σ_t Σ_i g_{t,i}(theta_{t,i} − theta*_i). I express g_{t,i}(theta_{t,i} − theta*_i) via the update. Write the update with bias correction folded in and the first-moment recursion expanded,

  theta_{t+1,i} = theta_{t,i} − alpha_t·m_hat_{t,i}/sqrt(v_hat_{t,i})
              = theta_{t,i} − (alpha_t/(1−beta_1^t))·( beta_{1,t} m_{t−1,i}/sqrt(v_hat_{t,i}) + (1−beta_{1,t}) g_{t,i}/sqrt(v_hat_{t,i}) ).

Subtract theta*_i from both sides and square — the classic complete-the-square so distance-to-optimum telescopes across t. For coordinate i,

  (theta_{t+1,i} − theta*_i)^2 = (theta_{t,i} − theta*_i)^2
     − 2·(alpha_t/(1−beta_1^t))·( beta_{1,t} m_{t−1,i}/sqrt(v_hat_{t,i}) + (1−beta_{1,t}) g_{t,i}/sqrt(v_hat_{t,i}) )·(theta_{t,i} − theta*_i)
     + alpha_t^2·(m_hat_{t,i}/sqrt(v_hat_{t,i}))^2.

Solve for the piece I want, g_{t,i}(theta_{t,i} − theta*_i). Isolate the (1−beta_{1,t}) g_{t,i} term, rearrange, and tame the cross term carrying m_{t−1,i}(theta*_i − theta_{t,i}) with Young's inequality ab ≤ a^2/2 + b^2/2 (so the m_{t−1,i} factor goes into an m_{t−1,i}^2/sqrt(v_hat) term plus a (theta*−theta)^2 term). Use sqrt(v_hat_{t,i}) ≤ ||g_{1:t,i}||_2 — which holds because v_hat_{t,i} is a *weighted average* of the g_{j,i}^2 with weights summing to one, hence at most their plain sum, whose square root is ||g_{1:t,i}||_2 — and beta_{1,t} ≤ beta_1. After the dust settles the per-step bound is

  g_{t,i}(theta_{t,i} − theta*_i) ≤ (1/(2 alpha_t(1−beta_1)))·((theta_{t,i}−theta*_i)^2 − (theta_{t+1,i}−theta*_i)^2)·sqrt(v_hat_{t,i})
     + (beta_{1,t}/(2 alpha_t(1−beta_{1,t})))·(theta*_i − theta_{t,i})^2·sqrt(v_hat_{t,i})
     + (beta_1 alpha_{t−1}/(2(1−beta_1)))·m_{t−1,i}^2/sqrt(v_hat_{t−1,i})
     + (alpha_t/(2(1−beta_1)))·m_hat_{t,i}^2/sqrt(v_hat_{t,i}).

When I sum over t = 1..T and i = 1..d, the leading bracket wants to telescope: successive distances-to-optimum cancel, ((theta_{t,i}−theta*)^2 − (theta_{t+1,i}−theta*)^2), but the coefficients in front of those distances are not constant. The coefficient is a_{t,i} = sqrt(v_hat_{t,i})/alpha_t. I cannot just say a_{t,i} grows because v_hat_t is an EMA; if a new run of small squared gradients arrives, an EMA second moment can decrease. So the clean telescoping argument needs the explicit condition a_{t,i} ≥ a_{t−1,i}, i.e. sqrt(v_hat_{t,i})/alpha_t is nondecreasing for each coordinate. Under that condition, summation by parts leaves a head term D^2/(2 alpha(1−beta_1))·Σ_i sqrt(T·v_hat_{T,i}) plus boundary pieces bounded by D_∞^2. Without that monotone-preconditioner condition, extra variation terms remain and this particular bound does not follow. The last two lines are exactly the m_hat^2/sqrt(v_hat) sums that the second lemma bounds by constants times ||g_{1:T,i}||_2, with alpha_t = alpha/sqrt(t) supplying the 1/sqrt(t) the lemma's denominator wants; using the sharper summation control, they contribute the middle regret term alpha(1+beta_1)G_∞/((1−beta_1)sqrt(1−beta_2)(1−gamma)^2)·Σ_i ||g_{1:T,i}||_2. The momentum-penalty term, summed, carries beta_{1,t} = beta_1·lambda^{t−1}; the sum is Σ_t (beta_{1,t}/(1−beta_{1,t}))·sqrt(t)·(something bounded), and since beta_{1,t} ≤ beta_1 < 1,

  Σ_t (beta_{1,t}/(1−beta_{1,t}))·sqrt(t) ≤ Σ_t (1/(1−beta_1))·lambda^{t−1}·sqrt(t) ≤ (1/(1−beta_1))·Σ_t lambda^{t−1}·t ≤ (1/(1−beta_1))·(1/(1−lambda)^2),

using the same arithmetic-geometric bound Σ_t t·lambda^{t−1} ≤ 1/(1−lambda)^2. So this term is *constant in T*, contributing Σ_i D_∞^2 G_∞ sqrt(1−beta_2)/(2 alpha(1−beta_1)(1−lambda)^2). This is precisely why beta_1 must *decay*: with a constant beta_{1,t} = beta_1 there is no lambda^{t−1} factor, the sum Σ_t sqrt(t) is Θ(T^{3/2}) and the bound falls apart; the geometric lambda < 1 schedule is exactly what collapses it to a constant. The deep-momentum folklore — "reduce the momentum coefficient near the end of training" — turns out to be a hard requirement of the proof, not just an empirical knob.

Putting the clean-condition pieces together,

  R(T) ≤ D^2/(2 alpha(1−beta_1))·Σ_{i=1}^d sqrt(T·v_hat_{T,i})
       + alpha(1+beta_1)G_∞/((1−beta_1)sqrt(1−beta_2)(1−gamma)^2)·Σ_{i=1}^d ||g_{1:T,i}||_2
       + Σ_{i=1}^d D_∞^2 G_∞ sqrt(1−beta_2)/(2 alpha(1−beta_1)(1−lambda)^2).

Look at the T-dependence under those proof conditions. The third term is constant in T. In the first, sqrt(v_hat_{T,i}) ≤ ||g_{1:T,i}||_2 ≤ sqrt(T)·G_∞, so that term is O(sqrt(T)). In the second, ||g_{1:T,i}||_2 ≤ sqrt(T)·G_∞, also O(sqrt(T)). So R(T) = O(sqrt(T)), and average regret R(T)/T ≤ O(sqrt(T))/T = O(1/sqrt(T)) → 0 as T → ∞. If I fall back to the unconditional Cauchy summation bound instead of the sharper S_i(T) control, the middle term picks up sqrt(1+log T), so the average regret still vanishes but at the slightly weaker O(sqrt(log T / T)) rate. Better than just the rate: the clean bound is written in terms of Σ_i ||g_{1:T,i}||_2 and Σ_i sqrt(T·v_hat_{T,i}) rather than d·G_∞·sqrt(T), so when the data is sparse and those per-coordinate gradient norms are small, the bound is much smaller than the worst case. AdaGrad's sparsity advantage carries straight over — the same Σ_i ||g_{1:T,i}||_2 ≪ dG_∞sqrt(T) story.

A tangent, because I notice the L2 norm in the denominator — sqrt(v_hat) is an L2-flavored average of past gradient magnitudes — is just one choice; why 2? Generalize: scale each coordinate's step inversely to an Lp norm of its current and past gradients. Define a p-th-power EMA, writing the decay as beta_2^p so the algebra stays clean, v_t = beta_2^p·v_{t−1} + (1−beta_2^p)|g_t|^p, which unrolls to v_t = (1−beta_2^p)Σ_{i=1}^t beta_2^{p(t−i)}|g_i|^p, and step proportional to 1/v_t^{1/p}. Large finite p is numerically nasty — |g|^p overflows for big p and small gradients underflow — so a generic Lp isn't usable. But watch the limit p → ∞, where the norm should turn into a max. Let u_t = lim_{p→∞} (v_t)^{1/p}:

  u_t = lim_{p→∞} ( (1−beta_2^p) Σ_{i=1}^t beta_2^{p(t−i)} |g_i|^p )^{1/p}.

The prefactor (1−beta_2^p)^{1/p} → 1. The remaining ( Σ_{i=1}^t (beta_2^{t−i}|g_i|)^p )^{1/p} is an Lp norm of the sequence (beta_2^{t−i}|g_i|)_i, and as p → ∞ an Lp norm of a finite sequence converges to its max element:

  u_t = max( beta_2^{t−1}|g_1|, beta_2^{t−2}|g_2|, ..., beta_2|g_{t−1}|, |g_t| ).

And that has a beautifully simple recursive form: u_t = max(beta_2·u_{t−1}, |g_t|), with u_0 = 0. So in the infinity-norm limit the messy power-EMA collapses to "decay the running max, compare to the current magnitude" — one max, no powers, nothing to overflow. The step is theta_t = theta_{t−1} − (alpha/(1−beta_1^t))·m_t/u_t. The denominator does not need bias correction, and I can see exactly why. The whole bias problem came from a weighted *sum* starting at zero: averaging in zeros drags the early estimate below the true magnitude. But a max doesn't average; once any |g| is nonzero, u_t simply tracks the largest decayed magnitude seen, with no shrink-toward-zero artifact from the u_0 = 0 start (the zero just loses every max comparison the moment a real gradient arrives). So u needs no 1/(1−beta_2^t) factor — I only keep the 1/(1−beta_1^t) on the first moment. The step-size story is cleaner too, but I should not overstate it. Since u_t is the decayed max, every past gradient obeys |g_k| ≤ beta_2^{-(t−k)}u_t, so for beta_1 < beta_2,

  |m_t|/((1−beta_1^t)u_t) ≤ ((1−beta_1)/(1−beta_1^t)) · (1 − (beta_1/beta_2)^t)/(1 − beta_1/beta_2).

With beta_2 very close to 1 this envelope is essentially one, and with the default pair it is only a hair above one in an adversarial all-same-sign sequence. The honest conclusion is an alpha-scale bound without the two-case split of the L2 rule, not an exact universal cap for every possible gradient stream. A tidy sibling — AdaMax. Good default stepsize here is a bit larger, alpha = 0.002, since the u_t denominator is a max (an upper envelope) rather than an RMS, so it tends to be larger and the safe step can be a touch bigger. In code the only change from the main method is replacing the squared-gradient EMA with the elementwise max u_t = max(beta_2·u_{t−1}, |g_t|+eps) and dropping one bias-correction factor on the denominator.

One more practical thought: the last iterate of any stochastic method is noisy — it rattles around the optimum at the noise floor — and Polyak–Ruppert averaging of the iterates is known to improve stochastic-approximation convergence (Polyak & Juditsky 1992; Ruppert 1988; Moulines & Bach 2011 for the non-asymptotic story). I can fold an EMA of the parameters themselves — theta_bar_t = beta_2·theta_bar_{t−1} + (1−beta_2)·theta_t, de-biased by the same 1/(1−beta_2^t) — into the same loop with a single extra line, weighting recent iterates more heavily than a uniform Polyak average. Cheap, and worth checking whether the averaged iterate generalizes better.

So let me put the whole reasoning into the code I'd actually ship, filling the one empty slot in the optimizer harness — the per-parameter update rule. Two state vectors per parameter, the elementwise EMAs, the folded bias correction, the fused divide-and-add step:

```python
import math
import torch


class Optimizer:
    """The harness from before: owns per-parameter state, applies an update in step()."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad                            # g_t = grad of stochastic objective f_t
            state = self.state[id(p)]
            if len(state) == 0:                      # initialize m_0 = v_0 = 0, t = 0
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)      # m: first-moment EMA (momentum)
                state['exp_avg_sq'] = torch.zeros_like(p)   # v: second-raw-moment EMA (denom)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            t = state['step']

            if self.weight_decay != 0:               # optional L2 penalty folded into the gradient
                grad = grad.add(p, alpha=self.weight_decay)

            # m_t = beta1*m_{t-1} + (1-beta1)*g_t    (smoothed first moment)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t = beta2*v_{t-1} + (1-beta2)*g_t^2  (smoothed second raw moment, elementwise)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)        # sqrt(v_t) + eps  (floors zero-grad coords)

            bias_correction1 = 1 - beta1 ** t        # (1 - beta1^t): undoes m's zero-init bias
            bias_correction2 = 1 - beta2 ** t        # (1 - beta2^t): undoes v's zero-init bias
            # fold both corrections into one scalar:
            # PyTorch-style folded bias correction; exact for eps=0, with eps kept as a fixed floor.
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            # theta_t = theta_{t-1} - alpha_t * m_t / (sqrt(v_t) + eps)  (trust-region / SNR step)
            p.addcdiv_(exp_avg, denom, value=-step_size)
```

I now have the causal chain. I started stuck with SGD's single global learning rate on noisy, sparse, non-stationary gradients, knowing it's structurally wrong because different layers need different scales. Momentum gave me a smoothed first-moment direction that averages noise and damps oscillation, but no per-parameter scale. AdaGrad gave me per-parameter scale by dividing by accumulated squared gradients — provably great on sparse data — but its monotonically growing sum decays the effective learning rate to zero, killing learning on the non-stationary problems I care about. RMSProp fixed that by forgetting, with an EMA of squared gradients, but had no bias correction, so pushing beta_2 near 1 — which sparsity demands, to average a rare signal over a long window — biases the zero-initialized estimate toward zero and blows up the first steps. Unrolling the EMA and taking its expectation showed the bias is exactly the multiplicative factor (1−beta_2^t), so dividing by it de-biases the estimate and makes the near-1 beta_2 regime safe; the same argument with (1−beta_1^t) de-biases the first moment. The resulting ratio m_hat/sqrt(v_hat) turned out scale-invariant (so alpha is a parameter-space trust region, |Delta| ⪅ alpha), and a signal-to-noise ratio that self-anneals as the optimum nears. AdaGrad and RMSProp both reappear as limiting cases — beta_1 = 0 for the latter, infinitesimal (1−beta_2) plus annealed alpha for the former — confirming I found the right generalization rather than a fourth competitor. The decay-rate defaults follow from EMA window lengths (short, responsive window for the numerator; long, smooth window for the noisier denominator I divide by); eps is the divide-by-zero floor sized below any healthy gradient RMS. The online-convex regret analysis gives the clean O(sqrt(T)) bound when the sharper gradient-summation and monotone-preconditioner controls hold, and the unconditional Cauchy fallback still gives vanishing average regret with a logarithmic factor in this proof route. The infinity-norm limit of the same construction collapses the power-EMA into a clean running max that needs no bias correction on that denominator and gives an alpha-scale envelope. And the whole thing drops into the standard minibatch training loop as two EMA vectors and a handful of elementwise ops, first-order, memory linear in the parameters.
