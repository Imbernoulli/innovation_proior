Let me start from what actually hurts. I have a stochastic policy `π_θ(a|s)` — a diagonal Gaussian whose mean comes out of an MLP, with a learned log-standard-deviation — and I want to maximize the discounted return `η(θ) = E[Σ_t γ^t r_t]` by climbing a Monte-Carlo estimate of its gradient. The estimator I trust is the score-function form, `∇_θ η = E[Σ_t Ψ_t ∇_θ log π_θ(a_t|s_t)]`, and among the choices for `Ψ_t` the advantage `A_π(s,a) = Q_π(s,a) − V_π(s)` is the one with nearly the lowest variance, because it asks the right question: was this action better or worse than what the policy does on average in this state. So a gradient step nudges up the log-probability of better-than-average actions and down for worse-than-average ones. Good. I'll estimate `A` from a learned value function and call the estimate `Â_t`.

The thing that bites is data efficiency. That gradient is only unbiased for data sampled from the very policy I'm differentiating. So the honest recipe is: roll out `π_θ`, form one gradient, take one step, throw the batch away, roll out again. Each batch of expensive environment interaction — and in MuJoCo a batch is thousands of simulator steps — buys exactly one gradient step. I'd love to be greedier: collect one batch and do several epochs of minibatch SGD on it, the way I would on any supervised objective. But the instant `θ` moves off `θ_old`, the data no longer comes from the current policy, the estimate goes stale, and what I see in practice is that the updates don't just degrade gracefully — they go destructively large. One batch can shove the policy off a cliff. So the real problem is sharp: how do I reuse a batch for many cheap first-order steps *without* the policy wandering so far that the objective I'm optimizing stops predicting the true return?

Let me get precise about why it goes wrong, because the fix has to attack the actual mechanism. Write the return of a candidate policy `π̃` relative to my current `π` exactly: `η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)`, where `ρ_{π̃}(s) = Σ_t γ^t P(s_t = s | π̃)` is the discounted state-visitation measure under `π̃`. This is exact and it's beautiful — the improvement is the advantage of `π` summed over the states `π̃` actually visits. But `ρ_{π̃}` depends on `π̃` in a way I can't sample, because I'd have to run `π̃` to know where it goes, and `π̃` is the thing I'm trying to choose. So I do the only tractable thing: freeze the visitation at the old policy's,

  `L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)`,

which I *can* estimate from old-policy rollouts. With importance sampling I turn the inner sum over actions into an expectation under the data: `Σ_a π̃(a|s) A = Σ_a π_old(a|s) [π̃(a|s)/π_old(a|s)] A = E_{a∼π_old}[ r(θ) A ]` with the ratio `r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)`. So the surrogate I can actually optimize on a batch is `L^CPI(θ) = Ê_t[ r_t(θ) Â_t ]`. At `θ = θ_old` every `r_t = 1`, and there `L` agrees with `η` to first order — its value and gradient match. That's the whole license to optimize it: near `θ_old`, climbing `L` climbs `η`. But the license is *local*. I substituted `ρ_π` for `ρ_{π̃}`; that substitution is exact only when `π̃ = π`, and the error grows as `π̃` pulls away, because the states `π̃` really visits drift from the ones `π` visited. So `L^CPI` is a tangent that's faithful in a neighborhood and lies about `η` outside it. Maximizing it with no leash walks straight out of the neighborhood and reports a huge "improvement" that isn't real. That's the destructive update, and now I can see it's not a numerical accident — it's me trusting a local model globally.

So the problem reduces to: maximize `L^CPI` but keep the policy inside the region where `L^CPI` is still a good model of `η`. I need to quantify "the region." How much can `η` and `L` disagree as `π̃` moves? There's a classical handle on exactly this. Kakade and Langford studied conservative policy iteration, where the new policy is a mixture `π_new = (1−α)π_old + α π'`, and proved a lower bound: `η(π_new) ≥ L_π(π_new) − (2εγ/(1−γ)²) α²` with `ε = max_s |E_{a∼π'}[A_π(s,a)]|`. So the gap between the true return and the surrogate is controlled by `α²`, the size of the mixing step, times a constant. That's the shape I want — surrogate minus a penalty for moving — but it's stuck on mixture policies, which I don't use; my policy is a Gaussian-MLP I update by gradient. I need this for general stochastic policies.

The mixture coefficient is really standing in for "how often the two policies can force different trajectories." For a general stochastic policy I can measure that directly per state. Let `α = D_TV^max(π,π̃) = max_s D_TV(π(·|s), π̃(·|s))`, where `D_TV(p,q) = ½Σ_i|p_i−q_i|`. A maximal coupling says two action distributions at TV distance at most `α` can be made to choose the same action with probability at least `1−α`. Then the probability that a coupled rollout has disagreed at least once by time `t` is at most `1−(1−α)^t`. When no disagreement has occurred, the two state histories match; when a disagreement has occurred, the expected advantage contribution can differ by at most the worst-case advantage scale. The TRPO bound makes that last step precise as `|E_{π̃}[Ā(s_t)] − E_π[Ā(s_t)]| ≤ 4εα(1−(1−α)^t)`, with `ε = max_{s,a}|A_π(s,a)|`. Summing the discounted error gives

  `Σ_{t≥0} γ^t · 4εα(1−(1−α)^t)
   = 4εα(1/(1−γ) − 1/(1−γ(1−α)))
   = 4εγ α² / ((1−γ)(1−γ(1−α)))
   ≤ 4εγ α²/(1−γ)²`.

So the CPI mixture penalty turns into a general-policy trust penalty:

  `η(π̃) ≥ L_π(π̃) − (4εγ/(1−γ)²) (D_TV^max)²`.

Now I want KL, because KL is what I can actually compute and differentiate for a Gaussian and it's the natural currency for "how much did the policy distribution move." Pinsker's inequality hands me the bridge: `D_TV(p,q)² ≤ D_KL(p,q)`. Substitute and the `(D_TV^max)²` becomes a clean `max_s KL`:

  `η(π̃) ≥ L_π(π̃) − C · max_s KL[π(·|s), π̃(·|s)]`,   with `C = 4εγ/(1−γ)²`, `ε = max_{s,a}|A_π(s,a)|`.

The true return is *at least* the surrogate minus a KL penalty. And because `L` and the bound both touch `η` at `π̃ = π` (where `L = η` and `KL = 0`) and match to first order there, this is a minorant that's tight at the current policy. So if I let `M_i(π) = L_{π_i}(π) − C·max_s KL`, then `η(π_{i+1}) ≥ M_i(π_{i+1})` and `η(π_i) = M_i(π_i)`, hence `η(π_{i+1}) − η(π_i) ≥ M_i(π_{i+1}) − M_i(π_i)`. Maximize `M_i` and the right side is `≥ 0`, so `η` cannot decrease. That's a genuine monotonic-improvement guarantee, a minorize-maximize scheme. And read what `M_i` *is*: an affine-in-the-probabilities objective `L` plus a KL regularizer that shrinks the step — this is exactly a proximal / mirror-descent update over the policy distribution, the KL playing the role of the Bregman distance from an entropy regularizer. The penalty isn't a hack bolted onto the surrogate; it falls out of the improvement theory as the price of the visitation approximation.

So the theory literally tells me what to do: each iteration, `maximize_θ L_{θ_old}(θ) − C · max_s KL[π_old, π_θ]`. First-order, simple, one objective. Let me just do that. And immediately I hit the wall the theory itself warns about. The constant is `C = 4εγ/(1−γ)²`. With `γ = 0.99`, `(1−γ)² = 10^{-4}`, so `C ~ 4·10^4·ε` — enormous. A penalty that big dominates the surrogate completely; the KL term pins `π_θ` to `π_old` and the policy barely moves. If I respect the theoretical `C`, my steps are uselessly tiny and learning crawls. The bound is honest but conservative — it's a worst-case minorant, and the worst case almost never happens, so the leash it prescribes is far too short. Wall.

What did TRPO do at this same wall? It abandoned the penalty form and switched to a hard constraint. Instead of subtracting `C·KL`, bound it: `maximize_θ L_{θ_old}(θ)` subject to `max_s KL ≤ δ`. The max over states is itself impractical — too many constraints — so use the *mean* KL over visited states, `Ê_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ] ≤ δ`. Now `δ` is an interpretable knob: a budget on how far, in policy-distribution space, one update is allowed to move. This is far easier to set than `C` because it directly controls the thing I care about — the size of the step — rather than a coefficient whose right value depends on the (unknown, drifting) advantage scale `ε`. TRPO solves it with a linear approximation to the objective and a quadratic approximation to the constraint, which gives a natural-gradient direction computed via conjugate gradient on Fisher-vector products, plus a backtracking line search to enforce the constraint exactly.

And that's a beautiful algorithm, but for *my* goal it's the wrong tool. I wanted first-order and simple. TRPO is second order: Fisher-vector products, conjugate gradient, a line search every update. It's a constrained solve per batch, not the cheap "run a few epochs of minibatch SGD" I was after. And the constrained/Hessian machinery is awkward exactly where I'd like to be flexible — sharing parameters between the policy and value networks, or using architectures with noise — because the Fisher and the line search assume a clean policy-only objective. So TRPO trades the penalty's pathology (bad `C`) for a constraint, but pays in complexity I don't want.

The tension is narrow now. The *penalty* form is what I want structurally — it's just SGD on a single differentiable scalar, trivially compatible with shared nets and several epochs per batch, no second-order anything. Its only sin is that I can't pick `β` (the role `C` plays). The *constraint* form solved the picking problem — by targeting a KL budget `δ` instead of guessing a coefficient — but at the cost of the second-order machinery. So why am I choosing between them? The thing that made the constraint usable was *targeting the KL*, not the constraint mechanism itself. What if I keep the cheap penalty form but borrow the constraint's idea of controlling the KL — by *adjusting* `β` to hit a target KL, instead of fixing it?

Let me think about whether a fixed `β` could ever work, to be sure adaptation is necessary and not just convenient. Suppose I pick some `β` and run `Ê_t[ r_t Â_t − β·KL ]`. The trouble is that the KL produced by a fixed `β` is not stable. `β` trades off the advantage signal against the policy-movement cost; the equilibrium KL depends on the magnitude of `Â`, which I standardize but which still varies, and on the local curvature of the policy — how much a unit of parameter change moves the distribution — which changes as training proceeds and differs across environments. So a `β` that lands me at a sensible KL early in training, when advantages are large and the policy is plastic, will produce a tiny KL late, when advantages shrink near convergence; and a `β` tuned on HalfCheetah will over- or under-shoot on Hopper. The right penalty is not a constant — it's whatever value happens, *right now, on this batch*, to produce the policy step I want. That's the case against fixed `β`, and it's decisive: there is no single good `β`, even within one run.

So make `β` chase a target. Pick a target KL `d_targ` — the policy-space step size I'm willing to take per update, the analogue of TRPO's `δ`. After each update, measure the actual KL `d` the policy moved. If I moved too far, `d > d_targ`, the leash was too loose, so tighten it: increase `β`. If I moved too little, `d < d_targ`, loosen: decrease `β`. The controlled quantity is the KL — the step size in distribution space — and `β` is just the dial I turn to keep it on target, batch by batch. This is the same logic TRPO used to justify a constraint over a fixed penalty, now realized inside a pure first-order penalty loop.

How should I adjust `β`? Additively or multiplicatively? Think about the response of KL to `β`. The achieved KL is, roughly, a decreasing, convex-ish function of `β` — doubling `β` doesn't subtract a fixed amount of KL, it roughly *scales* the step down. KL and `β` live on multiplicative scales (they range over orders of magnitude across training), so the natural controller is multiplicative: nudge `β` up or down by a factor. A factor of two each adjustment is aggressive enough to catch up to a drifting equilibrium within a few updates but not so violent that it oscillates wildly. And I don't want to react to every tiny miss, or `β` will thrash; I want a deadband around `d_targ` where I leave `β` alone. So: if `d > 1.5·d_targ`, the step was clearly too big — `β ← β·2`. If `d < d_targ/1.5`, clearly too small — `β ← β/2`. Otherwise leave it. The `1.5` band and the factor `2` are heuristic, but the algorithm isn't sensitive to them — they only set how fast and how tightly `β` tracks, and the tracking corrects itself regardless. For the same reason the *initial* `β` barely matters: whatever I start with, a couple of doublings or halvings carries it to the right neighborhood within the first few updates, after which it stays there. I'll start it at something modest like `0.5` and let the controller find the level. Occasionally a single update overshoots the KL badly, but those are rare and `β` snaps back.

Now I have to actually compute the KL penalty on a minibatch, and this is subtler than it looks. I have samples `a_t ∼ π_old`, and for each I can evaluate both log-probs, so I can form the log-ratio `logratio = log π_θ(a_t) − log π_old(a_t)` and the ratio `r_t = exp(logratio)`. The textbook KL `KL[π_old, π_θ] = E_{a∼π_old}[ log(π_old/π_θ) ] = E[−logratio]`. So the naive per-sample estimator of the KL is just `−logratio`, average it over the batch. Let me check it. It's unbiased — its mean is the KL by definition. But it's a bad estimator to *use*: KL is always nonnegative, yet `−logratio` is negative for any sample where the new policy happens to assign higher probability than the old, which is roughly half the samples near `θ_old`. So this estimator swings positive and negative sample-to-sample, has high variance, and can even come out negative on a finite batch — a nonsense value for a penalty I'm scaling by `β` and feeding to the optimizer. I want something that is unbiased, low-variance, and — ideally — always nonnegative, so that the penalty is honestly a penalty on every minibatch.

Lower the variance with a control variate. I want to add to `−logratio` some quantity with zero mean (so I don't bias the estimate) that's negatively correlated with `−logratio` (so it cancels the swings). The one obviously-zero-mean quantity available is `r − 1`: under `a ∼ π_old`, `E[r] = E[π_θ/π_old] = Σ π_old · (π_θ/π_old) = Σ π_θ = 1`, so `E[r − 1] = 0` exactly, for free. So for any scalar `c`, `−logratio + c(r − 1)` is still an unbiased estimator of the KL. Now pick `c` not by minimizing variance analytically (that needs `p` and `q` I don't have in closed form) but by a cleaner criterion: make the estimator nonnegative pointwise. Since `log` is concave, `log x ≤ x − 1` for all `x > 0`, i.e. `(x − 1) − log x ≥ 0` always, with equality at `x = 1`. So if I take `c = 1`,

  `KL estimate per sample = (r − 1) − log r ≥ 0`,

which is exactly the vertical gap between the tangent line `x − 1` and the curve `log x`. It's unbiased (sum of `−log r`, which is unbiased, and `r − 1`, which is mean-zero), it's nonnegative on every single sample, and because it's the gap to the tangent it's small and smoothly varying near `r = 1` rather than swinging in sign — so it has much lower variance than `−logratio`. That's the estimator I'll use for the penalty: `kl = mean( (r − 1) − logratio )`, where `logratio = log r`. And crucially I keep its gradient — this isn't a diagnostic, it's the term `β·KL` in the loss that the optimizer differentiates to pull `π_θ` back toward `π_old`. The very same expression, detached, is also the clean nonnegative number I read off to decide whether to raise or lower `β`.

I still need `Â_t`, and I should pin down its bias-variance behavior rather than reach for a default. The one-step option is the TD residual `δ_t = reward_t + γ V(s_{t+1}) − V(s_t)`. If `V` were exactly `V_π`, then `E[δ_t] = A_π(s_t,a_t)` exactly, so `δ_t` is a low-variance advantage estimate — but it's biased whenever `V` is imperfect, which it always is early on. The opposite extreme uses the empirical return: `Â_t = Σ_l γ^l reward_{t+l} − V(s_t)`, unbiased given a correct value baseline but high variance because it sums a whole noisy trajectory. Between them, sum `k` residuals: `Â_t^(k) = Σ_{l=0}^{k−1} γ^l δ_{t+l}`, which telescopes to `−V(s_t) + Σ_{l=0}^{k−1} γ^l reward_{t+l} + γ^k V(s_{t+k})` — bigger `k` leans more on real rewards (less value-function bias, more variance). I don't want to pick a single `k`; I want to blend them. Take the exponentially-weighted average with weight `λ^{k−1}` on `Â_t^(k)`, normalized by `(1−λ)`:

  `Â_t = (1−λ)( Â_t^(1) + λ Â_t^(2) + λ² Â_t^(3) + ... )`.

Substitute `Â_t^(k) = Σ_{l=0}^{k−1} γ^l δ_{t+l}` and collect the coefficient of each `δ_{t+l}`. The residual `δ_{t+l}` (carrying its `γ^l`) appears in every `Â_t^(k)` with `k > l`, so its total weight is `(1−λ) γ^l (λ^l + λ^{l+1} + ...) = (1−λ) γ^l λ^l / (1−λ) = (γλ)^l`. Everything collapses to

  `Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`.

One discounted sum of TD residuals, with `λ` as the bias-variance dial: `λ = 0` gives `Â_t = δ_t` (lowest variance, most value-function bias), `λ = 1` gives the Monte-Carlo `Σ γ^l r_{t+l} − V(s_t)` (unbiased, highest variance). In a finite length-`T` segment I truncate the sum, which I compute cheaply backward: `Â_t = δ_t + γλ·Â_{t+1}`, masking across episode boundaries with `(1 − done)`. Defaults `γ = 0.99`, `λ = 0.95` sit near the low-variance end where the learned value function does most of the work but a little real reward leaks in. And the value target for fitting `V` is just `Â_t + V(s_t)`, the GAE return.

Now assemble the full loss on a minibatch. The policy surrogate is `−Ê[ r_t Â_t ]` (negative because I minimize) plus the penalty `+ β·kl`. Concretely with the standardized advantages already in hand,

  `pg_loss = −mean( Â_t · r_t ) + β · kl`.

I want a value function too, and I'll fit it by plain regression to the GAE returns, `v_loss = ½·mean( (V_θ(s_t) − R_t)² )` — squared error, no clipping, the straightforward critic loss. And following the actor-critic tradition I add an entropy bonus to keep the policy from collapsing its exploration prematurely; for a diagonal Gaussian the entropy has a closed form and the bonus just rewards keeping the action distribution wide. Combine with coefficients: `loss = pg_loss − c_ent·entropy + c_vf·v_loss` (entropy enters with a minus because more entropy is good and I'm minimizing). Then the outer machinery is the actor-critic loop I already have: `N` parallel actors each collect `T` steps; compute `Â` and returns by GAE on the segment; snapshot `π_old`; run `K` epochs of minibatch SGD with Adam on `loss`; read the detached KL values produced by those policy evaluations and adjust `β` by the controller for subsequent minibatch calls; repeat. The only reason it's safe to do `K` epochs on one batch — the thing vanilla policy gradient couldn't do — is that the `β·KL` penalty keeps each pass proximal, with `β` self-tuned so the update stays near `d_targ` in policy space.

Let me write the slot exactly, the per-minibatch loss, since everything funnels into it. I keep the adaptive `β` as state on the agent, initialized once, clamped to a sane range so a runaway target can't drive it to zero or infinity:

```python
import torch
from torch.distributions import Normal


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Adaptive-KL-penalty policy update: r_t·Â_t surrogate minus β·KL, with β
    chased toward a target KL across successive minibatch calls. No ratio clipping."""
    # adaptive penalty state, created once and carried across updates
    if not hasattr(agent, "_kl_beta"):
        agent._kl_beta = 0.5       # initial β; the controller quickly finds the right level
        agent._target_kl = 0.01    # d_targ: the per-update policy-space step we aim for

    # re-evaluate the current policy on the stored (old-policy) actions
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs        # log r_t = log π_θ − log π_old
    ratio = logratio.exp()                      # r_t = π_θ(a)/π_old(a)

    # KL[π_old, π_θ] via the (r−1)−log r estimator: unbiased, ≥0 pointwise, low variance.
    # Kept WITH gradient — this IS the penalty term, not a diagnostic.
    kl = ((ratio - 1) - logratio).mean()

    with torch.no_grad():
        approx_kl = kl.detach()                 # the realized KL, read off to adapt β
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()  # diagnostic only

    # surrogate L^CPI = E[r_t Â_t], penalized by β·KL (no clip)
    beta = agent._kl_beta
    pg_loss = -(mb_advantages * ratio).mean() + beta * kl

    # adapt β toward d_targ: too-large a step ⇒ tighten (×2); too-small ⇒ loosen (÷2);
    # multiplicative because KL responds to β on a multiplicative scale; ×1.5 deadband
    # to avoid thrashing; clamp so β can't run away to 0 or ∞. The new β affects
    # subsequent minibatch calls; the current scalar loss has already captured beta above.
    with torch.no_grad():
        if approx_kl > 1.5 * agent._target_kl:
            agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
        elif approx_kl < agent._target_kl / 1.5:
            agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)

    # plain squared-error critic fit to the GAE returns (no value clipping)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    # total: minimize −surrogate, subtract entropy bonus, add value loss
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

So the causal chain closes. I started wanting to reuse each expensive batch of on-policy data for several first-order steps, and found that the surrogate `L^CPI = Ê[r_t Â_t]` — the importance-sampled local approximation to the return — is faithful only near `θ_old`, because freezing the state-visitation at the old policy is exact only there, so optimizing it without a leash produces destructively large, illusory updates. The improvement theory told me exactly the leash: `η ≥ L − C·KL`, a true lower bound that's tight at the current policy, so subtracting a KL penalty makes the update monotonically improving — but the theoretical coefficient `C = 4εγ/(1−γ)²` is far too large to take real steps. TRPO escaped by replacing the penalty with a hard average-KL constraint targeting a budget `δ`, which fixed the coefficient-picking problem but bought in second-order conjugate-gradient/Fisher machinery I don't want and that fights parameter sharing. The resolution is to keep the cheap first-order penalty form and borrow only the *targeting* idea: make `β` adaptive, chasing a target KL `d_targ` — raise `β` when the measured KL overshoots, lower it when it undershoots, multiplicatively with a deadband — so the controlled quantity is the policy-space step size, and the un-pickable coefficient tunes itself, batch by batch, run by run. The KL penalty itself I estimate with the `(r−1)−log r` control-variate estimator, which is unbiased, nonnegative on every sample, and low-variance, and which I keep differentiable so it actually pulls the policy back. Advantages come from GAE, the exponentially-weighted blend of TD residuals `Â_t = Σ(γλ)^l δ_{t+l}` that dials bias against variance. The whole thing is a single differentiable loss — penalized surrogate plus squared-error critic minus entropy bonus — optimized with `K` epochs of Adam minibatch SGD on each freshly collected batch of `N` actors × `T` steps, which is only safe to do because the self-tuned `β·KL` keeps every pass proximal.
