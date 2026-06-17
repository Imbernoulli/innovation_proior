OK, let me think this through from scratch. I have an SFT language model and I want to push it further with reinforcement learning against a reward model that scores whole responses. The standard move is PPO, and PPO works — it's stable, it does several gradient steps per batch of rollouts without blowing up because of the clipped surrogate. I don't want to throw that away. What I want to throw away is the part of PPO that hurts here: the learned value function. Let me first be precise about why it hurts, because the shape of the fix has to come out of the precise pain, not out of a wish to "simplify."

PPO's advantage `A_t` comes from GAE, `A_t = sum_l (gamma*lambda)^l * delta_{t+l}` with `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`. Every one of those TD residuals needs a per-step value `V(s_t)`, and `V` is a learned network, `V_psi`, of basically the same size as the policy. So I'm carrying two policy-sized networks and co-training both. That's the memory cost, and it's real on a GPU. But there's a second, sharper problem that's specific to my setting. The reward here is one scalar per response — the reward model looks at the whole `(q, o)` and emits a single number, and in the per-token bookkeeping that number lands at the *last* token; every interior token has reward zero. So what is `V(s_t)` at an interior token even supposed to be? It's the expected future reward from that prefix, and the only signal teaching `V_psi` what that is comes from a return that's nonzero only at the end of the sequence. I'm asking a network to densify, per token, a reward it only ever sees once per sequence, at the very end. That's the hardest, least observed quantity in the whole loop, and the entire advantage rides on it. The component I most want to trust is the one I can least fit. That's the thing to attack.

So: can I get a usable advantage *without* `V_psi`? Let me go back to what the value function is actually for. Strip GAE away and look at the bare policy gradient. The gradient of expected return is `E[ sum_t grad log pi_theta(a_t | s_t) * Psi_t ]`, where `Psi_t` is "how good was this action." If I plug in the raw return for `Psi_t`, the estimator is unbiased but the variance is brutal — every trajectory's whole return scales every token's gradient, good runs and bad runs alike, and the thing barely learns. The cure, the reason `V` is there at all, is the baseline. Subtract any function of the state `b(s)` from `Psi_t`. Does that bias the gradient? Let me actually check, because the whole plan hinges on it. The extra term is `E_{a~pi}[ grad log pi(a|s) * b(s) ]`. Pull `b(s)` out, it doesn't depend on `a`: `b(s) * E_{a~pi}[ grad log pi(a|s) ]`. And `E_{a~pi}[ grad log pi(a|s) ] = sum_a pi(a|s) * grad log pi(a|s) = sum_a grad pi(a|s) = grad sum_a pi(a|s) = grad 1 = 0`. So it's exactly zero. Any baseline that depends only on the state, not the action, leaves the gradient unbiased and only moves variance. The variance-minimizing flavor of this makes `Psi_t` into `Q(s,a) - V(s)` — the action's value relative to the *average* value at that state. So the only job `V` has to do, structurally, is be a good estimate of the *average return at this state*, so I can subtract it and center the signal.

Now stare at "the average return at this state" in my actual setting. The "state" at the start of generation is the prompt `q`. The "return" of a response `o` is just its scalar reward `r = r_phi(q, o)` — there's no discounting that matters across a single terminal reward, the return of the whole response *is* that scalar. So the baseline I want, at the level of whole responses, is `V(q) = E_{o ~ pi}[ r_phi(q, o) ]` — the expected reward of a response to this prompt under the current policy. That's what a perfect critic would tell me. And here's the thing: I'm *already sampling responses from the current policy*. In the on-policy loop I draw a group of `G` outputs per prompt, `{o_1, ..., o_G}`, and score every one of them. So I have `G` independent Monte-Carlo samples of exactly the quantity `V(q)` is trying to predict. Why am I training a network to estimate `E_{o~pi}[ r(q,o) ]` when I can just *average the rewards I already drew*?

Let me write that down. For a prompt `q` with its group of rewards `{r_1, ..., r_G}`, the obvious estimate of the baseline is the group mean, `mean(r) = (1/G) sum_i r_i`, and the centered signal for response `o_i` is

    A_i = r_i - mean(r_1, ..., r_G).

No critic. The baseline costs nothing — the samples are already in hand. And it sidesteps the per-token densification problem entirely, because I never needed a per-*token* value; the reward is a per-response scalar, so a per-response baseline is the honest object. This also lines up with something I know about the reward model itself: it was trained on *comparisons among outputs of the same prompt*. Its scores are calibrated as a within-prompt ordering; the absolute level is arbitrary and shifts from prompt to prompt. Centering each response against its own group's mean is using the reward model on exactly the scale it's meaningful on — relative to its peers on the same question. A response above the group mean gets a positive coefficient and is pushed up; one below gets a negative coefficient and is pushed down. Unlike just keeping the correct ones, this *signs* the gradient — wrong answers are actively penalized, not merely ignored — and it scales by how far above or below the peer average the response is, so a clearly-better correct answer is reinforced harder than a borderline one. That's strictly more information than a correctness indicator.

Wait — is the group mean a *valid* baseline, or did I just smuggle in bias? The unbiasedness argument above needs `b` to be a function of the state alone, not of the sampled action. My `mean(r)` includes `r_i` itself, the reward of the very response whose gradient I'm scaling. The clean exactly-unbiased version for response `i` would be the leave-one-out mean, `(1/(G-1)) sum_{j != i} r_j`, because that depends on the other same-prompt samples but not on `o_i`. If I ignore the std for a second and use the in-group mean, I can see the effect exactly: with `g_i = grad log pi(o_i|q)`, `E[(1/G) sum_i g_i (r_i - mean_j r_j)] = E[(1/G) sum_i g_i r_i] - E[(1/G^2) sum_i sum_j g_i r_j]`. The cross terms `j != i` vanish because `E[g_i]=0` and the samples are independent; the `j=i` terms subtract `1/G` of the original policy-gradient term. So the centered in-group mean gives `(1 - 1/G) E[g r]` in the mean-only case: same direction, smaller magnitude, not a new spurious direction, but not the exact state-baseline theorem either. Once I divide by the sample std, I'm deliberately choosing a normalized group-relative coefficient, not claiming an unbiased estimator of the raw-return gradient. That's the honest bargain: the exact theorem justifies why centering around a state value is the right object, and the group statistic is the practical same-prompt estimate I use for the update.

Now I'm not done, and the reason is scale. `A_i = r_i - mean(r)` has units of reward, and the *spread* of rewards within a group depends enormously on the prompt. Picture an easy prompt: the policy already nails it, every one of the `G` rewards is near the top, the spread is tiny, so all the `A_i` are tiny — almost no gradient, which is fine, there's little to learn. Now a prompt of medium difficulty where the policy is genuinely uncertain: rewards are spread out, some high, some low, and the `A_i` are large. And a prompt where every sample fails about equally: again tiny spread. So the magnitude of the update a prompt contributes is being set by the *reward model's scale on that prompt* — how spread out its scores happen to be — which is an arbitrary property of the reward model, not a measure of how much I should learn from that question. A prompt where the reward model assigns big numbers will dominate the gradient over a prompt where it assigns small ones, even if the second prompt is more informative. I want each prompt to contribute on a comparable footing, governed by the *relative* ranking of its responses, not by the raw numeric range the reward model chose for it.

I still need to normalize away the per-prompt spread. Divide each centered coefficient by the group's own standard deviation:

    A_i = ( r_i - mean(r_1, ..., r_G) ) / std(r_1, ..., r_G).

That's a within-group z-score when the group has more than one non-identical reward. The coefficients are dimensionless, centered inside the prompt, and scaled by that prompt's own spread; up to the epsilon guard, a prompt with a wide raw reward range and a prompt with a narrow raw reward range now hand back comparable order-1 coefficients whose signs and ordering reflect which responses beat their peers. The step magnitude a prompt drives is no longer set by the reward model's arbitrary per-prompt offset and gain; the update keeps the within-prompt comparison signal, which is exactly the information the comparison-trained reward model is calibrated to give. So: subtract the group mean as the sample baseline, divide by the group std for per-prompt scale invariance. That pair is the move.

Let me sanity-check the division for degenerate cases before I trust it. If a group has only one sample, the group-relative statistic is undefined: there is no peer comparison, no group mean estimate, and no group std. In normal use I choose `G > 1`, but the implementation still needs a finite fallback. The verl-style fallback sets `mean = 0` and `std = 1`, so the lone scalar is passed through up to the tiny denominator guard; that is a code fallback, not a meaningful z-score. If a group has multiple samples and every reward is identical, then the mean is that reward, the centered numerator is exactly zero, and `std = 0`; adding `epsilon` in `(r_i - mean) / (std + epsilon)` keeps the result at clean zero instead of `0/0`. When the spread is tiny but nonzero, `epsilon` also prevents an enormous coefficient from numerical noise. Set `epsilon` to something like `1e-6`. Good — edge cases don't poison the batch, and I don't pretend the one-sample fallback is a relative advantage.

Now, where exactly do I compute this? The reward arrives as a `(batch, response_length)` tensor with the scalar parked at each response's last valid token and zeros elsewhere, plus a `response_mask`. So to recover the per-response scalar `r_i` I just sum over the token axis: `scores = token_level_rewards.sum(dim=-1)` pulls the single nonzero value out of each row. Then I group the rows by their prompt `index` — same-prompt rollouts share an `index` — collect each group's scalars, compute that group's mean and std, and replace each `scores[i]` with its z-score. To hand this back to the rest of the pipeline as outcome supervision, I broadcast the per-response scalar advantage across all of that response's valid tokens — there is no token-level credit signal here, the reward is a single outcome, so the honest thing is to assign the same (normalized) outcome to every token of the response. Multiply by `response_mask` so only valid tokens carry it. That broadcast is the outcome-supervision choice: `A_{i,t} = A_i` for every token `t` of `o_i`.

I can fill the advantage slot without adding any other learned object:

```python
import torch
from collections import defaultdict


def group_relative_advantage(token_level_rewards, response_mask, index, epsilon=1e-6):
    # recover the per-response scalar: the reward sits at the last valid token,
    # zeros elsewhere, so summing the token axis pulls it out.
    scores = token_level_rewards.sum(dim=-1)          # (bs,)

    # group the per-response scalars by prompt: same-prompt rollouts share index.
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # per group: the mean centers same-prompt responses;
    # the std rescales the prompt's reward spread.
    for idx, vals in id2score.items():
        if len(vals) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        else:
            v = torch.stack(vals)
            id2mean[idx] = v.mean()
            id2std[idx] = v.std()                      # torch.std default matches verl's sample std

    # z-score each response within its own prompt group.
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    # outcome supervision: broadcast the scalar advantage to every valid token.
    return scores.unsqueeze(-1) * response_mask
```

That's the heart of it: replace the learned value baseline with the group mean, divide by the group std, broadcast. Now I have to wire it into a stable policy update, and I want to keep PPO's machinery because the reasons PPO clips haven't gone away. I'm still doing several gradient steps per batch of rollouts, the policy still drifts from the `pi_old` that generated the data, so the importance ratio `rho_{i,t} = pi_theta(o_{i,t}|q, o_{i,<t}) / pi_{theta_old}(o_{i,t}|q, o_{i,<t})` still needs containing. PPO's clipped surrogate `min( rho * A, clip(rho, 1-eps, 1+eps) * A )` is a pessimistic lower bound on the true objective — it stops rewarding ratio movement once it would help, so multiple inner steps don't run away. I keep that verbatim. So the per-token objective is

    (1/G) sum_i (1/|o_i|) sum_t min[ rho_{i,t} * A_{i,t}, clip(rho_{i,t}, 1-eps, 1+eps) * A_{i,t} ]

with `A_{i,t}` the broadcast group-normalized reward. The `1/G` averages over the group, the `1/|o_i|` averages over a response's tokens.

Now the KL regularizer. RLHF-PPO folds the KL penalty into the reward itself, `r_t = r_phi - beta * log( pi_theta / pi_ref )` per token. I should think about whether to keep it there, because I just spent a lot of care making the reward into a clean within-group normalized quantity. If I fold a per-token KL term into the reward *before* I group-normalize, then the thing I'm centering and z-scoring is no longer the reward model's score — it's the score tangled together with a per-token policy-vs-reference log-ratio, and that log-ratio gets swept into the group mean and the group std and the broadcast. The advantage stops being a clean comparison of responses on the same prompt; the regularizer contaminates the very statistic I built. So I don't want KL inside the reward. Pull it out and make it a separate term in the loss, added directly:

    objective_{i,t} = min[ rho * A, clip(rho) * A ] - beta * D_KL[ pi_theta || pi_ref ].

Now the advantage stays exactly the group-normalized reward, and the KL is its own additive penalty whose gradient I can reason about independently. `beta` trades off staying near the SFT reference against chasing reward.

What estimator do I use for the per-token `D_KL[pi_theta || pi_ref]`? I'm sampling tokens from the policy, so I want an unbiased Monte-Carlo estimate of `KL = E_{x~pi_theta}[ log(pi_theta(x)/pi_ref(x)) ]`. The naive estimator is the single-sample log-ratio `log(pi_theta/pi_ref)`, i.e. `-log(pi_ref/pi_theta)`. Write `r = pi_ref/pi_theta` for the per-token ratio of reference to policy. The naive estimator is `k1 = -log r`. It's unbiased — its expectation under `pi_theta` is the KL — but it's signed: any single token can come out negative even though KL is non-negative. A zero-mean control variate is sitting right there: `r - 1 = pi_ref/pi_theta - 1`. Its expectation under `pi_theta` is `E_{x~pi_theta}[ pi_ref(x)/pi_theta(x) ] - 1 = sum_x pi_theta(x) * pi_ref(x)/pi_theta(x) - 1 = sum_x pi_ref(x) - 1 = 1 - 1 = 0`. Zero mean — legitimate to add without changing the expected value. If I add it with coefficient 1, the first-order log-ratio fluctuation cancels near `r = 1`: `-log(1 + delta) + delta = delta^2/2 + O(delta^3)`. Now form

    k3 = r - 1 - log r = (pi_ref/pi_theta) - 1 - log(pi_ref/pi_theta).

Its expectation is `E[r - 1] + E[-log r] = 0 + KL = KL` — still unbiased. And it's always non-negative, because for all positive `x`, `log x <= x - 1`, so `x - 1 - log x >= 0` for every token, with equality only when `r = 1`, i.e. when the policy equals the reference. So `k3` is a per-token KL estimate that is unbiased, never negative, and less dominated by signed first-order log-ratio noise when the two policies are close. I'll use it:

    k3 = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1.

Let me make sure the gradient this whole thing produces is the one I think it is, because the KL-in-the-loss choice changes the gradient coefficient and I want to see what each piece contributes. Take the single-inner-step case where `pi_old = pi_theta`, so `rho = 1` and the clip is inactive (`min` and `clip` both pass through). The objective per token is `rho * A - beta * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)`. Differentiate with respect to `theta`. The first term: with `rho = pi_theta/pi_old` and `pi_old` fixed, `grad rho = (grad pi_theta)/pi_old = (pi_theta/pi_old) * grad log pi_theta`, and at `rho = 1` that's `grad log pi_theta`, so the first term contributes `A * grad log pi_theta` — the policy-gradient term with my group-normalized advantage as the coefficient. Good. Now the KL term. Let `u = pi_ref/pi_theta`. Note `grad log pi_theta = (grad pi_theta)/pi_theta`. Differentiate the two theta-dependent pieces:

`grad(pi_ref/pi_theta) = pi_ref * grad(1/pi_theta) = pi_ref * (-1/pi_theta^2) * grad pi_theta = -(pi_ref/pi_theta) * (grad pi_theta / pi_theta) = -u * grad log pi_theta.`

`grad(-log(pi_ref/pi_theta)) = grad(-log pi_ref + log pi_theta) = grad log pi_theta.`

(The `-1` constant differentiates to zero.) So `grad(k3) = grad(u - 1 - log u) ... ` — careful, write it directly: `grad( pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 ) = -u * grad log pi_theta + grad log pi_theta = (1 - u) * grad log pi_theta`. Then `grad( -beta * k3 ) = -beta * (1 - u) * grad log pi_theta = beta * (u - 1) * grad log pi_theta = beta * (pi_ref/pi_theta - 1) * grad log pi_theta`. Add the two contributions:

    grad J = E[ (1/G) sum_i (1/|o_i|) sum_t ( A_{i,t} + beta*(pi_ref/pi_theta - 1) ) * grad log pi_theta(o_{i,t} | q, o_{i,<t}) ].

So the effective per-token gradient coefficient is `A_{i,t} + beta*(pi_ref/pi_theta - 1)`. That's clean and interpretable: the first part is the group-normalized advantage — push up responses that beat their peers, push down those that don't — and the second is the KL pull, a per-token term proportional to the gap between the reference and the policy that nudges `pi_theta` back toward `pi_ref` exactly where it has drifted, vanishing when `pi_theta = pi_ref` (then `u = 1`, the term is zero). The sampled-loss regularizer enters the gradient additively, *without* having touched the advantage — which is precisely the benefit I wanted from pulling KL out of the reward. And notice the contrast with just keeping the correct samples: that method's coefficient is the indicator `1[correct] in {0,1}`, which never goes negative; mine is `A_{i,t}`, which is signed and magnitude-aware, so it penalizes wrong answers and reinforces by how far a response beat its group. The group-relative advantage is what buys that.

Let me also pin down a couple of implementation details so the advantage rule survives contact with a real trainer. The primary formula only says `std`, and the code has two nearby conventions: the verl outcome-advantage implementation uses `torch.std(scores_tensor)` with PyTorch's default sample std, while the reward-stage `group_std` hook uses `std(unbiased=False)` as a population-style batch statistic. Both are scale conventions around the same per-prompt normalization; for the advantage code I mirror verl, and for the reward-stage hook I mirror the hook. I keep `epsilon = 1e-6` inside the denominator as the guard against zero spread. The grouping is done on the per-response scalars recovered by the last-token sum, not on the token tensor, so it costs almost nothing. And the broadcast-then-mask at the end keeps the outcome-reward semantics: a single normalized number per response, replicated across its valid tokens, zero on padding.

There's one more thing I can extend cleanly if a denser reward exists. The outcome case broadcasts one scalar per response because that's all the signal there is. But suppose I have a process reward model that scores each *reasoning step*. Then I can normalize the per-step rewards the same way — pool all step rewards across the group, subtract their mean, divide by their std — `r~_i^{step j} = (r_i^{step j} - mean(R)) / std(R)`, and set a token's advantage to the sum of the normalized rewards of all steps at or after it, `A_{i,t} = sum_{j: index(j) >= t} r~_i^{index(j)}`. Same group-normalization principle, just applied per step and accumulated forward instead of broadcast flat. The outcome case is the special case where there's a single "step" at the end. But absent a process reward model, the outcome broadcast is the honest landing.

So the whole thing comes together as: for each prompt, sample a group, score it, subtract the group mean (the same-prompt sample baseline that removes the critic and the per-token-value-fitting problem), divide by the group std (per-prompt scale invariance so the reward model's arbitrary range doesn't set the step size), broadcast the normalized scalar across the response's tokens, feed it into PPO's clipped surrogate, and add the KL-to-reference as a separate loss term using the unbiased non-negative `k3` estimator. The trainer needs only this advantage stage:

```python
import torch
from collections import defaultdict


@torch.no_grad()
def group_relative_advantage(token_level_rewards, response_mask, index, epsilon=1e-6):
    """Group-relative advantage: per-prompt z-score of the response reward,
    broadcast across the response's tokens. No learned value function."""
    scores = token_level_rewards.sum(dim=-1)              # (bs,): per-response scalar reward

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx, vals in id2score.items():
        if len(vals) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        else:
            v = torch.stack(vals)
            id2mean[idx] = v.mean()                       # same-prompt sample baseline
            id2std[idx] = v.std()                         # verl-style sample std

    for i in range(bsz):
        # subtract group mean, divide by group std (+eps guards std=0)
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    return scores.unsqueeze(-1) * response_mask           # broadcast to all valid tokens
```

The causal chain: the learned per-token critic is both the most expensive and the least fittable component when the reward is one last-token scalar, so I asked what the critic is *for* — a state baseline — proved any state-only baseline keeps the gradient unbiased, checked that the in-group mean is a practical same-prompt sample baseline rather than a literal state-only baseline, and used it to center each response against its peers; I then noticed the per-prompt reward scale was leaking into the step size and divided by the group std to make every prompt contribute on the same footing; I kept PPO's clipped surrogate so multiple inner steps stay stable; and I moved the KL out of the reward into the loss — so the group-normalized advantage stays clean — estimating it with the unbiased, always-non-negative `k3 = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1`, whose sampled-loss gradient I verified adds a `beta*(pi_ref/pi_theta - 1)` pull-to-reference term and nothing more.
