Let me start from what actually hurts when I try to RL-fine-tune one of these math models. I have an instruction-tuned policy that already writes decent chain-of-thought solutions, and I have a reward model `r_phi(q, o)` that hands me a single scalar for a whole solution `o` to a question `q`. The plan is the usual one: sample solutions from the policy, score them, and nudge the policy to make the high-scoring ones more likely. The workhorse for this is PPO, and PPO works, but every time I stand it up in the LLM setting two things grind on me, and they're both about the same object — the value network.

PPO is actor-critic. Alongside the policy I'm training, I have to train a second network, the value function `V_psi`, and in practice it's the same size as the policy — a 7B critic shadowing a 7B actor. On a memory-bound box that roughly doubles the trainable footprint, and the critic has to be optimized in lockstep with the actor, which is its own headache. So there's a brute cost. But the deeper problem is *why* the critic is there and whether it can even do its job here. Let me think about what the critic is actually for, because I don't want to throw it away if it's load-bearing.

The reason a value function shows up at all is variance reduction in the policy gradient. The policy-gradient estimator is `g_hat = E[ grad_theta log pi_theta(a|s) * A_hat ]`, and the multiplier `A_hat` is what tells each token whether to go up or down. If I used the raw return there, the estimator would be unbiased but extremely noisy. The classic fix: subtract a baseline `b(s)` that depends only on the state, not the action. That subtraction is *free* in expectation — `E_a[ grad log pi(a|s) * b(s) ] = b(s) * grad_theta sum_a pi(a|s) = b(s) * grad_theta 1 = 0` — so any baseline leaves the gradient unbiased while it can slash the variance. And the variance-minimizing thing to put in front of `grad log pi` is the advantage `A^pi(s,a) = Q^pi(s,a) - V^pi(s)`: the action's value relative to the state's default value, so I push up better-than-average actions and push down worse-than-average ones. The "default value of the state" *is* `V^pi(s)`. So the critic isn't decoration — a good baseline, by definition, is a thing that approximates the state value.

Concretely PPO builds its advantages with GAE. Given a learned `V(s)`, the TD residual `delta^V_t = r_t + gamma*V(s_{t+1}) - V(s_t)` is a one-step advantage estimate, exact in expectation when `V = V^pi`, and GAE is the geometric blend `A_hat^{GAE(gamma,lambda)}_t = sum_{l>=0} (gamma*lambda)^l * delta^V_{t+l}`, dialing between high-variance Monte-Carlo (`lambda=1`, the return minus `V`) and low-variance-but-biased single-step (`lambda=0`). Stare at that formula and the dependence is glaring: *every* term needs `V` at *every* timestep. The whole estimator is only as good as a value function that's accurate token by token.

And that's exactly the thing that's broken in my setting. Where does my reward live? The reward model reads the *entire* solution and emits *one number*. Along the sequence there is no per-token reward — it's all concentrated at the end, effectively on the last token, with nothing in between. The value function is supposed to be the machinery that smears that terminal signal backward into a sensible per-token credit assignment. But I'm asking it to learn an accurate value at every intermediate token of a long reasoning chain from a single end-of-sequence scalar. That's a hard regression, and when `V` is badly fit, GAE feeds biased, noisy advantages straight into the policy update. So I'm paying for a policy-sized network whose central job — accurate per-token values — is precisely the job that the last-token-only reward makes hardest. That's the wall. I want PPO's stable clipped update and I want a variance-reducing baseline, but I do not want this learned per-token critic.

So the question sharpens: can I get a baseline — an approximation to "the default value of this state" — without learning `V`? The baseline only has to be (i) independent of the action being scored and (ii) close to the expected return from here, to actually cut variance. It does *not* have to be a learned network. What do I already have lying around that's free? Here's the thing I keep underusing: for each question I don't sample one solution, I sample several. I'm already drawing a *group* of outputs `{o_1, ..., o_G}` per question `q` from the old policy. Each gets a reward `r_i`. The expected reward of the rollout policy on this question is exactly what those samples are estimating. So the empirical mean `mean(r_1, ..., r_G)` *is* a Monte-Carlo estimate of the question's response-level value at collection time — the very quantity `V` was trying to approximate, at the only granularity the reward actually exists (the whole response). It costs nothing extra; the samples are already drawn. Use the group mean as the baseline.

Let me sanity-check that this is a legitimate baseline and not a trick. For a fixed question `q`, the baseline I'd subtract from output `o_i` is `mean over the group`. Is it independent of `o_i`? Not exactly — `o_i` is one of the terms in its own mean. If I wanted the cleanest REINFORCE-style baseline argument, I could use a leave-one-out mean; then the baseline for `o_i` would be formed from the other samples for the same question. But the full group mean is symmetric, cheap, and with a large `G` it differs from leave-one-out only by a `1/G` self-inclusion effect before the later normalization. I should not pretend this is a theorem that nothing changes; it is the practical estimator I want for this setting. It depends on the question and the sampled candidate set, not on any particular token position inside `o_i`, and it gives me the sign structure I need: `r_i - mean(r)` is positive if this solution beat the question's average and negative if it trailed.

There's a second reason this is the *right* baseline and not just a cheap one, and it comes from what the reward model actually is. Reward models are trained on *comparisons*: datasets of "output A is preferred to output B for the *same* prompt." Their scores are calibrated to be meaningful *relative to other responses to the same question*, not on any absolute scale. So a per-question, relative baseline isn't a hack to dodge the critic — it matches the comparative nature of the signal. Subtracting the group mean is computing exactly the kind of relative quantity the reward model was built to express.

Now, the spread of `r_i` differs wildly across questions. An easy question where all `G` samples are correct has rewards all near the top and tiny variance; a hard one has rewards splayed across the range. If I use the raw `r_i - mean(r)`, the easy question contributes microscopic advantages and the hard question contributes huge ones, just because of the scale of the rewards, not because one matters more for learning. I want each question to contribute on a comparable footing — to be telling me "this output was *this many standard deviations* better than typical for this question." So normalize by the group standard deviation too:

  A_hat_i = ( r_i - mean(r) ) / std(r).

A per-question z-score of the reward. This makes the advantage scale-invariant across questions, controls the magnitude of the update so one high-variance question can't dominate the batch, and keeps the comparative reading clean: how many sigmas above its peers this solution scored. (I'll need an epsilon in the denominator so a zero-variance group — all samples identical reward — doesn't divide by zero; and if a group somehow has one sample, there's no spread to speak of, so I'll just set mean to 0 and std to 1 and let the raw reward through.)

One more thing falls out and I should be explicit about it. PPO's GAE produces a genuinely *per-token* advantage `A_t`, varying along the sequence. My group construction gives me one scalar per *response*. With no critic and a single terminal reward, there is no information to distinguish tokens within a correct solution — I have no per-token signal at all. So the honest, minimal choice is to assign the whole normalized outcome to every token of `o_i`: `A_hat_{i,t} = A_hat_i` for all `t` in the response. Outcome supervision, broadcast across the sequence. It's the simplest thing consistent with the information I actually have; it makes every token in a response inherit the response-level credit because there is no finer signal to allocate. (If I ever had a process reward model scoring each *step*, I could do better — normalize the per-step rewards and set a token's advantage to the sum of normalized rewards from that step onward, so later tokens carry credit only for the steps still ahead of them. I'll keep that as the refined variant for when a step-level reward exists; the default with a single terminal reward is the broadcast.)

Good. That's the advantage half. Now the update objective. I'm keeping PPO's clipped surrogate — I have no reason to abandon it, it's the part that's stable. With the probability ratio `rho_{i,t} = pi_theta(o_{i,t}|q, o_{i,<t}) / pi_{theta_old}(o_{i,t}|q, o_{i,<t})`, the per-token term is `min[ rho_{i,t} * A_hat_{i,t}, clip(rho_{i,t}, 1-eps, 1+eps) * A_hat_{i,t} ]`, summed over tokens and averaged over the group:

  J(theta) = E_{q, {o_i} ~ pi_old} (1/G) sum_i (1/|o_i|) sum_t min[ rho_{i,t} A_hat_{i,t}, clip(rho_{i,t}, 1-eps, 1+eps) A_hat_{i,t} ].

That's PPO's machinery with my group-relative advantage slotted in where GAE used to be. So far so good. But I've quietly dropped something: the KL anchor. PPO keeps the policy near the SFT reference `pi_ref` so it doesn't reward-hack, and the standard way to do that bakes the KL *into the reward*: `r_t = r_phi(q, o_{<=t}) - beta * log( pi_theta(o_t|.) / pi_ref(o_t|.) )`, a per-token running cost for diverging from the reference. Can I just do the same — fold a per-token KL penalty into my `r_i` before I normalize?

Let me try it and watch what happens. If I push the KL into the per-token reward, then the thing I normalize per group is no longer a clean reward-model score — it's the score minus an accumulated per-token KL that depends on the *current* policy and varies token by token within a response. Then I subtract the group mean of *that* and divide by its std. Now my advantage entangles the reward-model signal with the KL regularizer, run through the group normalization, and the KL's effect on the update is filtered through a per-question mean-and-std I can't cleanly reason about. The KL was supposed to be a simple, interpretable pull toward the reference; after it goes through z-scoring it's neither simple nor interpretable. That's a mess. Wall.

Back up. The KL doesn't have to live in the reward. It's a regularizer on the policy — so put it where regularizers go, directly in the *loss*, as its own additive term, and leave the advantage to be a clean normalized reward:

  J(theta) = E (1/G) sum_i (1/|o_i|) sum_t { min[ rho A_hat, clip(rho,1-eps,1+eps) A_hat ] - beta * D_KL[ pi_theta || pi_ref ] }.

Now `A_hat_{i,t}` is *only* the normalized reward — the KL never touches the advantage computation — and the KL is a separate, transparent penalty I can tune with `beta`. This is cleaner and it decouples the two jobs: the advantage decides which tokens to push and how hard; the KL term independently keeps the whole policy from wandering off `pi_ref`.

But now I have to actually estimate `D_KL[ pi_theta || pi_ref ]` per token, from the single sampled token I have in hand. What does the standard reward-shaping KL term reduce to? The per-token penalty there is `log( pi_theta / pi_ref )` — that's the naive one-sample estimator of `KL[pi_theta || pi_ref]`, since `E_{x ~ pi_theta}[ log(pi_theta(x)/pi_ref(x)) ] = KL`. It's unbiased, fine. But look at its behavior on a single sample: it's *signed*. For a token where the current policy happens to be below the reference, `log(pi_theta/pi_ref) < 0`, so this "penalty" is *negative* — a single-sample KL estimate that can come out below zero even though the true KL is non-negative. That's high variance and it can momentarily reward divergence on a per-token basis, which is the opposite of what a KL anchor should do. I'd like a per-token KL estimate that is unbiased like this one but doesn't swing negative.

Here's the move. I have a single sample `x ~ pi_theta`; write `u = pi_ref(x)/pi_theta(x)` (the inverse ratio). The naive estimator is `-log u` (= `log(pi_theta/pi_ref)`), and `E_{pi_theta}[-log u] = KL[pi_theta||pi_ref]`. I want to add something with *zero* mean — a control variate — that drags the estimator's variance down and, if I'm lucky, also pins it non-negative. What has zero mean under `pi_theta`? The quantity `u - 1 = pi_ref/pi_theta - 1`, because `E_{pi_theta}[ pi_ref(x)/pi_theta(x) ] = sum_x pi_theta(x) * pi_ref(x)/pi_theta(x) = sum_x pi_ref(x) = 1`, so `E[u-1] = 0`. So `(-log u) + (u - 1) = (u - 1) - log u` has the *same expectation* (still unbiased for the KL) but a different per-sample value. And now the magic: `(u - 1) - log u >= 0` for all `u > 0`, because `log u <= u - 1` always (the log lies below its tangent at `u=1`). So

  D_KL[ pi_theta || pi_ref ] estimated by ( pi_ref/pi_theta - log( pi_ref/pi_theta ) - 1 ),

is unbiased *and* never negative, per token, from a single sample. The control variate `(u-1)` knocked out the part of `-log u` that made it swing below zero. I get a clean, always-positive per-token KL penalty. That's the estimator I want in the loss.

Now I should make sure I actually understand what these two pieces — the clipped surrogate and this KL term — *do to the gradient*, because that's where I'll see whether my method is meaningfully different from the cheap rejection-sampling alternatives or just a re-skin of them. Let me put everything into one common form. Almost every fine-tuning method's gradient can be written as a weighted log-likelihood gradient:

  grad_theta J = E_{(q,o) ~ D} ( (1/|o|) sum_t GC(q, o, t) * grad_theta log pi_theta(o_t | q, o_<t) ),

where `GC` is the per-token *gradient coefficient* — the scalar that says, for this token, how hard to push its log-probability up or down — and methods differ in the data source `D` and in how `GC` is built from the reward. This is a good ruler. Let me compute `GC` for the relevant methods and put them side by side.

SFT is the trivial one: `J_SFT = E[ (1/|o|) sum_t log pi_theta(o_t) ]`, so its gradient is just `sum_t grad log pi_theta`, i.e. `GC_SFT = 1` on every token of every curated example. Pure imitation, uniform push-up.

Rejection-sampling fine-tuning: it's SFT but only on the samples whose answer is correct, so `J_RFT = E[ (1/|o|) sum_t I(o correct) * log pi_theta(o_t) ]` and `GC_RFT = I(o correct) in {0, 1}`. A hard gate. It pushes up every token of every correct solution equally and does literally nothing to wrong ones (coefficient 0 — they vanish from the gradient). Online RFT is the same `GC` but with `o` sampled from the live policy instead of the SFT model.

PPO: take the simplifying case of a single update per rollout, so `pi_theta_old = pi_theta` and `rho_t = 1`; then `min` and `clip` are inactive (the ratio is exactly 1, inside the clip band), the surrogate is `E[ (1/|o|) sum_t rho_t A_t ] = E[ (1/|o|) sum_t A_t ]`, and its gradient is `E[ (1/|o|) sum_t A_t * grad log pi_theta ]` — wait, let me be careful: `grad_theta (rho_t A_t)` at `rho_t = 1` is `A_t * grad_theta rho_t = A_t * grad_theta (pi_theta/pi_old) = A_t * (pi_theta/pi_old) * grad log pi_theta`, and at `pi_theta = pi_old` that's `A_t * grad log pi_theta`. Good. So `GC_PPO = A_t`, the GAE advantage. A real-valued, signed, magnitude-aware coefficient — that's the expressiveness RFT's 0/1 gate is missing.

For my objective, use the same single-update simplification (`pi_old = pi_theta`, `rho = 1`). The objective is `(1/G) sum_i (1/|o_i|) sum_t [ rho A_hat_{i,t} - beta * ( pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 ) ]`. Differentiate term by term. The first piece is the PPO case again: it contributes `A_hat_{i,t} * grad log pi_theta`. The KL piece is the interesting one, and I want to do it carefully. Let me write `g = grad_theta log pi_theta`, and recall `grad_theta pi_theta = pi_theta * g`. The KL term inside the loss is `-beta * ( pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 )`. Differentiate the two `theta`-dependent pieces:

  grad_theta ( pi_ref / pi_theta ) = pi_ref * grad_theta ( pi_theta^{-1} ) = pi_ref * ( -pi_theta^{-2} ) * grad_theta pi_theta = -( pi_ref / pi_theta^2 ) * pi_theta * g = -( pi_ref / pi_theta ) * g.

  grad_theta ( - log( pi_ref / pi_theta ) ) = grad_theta ( - log pi_ref + log pi_theta ) = grad_theta log pi_theta = g.

  grad_theta ( -1 ) = 0.

So `grad_theta ( pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 ) = -(pi_ref/pi_theta) g + g = ( 1 - pi_ref/pi_theta ) g`, and the KL term's contribution to `grad J` is `-beta * (1 - pi_ref/pi_theta) g = beta * ( pi_ref/pi_theta - 1 ) g`. Adding the advantage piece, the full gradient coefficient is

  GC_GRPO = A_hat_{i,t} + beta * ( pi_ref/pi_theta - 1 ).

The control-variate KL estimator gives a clean per-token gradient term `beta(pi_ref/pi_theta - 1)` that pulls `pi_theta` back toward `pi_ref` in proportion to how far the ratio has drifted from 1, with no `log` and no sign-flip pathology in the gradient. And the advantage part, `A_hat_{i,t}`, is my signed, magnitude-aware, group-normalized reward.

Now line them up. `GC_RFT = I(correct) in {0,1}`; `GC_GRPO = A_hat + beta(pi_ref/pi_theta - 1)` with `A_hat = (r - mean(r))/std(r)`. The gradient-coefficient view makes the difference sharp: RFT/Online RFT can only *uniformly* reinforce all correct answers and is silent on wrong ones; this coefficient is *graded* — it reinforces an answer more when it scored well above its group, less when it barely cleared the bar, and it *pushes down* a low-scoring answer (negative advantage) instead of merely discarding it. It is using the reward model's magnitude, not just its sign. That graded, signed reinforcement is exactly the expressiveness PPO bought with its critic — and I've recovered it from the group statistics, with no value network at all. And against PPO, the only change in `GC` is `A_t` (GAE, needs `V`) becoming `A_hat` (group z-score, free) plus the explicit KL gradient term I moved out of the reward.

Let me also see what this paradigm tells me about a design choice I made almost in passing — online vs. offline. RFT and DPO draw their outputs once, from the frozen SFT model (offline); Online RFT and my method draw from the *live* policy each step (online). Early in training the live policy and the SFT model are nearly identical, so the sampled data looks the same and online buys little. But as training proceeds the live policy's outputs diverge from the SFT model's, and *those* — the policy's current mistakes and its current near-misses — are the data that actually carries gradient signal about where the policy is now. So online sampling should be roughly tied early and pull ahead late. That's an argument for keeping my sampling on-policy (from `pi_theta_old`, refreshed each rollout), not a free choice.

So I have the algorithm. Per question `q`: snapshot the old policy, sample `G` outputs, score each with the reward model to get `{r_1, ..., r_G}`, normalize within the group to advantages `A_hat_i = (r_i - mean(r))/std(r)` broadcast to every token, and take a clipped-surrogate step on `J(theta) = (1/G) sum_i (1/|o_i|) sum_t { min[rho A_hat, clip(rho,1-eps,1+eps) A_hat] - beta * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1) }`. Defaults that match the regime: a tiny policy learning rate (1e-6) because each step is a real RL update on a 7B model and I want it gentle; a KL coefficient `beta` around 0.04, small enough to let the reward drive learning but nonzero so the policy stays anchored; a large group, `G = 64` samples per question, because the group *is* my baseline and a noisy mean from a small group would inject variance straight into every advantage; a single policy update per exploration stage, which is also what makes the `pi_old = pi_theta` simplification in my gradient analysis nearly exact. I can iterate the whole thing — periodically reset `pi_ref` to the current policy and refresh the reward model with new samples while keeping 10% historical data in replay — when the reference and reward model go stale relative to a much-improved policy, but that's an outer loop around the same core.

Now let me write the core as the real code I'd ship, filling the empty credit-estimation and actor-loss slots on the flat `(batch, response_len)` tensors the loop already produces. I want the implementation split the same way the training code naturally splits it: one function forms the outcome advantages, one function forms the clipped actor loss, and one small KL primitive supplies the reference penalty.

```python
from collections import defaultdict

import numpy as np
import torch


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Implementation-side aggregation: average over valid response tokens."""
    denom = mask.sum().clamp_min(1)
    return (values * mask).sum() / denom


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_len); nonzero only at the last token
    response_mask: torch.Tensor,         # (bs, response_len); 1 on response tokens
    index: np.ndarray,                   # (bs,); which prompt each row belongs to (the group)
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot (a): group-relative advantage. The whole-sequence reward is the baseline.

    Per prompt-group: subtract the group mean (the Monte-Carlo state value, free from the
    samples we already drew), divide by the group std (per-question z-score so easy and hard
    questions contribute comparably), and broadcast the scalar advantage to every token.
    """
    scores = token_level_rewards.sum(dim=-1)        # one scalar reward per response r_i

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])    # collect rewards within each prompt-group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # a singleton group has no spread: no normalization, pass the raw reward
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                g = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(g)         # baseline = group mean reward
                id2std[idx] = torch.std(g)           # spread for the z-score
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # A_hat_i = (r_i - mean(r)) / (std(r) + eps)
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # without the std (a centered-only variant): A_hat_i = r_i - mean(r)
                scores[i] = scores[i] - id2mean[index[i]]
        # broadcast the per-response scalar advantage to every response token
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores                            # (advantages, returns)


def kl_penalty_forward(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_penalty: str = "k3",
) -> torch.Tensor:
    """KL primitive used by the actor update.

    For k3, with u = pi_ref/pi_theta = exp(ref_log_prob - log_prob), the unclamped
    expression (u - 1) - log u is unbiased for KL[pi_theta || pi_ref] under x ~ pi_theta
    and is non-negative because log u <= u - 1.
    """
    if kl_penalty in ("kl", "k1"):
        return log_prob - ref_log_prob
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (log_prob - ref_log_prob).square()
    if kl_penalty not in ("low_var_kl", "k3"):
        raise NotImplementedError(f"unsupported KL penalty: {kl_penalty}")

    kl = ref_log_prob - log_prob                     # log u
    kl = torch.clamp(kl, min=-20.0, max=20.0)        # numerical stability before exp
    u = torch.exp(kl)                                # pi_ref / pi_theta
    kld = u - kl - 1.0                                # (u - 1) - log u  =  pi_ref/pi_theta - log(...) - 1
    return torch.clamp(kld, min=-10.0, max=10.0)


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,     # (bs, response_len) under pi_theta_old (the sampler)
    log_prob: torch.Tensor,         # under the current pi_theta
    advantages: torch.Tensor,       # (bs, response_len) from compute_grpo_outcome_advantage
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,        # PPO clip eps
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,      # dual-clip cap for negative-advantage tokens
) -> torch.Tensor:
    """PPO-style clipped actor loss used after the group-relative advantages are computed."""
    # rho_t = pi_theta / pi_theta_old, computed in log space for stability
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # clipped surrogate: pessimistic lower bound on rho * A
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)   # = -min(rho*A, clip(rho)*A)

    # dual-clip: for negative-advantage tokens, cap the loss at -clip_ratio_c * A
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    return masked_token_mean(pg_losses, response_mask)


def grpo_actor_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    kl_coef: float = 0.04,           # beta
    kl_penalty: str = "k3",
) -> torch.Tensor:
    """Fill the actor-loss slot: clipped surrogate plus explicit reference-policy penalty."""
    pg_loss = compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask)
    if ref_log_prob is not None and kl_coef != 0.0:
        kld = kl_penalty_forward(log_prob, ref_log_prob, kl_penalty=kl_penalty)
        kl_loss = masked_token_mean(kld, response_mask)
        pg_loss = pg_loss + kl_coef * kl_loss

    return pg_loss
```

So the causal chain, start to finish. I wanted PPO's stable clipped update and a variance-reducing baseline, but PPO's baseline is a learned, policy-sized value network, and in the LLM setting that network is both a heavy memory cost and — because the reward arrives only at the last token — nearly impossible to fit accurately per token, which is exactly what GAE needs from it. The group of sampled outputs for the same question gives me a practical Monte-Carlo baseline at the response level: subtract the mean reward, normalize by the std to put every question on a comparable z-scored footing, and broadcast that scalar advantage to all tokens because a single terminal reward gives no within-sequence signal. Folding the KL anchor into the reward and then z-scoring it entangles the regularizer with the advantage, so I pull the KL out into an explicit loss term, and estimate it per token with the control-variate form `(pi_ref/pi_theta) - log(pi_ref/pi_theta) - 1`, which is unbiased like the naive `log(pi_theta/pi_ref)` before numerical clipping but always non-negative. Writing every method's gradient as a per-token coefficient times `grad log pi` shows the payoff: my coefficient is `A_hat + beta(pi_ref/pi_theta - 1)` — graded and signed where rejection-sampling's coefficient is a binary gate, and recovering PPO's expressiveness with the group z-score standing in for the GAE advantage and no value network anywhere. The whole thing drops into the existing generate-score-credit-loss-step loop as three primitives: a group-relative outcome advantage, a clipped actor loss, and a k3 reference-policy penalty.
