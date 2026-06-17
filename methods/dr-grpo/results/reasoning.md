Let me start from the thing that actually bothers me when I watch these runs. I am RL-tuning a base model on math with a verifiable reward — sample a group of responses to a question, give each a 1 if its final answer is right and a 0 otherwise, and push the policy toward what earned reward. And over training the responses get longer and longer. Everyone reads that as the model "learning to reason more." But I keep staring at the curves and something is off: the length keeps climbing well after the reward has flattened. The model writes more and gets no more right. Worse, when I split the length growth by correctness, it is the *wrong* answers that are ballooning — the incorrect responses get longer and longer while the correct ones don't move much. That is not reasoning emerging. That is tokens being burned for nothing, and I want to know what in my training signal is paying for it.

The only place that signal is shaped is the advantage estimator. I hand it a group of scored responses and it tells the actor loss how to weight each response, token by token. So whatever is inflating length has to be hiding in how I turn those scalar scores into per-token advantages, or in how the loss then sums those tokens up. Let me write down exactly what I'm running and look at it as if I'd never seen it before, because clearly I've been taking some piece of it on faith.

What I'm running is the group-relative objective. For a question q I draw G responses, score them R = {R_1, …, R_G}, and I set the advantage of *every* token of response i to the group-normalized score (R_i − mean(R)) / std(R). Then the loss is the average over the group of, for each response, one over its length times the sum over its tokens of the clipped PPO term: (1/G) Σ_i (1/|o_i|) Σ_t min[ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t}]. The group mean is doing the job a value network would do — it's my cheap baseline, no critic to train, which is the whole reason I'm here instead of running full PPO with a value model the size of the policy. Fine. I believe in the baseline. But there are two other things in that formula that I have been treating as "standard normalization," and I want to interrogate both, because they are the only candidates left.

The first is the per-response 1/|o_i| out front. The second is the std(R) in the denominator of the advantage. Let me take them one at a time and ask, very literally: what does this term do to the gradient on a single token?

Start with the 1/|o_i|. Here's a clean way to feel it. Two responses to the same question, both correct, so both have the same positive scalar advantage Â — but one is 20 tokens, the other 200. The loss contribution of each is (1/|o_i|) times the sum over its tokens. Each token carries roughly the same advantage Â, so the per-token gradient magnitude that survives the 1/|o_i| is proportional to Â/|o_i|. The 20-token response gets ten times the per-token push of the 200-token one. So among *correct* answers, shorter ones get the bigger gradient — the policy is being nudged toward brevity when it's right. Okay, that alone sounds almost benign, even nice. But now flip the sign: two *incorrect* responses, both with the same negative advantage Â<0, one short one long. The penalty per token is Â/|o_i|, and Â is negative, so the *long* wrong answer is penalized *less* per token — its 1/|o_i| is smaller, it divides the negative push down. The policy learns it can dodge the penalty for being wrong by being long. That is the engine. Short correct, long incorrect — both biases point the same way for length: shrink the right answers, and let the wrong answers sprawl. And it's the wrong-answer branch that's dangerous, because there's no correctness ceiling on it; a wrong response can grow without ever becoming right, so the length on incorrect responses can run away. That matches exactly what I see in the split-by-correctness curves. So 1/|o_i| is not innocent normalization. It is a length-dependent reweighting of the gradient, and its asymmetry across the sign of the advantage is precisely the runaway-length mechanism.

Now I want to be sure this isn't supposed to be there. Where does the 1/|o_i| even come from? Let me look at the objective I'm *trying* to implement, the PPO surrogate, written honestly: E Σ_t min[ρ_t Â_t, clip(ρ_t,1−ε,1+ε) Â_t]. That is a *sum* over the response's tokens. There is no 1/|o| in it. None. So the 1/|o_i| isn't a derived part of the objective at all — it's something the implementation bolted on. And once I go look, it's everywhere: the standard masked-mean these PPO codebases use is `(per_token_loss * mask).sum(-1) / mask.sum(-1)`, i.e. divide each response's token-sum by its own length. trl does it, OpenRLHF does it, verl does it, the R1-Zero reproductions do it. They're all dividing the per-response loss by `mask.sum`, which is `|o_i|`, a quantity that changes from response to response. Why would everyone do this? I think I can reconstruct it: in pretraining you pack tokens into a fixed-length context and you take `loss.mean(-1)` — divide by the context length, which is a *constant* — purely for numerical stability, so the loss scale doesn't depend on how many tokens you packed. That's harmless because the divisor is fixed. But carried into RL, `mask.sum(-1)` divides by the *response* length, which is not constant, and that turns a harmless constant rescale into a per-response length-dependent reweighting. The bias is an artifact of a habit, not a design.

So the fix for that piece is staring at me: if I want to match the actual PPO objective, the per-token loss should be *summed*, not length-averaged. If I need a divisor at all for numerical scale, it has to be a *constant* — the generation budget, the max token count, anything that doesn't vary with the response — so it rescales the whole loss uniformly and leaves no per-response length dependence. Replace `mask.sum(-1)` with a fixed `MAX_TOKENS`. That kills the length bias at the loss-aggregation level.

Now the std(R) in the advantage. The reading I've been carrying around is "advantage normalization, everybody does it." And it's true that whitening advantages to zero mean and unit variance is a standard stabilization trick. But I need to be precise about *scope*, because I have a feeling the scope is where this goes wrong. When you whiten advantages across an entire batch, you divide every advantage by one global number — the batch std — and that's just a uniform rescale of the whole gradient, which folds into the learning rate and changes nothing about the relative weighting of examples. But here the std is computed *per question, per group*. Each question's centered reward is divided by *that question's* std(R). So now different questions are divided by different numbers, and that does change their relative weight in the update.

Let me make that concrete. Pull out the un-normalized centered score Ã_{i,t} = R_i − mean(R), and write what GRPO actually uses as a reweighting of it: the GRPO advantage is Ã_{i,t} / std(R). So GRPO is the centered score scaled per-question by 1/std(R). Now think about which questions have small std. A question whose group of responses is almost all correct (rewards almost all 1) has tiny std. A question that's almost all wrong (rewards almost all 0) also has tiny std. A question with a mix — the genuinely informative ones, where the model sometimes gets it and sometimes doesn't — has large std. So dividing by std multiplies the update weight of the *too-easy* and *too-hard* questions up, and tamps the mixed-outcome questions down. That's backwards from what I'd want, and regardless of direction, it's a distortion: the per-question 1/std is silently deciding which questions dominate the gradient based on nothing but the spread of their reward, not on how much I should learn from them. That's a difficulty bias, and it exists purely because the normalization is scoped to the group instead of the batch. In the exactly unanimous case the centered numerator is zero, so the advantage itself is zero, but the scale factor has still become singular and has to be patched with epsilon; that numerical patch does not make the near-unanimous weighting right.

So both red terms — the 1/|o_i| and the std(R) — are doing the same kind of damage: each is a reweighting that I never intended, masquerading as standard practice. The 1/|o_i| reweights by length and inflates wrong-answer length; the std(R) reweights by per-question spread and distorts difficulty. The obvious move is just to delete both. But "just delete them because they look bad" isn't enough — I deleted them on a hunch about gradients; I need to check that what's *left* is something principled, that removing them lands me on an estimator I can actually defend as unbiased, not just on "GRPO minus two terms." Let me derive what the gradient is *supposed* to be from scratch and see whether the stripped-down thing is it.

I'm maximizing J(π_θ) = E_q E_{o ~ π_θ(·|q)}[R(q, o)]. (I've dropped the KL-to-reference term — with a rule-based verifier there's no reward-model distribution I need to stay near, so β = 0; that also spares me the reference forward pass.) The Monte-Carlo policy gradient is ∇_θ J = E[ ∇_θ log π_θ(o|q) · R(q, o) ]. The log-prob of the whole response factorizes over tokens, log π_θ(o|q) = Σ_t log π_θ(o_t | q, o_{<t}), so ∇_θ J = E[ Σ_t ∇_θ log π_θ(o_t | q, o_{<t}) · R(q, o) ]. Now the standard sharpening: a token at position t can only influence rewards that come at or after t — it has no effect on the past — so the R(q,o) multiplying the t-th token's score can be replaced by the reward-to-go Σ_{t'≥t} r(q, o_{≤t'}) without changing the expectation. That gives ∇_θ J = E[ Σ_t ∇_θ log π_θ(o_t|·) Σ_{t'≥t} r(q, o_{≤t'}) ].

Next, the baseline. I want to subtract some B(q, o_{<t}) from the reward-to-go to cut variance, and I need to confirm subtracting it doesn't bias the gradient. The condition is that B not depend on the action o_t. Then, taking the expectation over o_t at a fixed prefix: E_{o_t}[ ∇_θ log π_θ(o_t|·) · B(q,o_{<t}) ] = B · E_{o_t}[ ∇_θ log π_θ(o_t|·) ] = B · Σ_{o_t} π_θ(o_t|·) ∇_θ log π_θ(o_t|·) = B · Σ_{o_t} ∇_θ π_θ(o_t|·) = B · ∇_θ Σ_{o_t} π_θ(o_t|·) = B · ∇_θ 1 = 0. Good — that's the log-derivative trick run backward: the score function has zero mean under the policy, so any action-independent B washes out. So I'm free to subtract any such baseline, and ∇_θ J = E[ Σ_t ∇_θ log π_θ(o_t|·) ( Σ_{t'≥t} r(q, o_{≤t'}) − B(q, o_{<t}) ) ]. The thing in parentheses is the advantage. The textbook best B is the expected reward-to-go — the state value V(s_t) — which is exactly what the critic in PPO was estimating and what I'm trying to avoid training.

Now use the structure of *my* reward. It's outcome-level: zero everywhere except the last token, where it's the correctness bit. So the reward-to-go from *any* position t is the same — it's the whole-trajectory return: Σ_{t'≥t} r(q, o_{≤t'}) = R(q, o), for every t. That's a big simplification and it tells me something I'd half-noticed in the formula already: the advantage is the *same scalar for every token* of a response. There's no per-token credit to assign because there's no per-token reward; one number per response, broadcast across its tokens. So the entire estimator collapses to: pick a baseline B, compute R_i − B for each response, broadcast over its tokens. Everything is in the choice of B.

And here is where the group nearly does its job. I don't have a value network for B, but I have G samples from the same prompt. The expected return at the prompt — the state value of the initial state — is E_{o~π}[R(q,o)], and the cheap number sitting in front of me is the group mean. If I center response i by that mean, I get Ã_{i,t} = R(q, o_i) − mean(R), constant in t, broadcast over the tokens. No std has appeared. No 1/|o| has appeared. But I need to be careful before I call this an unbiased baseline: the mean includes R_i itself, so it is not action-independent in the literal baseline-proof sense. The clean multi-sample baseline is leave-one-out — for response i, average the *other* G−1 responses, B_i = (1/(G−1)) Σ_{j≠i} R_j — which by construction does not depend on o_i. So how far is the group-mean-centered advantage from the leave-one-out advantage? Let me compute it exactly. Take the centered advantage and scale it by G/(G−1):

  (G/(G−1)) · (R_i − (1/G) Σ_j R_j)
   = (G/(G−1)) R_i − (1/(G−1)) Σ_j R_j
   = (G/(G−1)) R_i − (1/(G−1)) R_i − (1/(G−1)) Σ_{j≠i} R_j
   = ((G − 1)/(G−1)) R_i − (1/(G−1)) Σ_{j≠i} R_j
   = R_i − (1/(G−1)) Σ_{j≠i} R_j.

That last line *is* the leave-one-out advantage. So the exact relationship is not vague: (G/(G−1)) · (R_i − mean(R)) = R_i − mean_{j≠i}(R_j). Equivalently, the group-mean-centered advantage is ((G−1)/G) times the RLOO advantage. RLOO is the action-independent-baseline estimator; the extra factor ((G−1)/G) is global for a fixed group size and just folds into the learning rate. That's the confirmation I wanted. The mean-only centered return is not a new biased object I have to defend separately; it is the unbiased leave-one-out policy gradient up to a constant scale.

Let me also sanity-check what I just claimed about GRPO from the other side, to make sure I diagnosed it right and didn't strawman it. GRPO's effective advantage is (R_i − mean(R))/std(R), and its loss carries 1/|o_i|. Compared to my Ã_{i,t} = R_i − mean(R), GRPO is *exactly* the RLOO-scaled centered advantage, reweighted: divided per-question by std(R), and (through the loss) divided per-response by |o_i|. So GRPO = RLOO-scaled centered advantage × (per-question 1/std) × (per-response 1/|o|). Both extra factors are reweightings of the correct thing. That's a tidy way to see it: I'm not proposing a different estimator, I'm removing two multiplicative distortions from the same estimator, and the distortions are exactly the two pathologies — the per-response 1/|o| (length) and the per-question 1/std (difficulty).

One more thing I want to make sure I'm not breaking: the easy-question / unanimous-group case. With std removed there is no singular scale factor when a group is all-correct or all-wrong — Ã_{i,t} = R_i − mean(R) just becomes zero for a unanimous group (every R_i equals the mean), which is exactly right: if every sample got the same score, there's no relative signal, the advantage should be zero, and the question simply contributes nothing this step. The degenerate case fixes itself once the std is gone. And there's still the genuine size-1 edge case — a group with a single response, where there's nothing to compare against; there I set the mean to zero so the advantage is the raw return, which is the only fallback available with one sample.

So the algorithm is: per question, sum the per-token rewards to get each response's scalar return; group the returns by question id; for each group compute the mean; subtract the group mean from each response's return; broadcast that scalar over the response's valid tokens; feed it as the advantage (and the return) into the PPO clipped actor loss, whose token-sum I aggregate with a *constant* divisor rather than per-response length. Two deletions and one substitution, all forced by the derivation.

Let me write it as the advantage estimator that drops into the verl registry, since that's where the one empty slot is. The harness hands me the per-token reward tensor (outcome scalar at the last valid token), the response mask, and the group index array; I return per-token advantages and returns, both masked. The grouping, centering, and broadcast happen under no-grad — this is a target computation, not part of the differentiable graph.

```python
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,         # (bs, response_length); 1 = valid response token
    index: np.ndarray = None,            # (bs,) group id; same id == same question
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # outcome reward => reward-to-go from any t equals the whole-trajectory return:
    # one scalar per response.
    scores = token_level_rewards.sum(dim=-1)                  # (bs,) per-sequence return R_i

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])              # bucket returns by question
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)              # single sample: no group baseline
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)      # B = mean(R), the cheap baseline
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # mean-only centered advantage: R_i - mean(R). No std (no difficulty reweighting).
            scores[i] = scores[i] - id2mean[index[i]]
        # broadcast the per-response scalar over its valid tokens; mask the rest.
        scores = scores.unsqueeze(-1) * response_mask         # (bs, response_length)

    # advantage == return here (outcome reward, constant per token).
    return scores, scores
```

And the matching loss-aggregation change, which lives in the actor loss rather than the estimator: where the per-token PPO loss gets reduced, divide each response's token-sum by a fixed constant (the generation budget) instead of by `mask.sum(-1)`, so the aggregation is a bare summed objective up to a uniform scale, with no per-response length factor:

```python
def masked_sum_normalized(per_token_loss, response_mask, max_tokens):
    # was: (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1)   # per-response length -> length bias
    return (per_token_loss * response_mask).sum(-1) / max_tokens               # constant divisor: no length bias
```

The causal chain, start to finish: I saw response length — especially on *wrong* answers — climb after reward had plateaued, pure token waste, and the only place that signal is shaped is the advantage estimator and its loss aggregation. Tracing the per-token gradient, the per-response 1/|o_i| turned out to penalize long wrong answers *less* and shrink short right ones, an asymmetry that drives runaway length, and it isn't even part of the PPO objective — it's a constant-divisor pretraining habit miscarried into a setting where length isn't constant. The per-question std(R) turned out to be a normalization scoped to the group rather than the batch, so it reweights questions by their reward spread — over-weighting the trivially-easy and trivially-hard, distorting difficulty — instead of uniformly rescaling. Deriving the policy gradient from scratch with an action-independent baseline, and using the outcome-reward fact that the advantage is one scalar per response, showed the gradient asks only for a centered return — neither std nor 1/|o| appears — and the algebra showed that R_i − mean(R) is the leave-one-out advantage scaled by ((G−1)/G). So the right estimator is R_i − mean(R) broadcast over tokens, the loss aggregated with a constant divisor: GRPO with its two unrequested reweightings removed, giving the same Monte-Carlo policy-gradient direction as the unbiased leave-one-out baseline up to a global scale.
