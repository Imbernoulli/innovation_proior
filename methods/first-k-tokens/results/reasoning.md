Let me start from what's actually breaking, because I have a concrete failure in front of me: I take a strong base model, I run the obvious critic-free RL recipe on verifiable math — sample a group of completions per problem, score each correct/incorrect with a rule, standardize the group reward into an advantage, and ascend the clipped policy-gradient objective — and it stalls far below the systems this setup is meant to reproduce, while the metrics show characteristic instabilities. That tells me to stare at exactly *where* the standard recipe leaks and patch each leak with the smallest change that's still principled. Let me first get the recipe completely concrete so I have something to poke at.

The behavior policy π_old generates a completion o token by token. The current policy π_θ assigns each generated token a probability, and the quantity every clipped method turns is the per-token importance ratio r_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t}) — how much more or less likely my current policy is to have emitted that token than the policy that actually emitted it. The advantage Â_{i,t} tells me whether that token's continuation beat a baseline. PPO's move, which I'm inheriting, is to ascend min(r·Â, clip(r, 1−ε, 1+ε)·Â): the clip kills the incentive to drag r outside [1−ε, 1+ε], and the outer min makes it a pessimistic lower bound so I never get rewarded for a ratio move that only *looks* good. Since I'm critic-free, I get Â not from a value function but group-relatively: draw G completions for one prompt, score them, and set Â_{i,t} = (R_i − mean({R})) / (std({R}) + ε_num) for every token of completion i — the whole completion shares one standardized scalar, with ε_num only preventing division by zero. And the version in front of me carries a KL penalty β·D_KL(π_θ ‖ π_ref) to a frozen reference, plus a particular way of averaging the per-token losses: within each completion average over its tokens with a 1/|o_i|, then average the G completions with a 1/G. Fine. That's the machine. Now where does it bleed?

The first thing I see on the metrics is entropy collapsing — fast. Within a few hundred steps the policy's token distribution sharpens, the G completions for a prompt come back nearly identical, and the model has effectively decided it's done exploring. That's fatal for this kind of training, because the long reasoning behaviors I'm trying to reinforce only exist if the policy keeps *sampling* them; a deterministic policy can't discover a better chain of thought it never emits. So I need to ask, mechanically, what in my update is squeezing entropy out. My instinct is to blame the clip, because the clip is the only thing in the objective that treats tokens asymmetrically depending on their probability — let me check that instinct by actually computing what the clip does to a token as a function of its probability.

Take a token with positive advantage — the update wants to make it more probable. The clip on the ratio is symmetric, r ∈ [1−ε, 1+ε], with ε = 0.2 the usual default. The ratio is π_θ / π_old, so the largest the clip lets the new probability become is π_old · (1 + ε). Now plug in two tokens. An exploitation token that the behavior policy already loved, π_old = 0.9: its cap is 0.9 · 1.2 = 1.08 — which is above 1, i.e. no active cap inside the probability simplex, so this token can be pushed essentially all the way to certainty before the upper ratio clip matters. An exploration token that the behavior policy barely considered, π_old = 0.01: its cap is 0.01 · 1.2 = 0.012. So in one update the confident token has room to move up to the simplex ceiling, while the rare token gets only an absolute 0.002 of room. The symmetric ratio clip, read in probability space, is wildly asymmetric in its *effect*: it lets winners run and pins down underdogs. And that's exactly the wrong way around for keeping a policy exploratory — it's a ratchet that concentrates probability mass on the already-likely tokens and throttles the rise of the unlikely ones. Of course entropy collapses; the clip is structurally an entropy sink for positive-advantage updates. And there's a measurement that backs this up rather than me just reasoning about it: among the tokens whose positive-advantage update actually gets clipped at the top, the typical token probability is low, below about 0.2. So the upper clip is biting precisely the exploration tokens. That's the leak, named precisely.

So how do I let the underdogs rise without also blowing up the whole trust region? The clip exists for a reason — it's what keeps off-policy reuse of the sampled batch stable, it's the proximal leash. I don't want to throw it away; I want to loosen it only where it's hurting me. The clip has two sides: 1−ε on the low end (how far a token's probability can be pushed *down*) and 1+ε on the high end (how far *up*). My problem is entirely on the high end — the upper bound is what pins the exploration tokens. The low end is doing something I actually want to keep tight: it's the bound that protects a token from being suppressed too aggressively, and if I loosened it, a token's probability could be driven toward zero in a single step, which would amputate parts of the sampling space rather than just sharpening it. So the fix writes itself once I separate the two sides instead of forcing them to share one ε. Decouple the clip:

```
min( r_{i,t} Â_{i,t},  clip(r_{i,t}, 1 − ε_low, 1 + ε_high) Â_{i,t} ),   ε_high > ε_low.
```

Raise only ε_high to give the low-probability tokens room to climb; leave ε_low where it is so I don't push probabilities to zero and collapse the sampling space. How far to raise it? It's a trust-region knob, so I can't go wild — too large and I lose the stability the clip buys. ε_low stays at the familiar 0.2, and ε_high I nudge up to 0.28: enough headroom that an exploration token at π_old = 0.01 now caps at 0.01 · 1.28 instead of 0.01 · 1.2 — still a small absolute room, but across a long run of updates the looser ceiling gives the exploration tokens a path to grow instead of hitting the old ceiling immediately. The asymmetry isn't arbitrary tuning; it's the direct consequence of noticing that one side of the clip is the entropy sink and the other side is load-bearing.

That gives me an entropy-side patch. Now the second leak is quieter because the run doesn't crash, it just gets weaker over time. With group-relative advantages, Â_{i,t} = (R_i − mean({R})) / (std({R}) + ε_num). Watch what happens under the binary correctness reward when a prompt becomes easy: all G completions come back correct, so all G rewards are equal, the group mean equals every R_i, the numerator is zero for the whole group — and the standard deviation in the denominator is zero too, with the numerical ε only keeping the division finite, so the entire group's advantage is exactly zero. Zero advantage means zero gradient from that prompt. Same thing if all G are wrong. The dynamic is corrosive rather than benign: as training succeeds, more and more prompts tip into the all-correct bucket. I watch the fraction of accuracy-1 groups climb steadily. So the count of prompts in each batch that actually carry a gradient *shrinks over training* — exactly when I'd most like a strong, clean signal, my effective batch is quietly emptying out into zero-gradient groups. A smaller effective batch means the surviving gradient is noisier and the update magnitude shrinks; sample efficiency degrades right as the model is getting good. It's not a crash, it's a slow starvation.

My first thought is the cheap one: just throw away the dead groups, filter out every prompt with accuracy 0 or 1. But then the number of *live* prompts per step jumps around — some batches happen to have many decisive groups, some have few — and now my effective batch size is varying step to step, which is its own source of instability (the gradient's scale and noise change run to run for reasons that have nothing to do with the policy). So filtering alone trades one problem for another. The fix is to filter *and refill*: keep over-sampling prompts and discarding the all-correct and all-wrong groups until the batch is full of groups that have a mix — at least one correct and at least one wrong completion, 0 < #correct < G. Then every prompt in the batch contributes a nonzero, well-defined standardized advantage, and the effective batch size is held constant at the target. As a constraint on the objective it reads

```
s.t.   0 < |{ o_i : is_equivalent(a, o_i) }| < G,
```

which is just "this prompt's group is neither all-right nor all-wrong." The natural worry is cost: doesn't over-sampling burn generation budget? The wall-clock of a synchronized rollout step is dominated by the few longest completions in the batch — the long-tail of long-CoT generations — so extra groups used for refill can hide under that tail. The important part is that I keep a constant supply of effective gradients instead of letting the live batch size fluctuate.

Third leak, and this one I have to derive carefully because it's about *how I'm averaging*, which is easy to get wrong by habit. The objective I inherit reduces the per-token losses sample-level: (1/G) Σ_i (1/|o_i|) Σ_t — average the tokens within a completion first, then average the completions. Let me write out what weight a single token actually gets. A token sitting in completion i contributes to the loss with coefficient (1/G)·(1/|o_i|). So a token in a 100-token answer gets weight 1/(100G); a token in a 10,000-token answer gets weight 1/(10000G). Same G. The long completion's tokens are down-weighted by a factor of 100 *per token* relative to the short one. Stare at what that does over a long-CoT run. The completions I most care about — the genuinely long, high-quality chains of reasoning — have all their tokens whispering, each contributing a hundredth of what a short answer's token contributes, so the model barely learns the reasoning patterns inside them. And symmetrically, the degenerate long completions — the ones that run away into repetition and gibberish — also get their per-token loss whispered, so when those bad patterns deserve a strong negative push, the sample-level averaging mutes the penalty exactly because the response is long. I watch the consequence: length and entropy drift upward, and the longest responses are the lowest quality, because nothing in the loss is penalizing length-padded garbage with any force. The averaging is rewarding length by neglecting it.

The cure follows straight from the diagnosis: I want every token to count the same regardless of which completion it happens to live in. So don't average within a completion and then across completions; pool all the tokens in the batch and average once, normalizing by the total token count:

```
(1 / Σ_i |o_i|) Σ_i Σ_t ( ... ).
```

Now a token's weight is 1/(Σ_i |o_i|) — identical for every token in the batch, long completion or short. Longer completions now have proportionally more influence on the update simply because they contain more tokens, which is right: more tokens of good reasoning *should* move the policy more, and more tokens of garbage *should* be penalized more. From a single token's point of view, a generation pattern that helps or hurts gets the same push whether it appeared in a short answer or buried deep in a long one. This is a token-level loss, and the only thing I changed is the normalization constant — but it's the constant that decides whether long-CoT learning is balanced or systematically biased toward shortness.

Fourth leak, in the reward itself, before advantages are even computed. I cap generation length and truncate completions that run over. By default a truncated completion gets a punitive reward — it doesn't finish, mark it wrong. But think about what that punishes. A completion might be reasoning perfectly soundly and simply not have wrapped up before the length cap; truncating it and slapping it with a penalty tells the model that a *correct line of reasoning* is bad, purely because of where the cap falls. That's noise injected straight into the reward — the cleanest signal I have, the verifiable correctness, is being corrupted by a length artifact. The minimal fix is to stop letting the truncation contaminate the gradient at all: mask out the loss of truncated samples, so a completion that gets cut off neither helps nor hurts. I am no longer teaching the model that long-and-unfinished equals wrong.

But pure masking leaves a gap: if overlong completions simply vanish from the gradient, the model gets no signal at all about length, and nothing stops it from drifting longer and longer until everything is truncated and masked. So I want a *soft* steer rather than a hard cliff or a blind mask — a length-aware penalty that is zero until the expected response length and then ramps through the reserved buffer just below the hard cap. Let the hard generation cap be L_max, let the cache width be L_cache, and therefore let the no-penalty boundary be L_max − L_cache. So

```
R_length(y) = 0,                                      if |y| ≤ L_max − L_cache
            = ((L_max − L_cache) − |y|) / L_cache,    if L_max − L_cache < |y| ≤ L_max
            = −1,                                     if |y| > L_max,
```

and this length term is *added* to the rule-based correctness reward, not substituted for it. Check the endpoints: at |y| = L_max − L_cache the middle branch gives 0, continuous with the first branch; at |y| = L_max it gives −L_cache/L_cache = −1, continuous with the third branch. So R_length slides smoothly from 0 down to −1 across the cache — a graded "you're getting too long" signal rather than a step function that punishes one extra token as harshly as a thousand. With L_max = 20480 and L_cache = 4096, responses up to 16384 tokens pay nothing, the next 4096 tokens pay a smoothly growing toll, and a response at the hard cap pays the full −1. The model is nudged away from runaway length without being told that any single long-but-sound chain is a mistake.

While I'm cleaning up the reward and the objective, the KL term is sitting there, and I should ask whether it's earning its place rather than leaving it on out of habit. The β·D_KL(π_θ ‖ π_ref) penalty is inherited from alignment training, where the entire point is to nudge a model's behavior while keeping it close to the initial model — you don't want the aligned model wandering off and forgetting how to be itself. But my situation is the opposite. I *want* the policy to wander far from the base model; learning to reason at length means the output distribution should diverge substantially from where it starts. So a leash to a frozen reference is fighting the very thing I'm training for — it's a regularizer toward a distribution I'm deliberately trying to leave. There's no alignment-preservation requirement here; the reward is verifiable correctness, not a learned preference I have to avoid over-optimizing in the alignment sense. So I drop the KL term entirely. The objective gets simpler and the policy is free to move.

Now let me assemble the whole objective from the four patches and the removed KL, and make sure the pieces compose. Sample a group {o_i}_{i=1}^G per prompt; add the soft-overlong term to each scalar reward before the group-relative standardization; use the truncated-sample mask only to remove those response tokens from the policy loss; form the per-token ratio r_{i,t}; apply the decoupled clip; reduce token-level; and only keep prompts whose group is mixed:

```
J(θ) = E_{(q,a)~D, {o_i}~π_old} [ (1 / Σ_i |o_i|) Σ_i Σ_t
            min( r_{i,t} Â_{i,t},  clip(r_{i,t}, 1 − ε_low, 1 + ε_high) Â_{i,t} ) ]
       s.t.  0 < |{ o_i : is_equivalent(a, o_i) }| < G,
```

with r_{i,t} = π_θ(o_{i,t}|q,o_{i,<t})/π_old(...) and Â_{i,t} = (R_i − mean({R})) / (std({R}) + ε_num). Four named changes off the GRPO baseline: decoupled clip (ε_low, ε_high), the mixed-group sampling constraint, token-level normalization, and the reward shaping — plus the KL removal. None of them is a new theoretical object; each is the minimal edit that closes a specific observed leak, and they're orthogonal enough to stack.

Let me make sure I actually understand the clip's interaction with the sign of the advantage before I write code, because the min and the clip together do something I want to be precise about, and it's the thing the implementation has to get right. The objective per token is min(r·Â, clip(r,·)·Â). When Â > 0, I'm trying to raise r; the unclipped term r·Â grows with r, the clipped term saturates at (1+ε_high)·Â once r exceeds 1+ε_high, so min picks the saturated one — the gain is capped, I get no credit for pushing r past the ceiling. When Â < 0, lowering r makes r·Â less negative, which would look better if I could drive r toward zero. The lower clip prevents that: once r falls below 1−ε_low, clip(r)·Â = (1−ε_low)·Â is more negative than r·Â, so the min picks the clipped branch and gives no extra credit for suppressing the token further. For bad tokens whose ratio becomes huge, r·Â becomes very negative, so the objective keeps penalizing that move rather than hiding it behind the upper clip. The sign cases are coherent, but the implementation is clearer if I convert to the loss form the optimizer actually minimizes, because maximizing J is minimizing −J.

In loss form, define pg_losses1 = −Â·r and pg_losses2 = −Â·clip(r, 1−ε_low, 1+ε_high). Maximizing min(a,b) over the pair (r·Â, clip·Â) is minimizing max(−r·Â, −clip·Â) = max(pg_losses1, pg_losses2). So the per-token loss is max(pg_losses1, pg_losses2), and the optimizer descends it. That single line — element-wise maximum of the unclipped and clipped negative-advantage products — is the entire clipped surrogate, sign-correct by construction, no case analysis needed. The fraction of tokens where the clipped branch wins, mean over the response of [pg_losses2 > pg_losses1], is the clip-fraction I want to monitor. And there's one more guard worth keeping from robust PPO implementations: a dual lower clip for the negative-advantage tail. If a token has Â < 0 and a huge ratio, −Â·r can be enormous and a single token can dominate the batch; clamp the loss for negative-advantage tokens at −Â·c for some c > 1 (c = 3 is the standard floor), i.e. for Â < 0 take min(loss, −Â·c). That's an orthogonal stability guard, not one of my four, but it's cheap insurance and it composes cleanly: where(Â < 0, min(clipped_loss, −Â·c), clipped_loss).

For numerical safety the ratio has to be formed from log-probs, not probabilities — π_θ and π_old are tiny per-token numbers and their quotient underflows or overflows. So compute the log-ratio log π_θ − log π_old, clamp it to a safe band like [−20, 20] before exponentiating (a token whose log-ratio is genuinely ±20 is already so far off-policy that the clip will saturate it anyway, so clamping costs nothing real and prevents exp from blowing up), then exponentiate. The negative log-ratio's masked mean over the response is also a free, useful diagnostic — it's an estimate of the per-token KL between old and new policy, which I want to watch even though I drop the KL *penalty*; monitoring divergence is not the same as penalizing it.

Now the token-level aggregation in code. I have a per-token loss matrix of shape (batch, T) and a response mask marking which positions are real response tokens. Token-level pooling is: sum the masked loss over all positions in the whole batch and divide by the total number of valid tokens — masked_sum(loss, mask) / total_valid_tokens. That's exactly the (1/Σ_i|o_i|) Σ_i Σ_t normalization, every token weighted equally. (The alternative reduction — average within each sequence, then average sequences — is the sample-level 1/G·1/|o_i| I'm rejecting; I just don't select that mode.) So I let the aggregator take a mode argument and pass "token-mean."

Let me write the loss the way I'd actually ship it, filling the objective slot in the trainer. Two log-prob tensors and the advantages and mask come in; the clip ranges come from config so ε isn't hardcoded; out goes the scalar loss plus the metrics I watch.

```python
import torch
from typing import Any, Optional


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, T)  log π_old(o_t | q, o_<t)
    log_prob: torch.Tensor,          # (bs, T)  log π_θ(o_t | q, o_<t)
    advantages: torch.Tensor,        # (bs, T)  group-relative Â_{i,t}
    response_mask: torch.Tensor,     # (bs, T)  1 on response tokens
    loss_agg_mode: str = "token-mean",   # token-level pooling (1 / Σ|o_i|)
    config: Optional["ActorConfig"] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)   # dual lower clip for Â<0 tail
    assert clip_ratio_c > 1.0

    # ratio from log-probs, clamped for stability before exp
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # clipped surrogate in loss (minimization) form: max of unclipped & clipped
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # dual lower clip: cap the negative-advantage loss so one huge-ratio token can't dominate
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # token-level aggregation: pool every token equally, normalize by total valid tokens
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
```

and the two pipeline slots — reward shaping and the dynamic, refill-to-full sampling — alongside it:

```python
def shape_reward(
    raw_correct,
    length,
    truncated,
    response_mask,
    max_response_length=20480,
    overlong_buffer_len=4096,
    penalty_factor=1.0,
):
    # +1 / -1 rule-based correctness
    length_f = length.to(torch.float32)
    R = torch.where(raw_correct, torch.ones_like(length_f), -torch.ones_like(length_f))

    # soft overlong penalty: 0 through max_len-buffer, then down to -penalty_factor at max_len
    expected_len = max_response_length - overlong_buffer_len
    exceed_len = length_f - expected_len
    R_length = torch.minimum(
        -exceed_len / overlong_buffer_len * penalty_factor,
        torch.zeros_like(length_f),
    ).clamp(min=-penalty_factor)

    # overlong filtering, when enabled, removes truncated sequences from the policy loss
    sequence_mask = (~truncated).to(response_mask.dtype).unsqueeze(-1)
    filtered_response_mask = response_mask * sequence_mask
    return R + R_length, filtered_response_mask


def build_training_batch(prompts, policy_old, reward_fn, sampler, G, target_prompts):
    # dynamic sampling: keep only mixed groups (0 < #correct < G), refill to a fixed size
    kept = []
    while len(kept) < target_prompts:
        for q, a in sampler(prompts):
            outs = policy_old.sample_group(q, G)
            correct = [is_equivalent(a, o) for o in outs]
            n_correct = sum(correct)
            if 0 < n_correct < G:                       # nonzero std -> nonzero advantage
                rewards, response_mask = reward_fn(q, a, outs)
                kept.append((q, a, outs, correct, rewards, response_mask))
                if len(kept) >= target_prompts:
                    break
    return assemble_group_relative_advantages(kept)     # Â = (R - mean) / (std + eps) per group
```

So the causal chain, end to end. I start from a critic-free, group-relative clipped-policy-gradient baseline that stalls on long-CoT math RL and watch it fail in four specific places. The symmetric ratio clip, read in probability space, caps low-probability exploration tokens far harder than high-probability exploitation tokens, so it's an entropy sink — decoupling the clip and loosening only the upper side (ε_low = 0.2, ε_high = 0.28) gives the exploration tokens room without letting probabilities be driven to zero. Under the binary correctness reward, group-relative advantages vanish whenever a group is all-correct or all-wrong, and the all-correct fraction grows over training, slowly emptying the effective batch — over-sampling and refilling with only mixed groups (0 < #correct < G) holds the effective batch constant. Sample-level loss reduction weights each token by 1/|o_i|, so long high-quality chains are under-learned and long garbage is under-penalized — pooling all tokens and normalizing by the total token count weights every token equally. Truncated overlong samples poison the verifiable reward with a length artifact — masking them and adding a smooth length-aware penalty that ramps from 0 to −1 across a buffer below the cap steers length without punishing sound long reasoning. And the alignment-era KL leash to a frozen reference fights the deliberate, large divergence that long-CoT learning requires, so it comes off. Each fix is the smallest principled change to one leak; together they make the objective J(θ) above — Decoupled Clip and Dynamic sAmpling Policy Optimization — which drops into the existing trainer as a clipped-surrogate loss in max-of-unclipped-and-clipped form with token-level aggregation, plus the two pipeline edits for sampling and reward.
