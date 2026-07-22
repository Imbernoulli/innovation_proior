OK, let me think about why generating from a big Transformer is so painfully slow, because the slowness has a specific shape and I want to understand the shape before I reach for a fix. I have a model that gives me `p(x_t | x_{<t})`, a distribution over the next token given everything so far. To produce a length-`K` continuation I sample `x_1`, append it, sample `x_2` conditioned on `x_1`, and so on. Every token requires its own forward pass, and pass `t+1` literally cannot begin until pass `t` has produced its token, because the token *is* part of the input to the next pass. So the wall-clock cost is `K` forward passes laid end to end, strictly serial. That's the whole problem in one sentence: `K` tokens, `K` serial runs of the model.

Now the instinct is "make each pass cheaper" — distill the model down, quantize it, sparsify it, swap in cheaper attention. All of that works, and all of it does the one thing I've been told I'm not allowed to do: it changes the model. The cheaper model is a *different* model with a *different* output distribution. I want the exact same tokens the big model would have produced, just faster. So shrinking the model is off the table. Same objection kills the adaptive-computation line — early-exit at a shallow layer when some confidence head fires, or route easy inputs to a small model in a committee. These lean on a real and useful observation, that not every next-token decision is equally hard (lots of positions are closing a bracket or finishing a common word, where even a tiny model nails it), but they all decide *when to take the shortcut* with a heuristic, and the moment you take the shortcut you've emitted a token the big model might not have. The distribution drifts. So: keep the model, keep its outputs, just stop wasting time.

To find the waste I have to look at where the time actually goes in a single decode pass, not just count passes. At batch size 1 — one sequence, decoding one token — what is a forward pass doing? It reads every weight of the model out of memory, reads the whole key/value cache, multiplies, and produces one token's logits. The number of multiply-adds is tiny relative to how much the accelerator *can* do; the thing that dominates the clock is moving bytes — streaming all those weights and the KV cache out of high-bandwidth memory. The pass is memory-bandwidth-bound. The arithmetic units are mostly sitting idle waiting for data.

Stare at that for a second, because it's strange and it's the opening. The bottleneck is reading the weights, and I read *the entire model* whether I use it to score one token or to score fifty. If I had a block of tokens `x_1, ..., x_K` already in hand and I wanted the model's distribution at every position, I could feed the whole block through in *one* causally-masked pass — that's exactly how the thing is trained — and get `p(·|x_{<1}), ..., p(·|x_{<K})` all at once. And because the pass is memory-bound, that one block pass costs almost the same wall-time as a single-token pass: same weights streamed once. Scoring fifty tokens is nearly as cheap as scoring one.

So the asymmetry is glaring. *Verifying* a guessed block of tokens is cheap — one pass. *Generating* the block is expensive — `K` passes — purely because generation discovers the tokens one at a time and verification gets them handed over. If only I had the tokens already, I could check them all for free. But of course I don't have them; producing them is the entire task.

Unless... something cheaper produces a *guess* at them. This is where the "easy vs. hard steps" observation stops being a curiosity and becomes leverage. If a small, fast model `M_q` gets most of the easy positions right, I can let it run ahead and propose the next `γ` tokens autoregressively — `γ` cheap passes — and then run the big model `M_p` *once* over that proposed block to score all `γ+1` positions in parallel. Where `M_q` guessed what `M_p` would have done, I keep the token essentially for free; where it guessed wrong, I catch it. The shape of this is familiar from processors: do a task in parallel with checking whether it was actually needed, and win concurrency whenever the guess turns out right — branch prediction does exactly this. The prerequisite there is a cheap accurate guesser, which is precisely what a small language model is for the easy positions. So the design is "guess a block cheaply, verify it in one expensive-but-amortized pass" — the question is whether I can make the verify step *correct*, not just fast.

But there's a gap between the processor version and mine, and it's the whole technical problem. In branch prediction the guessed task is either needed or not — a deterministic yes/no. Here "what `M_p` would have produced" isn't a single token, it's a *distribution* `p(·)`. `M_p` doesn't have one right answer I can check the guess against; it has a probability over all tokens. So what does it even mean to "accept" `M_q`'s guess? I can't just compare to an argmax, because then I'd only be correct for greedy decoding and I'd be silently changing the distribution for temperature-1 sampling — back to the forbidden territory. I need to generalize speculative execution to the stochastic setting: the guessed token *might be* the right one with some probability, and I have to accept or reject it in a way that leaves me with a sample that is *exactly* `p`-distributed.

Let me strip it down to a single position so I'm not confused by the block structure. I have the small model's distribution `q(x)` and the big model's distribution `p(x)` over the same vocabulary. I've drawn a guess `x ~ q(x)`. I want to end up holding a sample from `p(x)`. What's the rule?

First idea, the obvious one from classical statistics: rejection sampling. To sample `p` using proposal `q`, draw `x ~ q` and `r ~ U(0,1)`, accept if `r < p(x) / (M q(x))` with `M = max_x p(x)/q(x)`, else throw everything away and draw again. Accepted samples are provably `p`-distributed. Beautiful — but look at what `M` does to me. `M` is the worst-case ratio over the *entire* vocabulary. If there's even one token where `p` is much larger than `q` — and with a tiny `q` model and a huge vocabulary there always is — then `M` is large, and the accept probability `p/(M q) ≤ 1/M` is tiny. I'd reject almost everything. Worse, every rejection discards the draw and forces a fresh start, so the expensive parallel `M_p` computation I did on the block would be wasted on rejection. That's the opposite of what I want; I wanted to *reuse* `M_p`'s work. Rejection sampling fights me on both counts: the global constant `M` crushes the accept rate, and the retry-from-scratch throws away the verification.

Second idea: importance sampling. Keep every `x ~ q` but weight it by `p(x)/q(x)`. This is great for estimating expectations, but it gives me *weighted* samples, not unweighted draws — I can't emit a token "with weight 1.7." It doesn't produce a `p`-distributed token at all. Dead end for sampling.

So I need something that (a) accepts often, ideally with no global `M`, and (b) when it does reject, produces the correct token *immediately* from a quick fix-up rather than retrying. Let me reason about what acceptance rule could possibly be both cheap and exact.

Think about the two distributions overlaid. Where `q(x) ≤ p(x)`, the small model is being *cautious* — it assigns this token no more probability than the big model does. Every `q`-draw of such a token is "deserved": `p` wanted it at least this much. I should just accept those outright. Where `q(x) > p(x)`, the small model is being *greedy* — it over-proposes this token relative to `p`. I can't keep all of those `q`-draws or I'd over-represent the token. I should keep only a `p/q` fraction of them: accept with probability `p(x)/q(x)`, reject the rest. Combine the cases: accept the guess `x` with probability `min(1, p(x)/q(x))`. No global maximum anywhere — the acceptance for each token only looks at that token's own `p` and `q`. That already smells better than rejection sampling's `1/M`.

Now I have to handle the rejections, and this is the delicate part, because if I get the fix-up wrong the whole exactness guarantee collapses. When I reject, what's "missing"? Let me compute how much probability mass the accept step has *delivered* to each token `x`, and compare to the `p(x)` I'm supposed to deliver. The probability that I draw `x` from `q` and accept it is `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`. So after the accept step, token `x` has received exactly `min(p(x), q(x))` of probability. But it's *supposed* to end up with `p(x)`. The shortfall is `p(x) − min(p(x), q(x)) = max(0, p(x) − q(x))`. Precisely the tokens where `p` wanted more than `q` was willing to give are under-served, and by exactly the amount `p` exceeds `q`. So when I reject, I should resample the token from the distribution proportional to that shortfall: `p'(x) ∝ max(0, p(x) − q(x))`, normalized. Write it `p'(x) = norm(max(0, p(x) − q(x)))`. Intuitively this is "the part of `p` that `q` failed to cover," and it's a quick fix-up — I already have `p` and `q` in hand at this position, no model re-run, no retry loop.

Let me actually prove this lands me on `p` exactly, because "intuitively it patches the shortfall" is not a proof and I've been burned by off-by-a-normalizer mistakes before. Call `β` the overall probability of accepting the `q`-draw. The total probability that the final emitted token equals some specific `x'` splits into the accept branch and the reject branch:

`P(x = x') = P(accepted, x = x') + P(rejected, x = x')`.

The accept branch I already have: I draw `x'` from `q` and accept it, so

`P(accepted, x = x') = q(x') · min(1, p(x')/q(x')) = min(q(x'), p(x'))`.

For the reject branch I need two things: the probability of rejecting at all, and the resample distribution `p'`. Start with `p'`'s normalizer, since that's where these proofs usually go wrong. `p'(x) = max(0, p(x) − q(x)) / Z` with `Z = Σ_x max(0, p(x) − q(x))`. Now `max(0, p − q) = p − min(p, q)` (when `p ≥ q` it's `p − q`; when `p < q` it's `0 = p − p = p − min(p,q)` — yes, both cases give `p − min(p,q)`). So

`Z = Σ_x (p(x) − min(p(x), q(x))) = 1 − Σ_x min(p(x), q(x))`.

Hold onto `Σ_x min(p(x), q(x))` — it keeps showing up. And what is `β`, the accept probability? It's the chance I draw some `x` from `q` and then accept it: `β = Σ_x q(x) · min(1, p(x)/q(x)) = Σ_x min(q(x), p(x))`. So `β = Σ_x min(p,q)` exactly, which means `Z = 1 − β`. The reject probability is `1 − β`, and the resample is `p'`, so

`P(rejected, x = x') = (1 − β) · p'(x') = (1 − β) · (p(x') − min(q(x'), p(x'))) / (1 − β) = p(x') − min(q(x'), p(x'))`.

The `(1 − β)` cancels the normalizer — that's the satisfying part, the reject-probability and the resample-normalizer are the *same* number `1 − β`, so they annihilate and leave the bare shortfall. Add the two branches:

`P(x = x') = min(p(x'), q(x')) + p(x') − min(p(x'), q(x')) = p(x')`.

Exactly `p`. The algebra is clean, but I've miscancelled normalizers in proofs before, so let me not trust it until I've watched it work on actual numbers. Take a three-token vocabulary, `p = (0.5, 0.3, 0.2)` and a deliberately mismatched proposal `q = (0.2, 0.7, 0.1)` — note `q` badly over-proposes token 2 and under-proposes token 1, exactly the kind of skew that should stress the fix-up. The accept branch deposits `min(p, q)` on each token: `min(0.5,0.2)=0.2`, `min(0.3,0.7)=0.3`, `min(0.2,0.1)=0.1`, so `(0.2, 0.3, 0.1)`, summing to `β = 0.6`. The shortfalls are `max(0, p−q) = (0.3, 0, 0.1)`, which sum to `Z = 0.4` — and indeed `Z = 1 − β = 0.4`, so the normalizer-equals-reject-probability identity I leaned on holds here. The residual distribution is therefore `p' = (0.3, 0, 0.1)/0.4 = (0.75, 0, 0.25)`. Now add the branches token by token: token 1 gets `0.2` (accept) `+ 0.4·0.75 = 0.3` (reject) `= 0.5`; token 2 gets `0.3 + 0.4·0 = 0.3`; token 3 gets `0.1 + 0.4·0.25 = 0.1 + 0.1 = 0.2`. That is `(0.5, 0.3, 0.2)` — exactly `p`, on a `q` that gets the ranking of the two top tokens *backwards*. The proof survives contact with numbers, and the case where `q` is badly wrong is precisely where I'd have expected a normalizer slip to show up, so this is reassuring. Nowhere did I assume anything about `q` — not that it covers `p`'s support, not a bound on the ratio, nothing. *Any* `q` works; a bad `q` just rejects more often (here `40%` of the time), it never biases the output. That's a much stronger guarantee than rejection sampling, which needs the finite `M`. The output token is distributed identically to a draw from `p` alone, which is the non-negotiable requirement: the user gets exactly the big model's distribution.

Good. Now lift this from one position back to the block of `γ` guesses, and I have to be careful about *which* `p` and `q` go with each position, and about what happens after the first rejection. The small model produced `x_1, ..., x_γ` autoregressively, so `x_i` was drawn from `q_i(x) = q(·| prefix, x_1, ..., x_{i-1})`. The big model ran once over `prefix + [x_1, ..., x_γ]` and gave me, in parallel, `p_1(x) = p(·|prefix)`, `p_2(x) = p(·|prefix, x_1)`, ..., up to `p_{γ+1}(x) = p(·|prefix, x_1, ..., x_γ)` — one distribution per position including one extra past the last guess. For each `i` from `1` to `γ` I apply the single-position rule: draw `r_i ~ U(0,1)`, accept `x_i` iff `r_i ≤ p_i(x_i)/q_i(x_i)` (and recall if `p_i ≥ q_i` this always holds since the ratio is `≥ 1`).

The subtlety is the first rejection. Say positions `1, ..., n` were accepted and position `n+1` is the first reject. Can I keep checking `n+2, n+3`? No — and here's why, it's not optional. The big model's distribution `p_{n+2}` was computed conditioned on `prefix, x_1, ..., x_{n+1}`, i.e. conditioned on the *rejected* token `x_{n+1}` being in the context. But I'm not going to emit `x_{n+1}`; I'm about to replace it. So everything `M_p` computed downstream of the rejected token is conditioned on a prefix that won't exist. Those distributions are invalid. I must stop at the first rejection, discard `x_{n+1}, ..., x_γ`, and fix up position `n+1`. So the number of accepted guesses is `n = ` the index just before the first `i` with `r_i > p_i(x_i)/q_i(x_i)`, capped at `γ` if none reject.

The fix-up: at the rejected position `n+1` I resample from the residual `p'(x) = norm(max(0, p_{n+1}(x) − q_{n+1}(x)))` — that one extra token is correct by the single-position proof, and it's the token I emit after the `n` accepted ones. What if nobody rejected, all `γ` accepted? Then I've consumed `q_1, ..., q_γ` and there's a leftover, unused big-model distribution `p_{γ+1}` sitting right there — already computed in the same parallel pass, free. I just sample one token straight from `p_{γ+1}` (an ordinary `p`-draw, trivially exact) and emit it as a bonus. Either way I emit *at least one* token per `M_p` pass — the fix-up token in the worst case where even `x_1` is rejected (then `n=0` and I emit one token from the residual at position 1, which is still an exact `p`-draw) — and up to `γ+1` tokens when every guess sticks. That worst case matches standard decoding in target-model passes, but not necessarily in wall-clock time, because I still spent `γ` cheap draft passes; the cost ratio has to appear in the latency calculation.

One loose end before the analysis: real systems don't sample "from `p`," they argmax, or top-k, or nucleus-filter, or temperature-scale. Do I need a separate accept rule for each? No — every one of those is just sampling from *some* adjusted categorical. Argmax is sampling from the distribution that puts all its mass on the max token; top-k is sampling from the truncated-and-renormalized distribution; temperature rescales the logits first. So I standardize: let `p` and `q` already *be* the adjusted distributions for whatever scheme is in use, and the single rule above covers all of them. Clean.

Now, how much do I actually gain? I want the expected number of tokens emitted per `M_p` pass, because that's the factor by which the serial-pass count drops. Let me define the acceptance rate carefully. Given a prefix, `β` is the probability that a `q`-draw at that position gets accepted by speculative sampling — and I computed it above, `β = Σ_x min(p(x), q(x))`. This is a clean measure of how well `M_q` matches `M_p`. Let me sanity-check the form of `β` again from the definition rather than reusing the result: `β = E_{x~q}[ min(1, p(x)/q(x)) ]`. Split by the two cases: where `q(x) ≤ p(x)` the term is `1`, contributing `q(x)`; where `q(x) > p(x)` the term is `p(x)/q(x)`, contributing `q(x)·p(x)/q(x) = p(x)`. Summing, each token contributes `min(q(x), p(x))`, so `β = Σ_x min(p(x), q(x))`. Consistent.

This `Σ min(p,q)` has a nice geometric reading — it's one minus the overlap deficit. Define `D(p,q) = Σ_x |p(x) − q(x)| / 2`. Then since `min(p,q) = (p + q − |p − q|)/2`, summing gives `Σ_x min(p(x), q(x)) = (1 + 1 − Σ_x |p(x)−q(x)|)/2 = 1 − Σ_x |p(x)−q(x)|/2 = 1 − D(p,q)`. So `β = 1 − D(p,q)`, where `D` is the total-variation distance between the two models' distributions — symmetric, sitting in `[0,1]`, zero exactly when `p = q`, one exactly when their supports are disjoint. Acceptance is just "one minus how far apart the two distributions are." The `β`s vary position to position, so let `α = E(β)` be the expected acceptance rate — `α = E[Σ_x min(p(x),q(x))] = 1 − E(D)` — the single number that captures how good the approximation is across the task.

Expected tokens per pass. Make the simplifying assumption that the per-position acceptances are i.i.d. with probability `α` each. The `γ` guesses are checked in order, stopping at the first rejection — that's a geometric process, but capped at `γ` because I only have `γ` guesses, and then there's always the `+1` fix-up/bonus token. So the number of *accepted* guesses is `min(geometric, γ)`, and emitted tokens `= accepted + 1`. Let me just sum the expectation of the accepted count directly: position `i` is accepted only if all of `1, ..., i` were accepted, which under i.i.d. has probability `α^i`. So `E[# accepted] = Σ_{i=1}^{γ} α^i`, and

`E[# tokens] = 1 + Σ_{i=1}^{γ} α^i = Σ_{i=0}^{γ} α^i = (1 − α^{γ+1}) / (1 − α)`.

That's the target-pass payoff formula. Quick sanity checks: at `α → 0` (useless `M_q`) it gives `1` token per target pass — the standard target-pass count, good. At `α → 1` it gives `γ+1` — every guess sticks plus the bonus, the maximum, good. And it's increasing in both `α` and `γ`, as it must be.

But tokens-per-pass isn't the wall-time, because the draft passes aren't free. Let `T` be the wall-time of one `M_p` pass, and let `c` be the cost ratio — the time for one `M_q` run over the time for one `M_p` run. One step of the algorithm runs `M_q` `γ` times (serially, to draft) and `M_p` once (the parallel block verify), costing `T·c·γ + T = T(cγ + 1)`. I'm assuming here that I genuinely have the spare compute to run the `γ+1` positions of `M_p` concurrently without inflating `T` — which is exactly the memory-bound regime I started from, where parallel arithmetic was free. That step produces `(1 − α^{γ+1})/(1 − α)` tokens on average. So the expected cost *per token* is

`T(cγ + 1) · (1 − α) / (1 − α^{γ+1})`,

and standard decoding costs `T` per token, so the wall-time improvement factor is the ratio:

`(1 − α^{γ+1}) / ((1 − α)(cγ + 1))`.

When does this even beat 1? Try the smallest interesting case `γ = 1`: the factor is `(1 − α^2)/((1 − α)(c + 1)) = (1 + α)/(1 + c)`. So as soon as `α > c` — the approximation accepts more often than it costs — `γ=1` already gives `(1+α)/(1+c) > 1`, an improvement. That's a clean condition: `α > c`. I can pin the boundary down with numbers: at `α = 0.15, c = 0.10` the `γ=1` factor is `1.15/1.10 ≈ 1.045`, just above one; drop to `α = 0.05 < c` and it is `1.05/1.10 ≈ 0.955`, just below — the crossover sits exactly at `α = c` as the formula says. Conversely, if some larger `γ` beats 1, then the average `(\alpha + \alpha^2 + ... + \alpha^\gamma)/\gamma` must exceed `c`; that average is at most `α`, so `α > c` is necessary too. For larger `γ` the optimum trades off "more guesses catch more free tokens" against "more wasted draft passes when an early guess is rejected." I should check that this tradeoff actually *turns over* rather than rewarding ever-larger `γ`. Take `α = 0.8, c = 0.1` and walk `γ` up: the speedup goes `1.64, 2.03, 2.27, 2.40, 2.46, 2.47, 2.45, 2.41` for `γ = 1..8`. It climbs, peaks at `γ = 6`, and then *declines* — the `cγ` draft tax in the denominator eventually outruns the saturating `(1−α^{γ+1})` numerator, which tops out near `1/(1−α) = 5`. So there is a genuine interior optimum; `γ` is an integer so I just sweep it for given `α, c` and pick the max rather than pushing `γ` as high as possible. If `M_q` is essentially free, `c ≈ 0`, the factor collapses to `(1 − α^{γ+1})/(1 − α)`, the token count itself, which is monotone in `γ` and bounded above by `1/(1 − α)` as `γ → ∞` — only when drafts are free does "more lookahead" never hurt.

This also tells me how to *pick* `M_q`. Bigger approximation model → better match → higher `α`, but also more cost → higher `c`. Smaller → cheaper `c` but worse `α`. The product in the denominator, `(1−α)(cγ+1)`, is what I'm minimizing, so I want `M_q` at the sweet spot; a model one or two orders of magnitude smaller than `M_p` is the natural place to look, because it can keep a nontrivial amount of next-token structure while making `c` tiny. And nothing forced `M_q` to be a trained neural net at all: a bigram table, or even "copy the next token from earlier in the context" for repetitive text, is a valid `M_q` with `c ≈ 0` — the exactness proof never cared what `q` was, so any of these is safe and any positive `α` it scrapes together is pure speedup.

I should also be honest about the cost I'm trading away, because latency isn't the only currency. Concurrency went up by a factor of `γ+1` (that's `γ+1` positions of `M_p` evaluated at once), so the *peak* compute per step is `γ+1`×. And the *total* arithmetic over a run can go up, not down: when a guess is rejected, the `M_p` work past the rejection point and the `M_q` draft work for the discarded positions were wasted. Let `ĉ` be the per-token arithmetic-ops ratio of `M_q` to `M_p`. One step does `γ` `M_q` token-runs and `γ+1` `M_p` token-evaluations, i.e. ops `∝ ĉγ + (γ+1)` in units of one `M_p` token; dividing by the expected tokens per step `(1−α^{γ+1})/(1−α)`, the factor of increase in total operations is

`(1 − α)(γĉ + γ + 1) / (1 − α^{γ+1})`.

Low `α` → lots of rejected guesses → this blows up; high `α` → close to 1. So this is fundamentally a latency-for-compute trade, only worth it when compute is the spare resource — which, back at the start, is exactly the memory-bound regime: there, the arithmetic was idle anyway, and the thing that genuinely *drops* is memory traffic. The target weights and KV cache get read once per step instead of once per token, so reads shrink by that same `(1 − α^{γ+1})/(1 − α)` factor — precisely the bottleneck I identified at the very beginning. The method attacks the byte-movement that was actually setting the clock.

Let me close the loop on why this beats plain rejection sampling quantitatively, since that was my first instinct and I want to be sure the residual-resample rule was worth it. A non-iterative rejection step accepts with probability `p(x)/(M q(x))`, `M = max_x p(x)/q(x)`. Its expected accept rate is `E_{x~q}[ p(x)/(M q(x)) ] = (1/M) Σ_x p(x) = 1/M`. Writing `1/M = min_{x'} q(x')/p(x')`, this is `Σ_x p(x) · min_{x'} q(x')/p(x') ≤ Σ_x p(x) · min(1, q(x)/p(x)) = Σ_x min(p(x), q(x)) = β` for a fixed prefix, and its task average is at most `α`. So rejection sampling's accept rate is at most my accept rate — generally strictly worse, because the global `min_{x'}` is harsher than the per-token `min(1, ·)`. Speculative sampling accepts at least as often, needs no global `M`, and crucially reuses the parallel `M_p` computation on a rejection instead of throwing it away. The residual rule was worth it.

There is a tempting knob if exact equality is too strict: I could make the accept test lenient by comparing `l q(x)` to `p(x)` for some `l` in `[0,1]`. Then a proposed token is accepted with probability `1` when `l q(x) ≤ p(x)`, and with probability `p(x)/(l q(x))` when `l q(x) > p(x)`. The acceptance rate becomes

`E_{x~q}[p(x)/max(p(x), l q(x))] = Σ_x p(x)q(x)/max(p(x),lq(x)) = (1/l)Σ_x min(p(x),lq(x)) = Σ_x min(p(x)/l,q(x))`.

That does raise acceptance as `l` shrinks, but I pay by weakening the guarantee: no token can be sampled with probability more than `p(x)/l`, yet the distribution is no longer exactly `p`. For the core method I keep `l=1`; the whole point is exactness. If the adjusted distribution is deterministic argmax, this lenience cannot be applied after the one-hot collapse anyway — it has to happen before the logits are standardized.

Beam search also needs a different acceptance object. If ordinary decoding keeps `w` target beams, I can let the helper keep a wider beam `u ≥ w` for `γ` steps, score the `w + uγ` relevant target candidates in parallel, and accept a helper step only while `top_w(M_p) ⊆ top_u(M_q)`. That condition preserves the same beam set the target-only search would have kept; once it fails, I stop using downstream helper guesses because they are conditioned on a beam history I may not keep. The stochastic residual trick is for sampling from a categorical distribution, not for ranking a beam set.

So the whole thing assembles: draft `γ` tokens with a cheap `M_q`, verify them in one parallel `M_p` pass that's nearly free because decoding was memory-bound, accept each guess with probability `min(1, p/q)`, on the first rejection resample that one token from `norm(max(0, p − q))` and stop, and if all stick take a free bonus token from the spare `p_{γ+1}` — emitting between 1 and `γ+1` tokens per target pass, with the output distributed *exactly* as `p` alone for any `M_q` whatsoever, at an expected speedup of `(1 − α^{γ+1})/((1−α)(cγ+1))` when the draft cost and parallel target pass fit the hardware.

```python
import torch
from torch.nn import functional as F


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < kth[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
    return logits


def norm_logits(logits, temperature, top_k, top_p):
    # Standardize argmax / top-k / nucleus / temperature into one categorical.
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs.scatter_(1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs, num_samples=1):
    return torch.multinomial(probs, num_samples=num_samples)


@torch.no_grad()
def autoregressive_decoding(prefix, model, max_len, temperature=1.0, top_k=0, top_p=0.0):
    T = prefix.shape[1] + max_len
    while prefix.shape[1] < T:
        logits = model(prefix).logits
        probs = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
        prefix = torch.cat((prefix, sample(probs)), dim=1)
    return prefix


@torch.no_grad()
def fast_decoding(prefix, helper_model, target_model, max_len, lookahead=4,
                  temperature=1.0, top_k=0, top_p=0.0):
    assert prefix.shape[0] == 1
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]

        # Draft: the helper proposes lookahead tokens autoregressively.
        x = prefix
        for _ in range(lookahead):
            q_logits = helper_model(x).logits
            q_next = norm_logits(q_logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat((x, sample(q_next)), dim=1)

        # q_i for the drafted positions; recomputed to mirror the block check.
        q = helper_model(x).logits
        for i in range(q.shape[1]):
            q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)

        # One parallel target pass scores p_1 through p_{lookahead+1}.
        p = target_model(x).logits
        for i in range(p.shape[1]):
            p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

        # Accept each guess with prob min(1, p_i/q_i); stop at first rejection.
        is_all_accept = True
        n = prefix_len - 1
        for i in range(lookahead):
            drafted = x[:, prefix_len + i].unsqueeze(-1)
            p_i = p[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            q_i = q[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            accept_prob = torch.minimum(torch.ones_like(p_i), p_i / q_i.clamp_min(1e-12))
            if (torch.rand_like(accept_prob) <= accept_prob).item():
                n += 1
            else:
                # First rejection: resample from norm(max(0, p_{n+1} - q_{n+1})).
                residual = torch.clamp(p[:, n, :] - q[:, n, :], min=0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                t = sample(residual)
                is_all_accept = False
                break

        prefix = x[:, :n + 1]
        if is_all_accept:
            # All guesses accepted: the extra target distribution is already available.
            t = sample(p[:, -1, :])
        prefix = torch.cat((prefix, t), dim=1)

    return prefix[:, :T]
```
