The dr_grpo run confirmed the difficulty-bias diagnosis, and it did it on exactly the splits I
predicted. Deleting the per-group std moved the aggregate `score_mean` from grpo's −0.5166 up to
−0.5049, and the movement is concentrated where I said it would be: MATH-500 went 0.2973 → 0.3070 and
AMC went 0.0934 → 0.1009, while GSM8K barely twitched (0.4668 → 0.4671). Let me read those deltas before
I move, because their *sizes* are as informative as their signs. The aggregate gain is `+0.0117`. On
the benchmarks, GSM8K moved `+0.0003` — flat to three digits, which is what "the easy split is near-
unanimous either way" predicts. MATH-500 moved `+0.0097`, a `+3.3%` relative lift; AMC moved `+0.0075`,
which on a base of 0.0934 is `+8.0%` relative — the largest proportional gain of the three, exactly the
hardest split. So the direction and the *shape* of the recovery both match the difficulty-bias story:
strip the per-group std, and the prompts whose informative mixed-outcome groups were being tamped down
recover most. The per-group std *was* a distortion, not a stabilizer, and removing it was the right
call. Good.

But dr_grpo is still not the strongest thing I can run, and the reason is the other half of the trade I
made when I deleted the std. The std was doing one *legitimate* job badly: trying to put different
prompts' advantages on a comparable scale. By deleting it I removed the distortion but also removed all
cross-prompt scale normalization — different prompts' centered returns now enter the gradient on
incomparable raw scales. I priced this out at step 2: after mean-only centering the per-response
magnitudes range from about `0.06` to `0.94` across prompts depending on their solve rate, with no
common yardstick, so the batch gradient is dominated by whichever prompts happen to have large centered
scores. The aggregate ticked up, but only by `+0.0117` in score, and AMC at 0.1009 is still leaving a
lot on the table. I suspect the missing scale is why. So the question for this step is sharp: can I get
the legitimate scale normalization back *without* bringing back the difficulty bias?

Let me restate what is actually wrong with the std so I attack the right thing. The std was bad for two
reasons, and they share a root. First, it is *scoped to the group* — 16 samples — so dividing by it
reweights prompts by their reward spread, the difficulty bias the numbers confirmed. Second, even
ignoring scope, a 16-sample std is a brittle scale estimate: near-unanimous groups (common with outcome
rewards) make it tiny and noisy. Both failures come from the same place — the scale is estimated from
the same handful of samples as the numerator, and from too few of them. So I should not try to "estimate
the std more carefully within the group"; the group is small by construction and I cannot enlarge it
without paying 16× the rollout cost.

I want to know how the trouble depends on sample count, because that will tell me the fix. Model a group
as `r_i = θ + ε_i` with `ε_i ∼ N(0,σ²)` i.i.d., `θ` the true prompt baseline. The std-normalized
advantage is `A_i = (ε_i − ε̄)/D` with `D = sqrt((1/N)Σ_j(ε_j − ε̄)²)`. Condition on `ε_i` and ask
whether the denominator behaves like a constant. Computing the conditional second moment of `D²` gives
`E[D²|ε_i] = ((N−1)²/N²)σ² + ((N−1)/N²)ε_i²` — call it `α + βε_i²` with `β = (N−1)/N² > 0`. Let me not
just assert that; the pieces are worth seeing, because the whole fix hinges on the `N`-dependence.
`Σ_j(ε_j−ε̄)² = Σ_j ε_j² − Nε̄²`. Conditioning on `ε_i` and using `E[ε_j]=0`, `E[Σε_j²|ε_i] = ε_i² +
(N−1)σ²`, and `E[Nε̄²|ε_i] = (N−1)σ²/N + ε_i²/N`, so the difference is `(1−1/N)[ε_i² + (N−1)σ²] =
((N−1)/N)[ε_i² + (N−1)σ²]`, and dividing by `N` gives exactly `α + βε_i²`. The reading is that the
denominator's conditional scale *grows with* `ε_i²`: a sample with large `|ε_i|` inflates the very
spread it is then divided by, which compresses large advantages. (A cleaner non-asymptotic way to see
the compression: with `z_j = ε_j − ε̄`, `Σz_j = 0`, so given `z_i` the others sum to `−z_i` and
`Σz_j² ≥ Nz_i²/(N−1)`, hence `|A_i| ≤ sqrt(N−1)` — bounded for every finite `N`, while `ε_i` is
unbounded, so the std-normalized estimator is biased. With the sample std the code actually uses, that
ceiling is `(N−1)/√N = 15/4 = 3.75`, which is precisely the extremal z-score I kept seeing on `k = 1`
groups.) Now watch the `N` dependence: as `N → ∞`, `β = (N−1)/N² → 0` and `α → σ²`, so the dependence on
the individual sample vanishes and `D` converges to the true `σ`. The disease is *small N*. The cure is
more samples in whatever pool I standardize over.

I cannot grow the per-prompt group, so before I reach for a bigger pool let me make sure I have the
right lever and not a patch. One option is to floor the group std so the near-unanimous blowups are
capped — but that treats only the brittleness and leaves the per-prompt scope, so the difficulty bias
survives; step 2 already told me deletion beat that. Another is to keep a running scale across steps, an
EMA of the advantage std — a large effective pool, but it pulls in statistics from a policy that has
already moved, and with the policy shifting every one of 100 steps that staleness is a real cost for no
clear gain. What other pool do I have *right now*, in-distribution? The whole batch. In one optimization
step the loop gives me 128 prompts × 16 = a global batch of ~2000 responses across many prompts. If I
standardize over *that* pool instead of over each 16-sample group, the mean and std I divide by are
computed from `N_global ≈ 2000`, not `N_group = 16`, and they are computed from the *current* policy's
rollouts, no staleness. Put the numbers to the `β` I just derived: for `N = 16`, `β = 15/256 ≈ 0.059`;
for `N = 2048`, `β = 2047/2048² ≈ 4.9×10⁻⁴`. That is the small-sample bias term shrinking by about
`120×`. (I should be honest that the tokens within a response are perfectly correlated — they share one
scalar — so the effective independent count for the *scale* is the number of responses, ~2048, not the
raw token count; even taking the conservative per-prompt count of 128 gives `β ≈ 0.0078`, still ~7.5×
smaller than the group's.) Either way, a global-batch scale behaves like a near-constant, the
small-sample bias collapses, and no single uninformative prompt can set the scale for itself. Both
failures of the group std weaken together, for the same reason: a large pool makes the normalization
statistics behave like constants. And this is not exotic — subtracting the mean and dividing by the std
of advantages over a minibatch (whitening) before they hit the policy loss is a standard PPO
implementation detail (Andrychowicz et al. 2020 list per-minibatch advantage normalization among the
common design knobs). What I am doing is taking that established whitening and choosing the pool to be
the *global batch* rather than the prompt group — precisely because pool size is what controls the bias
and the stability.

So a first cut: drop the per-group std, and whiten advantages across the full batch. But there is an
order-of-operations question I have to get right, because whitening *raw rewards* over the batch and
whitening *centered* advantages over the batch are not the same thing, and one of them is a disaster.
If I whiten the raw `r_i` across the batch first, I am standardizing rewards across prompts of different
difficulty, so an easy prompt's high raw reward reads as a high advantage even when *every* response to
it was correct and there was nothing to learn — the difficulty-as-quality error I disqualified back at
step 1. Make it concrete: an easy prompt where all 16 are correct (`r = 1`) and a hard one where all 16
are wrong (`r = 0`); raw-whitening pushes the easy prompt's already-correct tokens *up* and the hard
prompt's tokens *down*, spending gradient on prompts with zero relative signal. Center first, and both
prompts are unanimous so `A' = 0` and they correctly drop out. So the per-prompt mean subtraction is not
optional and it must come *before* the global scale — it is the one good thing the group gives me, and
the dr_grpo numbers proved it: "this response beat the typical response *to this prompt*" is the signal;
"this raw reward is high" is mostly difficulty. Group-mean subtraction has a second benefit in this RLVR
setting: it *reshapes* the reward. A 0/1 verifier with group mean 0.5 centers advantages about zero; a
−1/1 verifier also centers about zero. So group-mean subtraction makes the method robust to the reward
scale — I do not have to redesign rewards for 0/1 vs −1/1. The group mean still includes the sample
itself, so it carries the same finite `(1 − 1/N)` shrinkage I accounted for in dr_grpo, but it is a
*stable centering* step. It is the *denominator* — the local std — that was biased and brittle. So keep
the group mean as local reward reshaping; throw away the local std; replace it with a global one.

That gives a two-step estimator. Step one, reshape with a local baseline: `A'_i = r_i − mean_group(r)`
— exactly the dr_grpo advantage, the thing that already worked. Step two, stabilize with global
statistics: standardize `A'` over the whole batch, `A^norm = (A' − mean_batch(A'))/(std_batch(A') + ε)`.
The group mean handles per-prompt difficulty and reward-scale reshaping; the batch std handles
cross-prompt scale and is computed from a pool large enough that any one sample has little leverage. The
dangerous small-sample std is simply gone — I get scale normalization back *without* the difficulty
bias, because the scale is no longer per-prompt. That is the precise answer to the question this step
opened with.

I have to confront a tension I created at step 2, or I am fooling myself. Back then I argued that
dividing every advantage by one global number "folds into the learning rate and changes nothing about
the relative weighting of examples." But that is exactly what a global whiten does — divide the whole
batch by one std. So does it do anything, or am I about to add a no-op? Work it through honestly.
*Within a single step*, yes, dividing all advantages by the scalar `std_batch` is a uniform rescale;
the ratio between prompt A's advantages and prompt B's is untouched. And that is not a bug — it is the
safety property I need. dr_grpo restored the *correct* relative weighting (mixed prompts dominate,
unanimous ones vanish), and because a uniform divide preserves every ratio, the global whiten keeps
that weighting exactly intact; it cannot re-introduce the difficulty bias the way a per-prompt divide
did. What it changes is not within-step relative weighting but *across-step absolute scale*. Over 100
steps the raw magnitude of `A'` drifts — early on the policy is bad and reward spreads are wide, later
they narrow — so with a fixed learning rate and a fixed clip, an un-whitened advantage delivers a
different effective step size at step 5 than at step 95. Per-step whitening pins the batch to unit
variance every step, decoupling the effective step size from that drift so the one fixed LR and the one
fixed clip stay well-calibrated throughout the run. So the step-2 statement was right and this is not a
contradiction: the global divide *is* a per-step LR rescale, and the point is to make that rescale a
*constant* across steps instead of a drifting one. "Cross-prompt scale" is really "one stable overall
scale that preserves the good relative weighting" — which is exactly what the per-group std could never
be.

The place this should matter most is the tail of the run, and it is worth spelling out because it is a
mechanism dr_grpo cannot have. As the policy strengthens over 100 steps, more groups go near-unanimous,
so the raw centered advantages `A'` shrink toward zero — under dr_grpo the update magnitude fades with
them and late-training learning stalls just as the fixed LR expects a normally-scaled signal. Batch
whitening renormalizes the surviving spread back to unit variance every step, so the still-informative
mixed prompts keep delivering a full-scale gradient even when the batch-average spread has collapsed.
That is the concrete thing the across-step scale stabilizer buys over dr_grpo, and it says the two
estimators should diverge more the longer training runs — at 100 steps the gap may be modest, but it is
in the direction of the whitened estimator sustaining learning where the un-whitened one tapers.

That reframing also gives me a cheap correctness check on the two-step map: whitening is an affine
transform `x ↦ (x − m)/s` with `s > 0`, hence monotone, so within any prompt a correct response's
`A' = +0.5` stays strictly above a wrong response's `−0.5` after the whiten — the sign structure the
group mean established is preserved, not scrambled. And I can pin down what the whiten's *centering*
step actually removes: the per-sequence advantages sum to zero within each group by construction, so
their batch mean over sequences is exactly zero; the only thing `masked_whiten` subtracts is the small
*length-weighted* offset (the `+0.42`-style tilt from correct answers running longer). So the global
mean-subtraction is a minor length de-biasing and the real work of step two is the division — the
across-step scale stabilizer I just argued for.

Edge case before I trust it: a prompt with only one sample in the batch (group size 1). There is no
"other responses" to form a mean, and subtracting the sample from itself would zero its advantage,
wiping the signal. The honest fallback is to set the group mean to 0 for singletons — keep the raw
score, let the *global* whitening center it relative to the rest of the batch. Singletons get no local
baseline but still get the batch-level standardization. (With the fixed loop handing 16 per group this
never fires, but the two-step structure makes the fallback matter more than it did in dr_grpo, so it is
worth stating.)

Now the token question, because the actor loss is per-token and I have one scalar per response. The
reward is a scalar at EOS, `A'_i` is one scalar per response, and the standard move is to broadcast it
to every valid token. But there is a choice in *when* I whiten, and it changes the statistics. If I
whiten the per-sequence scalars and then broadcast, every response counts once regardless of length. If
I broadcast first and whiten over the pool of *tokens*, a 600-token response contributes 600 entries to
the mean and std while a 50-token one contributes 50 — the statistics become *length-weighted*. Let me
see what that does with a small example: take two centered responses from the same prompt, a correct one
with `A' = +0.5` over 600 tokens and a wrong one with `A' = −0.5` over 50 tokens. Their per-sequence
mean is 0, but their token-weighted mean is `(600·0.5 + 50·(−0.5))/650 = +0.42` — non-zero, because the
long correct response dominates the token pool. So token-level whitening subtracts a mean that reflects
which advantages actually fill the gradient, and if correct solutions run systematically longer (they
usually do in math), it removes that length-correlated offset from the baseline. For a token-level
policy loss that is the consistent choice: I am standardizing the actual per-token quantities that enter
the gradient, over the actual set of tokens that enter the gradient. So: broadcast the centered
per-sequence advantage to all valid tokens, mask the padding, whiten over all valid response tokens in
the batch via `verl_F.masked_whiten`, then mask again. The variance floor inside the whiten is now just
a numerical guard under a *batch* variance, not a load-bearing scale — with ~2000 responses' worth of
tokens the batch std is nowhere near zero, so unlike grpo's `ε` this floor never actually shapes an
advantage.

Let me dimension-check the code path so the masking is right. `token_level_rewards` is `(bs,
response_length)` with `bs = 2048`; `scores = token_level_rewards.sum(-1)` is `(2048,)`. After
subtracting the per-prompt mean, `scores.unsqueeze(-1).tile([1, response_length])` re-expands to
`(2048, response_length)` with `A'_i` copied across the row, and `* response_mask` zeros the padding.
`masked_whiten(scores, response_mask)` then computes a single scalar mean and scalar std over the
masked (valid) entries — one number each, pooled over ~2000 responses' tokens — and returns `(scores −
mean)/(std + ε)`, so the output has batch-token mean 0 and unit variance; the trailing `* response_mask`
re-zeros padding the whiten's affine shift may have disturbed. Returns equal advantages, no bootstrap.
The shapes close and the two masks bracket the whiten exactly as they must.

I have to be honest about a piece of the full recipe that I *cannot* apply here. The reasoning-task
variant of this method also adds a separate KL-to-reference loss term, and the right estimator for that
is subtle — sampling from `π_θ` means I am constraining the reverse KL, and of the usual k1/k2/k3
candidates only k2 = `½(log(π_θ/π_ref))²` differentiates to the reverse-KL gradient without an exploding
importance weight. But the KL-loss setting is *fixed outside my edit surface* — the only editable region
is `compute_custom_advantage`. So I do not write the KL term; the loop applies whatever KL it applies,
and my job is purely the advantage. My step-3 edit is exactly the two-step advantage: group-mean center,
broadcast to tokens, masked-whiten over the batch's valid tokens, mask. (The fill is in the answer.)

At this point the construction is dr_grpo plus one operation: where dr_grpo stopped after group-mean
centering, I broadcast to tokens and whiten over the global batch. So the delta is the global scale
normalization the deleted std could never safely provide. Reading the dr_grpo numbers, here is what I
expect and where I am unsure. dr_grpo already fixed the difficulty bias, so I do not expect a second
large jump on MATH/AMC from the *same* mechanism; the `+0.0117` aggregate gain there was the
difficulty-bias money, and it is already collected. The gain this step should come from putting all
prompts' advantages on one stable scale, which should most help wherever incomparable raw scales were
still injecting noise into the update — I expect the aggregate `score_mean` to clear dr_grpo's −0.5049,
with the strongest token-level whitening benefit showing up on the splits where response lengths and
reward scales vary most (MATH and AMC again). GSM8K should hold near 0.4671 or drift slightly, since its
prompts are short and uniform and have least to gain from length-weighted global whitening — I would not
be shocked to see it dip a hair if the global scale trades a touch of easy-split accuracy for stability
elsewhere. The move is again small in the algebra — one added whitening op on top of the step-2
advantage, and compute-neutral (two masked reductions over the batch tokens against the forward/backward
cost) — so I expect a modest aggregate lift, not a leap. I can even bound my expectation quantitatively:
the step-1 → step-2 change, which collected the whole difficulty-bias correction, moved the unweighted
3-accuracy mean from `0.2858` to `0.2917`, a `+0.0058` gain. Since that mechanism is already spent and
this step only adds scale stabilization, I expect the step-2 → step-3 gain on that same 3-accuracy mean
to be *smaller* than `+0.0058` — a diminishing return, not a repeat of the first jump. If instead I saw
a gain as large or larger, that would tell me scale stabilization was doing something bigger than I
think and I would have mis-attributed the step-2 win. On a single seed (42) any one benchmark could
wobble; the load-bearing prediction is the aggregate. If the batch-whitened estimator clears
dr_grpo on the aggregate, the missing-scale diagnosis is confirmed — the std was never the problem, its
*scope* was, and the fix is to keep the legitimate scale normalization but compute it from a pool large
enough to be a constant.
