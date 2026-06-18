The dr_grpo run confirmed the difficulty-bias diagnosis, and it did it on exactly the splits I
predicted. Deleting the per-group std moved the aggregate `score_mean` from grpo's −0.5166 up to
−0.5049, and the movement is concentrated where I said it would be: MATH-500 went 0.2973 → 0.3070 and
AMC went 0.0934 → 0.1009, while GSM8K barely twitched (0.4668 → 0.4671). So the harder splits recovered
and the easy split held — the per-group std *was* a distortion, not a stabilizer, and removing it was
the right call. Good. But dr_grpo is still not the strongest thing I can run, and the reason is the
other half of the trade I made when I deleted the std. The std was doing one *legitimate* job badly:
trying to put different prompts' advantages on a comparable scale. By deleting it I removed the
distortion but also removed all cross-prompt scale normalization — different prompts' centered returns
now enter the gradient on incomparable raw scales. The aggregate ticked up, but only by ~0.012 in
score, and AMC at 0.1009 is still leaving a lot on the table. I suspect the missing scale is why. So the
question for this step is sharp: can I get the legitimate scale normalization back *without* bringing
back the difficulty bias?

Let me restate what is actually wrong with the std so I attack the right thing. The std was bad for two
reasons, and they share a root. First, it is *scoped to the group* — 16 samples — so dividing by it
reweights prompts by their reward spread, the difficulty bias the numbers confirmed. Second, even
ignoring scope, a 16-sample std is a brittle scale estimate: near-unanimous groups (common with outcome
rewards) make it tiny and noisy. Both failures come from the same place — the scale is estimated from
the same handful of samples as the numerator, and from too few of them. So I should not try to "estimate
the std more carefully within the group"; the group is small by construction and I cannot enlarge it
without paying 16× the rollout cost.

I want to know how the trouble depends on sample count, because that will tell me the fix. Model a
group as `r_i = θ + ε_i` with `ε_i ∼ N(0,σ²)` i.i.d., `θ` the true prompt baseline. The std-normalized
advantage is `A_i = (ε_i − ε̄)/D` with `D = sqrt((1/N)Σ_j(ε_j − ε̄)²)`. Condition on `ε_i` and ask
whether the denominator behaves like a constant. Computing the conditional second moment of `D²` gives
`E[D²|ε_i] = ((N−1)²/N²)σ² + ((N−1)/N²)ε_i²` — call it `α + βε_i²` with `β = (N−1)/N² > 0`. The
denominator's conditional scale *grows with* `ε_i²`: a sample with large `|ε_i|` inflates the very
spread it is then divided by, which compresses large advantages. (A cleaner non-asymptotic way to see
the compression: with `z_j = ε_j − ε̄`, `Σz_j = 0`, so given `z_i` the others sum to `−z_i` and
`Σz_j² ≥ Nz_i²/(N−1)`, hence `|A_i| ≤ sqrt(N−1)` — bounded for every finite `N`, while `ε_i` is
unbounded, so the std-normalized estimator is biased.) Now watch the `N` dependence: as `N → ∞`,
`β = (N−1)/N² → 0` and `α → σ²`, so the dependence on the individual sample vanishes and `D` converges
to the true `σ`. The disease is *small N*. The cure is more samples in whatever pool I standardize over.

I cannot grow the per-prompt group. What other pool do I have? The whole batch. In one optimization
step the loop gives me 128 prompts × 16 = a global batch of ~2000 responses across many prompts. If I
standardize over *that* pool instead of over each 16-sample group, the mean and std I divide by are
computed from `N_global ≈ 2000`, not `N_group = 16`. By the limit I just derived, those batch statistics
behave like stable constants, the small-sample bias shrinks, and no single uninformative prompt can set
the scale for itself. Both failures of the group std weaken together, for the same reason: a large pool
makes the normalization statistics behave like constants. And this is not exotic — subtracting the mean
and dividing by the std of advantages over a minibatch (whitening) before they hit the policy loss is a
standard PPO implementation detail (Andrychowicz et al. 2020 list per-minibatch advantage normalization
among the common design knobs). What I am doing is taking that established whitening and choosing the
pool to be the *global batch* rather than the prompt group — precisely because pool size is what
controls the bias and the stability.

So a first cut: drop the per-group std, and whiten advantages across the full batch. But what do I feed
the whitening — raw rewards, or something centered first? Here I must not throw away the one good thing
the group gives me, and the dr_grpo numbers prove it is good: the per-prompt mean carries real
information. "This response beat the typical response *to this prompt*" is far more meaningful than
"this raw reward is high," which mostly tells me the prompt was easy — that is exactly the centering
that moved MATH and AMC up at step 2. Group-mean subtraction has a second benefit in this RLVR setting:
it *reshapes* the reward. A 0/1 verifier with group mean 0.5 centers advantages about zero; a −1/1
verifier also centers about zero. So group-mean subtraction makes the method robust to the reward scale
— I do not have to redesign rewards for 0/1 vs −1/1. The group mean still includes the sample itself, so
it carries the same finite `(1 − 1/N)` shrinkage I accounted for in dr_grpo, but it is a *stable
centering* step. It is the *denominator* — the local std — that was biased and brittle. So keep the
group mean as local reward reshaping; throw away the local std; replace it with a global one.

That gives a two-step estimator. Step one, reshape with a local baseline: `A'_i = r_i − mean_group(r)`
— exactly the dr_grpo advantage, the thing that already worked. Step two, stabilize with global
statistics: standardize `A'` over the whole batch, `A^norm = (A' − mean_batch(A'))/(std_batch(A') + ε)`.
The group mean handles per-prompt difficulty and reward-scale reshaping; the batch std handles
cross-prompt scale and is computed from a pool large enough that any one sample has little leverage. The
dangerous small-sample std is simply gone — I get scale normalization back *without* the difficulty
bias, because the scale is no longer per-prompt. That is the precise answer to the question this step
opened with.

Edge case before I trust it: a prompt with only one sample in the batch (group size 1). There is no
"other responses" to form a mean, and subtracting the sample from itself would zero its advantage,
wiping the signal. The honest fallback is to set the group mean to 0 for singletons — keep the raw
score, let the *global* whitening center it relative to the rest of the batch. Singletons get no local
baseline but still get the batch-level standardization.

Now the token question, because the actor loss is per-token and I have one scalar per response. The
reward is a scalar at EOS, `A'_i` is one scalar per response, and the standard move is to broadcast it
to every valid token. But there is a choice in *when* I whiten. If I whiten the per-sequence scalars and
then broadcast, every response counts once regardless of length. If I broadcast first and whiten over
the pool of *tokens*, a 600-token response contributes 600 entries to the mean and std while a 50-token
one contributes 50 — the statistics become *length-weighted*. For a token-level policy loss that is the
consistent choice: I am standardizing the actual per-token quantities that enter the gradient, over the
actual set of tokens that enter the gradient. So: broadcast the centered per-sequence advantage to all
valid tokens, mask the padding, whiten over all valid response tokens in the batch via
`verl_F.masked_whiten`, then mask again. The variance floor inside the whiten is now just a numerical
guard under a *batch* variance, not a load-bearing scale.

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
large jump on MATH/AMC from the *same* mechanism; the gain this step should come from putting all
prompts' advantages on one stable scale, which should most help wherever incomparable raw scales were
still injecting noise into the update — I expect the aggregate `score_mean` to clear dr_grpo's −0.5049,
with the strongest token-level whitening benefit showing up on the splits where response lengths and
reward scales vary most (MATH and AMC again). GSM8K should hold near 0.4671 or drift slightly, since
its prompts are short and uniform and have least to gain from length-weighted global whitening. If the
batch-whitened estimator clears dr_grpo on the aggregate, the missing-scale diagnosis is confirmed —
the std was never the problem, its *scope* was, and the fix is to keep the legitimate scale
normalization but compute it from a pool large enough to be a constant.
