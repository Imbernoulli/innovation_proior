The margin run confirmed the half of my hypothesis I most wanted confirmed, and it left the other half
standing as a wall, and both are in the numbers. On open-ended text the margin rule did exactly what I
predicted: MAUVE climbed from confidence-greedy's 0.0321 to 0.1124, more than tripling — being careful
about *which* near-tie tokens I freeze raised distributional similarity to the reference, just as the
not-poisoning-the-context argument said it should. And the tension I flagged showed up too: gen_ppl did
not fall with MAUVE, it *rose*, from 170.6 to 237.0, exactly the "lower perplexity is not the same as
higher distributional match" caveat — margin made the continuation read more like real C4 text without
making a fixed external GPT-2 scorer happier, because it is still a greedy one-block decoder with no
diversity injection. On the accuracy tasks the small lift arrived where I said it would: HumanEval moved
from 0.3659 to 0.3902 (code has many genuine near-ties where tie-avoidance helps), math edged up 0.316 to
0.322. So the selection-signal story is settled: best-versus-second-best is a better certainty scalar than
the winning mass, and it buys real quality.

But look at the cost column, because that is the wall. `avg_steps` is *still* pinned at the full budget —
256, 256, 224 — identical to confidence-greedy, to the last decimal. Margin changed which `k` positions I
commit; it could not change *how many*, because the count is dictated by a static uniform schedule, one
forward pass per slot, the entire budget spent on every example. So two rules now sit at the worst
possible value of the efficiency metric while the task grades me on it. And the diagnosis I wrote closing
the margin step is the one to act on: the way off the full budget is to make the count *adaptive* — commit
*many* positions in one step when the model is settled and sure, fall back to the schedule only when it is
not — so I can pull `avg_steps` down without giving back the quality the better selection signal bought.

So the real tension I am now trying to resolve is not which scalar to rank by — margin already won that —
but *how many* to commit per step, and that reframes everything. Committing one position per step would be
maximally safe and maximally slow: one forward pass per token, autoregressive speed, throwing away the one
thing a diffusion model is for. Committing many per step on the fixed schedule is what both prior rungs
did: fast in the sense that it spends the budget evenly, but it commits tokens on a clock rather than on
evidence, and since each commit is permanent it locks in whatever the schedule forced this step. I want to
commit many tokens per step *exactly when it is safe* and otherwise fall back to the schedule-sized
careful choice. The question is what "safe" means, beyond the single-step certainty I have already wrung
out.

Here is the thing that has been nagging me, and the margin numbers sharpen it: confidence at one step —
whether read as `max_v p(v)` or as the margin — is not the same as being *right*. Both prior rungs key off
a number read at a single frozen step. I keep being able to imagine the same failure: the model commits a
token it is highly confident about *this step*, and it is wrong, because the correct token had a lower
score at this step and the rule passed it over. The 237 perplexity on the text block is partly this — a
position can be confidently, margin-decisively wrong while the context that would overturn it is still
masked. So single-step certainty is necessary — I am not going to commit a position the model is unsure of
— but it is plainly not sufficient. There is a second axis I am not using, and to find it I should stop
staring at the formula and stare at the failures across steps.

When I take a position where a confidence-or-margin rule commits a high-score wrong token, and I watch
that position *over the denoising steps* rather than at one frozen step, something jumps out. Its
distribution is still moving. From one step to the next the model's categorical there keeps shifting — the
argmax may even stay the same and look confident, but the shape underneath is churning, because the
surrounding context is still being resolved and each newly revealed neighbor nudges the model's belief
here. Now take a position whose commitment no longer looks fragile: its distribution has stopped moving.
The model decided what goes there several steps ago and every subsequent step leaves it essentially
unchanged. So the discriminating thing is not how tall the peak is at this instant, nor how big the gap to
the runner-up is — it is whether the model's belief at this position has *settled*. Confidence and margin
are snapshots; what I actually want is a measure of *temporal consistency*. And I can compute it from the
distributions the decoder already has on hand: it is a divergence between two categoricals, the one I had
at the last step and the one I have now. KL divergence is the natural choice — `d_t^i = D_KL(p_t^i ||
p_{t+1}^i)`, the current step's distribution against the previous step's at the same position. Near zero
means the model is not changing its mind there; large means it still is. That gives me a second score to
put beside the confidence score.

I want to be sure I am not just curve-fitting an observation, so let me argue from first principles that a
wrong-but-should-be-correctable token *must* be unstable — that high KL is forced on it, not incidental.
Set up the cleanest version. Fix a position `i`. Suppose the model is a decent approximation of the task:
for every context `c`, its distribution at `i` is within total variation `δ` of a task-correct one.
Suppose at `i` there is a genuinely correct token `x*` that, at the resolved context `c*`, beats a
suboptimal `x†` by margin `γ`: `π(x*|c*) ≥ π(x†|c*) + γ`. And suppose right now, at the under-resolved
context `c_M` where most neighbors are still masked, the model actually prefers the wrong token,
`p(x†|c_M) ≥ p(x*|c_M) + β` for some `β ≥ 0` — exactly the confident-but-wrong situation, the one driving
the 237 perplexity. The denoising process walks the context from `c_M` down to `c*` as the other positions
fill in, `c_M → … → c_0 = c*`, changing only variables outside `i`. Write `P_t = p(·|c_t)`. How much must
the distribution at `i` move along this path?

Pick a test function that reads off exactly the `x†`-versus-`x*` preference:
`f = 1{x_i = x†} - 1{x_i = x*}`, so `||f||_∞ ≤ 1`. Total variation controls how much any bounded
function's expectation can differ between two distributions. Let `A_M = p(x†|c_M) - p(x*|c_M)` and
`A_0 = p(x†|c*) - p(x*|c*)`. Then `2·TV(P_M, P_0) ≥ |E_{P_M}[f] - E_{P_0}[f]| = |A_M - A_0|`. The current-
preference assumption gives `A_M ≥ β`. At `c*` the δ-approximation gives `p(x*|c*) ≥ π(x*|c*) - δ` and
`p(x†|c*) ≤ π(x†|c*) + δ`, so `p(x*|c*) - p(x†|c*) ≥ γ - 2δ`, i.e. `A_0 ≤ -(γ - 2δ)`. Therefore
`A_M - A_0 ≥ β + γ - 2δ`, and taking the positive part, `TV(P_M, P_0) ≥ ½(β + γ - 2δ)_+ =: Δ`. The
endpoints are far apart in total variation whenever the true margin and the current wrong-preference
together beat twice the model's approximation error.

Now turn endpoint separation into per-step movement, because per-step movement is what my KL score
measures. Total variation obeys the triangle inequality along the path, so the per-step TVs sum to at
least the endpoint TV: `Σ_t TV(P_{t+1}, P_t) ≥ TV(P_M, P_0) ≥ Δ`. Let `T_t = TV(P_{t+1}, P_t)`. To connect
TV to KL I use Pinsker, `D_KL(P_t || P_{t+1}) ≥ 2 T_t²`. Then by Cauchy–Schwarz `(Σ_t T_t)² ≤ M Σ_t T_t²`,
so `Σ_t T_t² ≥ Δ²/M`, giving `(1/M) Σ_t D_KL(P_t || P_{t+1}) ≥ 2Δ²/M²`. There it is: a token wrong at the
under-resolved context but correct at the resolved one cannot keep its average per-step KL near zero — it
is *forced* to be dynamically unstable somewhere along the path, average KL bounded below by `2Δ²/M²`. So
if a position's per-step KL has stayed near zero across the recent steps, that is exactly the evidence I
want before treating it as unlikely to be one of these will-flip tokens. Instability is the mathematical
fingerprint of a token that is going to change its mind; the stability gap is not a coincidence.

So now I know what to gate on. I commit a masked position only if it is *stable* — recent step-to-step KL
low — *and* confident — top probability high. Both, not either. Confidence alone is the prior art (the two
rungs behind me) and admits the confident-but-wrong tokens; KL alone would let me commit a position whose
belief has frozen but at a low, mushy probability where the model is not actually committing to anything.
Together they say "the model has settled on this and is sure of it." One KL reading bothers me, though: a
single low KL between two consecutive steps could be a fluke — the distribution might happen not to move
for one step and lurch the next, especially early when lots of context is masked. The theory bounds the
*average* KL away from zero, which is consistent with it being momentarily low; I want evidence of
*sustained* stillness. So I demand that the last `n` consecutive-step KLs are all below threshold, not just
the latest. That turns "it did not move this step" into "it has not moved for `n` steps," which is exactly
the robustness the average-KL bound says I need. How big should `n` be? `n = 1` is trigger-happy, the
fluke version. Large `n` is safe but stingy — requiring a long run of low KLs means I rarely declare
anything stable, so I commit few per step and lose the speedup I came for. The sweet spot is small-but-
greater-than-one; `n = 2` is the natural first stop, enough to kill the single-step fluke without paying
for a long history. I keep `n` a knob (`history_length`) but default it to 2. A consequence: with a window
of length `n` I should not evaluate stability until the window has filled, so for the first `n-1` steps of
a block nothing is eligible to be called stable — and the very first KL is computed against an all-zero
"previous" buffer, which is huge, so it fails the all-below-threshold test until it rolls out. The guard
is deliberately conservative.

Now the unmasking rule, where the speedup finally comes from. At each step I commit *every* position that
is stable-and-confident — all of them, in parallel, however many there are. At a settled step many
positions cross both thresholds at once and get written together in a single forward pass; at a churning
step the ready set is small or empty. The count adapts to the model's own state instead of a clock — that
is the lever both prior rungs lacked. The remaining worry is the empty case: if nothing is ready,
committing nothing means no progress, and a too-strict threshold pair could stall the chain forever. So I
need a fallback that guarantees progress and degrades gracefully to a sane baseline: when the ready set is
empty, unmask the top-`u` masked positions by confidence, where `u` is the fixed schedule count — exactly
what confidence-greedy would have done at that step. So the worst case for this rule is "behaves like the
confidence baseline I already measured," and the best case is "commits a whole settled block in one shot,"
and it interpolates automatically. That is precisely how I get `avg_steps` off the full budget without
risking the quality floor: on easy, settled examples it commits many per step and `used` drops far below
`steps`; on hard ones it falls back to the careful schedule-sized choice and `used` stays near the budget.

A few implementation points will bite if I am sloppy, and they all live where the stability signal is
built on tiny differences between near-identical distributions. The KL is
`Σ_v p_t(v)(log p_t(v) - log p_{t+1}(v))`, and in the settled regime — the one I most care about — this is
a small difference of large logs, so float32 gives noise where I need to read a threshold of order 1e-2.
And low-precision softmax already hurts MDM quality. So softmax and KL go in float64, with a 1e-12 floor
inside both logs against `log 0`. I maintain the KL history as a rolling per-position buffer: roll left by
one, write the new KL into the last slot, so after warmup the window holds the most recent `n` real values.
I form `p_curr`, compute its KL against the saved `p_prev`, push it, then overwrite `p_prev ← p_curr`; with
diffusion steps counted downward this makes the comparison `D_KL(p_t || p_{t+1})`, current against
previous, matching the path argument's orientation. And the whole thing stays training-free: every signal
is the base model's own successive predictions — no second model, no planner, no growing state, just the
previous-distribution buffer and the KL history.

And it has to serve both regimes from one decoder, which falls out for free because the rule is local to
the current block. In semi-autoregressive block decoding the stability machinery runs within the active
block — I restrict the masked set, the KL/confidence tests, and the fallback count to it. In fully-parallel
decoding the block is the whole region and the identical code runs with a single block. Nothing in the
stable-set logic cares which regime it is in. The full scaffold module is in the answer.

So the delta from the margin rung is concrete: where both prior rungs ranked a single-step score and
committed a fixed schedule-sized `k`, I add the temporal axis — gate commits on stability *and* confidence,
require the last `n` KLs all low, commit the whole ready set per step, and fall back to top-`u` by
confidence when nothing is ready. Reading the margin numbers, here is what I expect and where I am unsure.
The headline I am betting on is `avg_steps`: it should fall well below the pinned 256/256/224 on every
setting, because settled blocks commit many tokens per forward pass — this is the metric the prior two
rungs could not touch. I expect math accuracy to *hold or improve* over margin's 0.322 despite using fewer
forward passes, because the stability gate refuses exactly the confident-but-wrong commits that single-step
rules made — the theory says will-flip tokens are unstable, so gating on stability should remove them
rather than just decode faster. On open-ended text I expect the largest perplexity win: the fully-parallel
single block is where confident-but-wrong frozen tokens did the most damage (margin's 237 gen_ppl), and
committing only settled positions should let that block converge to a far more coherent continuation at far
fewer steps — I would expect gen_ppl to drop sharply from 237 and MAUVE to stay in margin's improved range
rather than collapse. Where I am unsure is the threshold sensitivity: if `conf_threshold` and
`kl_threshold` are too strict the ready set is often empty and the rule degrades to the confidence baseline
(no speedup, floor quality); too loose and it commits churning tokens (the failure I am trying to remove).
The falsifiable bar is sharp: this rung must beat margin's 0.322 math / 0.3902 HumanEval and 237 gen_ppl /
0.1124 MAUVE *while* spending strictly fewer forward passes than the pinned full budget. If it cannot pull
`avg_steps` down without giving back quality, the stability axis was the wrong second axis and I would go
back to the per-step score; but the average-KL bound says it should, and that is the prediction to test.
