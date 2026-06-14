The strategy is the whole point, but it sits inside a fixed decode loop, and the cheapest rule that
respects that loop is the floor I have to start from. So let me be precise about what the loop already
hands me and what the one empty slot actually has to decide, because the entire problem lives in that
slot. The harness builds `x = [prompt | gen_length masks]`, splits the generation region into blocks,
gives me a uniform per-step budget from `get_num_transfer_tokens`, walks the blocks left to right, and
at each step runs exactly one `model(x)` forward pass. From that pass I get per-position logits over the
vocabulary at every masked slot at once. The slot I fill is: turn those logits into a decision — which
of the still-masked positions in the current block to commit this step, and what token to write into
each. Everything else — the mask layout, the block loop, the budget, the one-forward-pass cost — is
given. And one structural fact from the loop is load-bearing for every rung that follows: once a
position is written it is frozen. The loop only ever writes into positions where `x == mid` and
re-checks that mask every step, so a committed token is never revisited. Commit is permanent. That is
the absorbing-state carry-over property made concrete in code, and it means whatever I freeze this step
becomes irreversible bidirectional context for every remaining masked position on every later step.

Let me settle the three sub-decisions in order, starting with the easy ones. The *schedule* — how many
to unmask per step — is not really mine to invent here: the forward masking schedule is linear, so the
expected number of positions that transition per step is constant, and the helper already returns that
uniform budget (`mask.sum() // steps` per step, remainder front-loaded). I take it as given; the budget
`k` for this step is `num_xfer[:, step]`. The *token assignment* is also nearly forced. The model hands
me a distribution at each masked position; the natural token to write is its own prediction. At the
default temperature zero that is the argmax — the mode of the predicted categorical. There is a real
trade-off buried here that I want to name now because it returns later: greedy/low-temperature decoding
suppresses diversity, which is exactly what I want when there is a single correct answer (math, code —
diversity there just means more ways to be wrong) and exactly what I do *not* want for open-ended text,
where greedy decoding degenerates into dull, repetitive output. The constructor exposes a `temperature`
knob for that reason, but the default for the accuracy tasks is zero, so the token is the argmax.

That leaves the one decision that is genuinely open and that the whole strategy turns on: *position
selection*. Out of the masked positions in the current block, which `k` do I commit this step? The
training objective is silent here — it only ever asks the model to predict the clean token at a masked
slot, never which slots to fill in what order — so any order that ends with everything unmasked is a
legal sampler. The question is which order is *good*.

The schedule-faithful answer is random: pick `k` masked positions uniformly at random and fill them.
This is not crazy — it is the unbiased reverse-process sampler, the one whose remasking marginal matches
the forward corruption exactly. But walk it forward one step. At the first step everything is masked, so
the model's per-position distributions are mostly flat — it has almost no context to condition on, so
many of its predictions are near-uniform guesses. Random selection now commits `k` of these, scattered
without regard to how sure the model is: some land where the model happened to be confident, but many
land where it was essentially flipping a coin. Those coin-flip commitments are frozen by carry-over, and
the next step conditions on them. An early random commitment is an early mistake, and an early mistake
here is expensive because it poisons the context for everything decoded after it. So random selection
gets the masking rate right but throws away the one piece of information I get for free on every step:
how confident the model is at each position. That is the wall.

Let me stare at *why* it is wasteful, because the fix should fall out of the diagnosis rather than
being grafted on. The model gave me a full distribution at *every* masked position, and the random rule
used none of it for selection. But those distributions are not equal. At some positions the distribution
is sharply peaked — the surrounding visible context already pins the token, the model is nearly certain.
At others it is flat — genuinely ambiguous given what is currently visible. The carry-over freeze makes
the *cost* of being wrong identical everywhere (every commit is equally permanent), but the
model-assigned *risk* of being wrong varies sharply across positions: it is roughly `1 - max_v P(v)`. If
I am forced to commit `k` positions this step and freeze them forever, I should commit the ones I am
least likely to regret — the peaked, high-confidence ones — and leave the flat ones masked, so that next
step, after the certain tokens have sharpened the context, the ambiguous ones may have become certain
too. That is an easy-first curriculum: fill the slots the context determines, let them become context,
and ambiguity collapses inward.

There is a second, sharper argument hiding in the parallel structure, and it points the same way. The
reverse step factorizes across positions, so when I fill several positions in one step I am treating
their proposals as conditionally independent given the current context. That is the tokenwise-
independence approximation that makes parallel decoding fast, but it is approximate — once I commit
position `i`, the correct conditional for position `j` may shift. The approximation error grows with how
many positions I commit at once and with how uncertain each one is: committing a peaked position barely
moves the conditional picture, while committing a flat position is exactly the case where another
position could have changed the answer. So the independence error is *smallest precisely at the
high-confidence positions*. Irreversibility and independence error converge on one prescription: commit
the positions the model is most sure about, defer the rest.

So I want a per-position confidence score and I want to commit the top-`k` by that score. What is the
score? The cleanest scalar is the model's own belief in the token it would write. I run the predictor,
take `x0 = argmax_v p(v)` at each masked position, and let confidence be `conf = p(x0)`, the probability
mass on the chosen token — at temperature zero this is just `max_v p(v)`. A peaked distribution gives
confidence near 1; a flat one gives it small. Then this step's rule is: among the masked positions in
the active block, take the `k` with the highest confidence, write their argmax tokens, leave everyone
else masked; next step, re-run, re-score, commit the next-most-confident `k`. To encode carry-over and
the block discipline in one stroke, I score only the masked positions in the active block and drive
every other position's selection score to `-inf` — a frozen token or a future-block position can then
never win the top-`k`. This matters because the parallel predictor sees the whole sequence and would
happily fill ahead of the current block if I let it, breaking the left-to-right order.

I should check this confidence-keeping idea against a place it has already been made to work, so I am
not reinventing a broken wheel. The exact move — predict all masked locations in parallel, score each by
the predicted probability of its sampled token, keep only the most confident, remask the rest, repeat —
is the iterative-decoding recipe for masked image transformers (Chang et al. 2022). The structure
transfers directly. But two of their choices are tuned for image *diversity* and I should not import them
blindly. They sample the token stochastically with temperature annealing, because in image synthesis
diverse samples are the goal; I want the mode for deterministic tasks, so the default token is the argmax.
And they use a *cosine* schedule for how many to keep masked — concave, "few confident early, many late"
— chosen for aesthetics. My schedule is dictated by my generative model, not chosen: the forward masking
probability is linear in time, so the principled per-step budget is the *uniform* one the helper already
hands me. So I take their confidence-keeping skeleton with a likelihood-consistent uniform schedule and
a greedy token — exactly what this setting demands. This is also LLaDA's `low_confidence` remasking rule:
keep the highest-probability predictions, remask the lowest, the same approach as MaskGIT carried into
the language setting.

Now the regime question, which the contract forces me to answer with one decoder. Do I decode the whole
response as a single parallel block, or carve it into pieces? Pure parallel decoding fills positions
anywhere in the response, ordered only by confidence — fully bidirectional, maximally parallel, best for
open-ended continuation where there is no strong left-to-right backbone. For long structured outputs — a
multi-step derivation, a function body — a coarse left-to-right ordering helps: I want the early
reasoning settled before the later steps that depend on it. The block size handles both: when
`block_length == gen_length` there is a single block and this is exactly fully-parallel decoding; when it
is smaller it is semi-autoregressive, blocks in order with confidence-keeping diffusion inside each. The
constraint `gen_length % block_length == 0` keeps the blocks even and I split the step budget evenly
across them. The same confidence rule serves both regimes — only the in-block eligibility mask changes —
which is precisely the property the task demands.

Let me be careful with numerics, because the whole strategy is a ranking and a ranking is only as good
as the scores. I compute the softmax in float64 before gathering the confidence and taking the argmax;
low-precision softmax is known to hurt generation quality in these models, and a float64 score keeps the
top-`k` ranking clean when several confidences are close. That is the one deliberate precision choice; in
this floor strategy I do not add a Gumbel sampling path or end-of-sequence suppression — the harness
constructor still accepts `temperature` (and the KLASS-shaped `conf_threshold`, `kl_threshold`,
`history_length` keywords, which this rule simply ignores), but the slot itself is just the argmax over
the float64 softmax and the confidence read off the same softmax. Keeping it minimal is the point: this
is the floor every later rung will be measured against.

So the per-step body that fills the slot is concrete: run one forward pass, softmax the logits in
float64, take `x0 = argmax(p)` as the token and `conf = p[x0]` as the score, set the confidence of every
non-eligible position to `-inf`, take the top-`k` eligible positions by confidence where `k` is the
schedule's budget for this step, and write `x0` at those positions. One forward pass per step, one
softmax, one top-`k`, one masked write — no extra model calls, so `used` is exactly the number of steps,
and on this floor strategy the loop never breaks early because confidence-keeping always has masked
positions to commit until the block's full step budget is spent. The full scaffold module is in the
answer.

Now reason about what this floor will do, because that is the entire point of running it. Confidence-
greedy is the simplest rule that uses the model's own certainty at all, and it has one structural
feature I should call out before the numbers come in: it commits exactly `k` positions per step, with
`k` fixed by the uniform schedule, so it spends the *entire* step budget — `used` equals `steps` on
every example. There is no early stopping, no adaptive batching of confident positions; one schedule
slot, one forward pass, always. So on the efficiency metric this rule is pinned at the worst possible
value: `avg_steps` will sit at the full budget (256 in the block regime, 224 in the parallel regime,
where one block holds the whole region). Anything later that commits more than `k` confident positions
in a single settled step will beat it on cost for free.

On quality I expect it to be a real improvement over random — it is finally using the predictor's
confidence — but to be the *weakest* of the confidence-aware rules, for one reason I can already see in
the construction. The score is `max_v p(v)`, a function of a single number, the winning mass. It cannot
distinguish a position where the top token is at 0.45 and the runner-up at 0.02 (a decisive, safe commit)
from one where the top token is at 0.45 and the runner-up at 0.44 (a near-tie, the most dangerous commit
there is). Both score 0.45; confidence-greedy calls them equally safe and may freeze the near-tie. The
information that separates them — the runner-up probability — is sitting right there in the logits and
this rule throws it away. So I expect confidence-greedy to land respectable but unspectacular accuracy on
math and code, to spend the full step budget everywhere, and on open-ended text — where greedy argmax
plus a single block tends to produce repetitive, low-diversity continuations — I expect the conditional
perplexity and the distributional-similarity score to be poor, because greedy decoding of a one-block
parallel region with no diversity injection is exactly the degenerate regime. Whatever the precise split,
the diagnosis points at the next rung: I have a *selection-signal* problem, and the cheapest fix is to add
back the one piece of the distribution this score discards — the second-place probability — which turns
`max_v p(v)` into the margin `P(top1) - P(top2)`.
