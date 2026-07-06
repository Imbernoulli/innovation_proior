IPO did roughly what I predicted, and the size of the move tells me the diagnosis was right but the
remedy was timid. Seed 42: GSM8K 85.9, MATH-500 74.4, AIME 6.67, average 55.66. Line the three columns up
against SimPO's 86.05 / 74.0 / 3.33 and read the deltas. GSM8K: −0.15, noise on a saturated benchmark.
MATH-500: +0.4, a whisker. AIME: +3.34, which is *two* correct problems where SimPO had one. Now decompose
the average gain to see where it actually came from: `(−0.15 + 0.40 + 3.34)/3 = 3.59/3 = 1.20`, matching
the 54.46 → 55.66 jump, and AIME's `3.34/3.59 = 93%` of it. So essentially the entire IPO improvement is
one recovered AIME problem. That is the diagnosis confirmed in the cleanest possible way: bringing back
the reference and swapping the saturating sigmoid for a bounded target stopped the unbounded push that was
dragging the correct chains down, and it showed up exactly where I said it would — on the most fragile
benchmark, off its floor.

But look at *how small* the recovery is, because that is the signal for this rung. AIME went from one
problem to two; MATH-500 barely twitched. This is consistent with the reading I closed the IPO derivation
on: the target `1/(2β) = 5.0` sits so far above the per-token gaps this data actually produces that the
finite-target brake almost never engages, and what carried the gain was the reference *anchor* alone — a
conservative anchor at that. IPO regresses the reference-corrected log-ratio gap toward a far target and
its gradient `2(h_π − 5)` is a gentle, roughly-constant upward nudge; it *defends* the correct chain's
likelihood from falling below the reference, but it never *actively grows* it above where `π_ref` already
had it. On near-saturated GSM8K that is fine. On the harder benchmarks I am leaving improvement on the
table: IPO holds the line, it does not advance it. So the question for this rung is what would actively
*grow* the correct chain's likelihood while still contrasting against the rejected one — and, I would like,
could I get it for less than IPO is paying?

Because there is a cost I have not questioned, and I diagnosed it back at the floor. IPO, like DPO, drags a
frozen reference around: a second resident model, ~3 GB per GPU, and a second forward pass every batch —
four forwards per step (chosen and rejected, each through policy and reference) versus SimPO's two. SimPO
was reference-free and cheap but had no anchor; IPO has an anchor but pays for a whole extra model to
provide it, and the anchor it bought is a conservative defend-only one. I want both halves at once: an
anchor on the correct chain's absolute likelihood, *and* active growth of it, *without* a frozen
reference. That is greedy — SimPO says you can be cheap but unanchored, IPO says you can be anchored but
expensive-and-passive — so let me see whether there is an objective that refuses the trade.

Let me lay out the options honestly before committing, because "grow and anchor without a reference"
admits more than one answer. The most obvious is two-stage: run an SFT epoch on the chosen chains to grow
their likelihood, then run DPO or IPO to contrast — SFT gives the active growth, the preference stage gives
the anchor. It would work, and it is what most pipelines do. But count what it costs against what I am
trying to save. The preference stage still loads the frozen reference (the ~3 GB and the doubled forward I
just complained about), and now there is a *separate* SFT pass on top, so I have made the pipeline longer
and kept the reference — the opposite of the "for less than IPO is paying" goal I set. A second option is
to stay inside IPO and force it to grow by cranking `β` down so the target `1/(2β)` moves *up* and the
gradient stays active longer. But that fights the anchor directly: smaller `β` means weaker regularization
(I showed the target-limit is "separate at all costs" as `β → 0`), so I would be buying growth by throwing
away the very reference-anchoring that IPO's whole point was — and I would still be paying for the frozen
reference to compute an anchor I am then defeating. Both tempting options either keep the cost or break the
anchor. What I actually want is a *single-stage, reference-free* objective that contains both the growth and
the contrast, and that pushes me to ask what the smallest such object is. So let me build up from the one
term I already know grows the chosen chain for free.

Start from what SFT alone does, because if supervised fine-tuning on the chosen chains already grew their
likelihood I would be most of the way there and could stop looking for exotic losses. Write the causal-LM
NLL on a chosen response of length `m`, `L = −(1/m) Σ_k Σ_i y_i^{(k)} log p_i^{(k)}`, with `y_i^{(k)}` the
one-hot label at position `k`. The inner sum over the vocabulary only survives where `y_i = 1` — the single
label token at each position. For every other token, including every token that would build a *rejected*
chain, the term vanishes. So cross-entropy *rewards* the label tokens of the correct chain and is completely
silent about the wrong ones — it is one-sided, all reward, no penalty. And I know what that one-sidedness
does in a domain like this: raising the probability of chosen responses raises the probability of the
*whole neighborhood*, and the rejected responses live in exactly that neighborhood — same problem, same
notation, near-identical chains branching at one step. The log-prob of the rejected chain climbs right
alongside the chosen one. So plain SFT gives me the active likelihood growth IPO lacked (good) but *also*
grows the wrong chain's likelihood (bad — that is the math-reasoning trap). I need SFT's growth on the
chosen chain *plus* a penalty that runs alongside it and pushes the rejected chain down, in one stage,
with no reference.

What does a penalty that *appends* to NLL look like, and which one? The natural contrast is a probability
ratio, `P(y_w)/P(y_l)`, wrapped in a log-sigmoid: `−log σ(log[P(y_w)/P(y_l)])`. But I have to think about
running this *during* SFT, on a model still adapting, not after SFT the way DPO and IPO run on a settled
policy. For any ratio `R`, minimizing `−log σ(log R)` does not just want `R > 1`, it wants `R` *large*,
into the saturating tail — and how hard it pushes to get there depends on how spread out `log R` is across
the data. If `log R` is tightly concentrated near zero, the only way to move the loss appreciably is to
force each example to an *extreme* margin, and in the probability-ratio case an extreme margin means
crushing `P(y_l)` toward zero — slamming the rejected tokens' logits way down. On a model still learning,
those tokens overlap heavily with good tokens it still needs (the rejected math chain is a near-duplicate
of the correct one), so this degenerates generation — the same erosion that ate SimPO, reappearing from
the optimization side rather than the objective side.

Let me make "too sharp" quantitative, because the choice between probability ratio and odds ratio should be
a computed one, not a taste. Take two probabilities from a flat prior, `X_1, X_2 ∼ Unif(0,1)`, as stand-ins
for `P(y_w), P(y_l)` before the model has an opinion, and compare the spread of the two candidate contrast
statistics. The log probability ratio is `log X_1 − log X_2`. For `X ∼ Unif(0,1)`, `−log X ∼ Exp(1)`, so
`log X` has mean −1 and variance 1, and the difference `log X_1 − log X_2` has mean 0 and variance 2. Now
the log *odds* ratio: the odds `P/(1−P)` is the logistic transform stretching `(0,1)` onto `(0,∞)`, and its
log — the logit `log(P/(1−P))` — is exactly the inverse of the sigmoid, so `logit(X)` for `X ∼ Unif(0,1)` is
a *standard logistic* random variable, mean 0, variance `π²/3 ≈ 3.29`. The log odds ratio `logit(X_1) −
logit(X_2)` therefore has variance `2·π²/3 ≈ 6.58`, versus the probability ratio's 2. The odds ratio is
`√(6.58/2) ≈ 1.81×` wider in standard deviation, and the reason is structural: the `log(1−X)` piece in the
logit explodes toward `+∞` as `X → 1`, which the bare `log X` has no counterpart for.

That factor flips the design choice, and I can now say why in one sentence of mechanism. With the
wide-ranging odds ratio, a given target sigmoid output is reached by a *modest* per-example margin — the
statistic is already spread out, so I do not have to overshoot any single example to move the aggregate
loss. With the concentrated probability ratio, the only way to move the loss is an *extreme* per-example
margin, the rejected-logit-crushing I must avoid. So the odds ratio gives a *mild* discrimination of the
rejected chain — exactly the right intensity for penalizing during SFT without degenerating a model still
learning — and on near-identical math pairs, mild is precisely what protects the correct chain that shares
those tokens. The 1.81× spread is the quantitative reason the odds ratio is the safe contrast and the
probability ratio is the dangerous one.

Pin down the objects. For a response `y` of length `m`, take the length-normalized sequence log-prob
`log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t|x,y_{<t})` — the log of the geometric mean of the per-token
probabilities — so `P_θ ∈ (0,1)`, the odds `P/(1−P)` is finite, and different-length chains are comparable.
The odds is `odds_θ(y) = P_θ(y)/(1 − P_θ(y))`, the odds ratio `OR = odds_θ(y_w)/odds_θ(y_l)`, and the
penalty wraps its log in a negative log-sigmoid so driving the loss down drives the log odds ratio up:
`L_OR = −log σ(log[odds_θ(y_w)/odds_θ(y_l)])`. The full single-stage objective is
`L_ORPO = E[ L_SFT + λ·L_OR ]`, where `L_SFT = −log P_θ(y_w)` is the ordinary NLL on the chosen chain —
this is the **active likelihood growth** IPO never had — and `L_OR` is the new contrast. And crucially:
**no `π_ref`**. The contrast is between `y_w` and `y_l` under the *same current* parameters, and the "don't
generate the wrong chain" pressure comes from comparing each chain's probability with its own complement
`1 − P`, not with a frozen model's. One model, one stage; the anchor on the correct chain is the SFT term,
not a reference. That is the trade I wanted off IPO: the anchor *and* the growth, without the second model.

One more choice to justify: why wrap the log odds ratio in `−log σ(·)` rather than just penalize `−log OR`
directly? The raw log odds ratio is unbounded *below* — as the model gets a pair badly wrong, `log OR →
−∞`, and `−log OR → +∞` with a constant-magnitude gradient. That would let a single very-wrong pair
dominate the batch with an unbounded pull, the exact unboundedness disease I have been fighting since
SimPO, now on the penalty side. The `−log σ` wrapper bounds the per-example loss (it flattens as the pair
gets more wrong, because `σ` saturates) and, more importantly, is what *produces* the self-pacing weight
`δ = [1 + odds_w/odds_l]^{−1}` in the first place — the sigmoid's derivative is what makes the weight
shrink on solved pairs and grow on wrong ones. So the wrapper is not decoration; it is what keeps the
contrast bounded and self-pacing instead of runaway. Let me verify the gradient does the right thing, because
I am asserting it both grows the chosen and stays mild on the rejected, and those are two separate claims. With `u = log g`, `g = odds_w/odds_l`, the descent
step moves in `+δ·h`. The example weight is `δ = [1 + odds_w/odds_l]^{−1}` — large when the model wrongly
prefers the rejected chain (odds ratio small) and `→ 0` once the example is solved (odds ratio large), so
the objective self-paces, pouring gradient into hard examples and releasing solved ones. The direction is
`h = ∇log P(y_w)/(1−P(y_w)) − ∇log P(y_l)/(1−P(y_l))` — a `+∇log P(y_w)` on the chosen (the growth) and a
`−∇log P(y_l)` on the rejected (the discrimination SFT alone could not do), each scaled by the log-odds
sensitivity `1/(1−P)`. That `1/(1−P)` factor is the odds ratio's signature: it grows as `P → 1`, so a
rejected chain that has become *too plausible* — the math trap — gets a *sharper* negative push exactly
when it is becoming dangerous, and a chosen chain that is already very likely has its own growth
sharpened. At `P = 0.5` the factor is 2, at `P = 0.9` it is 10; the penalty automatically intensifies on
the chains that have drifted toward high probability, which is where the trap lives. The derivation, not
just the intuition, says this grows the correct chain and trims the wrong one, and nothing in `δ` or `h`
touches a second model: one model in memory, two forwards per batch versus IPO's four. That is the same
memory and compute profile SimPO had — the ~3 GB reference copy gone, the preference-stage forwards halved
— except that where SimPO paid for cheapness with *no anchor at all*, ORPO keeps the anchor by folding SFT
into the loss. So I have recovered SimPO's cost while adding back the growth-and-anchor IPO paid a whole
extra model for. No SFT warm-up either, since `L_SFT` is *in* the objective — the growth and the contrast
run in the same step on the same batch, which is the whole appeal of the single-stage form over the
two-stage pipeline I eliminated above.

Put a couple of numbers through the self-pacing weight `δ = [1 + odds_w/odds_l]^{−1}` to see it allocate
gradient, because "self-pacing" is only a virtue if the allocation is the one I want. Suppose the model has
the pair badly wrong: `P_w = 0.3`, `P_l = 0.7`, so `odds_w = 0.43`, `odds_l = 2.33`, `odds_w/odds_l =
0.184`, `δ = 1/1.184 = 0.84` — a large weight on a wrongly-ordered pair, most of the available gradient.
Now suppose it has the pair right: `P_w = 0.7`, `P_l = 0.3`, `odds_w/odds_l = 5.44`, `δ = 1/6.44 = 0.155` —
the weight has dropped more than five-fold, the objective easing off a solved example. And the direction
sensitivity `1/(1−P)` layers on top: on that wrong pair the rejected chain sits at `P_l = 0.7`, sensitivity
`1/0.3 = 3.3`, so the `−∇log P(y_l)` push is amplified 3.3× — the wrong chain is both plausible and being
actively trimmed, exactly when it is dangerous. The two factors compose to concentrate learning on
plausible-but-wrong rejected chains and released-once-correct pairs, which is the allocation the math trap
calls for. And because the SFT term shares the *same* `P_θ(y_w)` the odds ratio uses, the growth and the
contrast are coupled through one quantity rather than fighting across two objectives — raising `P_θ(y_w)`
helps the SFT term and the odds ratio simultaneously.

Now the numerical shape and the substrate. The harness hands me, per response, the length-averaged token
log-prob, so `c = log P_θ(y_w)` and `r = log P_θ(y_l)` are each a mean-per-token log-prob, `≤ 0`, with
`P = e^c`. The log odds ratio is `(c − r) − [log(1 − e^c) − log(1 − e^r)]`. The term `log(1 − e^c)`
underflows or hits `log 0` naively when `c` is near 0; the stable primitive is `log1p(−exp(c))` for
`c ≤ 0`. Let me confirm it computes the right thing on a concrete value: at `c = −0.3`, `P = e^{−0.3} =
0.7408`, `1 − P = 0.2592`, and `log(0.2592) = −1.350`, while `log1p(−exp(−0.3)) = log1p(−0.7408) =
log(0.2592) = −1.350` — identical, and stable because `log1p` keeps precision when its argument is near
zero (the `P → 0` end) and `exp(c)` never overflows for `c ≤ 0`. Then `L_OR = −logsigmoid(log_odds)` and
`L_SFT = −c`, per-example `loss = −c + λ·(−logsigmoid(log_odds))`. The substrate makes this drop straight
in: `orpo` is in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` hands
`compute_preference_loss` the **average** per-token log-probs the odds needs; and `orpo` *is* in the
reference-free set in `finetuning_args.py`, so `use_ref_model` is False, no reference model is loaded, and
my loss lands in the top branch, dispatched to the `odds_ratio_loss` helper — which computes exactly
`(c − r) − (log1p(−exp(c)) − log1p(−exp(r)))`, then `−c + β·(−logsigmoid(log_odds))`, with `self.beta`
playing the role of `λ`. (The full scaffold dispatch is in the answer.)

Two limit-checks on `L_OR` before I set the balance, because a contrast term that misbehaves at the
extremes would poison the fused loss. At a solved pair — the correct chain nearly certain, the wrong chain
nearly impossible, `P_w → 1`, `P_l → 0` — the odds `odds_w → ∞`, `odds_l → 0`, so `log OR → +∞` and
`L_OR = −logsigmoid(+∞) → 0`: the penalty vanishes on pairs the model already has right, so it will not keep
shoving a solved example and cannot manufacture the runaway suppression I am trying to avoid. At an
undecided pair, `P_w = P_l`, `log OR = 0` and `L_OR = −logsigmoid(0) = 0.69`, a finite nonzero push — the
penalty engages when the model is genuinely torn and releases when it is not. And the length normalization
is not optional for any of this: without dividing by length, a long chain's summed log-prob can be very
negative, `P = e^{summed}` underflows to 0, the odds `P/(1−P)` is 0, and `log OR` is `−∞` or `NaN`. The
geometric-mean (per-token averaged) `P ∈ (0,1)` is exactly what keeps the odds finite and the two limits
above well-defined, which is why the harness handing me averaged log-probs for `orpo` is a correctness
requirement here, not a convenience.

I should sanity-check the balance between the two terms at `λ = β = 0.1`, because if the penalty dwarfed
the SFT term I would be back to a contrast-dominated loss that could erode the chosen chain, and if it
vanished I would have plain SFT. `L_SFT = −c` with `c` a per-token log-prob around −0.3 to −0.7, so the SFT
term is order 0.3–0.7. `L_OR` at initialization, where the model has no opinion and `log_odds ≈ 0`, is
`−logsigmoid(0) = 0.69`, so `λ·L_OR ≈ 0.1 × 0.69 ≈ 0.07`. The SFT term outweighs the penalty by roughly
5–10×. So at the default small `λ` the objective is *SFT-led with a mild contrast rider* — the active
growth dominates and the odds-ratio penalty gently discourages the wrong chain — which is exactly the
regime I argued for: grow the correct chain, trim the wrong one mildly, do not let the contrast run the
show. If `λ` were large the mild penalty would stop being mild and the near-identical rejected chains would
drag the chosen down again, the SimPO failure from the optimization side, so `λ` is the knob and small is
the safe setting.

Now the falsifiable expectations against IPO's 85.9 / 74.4 / 6.67. The new ingredient relative to IPO is
the *active SFT term growing the correct chain's likelihood*, paired with a *mild* odds-ratio penalty
instead of IPO's conservative defend-only far target. So I predict ORPO improves on the benchmarks where
IPO left headroom on the table — MATH-500 most plausibly, where IPO sat at 74.4 with a purely defensive
objective and the SFT term can push the correct chains higher rather than merely holding them. GSM8K stays
saturated near 85–86. AIME is the wild card: it moves in 3.33-point quanta on thirty problems, so it is as
plausibly a step up of one or two more problems (from genuinely growing correct long chains) as it is to
sit at 6.67; I will not over-read a single AIME problem either way. The risk I am watching is the mirror of
SimPO's: ORPO is *also* reference-free, so if `λ` is too high the mild penalty stops being mild and the
correct chain erodes — but at the default small `λ` the SFT balance above says growth should dominate. So
the signature I am betting on is "MATH-500 up, average up past IPO's 55.66, GSM8K flat, AIME noisy-to-up."
If MATH-500 fails to move, the SFT term is not buying the active growth I claimed, and the next rung would
have to grow the correct chain's likelihood more directly — and, given that reference-free growth has now
been tried, anchored against a reference so the growth cannot be quietly undone by the contrast. That is
the test ORPO is running.
