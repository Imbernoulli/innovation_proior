IPO did roughly what I predicted, and the size of the move tells me the diagnosis was right but the
remedy was timid. Seed 42: GSM8K 85.9 (flat, as it had to be — saturated), MATH-500 74.4 (a hair up from
SimPO's 74.0), AIME 6.67 — *two* correct problems where SimPO had one. So bringing back the reference and
replacing the saturating sigmoid with a finite target did exactly what I bet it would: it stopped the
unbounded push that was dragging the correct chains down, and the most fragile benchmark recovered. The
average climbed from 54.46 to 55.66. But look at how small the AIME recovery is — 3.33 to 6.67 is one
extra problem out of thirty, and MATH-500 barely moved. The reference anchor at β = 0.1 is holding the
correct chain near `π_ref`, but that is a weak ask: IPO regresses the *reference-corrected log-ratio gap*
onto a fixed target and goes silent the instant the gap is met, so once a pair clears `1/(2β)` it stops
learning from it. The objective is conservative by construction — it never pushes the correct chain
*above* where the reference already had it, it just keeps it from falling much below. On a near-saturated
GSM8K that is fine, but on the harder benchmarks I am leaving improvement on the table: IPO defends the
correct chain's likelihood, it does not actively *grow* it.

So the question for this rung is: what would actively grow the correct chain's likelihood while still
contrasting against the rejected one — and could I get that for less than IPO is paying? Because there is a
cost I have not questioned. IPO, like DPO, drags a frozen reference around: a second resident model and a
second forward pass every batch, four forwards per step (chosen and rejected, each through policy and
reference). On a memory-tight box doing full-parameter 1.5B training, that is real. SimPO was reference-
free and cheap but had no anchor; IPO has an anchor but pays for a whole extra model to provide it. I want
both: an anchor on the correct chain's absolute likelihood, *without* a frozen reference. Let me derive a
loss that gets it.

Start from what SFT alone does, because if supervised fine-tuning on the chosen chains already grew their
likelihood I would be most of the way there. Write the causal-LM NLL on a chosen response of length m,
`L = −(1/m) Σ_k Σ_i y_i^{(k)} log p_i^{(k)}`, with `y_i^{(k)}` the one-hot label at position k. The inner
sum only survives where `y_i = 1` — the single label token at each position. For every other token,
including every token that would build a *rejected* chain, the term vanishes. So cross-entropy *rewards*
the label tokens of the correct chain and is completely silent about the wrong ones — it is one-sided, all
reward, no penalty. And I know what that one-sidedness does: raising the probability of chosen responses in
a domain raises the probability of the *whole neighborhood*, and rejected responses live in exactly that
neighborhood — same problem, same notation, near-identical chains branching at one step. The log-prob of
the rejected chain climbs right alongside the chosen one. So plain SFT grows the correct chain's
likelihood (good — that is the active growth IPO lacked) but *also* grows the wrong chain's, which is
exactly the math-reasoning trap. I need SFT's active likelihood growth on the chosen chain *plus* a penalty
that runs alongside it and pushes the rejected chain down — in one stage, with no reference.

What does a penalty that *appends* to NLL look like? The natural contrast is a probability ratio,
`P(y_w)/P(y_l)`, wrapped in a log-sigmoid: `−log σ(log[P(y_w)/P(y_l)])`. But I have to think about running
this *during* SFT, on a model still adapting, not after SFT like DPO/IPO. For any ratio R, minimizing
`−log σ(log R)` does not just want `R > 1`, it wants R *large*, into the saturating tail — and how hard it
pushes depends on how spread out `log R` is across the data. If `log R` is tightly concentrated near zero,
the only way to move the loss is to force each example to an *extreme* margin, and in the probability-ratio
case that means crushing `P(y_l)` toward zero — slamming the rejected tokens' logits way down. On a model
still learning, those tokens overlap heavily with good tokens it still needs (the rejected math chain is a
near-duplicate of the correct one!), so this degenerates generation — the very failure that ate SimPO,
reappearing from the optimization side.

Let me make "too sharp" quantitative. Take two probabilities from a flat prior, `X_1, X_2 ∼ Unif(0,1)`, as
stand-ins for `P(y_w)`, `P(y_l)` before the model has an opinion. The log probability ratio is
`log X_1 − log X_2`; each `log X` piles up near 0 with a tail to `−∞`, so the difference is symmetric and
*concentrated* around 0. Now the log of the *odds* ratio: the odds `P/(1−P)` is the logistic transform
stretching `(0,1)` to `(0,∞)`, and its log, the logit `log(P/(1−P))`, blows up toward both `±∞` as P
approaches 1 or 0. So `log OR = [logit(X_1)] − [logit(X_2)]` is far more *spread out* than `log PR` for the
same inputs, because the `log(1−X)` piece explodes near `X = 1`. That flips the design choice. With the
wide-ranging odds ratio, a given target sigmoid output is reached by a *modest* per-example margin — I do
not have to overshoot any single example. With the concentrated probability ratio, the only way to move the
loss is an *extreme* per-example margin, the rejected-logit-crushing I must avoid. So the odds ratio gives
a *mild* discrimination of the rejected chain, exactly the right intensity for penalizing during SFT
without degenerating a model still learning — and on near-identical math pairs, mild is what protects the
correct chain that shares those tokens.

Pin down the objects. For a response y of length m, take the length-normalized sequence log-prob
`log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t|x,y_{<t})` (the geometric mean of per-token probs) — length-
normalized so `P_θ ∈ (0,1)` and the odds `P/(1−P)` is finite, and so different-length chains are
comparable. The odds is `odds_θ(y) = P_θ(y)/(1 − P_θ(y))`, the odds ratio `OR = odds_θ(y_w)/odds_θ(y_l)`,
and the penalty wraps its log in a negative log-sigmoid so driving the loss down drives the log odds ratio
up: `L_OR = −log σ(log[odds_θ(y_w)/odds_θ(y_l)])`. The full single-stage objective is
`L_ORPO = E[ L_SFT + λ·L_OR ]`, where `L_SFT = −log P_θ(y_w)` is the ordinary NLL on the chosen chain —
this is the **active likelihood growth** on the correct chain that IPO never had — and `L_OR` is the new
contrast. And crucially: **no `π_ref`**. The contrast is between `y_w` and `y_l` under the *same current*
parameters, and the "don't generate the wrong chain" pressure comes from comparing each chain's
probability with its own complement `1 − P`. One model, one stage; the anchor on the correct chain is the
SFT term, not a frozen reference. That is the trade I wanted off IPO: the anchor without the second model.

Let me verify the gradient does the right thing, because I am asserting it both grows the chosen and is
mild on the rejected. With `u = log g`, `g = odds_w/odds_l`, the descent step moves in `+δ·h`, where the
example weight `δ = [1 + odds_w/odds_l]^{−1}` is large when the model wrongly prefers the rejected chain
and `→ 0` once the example is solved (automatic difficulty weighting, self-pacing), and the direction
`h = ∇log P(y_w)/(1−P(y_w)) − ∇log P(y_l)/(1−P(y_l))` is a contrast — a `+∇log P(y_w)` on the chosen, a
`−∇log P(y_l)` on the rejected (the discrimination SFT alone could not do), each scaled by the log-odds
sensitivity `1/(1−P)` that grows as P approaches 1. So a rejected chain that has become *too plausible* —
the math trap — gets a sharper negative push, and a chosen chain becoming plausible has its odds margin
sharpened. The derivation, not just the intuition, says this grows the correct chain and trims the wrong
one. And nothing in `δ` or `h` touches a second model: one model in memory, two forwards per batch versus
IPO's four. No SFT warm-up either, since `L_SFT` is in the objective.

The numerical shape and the substrate. The harness hands me, per response, summed token log-probs and a
valid length; I divide to get `c = log P_θ(y_w)` and `r = log P_θ(y_l)`, each a mean-per-token log-prob,
`≤ 0`, with `P = e^c`. The log odds ratio is `(c − r) − [log(1 − e^c) − log(1 − e^r)]`. The term
`log(1 − e^c)` underflows or hits `log 0` naively; the stable primitive is `log1p(−exp(c))` for `c ≤ 0`.
Then `L_OR = −logsigmoid(log_odds)` and `L_SFT = −c`, per-example `loss = −c + λ·(−logsigmoid(log_odds))`.
The substrate makes this drop straight in: `orpo` is in the `["ipo","orpo","simpo"]` set, so
`concatenated_forward` hands `compute_preference_loss` the **average** per-token log-probs the odds needs;
and `orpo` *is* in the reference-free set in `finetuning_args.py`, so `use_ref_model` is False, no reference
model is loaded, and my loss lands in the top branch, dispatched to the harness `odds_ratio_loss` helper —
which computes exactly `(c − r) − (log1p(−exp(c)) − log1p(−exp(r)))`, then `−c + β·(−logsigmoid(log_odds))`,
with `self.beta` playing the role of λ. (The full scaffold dispatch is in the answer.)

Now the falsifiable expectations against IPO's 85.9 / 74.4 / 6.67. The new ingredient relative to IPO is
the *active SFT term growing the correct chain's likelihood*, paired with a *mild* odds-ratio penalty
instead of IPO's conservative defend-only target. So I predict ORPO improves on the benchmarks where there
was headroom IPO left on the table — MATH-500 most plausibly, where IPO sat at 74.4 with a conservative
objective and the SFT term can push the correct chains higher. GSM8K stays saturated near 85–86. AIME is
the wild card: it moves in 3.33-point quanta on thirty problems, so it is as likely to tick to 13.33 (a
genuine gain from growing correct long chains) as to stay where it is; I will not over-read a single AIME
problem either way. The risk I am watching is the mirror of SimPO's: ORPO is *also* reference-free, so if
λ is set too high the mild penalty stops being mild and the near-identical rejected chains drag the chosen
down again — but at the default small λ I expect the SFT term to dominate and the net to be a likelihood
*gain* on the correct chains. So the signature I am betting on is "MATH-500 up, average up past IPO's
55.66, GSM8K flat, AIME noisy." If MATH-500 fails to move, the SFT term is not buying the active growth I
claimed, and the next rung would have to grow the correct chain's likelihood more directly — anchored
against a reference so the growth cannot be undone by the contrast.
