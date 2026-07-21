IPO did roughly what I predicted, and the size of the move says the diagnosis was right but the remedy was
timid. Seed 42: GSM8K 85.9, MATH-500 74.4, AIME 6.67, average 55.66. Deltas against SimPO's
86.05 / 74.0 / 3.33: GSM8K −0.15 (noise on a saturated benchmark), MATH-500 +0.4 (a whisker), AIME +3.34,
which is *two* correct problems where SimPO had one. The average gain decomposes as
`(−0.15 + 0.40 + 3.34)/3 = 1.20`, and AIME is 93% of it. So essentially the entire IPO improvement is one
recovered AIME problem — the diagnosis confirmed cleanly: bringing back the reference and swapping the
saturating sigmoid for a bounded target stopped the unbounded push, exactly on the most fragile benchmark.

But look how small the recovery is. AIME went one problem to two; MATH-500 barely twitched. This matches
the reading I closed IPO on: the target `1/(2β) = 5.0` sits so far above the per-token gaps this data
produces that the brake almost never engages, and what carried the gain was the reference *anchor* alone —
a conservative one. IPO's gradient `2(h_π − 5)` is a gentle, roughly-constant nudge; it *defends* the
correct chain's likelihood from falling below the reference but never *grows* it above where `π_ref` already
had it. On near-saturated GSM8K that is fine, but on the harder benchmarks I am leaving improvement on the
table: IPO holds the line, it does not advance it. So the question for this attempt is what would actively
*grow* the correct chain's likelihood while still contrasting against the rejected one — and, I would like,
for less than IPO is paying, since IPO drags a frozen reference around: a second resident model, ~3 GB per
GPU, four forwards per step versus SimPO's two, all to provide a defend-only anchor.

I want both halves at once: an anchor on the correct chain's absolute likelihood *and* active growth of it,
*without* a frozen reference. That is greedy — SimPO says cheap-but-unanchored, IPO says anchored-but-
expensive-and-passive — so let me see whether an objective refuses the trade. The obvious alternatives do
not. Two-stage (an SFT epoch to grow, then DPO/IPO to contrast) works but the preference stage still loads
the reference and now there is a separate SFT pass on top — a longer pipeline that keeps the reference, the
opposite of the goal. Staying inside IPO and cranking `β` down to move the target up fights the anchor
directly: smaller β is weaker regularization, so I would buy growth by throwing away the very reference-
anchoring that was IPO's point, while still paying for the reference. Both either keep the cost or break the
anchor. What I want is a *single-stage, reference-free* objective containing both growth and contrast, so
let me build up from the one term I know grows the chosen chain for free.

Start from SFT. The causal-LM NLL on a chosen response, `L = −(1/m) Σ_k Σ_i y_i^{(k)} log p_i^{(k)}`, has
its inner vocabulary sum survive only where `y_i = 1` — the single label token at each position. For every
other token, including every token that would build a *rejected* chain, the term vanishes. So cross-entropy
rewards the label tokens of the correct chain and is completely silent about the wrong ones — one-sided,
all reward, no penalty. And I know what that one-sidedness does here: raising the probability of chosen
responses raises the probability of the *whole neighborhood*, and the rejected responses live in exactly
that neighborhood — same problem, near-identical chains branching at one step. The rejected chain's log-prob
climbs alongside the chosen one. So plain SFT gives the active growth IPO lacked but also grows the wrong
chain — the math-reasoning trap. I need SFT's growth on the chosen chain *plus* a penalty running alongside
it that pushes the rejected down, in one stage, no reference.

What penalty *appends* to NLL, and which one? The natural contrast is a probability ratio `P(y_w)/P(y_l)`
wrapped in a log-sigmoid. But I have to run this *during* SFT, on a model still adapting, not after it on a
settled policy. Minimizing `−log σ(log R)` wants `R` not just above 1 but *large*, into the saturating
tail, and how hard it pushes depends on how spread out `log R` is across the data. If `log R` is tightly
concentrated near zero, the only way to move the loss appreciably is to force each example to an *extreme*
margin — and in the probability-ratio case an extreme margin means crushing `P(y_l)` toward zero, slamming
the rejected tokens' logits down. On a model still learning, those tokens overlap heavily with good tokens
it still needs, so this degenerates generation — SimPO's erosion, reappearing from the optimization side.

Make "too sharp" quantitative, because the choice between probability ratio and odds ratio should be
computed, not tasted. Take `X_1, X_2 ∼ Unif(0,1)` as stand-ins for `P(y_w), P(y_l)` before the model has an
opinion. The log probability ratio `log X_1 − log X_2`: since `−log X ∼ Exp(1)`, `log X` has variance 1 and
the difference has variance 2. The log *odds* ratio uses the logit `log(P/(1−P))`, which for `X ∼ Unif(0,1)`
is a *standard logistic* variable, variance `π²/3 ≈ 3.29`, so the difference has variance
`2·π²/3 ≈ 6.58` — `√(6.58/2) ≈ 1.81×` wider in standard deviation. The reason is structural: the
`log(1−X)` piece in the logit explodes toward `+∞` as `X → 1`, which the bare `log X` has no counterpart
for. That factor flips the choice. With the wide-ranging odds ratio, a given target sigmoid output is
reached by a *modest* per-example margin — the statistic is already spread out, so I never overshoot a
single example. With the concentrated probability ratio, the only way to move the loss is the rejected-
logit-crushing I must avoid. So the odds ratio gives a *mild* discrimination of the rejected chain, exactly
the right intensity for penalizing during SFT without degenerating a model still learning, and on near-
identical math pairs mild is what protects the shared correct tokens.

Pin down the objects. For a response of length `m`, take the length-normalized log-prob
`log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t|·)` — the log of the geometric mean of per-token probabilities — so
`P_θ ∈ (0,1)`, the odds `P/(1−P)` is finite, and different-length chains are comparable. The penalty wraps
the log odds ratio in a negative log-sigmoid, `L_OR = −log σ(log[odds_θ(y_w)/odds_θ(y_l)])`, and the full
single-stage objective is `L_ORPO = E[L_SFT + λ·L_OR]` with `L_SFT = −log P_θ(y_w)` the ordinary NLL — the
active growth IPO never had. And crucially **no `π_ref`**: the contrast is between `y_w` and `y_l` under the
same current parameters, and the "don't generate the wrong chain" pressure comes from comparing each chain
with its own complement `1 − P`, not with a frozen model. One model, one stage; the anchor on the correct
chain is the SFT term. That is the trade I wanted off IPO — anchor *and* growth, without the second model.

Why wrap the log odds ratio in `−log σ` rather than penalize `−log OR` directly? The raw log odds ratio is
unbounded *below* — a badly-wrong pair sends `−log OR → +∞` with constant-magnitude gradient, a single pair
dominating the batch, the unboundedness disease again on the penalty side. The `−log σ` wrapper bounds the
per-example loss and, more importantly, *produces* the self-pacing weight in the first place. The gradient
is `+δ·h`: the example weight `δ = [1 + odds_w/odds_l]^{−1}` is large when the model wrongly prefers the
rejected chain and `→ 0` once solved, so the objective pours gradient into hard examples and releases
solved ones. The direction `h = ∇log P(y_w)/(1−P(y_w)) − ∇log P(y_l)/(1−P(y_l))` is `+∇log P(y_w)` growth on
the chosen and `−∇log P(y_l)` discrimination on the rejected, each scaled by `1/(1−P)` — the odds ratio's
signature, growing as `P → 1` (2 at P=0.5, 10 at P=0.9). So a rejected chain that has become *too plausible*
gets a *sharper* negative push exactly when it is becoming dangerous. Nothing in `δ` or `h` touches a second
model: one model in memory, two forwards per batch versus IPO's four — SimPO's cost profile, except where
SimPO paid for cheapness with no anchor at all, ORPO keeps the anchor by folding SFT into the loss. No SFT
warm-up either, since the growth and the contrast run in the same step on the same batch.

The numerical shape and the substrate. The harness hands me, per response, the length-averaged token
log-prob, so `c = log P_θ(y_w)` and `r = log P_θ(y_l)` are mean-per-token log-probs `≤ 0` with `P = e^c`.
The log odds ratio is `(c − r) − [log(1 − e^c) − log(1 − e^r)]`, and `log(1 − e^c)` underflows naively when
`c` is near 0, so the stable primitive is `log1p(−exp(c))` for `c ≤ 0` (`log1p` keeps precision near zero,
`exp(c)` never overflows for `c ≤ 0`). Then `L_OR = −logsigmoid(log_odds)`, `L_SFT = −c`, per-example
`loss = −c + λ·(−logsigmoid(log_odds))`. This drops straight in: `orpo` is in the `["ipo","orpo","simpo"]`
set, so `concatenated_forward` hands the average per-token log-probs the odds needs; and `orpo` *is* in the
reference-free set, so `use_ref_model` is False, no reference loaded, and my loss lands in the top branch
dispatched to `odds_ratio_loss`, with `self.beta` playing the role of `λ`. (The full dispatch is in the
answer.)

The length normalization is not optional: without dividing by length, a long chain's summed log-prob is
very negative, `P = e^{summed}` underflows to 0, the odds is 0, and `log OR` is `−∞` or `NaN`. The geometric-
mean `P ∈ (0,1)` keeps the odds finite — a correctness requirement, not a convenience. And the limits are
benign: at a solved pair (`P_w → 1, P_l → 0`) `log OR → +∞` and `L_OR → 0`, so the penalty vanishes on
pairs the model already has right and cannot manufacture runaway suppression; at an undecided pair
`L_OR = −logsigmoid(0) = 0.69`, engaging only when the model is genuinely torn.

I should check the balance between the two terms at `λ = β = 0.1`, because a penalty that dwarfed the SFT
term would be back to a contrast-dominated loss that erodes the chosen chain. `L_SFT = −c` with `c` around
−0.3 to −0.7 is order 0.3–0.7; `λ·L_OR` at initialization (`log_odds ≈ 0`) is `0.1 × 0.69 ≈ 0.07`. The SFT
term outweighs the penalty 5–10×, so at the default small `λ` the objective is *SFT-led with a mild contrast
rider* — active growth dominates, the odds-ratio penalty gently discourages the wrong chain — exactly the
regime I argued for. If `λ` were large the near-identical rejected chains would drag the chosen down again,
so `λ` is the knob and small is the safe setting.

Against IPO's 85.9 / 74.4 / 6.67, the new ingredient is the *active SFT term* growing the correct chain,
paired with a *mild* odds-ratio penalty instead of IPO's defend-only far target. So I predict ORPO improves
where IPO left headroom — MATH-500 most plausibly, where IPO sat at 74.4 with a purely defensive objective
and the SFT term can push the correct chains higher. GSM8K stays saturated near 85–86. AIME is the wild
card: on thirty problems it is as plausibly a step of one or two more problems as it is to sit at 6.67, and
I will not over-read a single AIME problem. The risk I am watching is the mirror of SimPO's — ORPO is also
reference-free, so if `λ` is too high the mild penalty stops being mild and the correct chain erodes — but
at the default small `λ` the balance says growth should dominate. If MATH-500 fails to move, the SFT term is
not buying the growth I claimed, and the next attempt would have to grow the correct chain more directly,
and — reference-free growth now tried — anchored against a reference so it cannot be quietly undone by the
contrast.
