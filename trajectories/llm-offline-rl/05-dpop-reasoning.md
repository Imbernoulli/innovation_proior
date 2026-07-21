DPO topped the ladder, and the way it topped it is the opening for the next move. Seed 42: GSM8K 85.9,
MATH-500 74.2, AIME 13.33, average 57.81 — the best, and exactly the synthesis I predicted. Deltas from
ORPO's 85.37 / 72.4 / 13.33: GSM8K +0.53, MATH-500 +1.8, AIME 0, average +0.78. DPO did the two things the
table said it would: it *repaired MATH-500* back to 74.2 where ORPO regressed it (the reference anchor doing
its job on the chains ORPO left un-anchored) while *holding AIME* at 13.33 (the self-pacing growth still
cracking the long correct chains). Both ingredients, together, beating every rung that had only one — the
fourth cell of the {anchor, growth} table confirmed.

Now read the whole ladder as four columns, because the plateaus are as informative as the gains. Averages
54.46 → 55.66 → 57.03 → 57.81, positive but *diminishing* (+1.20, +1.38, +0.78). AIME 3.33 → 6.67 → 13.33
→ 13.33 — climbed three rungs then *stopped*, stuck at four problems. MATH-500 74.0 → 74.4 → 72.4 → 74.2 —
bounced around but *never exceeded* IPO's 74.4; DPO's repair brought it back, not forward. GSM8K flat noise
throughout. So the remaining headroom is not GSM8K, pinned at its ceiling. It is in the two benchmarks that
have gone quiet — AIME plateaued at 13.33, MATH-500 capped at ~74.4 — and both are the *near-duplicate-pair*
regime: the long competition chains on AIME, the dense middle-band math on MATH-500, exactly where, since
the first rung, the correct chain's absolute likelihood is most fragile. The ladder has climbed as far as
"have both ingredients" can take it and stalled precisely where the residual failure I flagged at the top of
the DPO derivation lives.

That residual, re-read: DPO's loss is satisfied by making the *difference* `ρ(y_w) − ρ(y_l)` positive,
where `ρ(y) = log π_θ(y) − log π_ref(y)`. On near-identical math pairs that can be met by raising `ρ(y_w)`
or by *lowering* `ρ(y_l)`, and lowering the rejected chain, which shares almost every token with the
correct one, drags `ρ(y_w)` down too — the two worlds `ρ(y_w) = +0.5, ρ(y_l) = −0.5` and
`ρ(y_w) = −1.0, ρ(y_l) = −2.0` produce the identical DPO logit `β·1.0` even though in the second the correct
chain has fallen a full nat below the reference. DPO's `σ` brake stops the push once the pair is *ordered*,
which is why DPO does not collapse; but "stop once ordered" is not "never let the correct chain's absolute
probability fall." The logit is blind to whether the gap was won by pushing the chosen up or the rejected
down, and greedy correctness depends on the absolute likelihood of the correct chain, not the gap. That
blindness is the residual sitting underneath 57.81, and the AIME/MATH-500 plateau is what it looks like from
the leaderboard.

Before committing to a fix, lay out the options, because the residual can be attacked from more than one
angle. Option one: length-normalize DPO, dividing the summed log-ratios by length, which kills the length-
amplification I worried about. It is a real improvement to that problem, but does it fix the gap-blindness?
After averaging, the logit is still `β·(ρ̄(y_w) − ρ̄(y_l))`, a function of the *difference* of the per-token
reference-relative log-ratios alone, so worlds A and B still map to the same logit whenever their averaged
gaps match. Normalizing rescales the coordinate; it adds no term reading `ρ(y_w)` by itself, so the
blindness survives — it treats the length symptom, not the disease. Option two: go back to ORPO and add a
reference to its odds-ratio — but that abandons DPO's clean Bradley-Terry synthesis to rebuild the anchor on
a different, mildly-corrosive contrast, throwing away the rung I just proved strongest. Option three: keep
DPO exactly and add a term that reads `ρ(y_w)` directly and objects when it falls. Only option three touches
the actual mechanism, and it does so without disturbing the anchor or the self-pacing. So the move is to
take DPO — which already has the two ingredients the whole ladder proved necessary — and *add a term that
forbids the correct chain's likelihood from falling below the reference's*. Keep everything that made DPO
the top; close the one hole its gap-blindness leaves open. Let me derive the term from the failure so it
composes with DPO's anchor and self-pacing rather than fighting them.

The pathology, stated in `ρ`, is `ρ(y_w) < 0` — the policy has made the correct chain *less* likely than
the reference — while `ρ(y_w) − ρ(y_l)` stays positive because `ρ(y_l)` fell further. The fix must make the
loss unhappy when `ρ(y_w) < 0` regardless of the gap. The cleanest such term is a one-sided penalty on the
correct chain's reference-relative log-ratio: `max(0, −ρ(y_w)) = max(0, log π_ref(y_w) − log π_θ(y_w))`.
When `ρ(y_w) ≥ 0` it is exactly zero — nothing to fix, no gradient; when `ρ(y_w) < 0` it equals `−ρ(y_w)`
and grows linearly as the correct chain falls further below the reference. One-sided is the whole point. A
symmetric term would also penalize `ρ(y_w) > 0`, punishing the model for making the correct chain *more*
likely than the reference — exactly the active growth that cracked AIME. And a symmetric term forcing
`ρ(y_w)` up everywhere is essentially an SFT term on the chosen chain, which I saw at rung three raises the
whole neighborhood including the near-duplicate rejected chain — the erosion I am trying to escape. The
`max(0, ·)` engages only where the chosen chain has *already fallen below the reference*, a regime where
lifting it back is unambiguously good and there is no runaway SFT-style growth to re-import the neighborhood
problem. So it is a *floor*, not a target: forbid the fall, permit the rise.

Where does it go? Two places, not equivalent. As a separate term outside the sigmoid,
`L_DPO + λ·max(0, −ρ(y_w))`, the barrier fires with a constant gradient whenever `ρ(y_w) < 0` regardless of
whether the pair is already correctly ordered — a slow drift back toward SFT-on-chosen even on pairs DPO has
comfortably solved. Subtracted *inside* the DPO logit, the barrier's gradient is gated by the same `σ`
self-pacing weight as the contrast, so it is strong on pairs the model is getting wrong and fades on ones it
has right — the barrier inherits DPO's brake for free. So it goes inside, scaled by a new coefficient `λ`:
`L_DPOP = −E[ log σ( β·( ρ(y_w) − ρ(y_l) − λ·max(0, −ρ(y_w)) ) ) ]`. Read the gradient. When `ρ(y_w) ≥ 0`
the penalty and its gradient are zero — DPOP *is* DPO, losing nothing that made it the top baseline. When
`ρ(y_w) < 0` the logit gains `−λ·(−ρ(y_w)) = +λ·ρ(y_w)` (negative), which shrinks the logit, raises the
loss, and — through the `σ` weight — pours gradient into `+∇log π_θ(y_w)`, pushing the correct chain back
above the reference. The logit's derivative w.r.t. `log π_θ(y_w)` is `1` from the DPO term plus `λ` from the
barrier, so when active the chosen chain is pulled up `(1 + λ)×` as hard as under plain DPO. Invisible while
the correct chain stays anchored, sharply restoring once it slips — the "never let the correct chain's
absolute probability fall" guarantee the DPO logit was blind to, in the same reference-relative coordinate
DPO already uses.

Watch it flip a "won" gap on a concrete pair. Take `β = 0.1`, `λ = 5`, the pathological world
`ρ(y_w) = −0.5, ρ(y_l) = −2.0`. Under DPO the logit is `0.1·(−0.5 + 2.0) = 0.15`, loss `−log σ(0.15) = 0.62`
— DPO is nearly content though the correct chain has fallen. Under DPOP the penalty is `max(0, 0.5) = 0.5`,
so the logit is `0.1·(1.5 − 5·0.5) = −0.10`, loss `0.74`, and now the logit is negative so the self-pacing
weight `σ(0.10) = 0.52` is above one-half, pouring gradient into raising `ρ(y_w)`, amplified `6×`. A pair DPO
would have signed off becomes an active correction pointed exactly at the greedy-accuracy quantity. On the
healthy world `ρ(y_w) = +0.5, ρ(y_l) = −0.5` the penalty is zero and DPOP equals DPO identically — the
barrier fires only where it should. As `λ → ∞` it becomes a hard constraint `ρ(y_w) ≥ 0`; finite `λ` softens
that into a penalty that lets the optimizer pay a growing price if the correct chain must temporarily dip.
And the `σ` self-pacing composes cleanly: on a wrong pair the contrast wants `+∇log π_θ(y_w)` and, if
`ρ(y_w) < 0`, so does the barrier — they point the same way, so the barrier reinforces the growth exactly
where it is needed.

Two safety properties. Set `λ = 0` and the penalty vanishes identically — DPOP is a strict generalization
of DPO with one extra non-negative knob, so it cannot do worse than DPO on pairs that do not trigger the
barrier, and the ladder's strongest result is recoverable by construction. I am not risking the 57.81 to
chase more; I am adding a one-directional correction on top of it. And the barrier is never
counterproductive: the only thing it touches is `ρ(y_w) < 0`, and there is no benchmark story where making
the *correct* chain less likely than a competent math-SFT reference is desirable, so its active region is
unambiguously the failure region.

The substrate: this finale is a real fill of `compute_preference_loss`, the `custom` slot the task reserves.
`custom` is *not* in the `["ipo","orpo","simpo"]` set, so the loss receives **summed** sequence log-probs —
correct, because `ρ(y) = log π_θ(y) − log π_ref(y)` lives in log-likelihoods, the coordinate in which the
`β log Z` cancellation holds and DPO's own reward lives. And `custom` is *not* in the reference-free set
(`use_ref_model = stage=="dpo" and pref_loss not in ["orpo","simpo"]`), so `use_ref_model` stays True, the
reference is loaded, and all four log-probs arrive — exactly what the floor `ρ(y_w) = policy_chosen_logps −
reference_chosen_logps` needs. So I leave the `finetuning_args.py` line untouched and add an
`elif self.loss_type == "custom"` branch on the reference-based side: form the DPO log-ratios, the chosen-
side `ρ(y_w)`, the one-sided penalty `relu(reference_chosen_logps − policy_chosen_logps)`, subtract
`λ·penalty` inside the logits, and return `−logsigmoid(β·logits)` with DPO's detached implicit rewards. (The
full fill is in the answer.) Numerically the term is safe: the relu is a bounded non-negative subtraction
inside the logit and `logsigmoid` is stable for logits of either sign, so even a large slip gives only a
large positive loss and a strong-but-finite corrective gradient.

The one genuinely new number is `λ`, and I reason it rather than guess. The barrier subtracts `λ·(−ρ(y_w))`
from the logit, then `β` multiplies, so a slip of `δ` nats produces a logit correction of `β·λ·δ`. For the
barrier to bite — an order-one shift in the sigmoid argument — at a modest slip `δ ≈ 1` with `β = 0.1`, I
need `λ` on the order of `1/(β·δ) = 10`. Settings where this barrier was first explored used `λ` around 50
with a larger `β` on much larger models (34B/72B), but those do not transfer: the product `β·λ` sets the
correction strength, and my `β` is small, my model small, my summed log-ratios on the scale of these short-
to-medium math completions. So I start `λ = 5` (`β·λ = 0.5` per nat of slip, firm but not overwhelming) and
treat it as the one knob. Too large and the loss degrades to SFT-on-chosen with a vestigial contrast,
shoving GSM8K and MATH-500 around and blunting the AIME growth; too small and it never engages and I am back
to plain DPO. The sweep is one-directional and cheap: start at 5, watch the training-time mean of `ρ(y_w)`,
increase only if it is still dipping negative — the knee where the barrier just holds the floor without the
loss looking like SFT.

The bar is DPO's 85.9 / 74.2 / 13.33, average 57.81. My falsifiable claim is that DPOP holds DPO's gains
everywhere the barrier is inactive — GSM8K stays ~85–86, and by construction it cannot regress below DPO at
`λ = 0` — and *adds* on the pairs where DPO was silently letting the correct chain slip, the near-identical
math pairs densest on MATH-500 and AIME. The barrier does not invent new correct chains; it converts pairs
from world B (chosen fallen below reference) into world A (at or above) by lifting `ρ(y_w)`, so the gain is
bounded by the fraction of pairs DPO was solving the world-B way *and* whose problems sit near the boundary
of greedy correctness — a fallen `ρ(y_w)` only flips an answer if lifting it tips that problem's greedy chain
from wrong to right. On near-saturated GSM8K almost no problems sit near that boundary, so I expect
essentially nothing. On MATH-500, the dense middle, the marginal-problem population is largest, so I most
expect to finally clear the 74.4 cap. On AIME, stuck at four problems for two rungs, each recovered problem
is +3.33, so a quantized step up is possible if even one or two unsolved problems have a correct chain DPO
was letting slip. The cleanest validation is the diagnostic the penalty was built from: log the mean of
`ρ(y_w)` over training — under DPO it should trend negative early (the pathology), under DPOP pinned at or
above zero, with the benchmark gain tracking the fraction of pairs DPO would have driven negative. And the
honest falsifier: if `ρ(y_w)` never goes negative under DPO on this data, the barrier is inactive, DPOP
collapses to DPO, and there is no gain. I expect it to go negative, because the plateau is exactly the
symptom a fallen `ρ(y_w)` would produce — but the training log is what turns that expectation into a
measurement rather than a hope.
