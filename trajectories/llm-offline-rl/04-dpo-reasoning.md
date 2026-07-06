ORPO's number is the most interesting one yet, because it broke the pattern in a way that tells me what
actually matters. Seed 42: GSM8K 85.37, MATH-500 72.4, AIME 13.33, average 57.03. Decompose the move from
IPO's 85.9 / 74.4 / 6.67 the way I have been doing: GSM8K −0.53, MATH-500 −2.0, AIME +6.66, average
`(−0.53 − 2.0 + 6.66)/3 = 4.13/3 = 1.38`, matching 55.66 → 57.03 — the biggest single jump on the ladder.
But look *where* it came from. AIME leapt from 6.67 to 13.33, which is two more correct problems (four out
of thirty now, where IPO had two and SimPO had one); that is the active SFT growth on the correct long
chains paying off exactly as I bet. And MATH-500 *fell*, 74.4 → 72.4, the first time a benchmark went
backward on the ladder. So ORPO is not a clean win; it is a trade. The reference-free fused objective grows
the correct chains hard enough to crack AIME, but the mildness of the odds-ratio penalty is not free —
without a reference holding the line, pushing the SFT term and the contrast together has cost accuracy on
the middle benchmark where SimPO and IPO both sat at ~74. The AIME gain (+6.66) simply outweighs the
MATH-500 loss (−2.0) in the average, so 57.03 > 55.66. But the regression is a real signal, and it is the
one I flagged as ORPO's risk: reference-free active growth is powerful on the hardest chains and slightly
corrosive on the broad middle, because nothing anchors the chains it is *not* actively growing.

So the diagnosis sharpens into a clean synthesis problem, and it is worth writing the whole ladder as a
table because the empty cell is the answer. I have been toggling two ingredients. Call them *anchor* — a
reference the correct chain's likelihood is measured against, so chains I do not actively push cannot
silently erode — and *growth* — a term that actively raises the correct chain's likelihood rather than
merely defending it. SimPO: no anchor (reference-free, purely relative), no real growth (relative margin
only) — and it collapsed AIME. IPO: anchor yes (reference-corrected regression), growth no (conservative
far target that only defends) — and it barely moved, +1.20, almost all one AIME problem. ORPO: anchor no
(reference-free), growth yes (the SFT term) — and it cracked AIME but regressed MATH-500. Three of the four
cells of the {anchor, growth} table are filled, each with a characteristic failure, and the pattern across
them says the same thing three times: the two ingredients are *both* necessary and I have never had them
together. The averages trace the story too — 54.46, 55.66, 57.03, each step positive but the mechanism
behind each different: SimPO→IPO bought the anchor (+1.20, almost all AIME), IPO→ORPO bought the growth
(+1.38, AIME up two problems but MATH-500 down). Neither rung has ever *not* given something back, because
neither ever held both ingredients; the fourth cell is the first chance to add without subtracting. The empty cell is (anchor yes, growth yes). ORPO got growth by dropping the anchor; IPO got the
anchor by dropping the growth; SimPO had neither. What I want is the objective in the fourth cell — the
reference anchor *and* a self-pacing growth term in one loss — and the cleanest such object is the one I
have been circling the whole ladder without running: DPO itself, the reference-based Bradley-Terry MLE that
every other rung was a reaction to. Let me derive it properly, because if it is the object that fills the
empty cell I should understand exactly why, and it is the natural top of this ladder.

There is an obvious cheaper way to fill the empty cell that I should rule out before deriving DPO, because
if it worked I would prefer it. I already have IPO (anchor, no growth) and ORPO (growth, no anchor); why not
literally add them — take IPO's reference-corrected regression and append ORPO's SFT growth term, or add an
explicit `−log π_θ(y_w)` to IPO? It would nominally put both ingredients in one loss. But it is a
two-term Frankenstein with two coefficients to balance — the regression weight and the SFT weight — and the
two terms pull on different scales (a squared per-token residual against a summed NLL), so tuning them to
cooperate rather than fight is its own project, and I would still be paying for the reference IPO needs. The
appeal of finding a *principled* single loss is that both ingredients come from one term with one
coefficient, self-consistently, rather than from a weighted sum I have to hand-balance. That is what makes
the DPO derivation worth doing rather than gluing two rungs together: if the anchor and the growth both fall
out of a single Bradley-Terry loss, I get the fourth cell for free, with `β` as the only knob.

The objective everyone writes is `max_π E_{y∼π}[r(x,y)] − β·KL(π‖π_ref)` — chase reward, stay near the
reference. The reward comes from preferences through Bradley-Terry, `p*(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`,
fit by MLE. The painful part is the next stage: maximizing `E_{y∼π}[r]` over a discrete autoregressive LM
needs RL (PPO) — a separate reward model, a critic, on-policy sampling in the loop. I want to collapse the
two stages into one supervised loss, the way the whole offline ladder does, so let me look at the
closed-form optimum and see whether it can be inverted. Write the KL out, pull out `−β`, flip max to min:
`min_π E_{y∼π}[log(π/π_ref) − (1/β)r]`. Now manufacture the policy
`π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)` precisely so that `−(1/β)r = −log(Z(x)·π*/π_ref)`;
substitute, and the bracket becomes `log(π/π_ref) − log(π*/π_ref) − log Z(x) = log(π/π*) − log Z(x)`, so
the whole objective is `min_π E_x[KL(π‖π*) − log Z(x)]`. `Z` does not depend on `π`, and `KL ≥ 0` with
equality at `π = π*`, so the optimum is the exponential tilt
`π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)`. Tilt mass toward high-reward completions; `β` controls how
hard. This is famous — and famously unusable directly, because `Z(x) = Σ_{y'} π_ref(y')exp(r(y')/β)` sums
over all sequences, intractable. That intractable `Z` is the wall the reward-weighted-regression approaches
hit. And read `β` in the tilt for what it is: the temperature of the KL leash. As `β → ∞` the exponent
`r/β → 0`, `exp(·) → 1`, and `π* → π_ref` — infinite regularization pins the policy at the reference. As
`β → 0` the tilt concentrates all mass on the argmax-reward completion — no regularization. So `β = 0.1` is
a fairly *cold* leash: it lets the policy move a real distance toward high-reward completions, which is the
setting I want on a strong SFT start where the reference is good but improvable. That is the same `β` that
appears in the implicit reward `β log(π_θ/π_ref)`, so the coefficient controlling how far the policy may
drift from the reference and the coefficient scaling the reward gap into the sigmoid are *the same number* —
one knob doing both jobs, which is the single-coefficient economy I said made the principled derivation
worth the trouble.

Here is the move, and it is a change of reading direction. The form `π* = π_ref exp(r/β)/Z` I have been
reading left-to-right ("given `r`, here is `π*`"). Read it right-to-left: it is one equation relating `r`,
`π*`, `π_ref`, `Z` — solve it for *`r`*. Take logs:
`r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)`. So *any* reward can be written as `β` times the log-ratio
of its own optimal policy to the reference, plus an `x`-only term. Parameterize the *policy* directly and
read off its implicit reward `r̂(x,y) = β log(π_θ/π_ref) + β log Z(x)`. `Z` is still there — but ask where
`r` actually enters the data. Only through Bradley-Terry, which depends on the *difference* `r(y_w) −
r(y_l)`. The `β log Z(x)` term is a function of `x` only, identical for `y_w` and `y_l` (same prompt), so in
the difference it cancels: `r̂(y_w) − r̂(y_l) = β log(π_θ(y_w)/π_ref(y_w)) + β log Z(x) − β
log(π_θ(y_l)/π_ref(y_l)) − β log Z(x) = β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l))`, and the
`β log Z(x)` has vanished. Substitute into Bradley-Terry:
`p*(y_w ≻ y_l|x) = σ(β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)))`. No `Z`, no sum over
sequences. The thing that made the closed form unusable evaporated the moment I expressed preferences
through it, because preferences only ever see reward *differences* and `Z` was a reward *offset*. So the
implicit reward is just `r̂(x,y) = β log(π_θ(y|x)/π_ref(y|x))` — the policy is secretly a reward model, and
the secret reward is `β` times how much more likely the policy makes a completion relative to the reference.
And I can see the whole ladder folded into this one expression by asking what happens if I throw the
reference away: set `π_ref` uniform (constant in `y`), and `r̂` collapses to `β log π_θ(y) + const`, so the
DPO logit becomes `β·(log π_θ(y_w) − log π_θ(y_l))` — the reference-free Bradley-Terry loss on *summed*
policy log-probs, which is exactly SimPO before the length-normalization and the margin. So SimPO is the
`π_ref`-uniform corner of this same object, and the reason it had no anchor is literally that it set the
reference to a constant. DPO is the version that keeps the reference informative, and that is the entire
difference the {anchor} axis was tracking.

Now flip the pipeline: write the preference NLL with `r` already replaced by the implicit reward, and fit
`π` directly:
`L_DPO = −E_{(x,y_w,y_l)}[ log σ( β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) ) ]`. One stage,
a supervised classification loss on the policy, no reward model, no critic, no sampling — `y_w`, `y_l` come
straight from the fixed dataset. And now check it against the empty cell of my table, because that is the
whole reason I derived it. The reference is in *every* term — `β log(π_θ/π_ref)` is measured against
`π_ref`, so this is the **anchor** ORPO's MATH-500 regression showed was missing, the thing that keeps the
chains I am not actively pushing from eroding. And the gradient is the **self-pacing growth** IPO lacked.
Differentiate: with `s = r̂(y_w) − r̂(y_l)`, the per-example loss `−log σ(s)` has gradient
`∇L_DPO = −β·E[σ(r̂(y_l) − r̂(y_w))·(∇log π_θ(y_w) − ∇log π_θ(y_l))]`. The bracket *raises* the correct
chain's log-probability and *lowers* the wrong one — active growth on `y_w`, the thing IPO's far target
never delivered. And the scalar weight `σ(r̂(y_l) − r̂(y_w))` is large precisely when the implicit reward
orders the pair *wrong* and `→ 0` once the pair is correctly ordered with margin — self-pacing, pouring
gradient into the examples the model gets wrong and stopping on the ones it has right.

Two checks on that gradient before I trust it. First a degenerate limit: at the start of training the
policy *is* the reference, `π_θ = π_ref`, so `r̂(y_w) = r̂(y_l) = 0`, the logit `s = 0`, the loss is
`−log σ(0) = 0.69`, and the weight `σ(0) = 0.5` — a nonzero gradient on an untrained policy, so DPO does
not sit inert at initialization; it begins separating the pair immediately, and it begins from a state where
every implicit reward is exactly zero, which is the cleanest possible starting point. Second, watch the
self-pacing weight `σ(r̂(y_l) − r̂(y_w))` move on concrete numbers. Suppose the model has a pair wrong: the
implicit reward puts the loser above the winner, `r̂(y_l) − r̂(y_w) = +1.0`, weight `σ(1.0) = 0.73` — most
of the available gradient poured into fixing it. Once the model has separated the pair, `r̂(y_l) − r̂(y_w) =
−2.0`, weight `σ(−2.0) = 0.12` — the gradient has dropped six-fold, the pair nearly released. At a
comfortably-solved `−4.0` the weight is `0.018`, essentially off. So the same loss that pushes hard on
wrongly-ordered pairs walks away from ones it has settled, and it does so smoothly with the reward margin —
this is the brake IPO's far target approximated crudely and SimPO's unbounded sigmoid never had.

That weight is the crux that separates DPO from the naive "just raise log p(y_w), lower log p(y_l)"
objective — the unweighted unlikelihood baseline — which has no brake on the likelihood *minimization* of
`y_l` and degenerates, driving the rejected probability to zero and, on near-duplicate pairs, the chosen
with it. The `σ` weight is that brake: it stops pushing the moment the pair is correctly ordered, so on a
pair the model already ranks right the gradient is essentially zero and nothing gets crushed. Anchored to
`π_ref` through the log-ratio and scaled by `β`, "lowering p(y_l)" is measured relative to the reference
rather than in absolute terms, so DPO trims the wrong chain *toward the reference* rather than *toward
zero* — the mechanism that keeps the collapse that ate SimPO at bay. So DPO is exactly the synthesis the
table demanded: IPO's reference anchor plus ORPO's active, self-pacing growth, in a single loss whose
saturating `σ` stops paying out once the implicit-reward margin is met. The only cost is the one ORPO
escaped and IPO paid — the frozen reference, ~3 GB per GPU and four forwards per step. On this ladder that
cost buys precisely the thing ORPO's MATH-500 regression said was worth buying: the anchor.

The substrate wiring is the one place DPO differs from the reference-free rungs, and I should get the
length handling right because it is a real decision, not a default. The `sigmoid` loss is *not* in the
`["ipo","orpo","simpo"]` set, so `concatenated_forward` does **not** length-average — DPO sees the
**summed** sequence log-probs. Is that correct? Trace the derivation: the Bradley-Terry step and the
`β log Z` cancellation are written in `log π(y)`, the log-likelihood of the whole sequence, which for an
autoregressive model *is* the sum of per-token log-probs. The implicit reward `β log(π_θ/π_ref)` is a
statement about the sequence likelihood, not a per-token average, and the `log Z` offset that cancels is a
property of that summed quantity. So summed is not a quirk of the harness; it is what the math wrote. And
`sigmoid` is *not* in the reference-free set in `finetuning_args.py`, so `use_ref_model` is True, the frozen
reference is loaded, and the loss routes through `self.dpo_loss`, which receives all four log-probs. In
TRL's `dpo_loss`, `loss_type="sigmoid"` forms `logits = (chosen_logps − rejected_logps) −
(reference_chosen_logps − reference_rejected_logps)` — that is `(r̂(y_w) − r̂(y_l))/β` — and returns
`−logsigmoid(β·logits)·(1 − label_smoothing) − logsigmoid(−β·logits)·label_smoothing` with
`label_smoothing = 0` here, plus the detached implicit rewards `β·(policy − reference)` for logging. So for
this named baseline my edit is the default dispatch into `dpo_loss` with `loss_type="sigmoid"`; the summed
log-probs and the reference are supplied by the frozen loop. (The full scaffold dispatch is in the answer.)

While I am at the `β`, let me reconcile it with the ladder, because `β = 0.1` here versus SimPO's `β = 2.0`
is not an inconsistency, it is the same calibration principle I used at the floor. `β` multiplies the reward
gap to set the logit scale of the sigmoid, so it must scale inversely with the magnitude of that gap.
SimPO's reward was a *per-token average* log-prob, differences of order 0.1–0.3, so `β = 2.0` lifted them
into the responsive region. DPO's reward is a *summed* sequence log-ratio, which over a completion of
hundreds of tokens can accumulate to differences of order 1–20 nats, so `β = 0.1` brings the logit back
down into the responsive region (`0.1 × [1..20] = 0.1..2`). Same target scale into the sigmoid, `β` ~20×
smaller because the reward is ~20× larger. The two settings agree once I account for summed-versus-average,
which is a small confirmation that I understand what each loss is feeding its sigmoid.

Now the falsifiable expectations against ORPO's 85.37 / 72.4 / 13.33. DPO restores the reference anchor ORPO
dropped while keeping a self-pacing growth term, so my specific prediction is: it *repairs MATH-500* back
toward the ~74 that SimPO and IPO both held — the anchor stops the middle-benchmark erosion that was
ORPO's cost — while *holding AIME* near ORPO's strong 13.33, because the self-pacing growth still cracks the
long correct chains. GSM8K stays saturated ~85–86. If both hold, DPO clears ORPO's 57.03: MATH-500 up ~+1.5
to ~74 with AIME flat at 13.33 nets an average comfortably above ORPO's 57.03, which would be the best on the ladder and would
explain why plain DPO is the strongest baseline despite being the one the others were all reacting to — it
is the only one that ever held both ingredients at once. The risk I will be reading against that
prediction: DPO's summed-log-prob reward is *not* length-normalized, so on math, where correct chains are
long, the un-normalized gradient `∇log π(y_w) − ∇log π(y_l)` lets long responses contribute
disproportionately to a batch — the very length sensitivity SimPO was built to remove, now back because I
chose summed for the derivation's sake. Put a number on the concern: a 300-token correct chain contributes 300 per-token gradient terms to
`∇log π(y_w)` while a 260-token wrong chain contributes 260 to `∇log π(y_l)`, so on a long pair the raw
magnitude of the update is ~10× a 30-token GSM8K pair's — long math pairs simply shout louder in every
batch, and the ones that shout loudest are exactly the AIME/MATH-500 derivations where the shared-token
erosion does the most damage. If that bites, MATH-500 or AIME could come in below my prediction,
and the signature would be a length-correlated error: the loss winning the margin on long pairs by moving a
lot of shared tokens, which is exactly the erosion mechanism one more time, now length-amplified. Note the
DPO logit is also blind in a subtler way — it only sees the winner-minus-loser *gap*, so it cannot tell
whether that gap was won by raising the chosen chain or lowering the rejected one, and greedy correctness
depends on the chosen chain's *absolute* likelihood, not the gap. Make the blindness concrete. Define the
per-response reference-relative log-ratio `ρ(y) = log π_θ(y) − log π_ref(y)`; the DPO logit is
`β·(ρ(y_w) − ρ(y_l))`. Two very different worlds produce the identical logit `β·1.0`: world A with
`ρ(y_w) = +0.5, ρ(y_l) = −0.5` — the correct chain has risen above the reference, healthy — and world B
with `ρ(y_w) = −1.0, ρ(y_l) = −2.0` — the correct chain has *fallen a full nat below* the reference and the
loss is just as content, because the rejected chain fell twice as far. DPO cannot distinguish A from B; the
loss is a function of the difference `ρ(y_w) − ρ(y_l)` alone and never reads `ρ(y_w)` by itself. On
near-duplicate math pairs, where lowering `ρ(y_l)` drags the shared-token `ρ(y_w)` down with it, world B is
not hypothetical — it is the default way the optimizer wins the easy margin, and it is precisely a fallen
`ρ(y_w)` that greedy accuracy punishes. I should be even-handed about the length choice, though, because it is not obviously a defect: on math the
longer chain is usually the *correct* one, so an un-normalized gradient pours *more* growth into the long
correct chains — which is arguably why DPO cracks AIME as well as ORPO did despite lacking the SFT term. So
summed log-probs may be helping AIME even as I worry it hurts the middle; the two effects could partly
cancel, and only the per-benchmark split will tell me which dominates where. That ambiguity is itself
informative: it says the length behavior and the gap-blindness both trace to the *same* summed-log-prob
reward, so whatever comes after DPO cannot simply re-normalize without thinking about the anchor, or
re-anchor without thinking about length — the two are entangled in that one design choice. The `σ` brake
keeps the gap-blindness milder than SimPO's unbounded version, which is why I expect DPO to top the ladder
rather than collapse, but "milder" is not "cured." That residual — the gap-blindness underneath even the strongest baseline — is the gap I will
be reading at the top of the ladder for whatever comes next to fix.
