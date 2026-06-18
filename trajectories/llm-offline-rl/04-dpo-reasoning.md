ORPO's number is the most interesting one yet, because it broke the pattern in a way that tells me what
actually matters. Seed 42: GSM8K 85.37 (flat, saturated as always), MATH-500 72.4, AIME 13.33. The
average jumped to 57.03 — the biggest move so far — but look *where* it came from. AIME leapt to 13.33,
four correct problems where IPO had two and SimPO had one; that is the active SFT growth on the correct
long chains paying off exactly as I bet. But MATH-500 *fell*, 74.4 → 72.4, the first time a benchmark went
backward on the ladder. So ORPO is not a clean win; it is a trade. The reference-free fused objective grows
the correct chains hard enough to crack AIME, but the mildness of the odds-ratio penalty is not free —
without a reference holding the line, pushing the SFT term and the contrast together has cost it accuracy on
the middle benchmark where SimPO and IPO were both at ~74. The AIME gain (+four problems ×3.33) simply
outweighs the MATH-500 loss (−2.0) in the average, so 57.03 > 55.66. But the regression is a real signal:
reference-free active growth is powerful on the hardest chains and slightly corrosive on the broad middle,
because nothing anchors the chains it is *not* actively growing.

So the diagnosis sharpens into a clean synthesis problem. SimPO: reference-free, no anchor, unbounded —
collapsed AIME. IPO: reference anchor, finite target — defended everything but grew nothing, tiny gains.
ORPO: reference-free active growth — cracked AIME but regressed MATH-500. The pattern across all three says
the two ingredients I have been toggling are *both* necessary and I have never had them together: an
**anchor against a reference** (so the chains I am not actively pushing do not erode — the thing ORPO's
MATH-500 regression shows is missing) *and* a **mechanism that keeps growing the correct chain** (the thing
IPO lacked). ORPO got growth by dropping the reference; IPO got the anchor by dropping the growth. What I
want is the objective that has the reference anchor *and* a self-pacing growth term — and the cleanest such
object is the one I have been circling the whole ladder without running: DPO itself, the reference-based
Bradley-Terry MLE that every other rung was a reaction to. Let me derive it properly, because if it is the
strongest baseline I should understand exactly why, and it is the natural top of this ladder.

The objective everyone writes is `max_π E_{y∼π}[r(x,y)] − β·KL(π‖π_ref)` — chase reward, stay near the
reference. The reward comes from preferences through Bradley-Terry, `p*(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`,
fit by MLE. The painful part is the next stage: maximizing `E_{y∼π}[r]` over a discrete autoregressive LM
needs RL (PPO) — a separate reward model, a critic, on-policy sampling in the loop. I want to collapse the
two stages into one supervised loss, the way the whole offline ladder does. Look at the closed-form optimum.
Write the KL out, pull out `−β`, flip max to min: `min_π E_{y∼π}[log(π/π_ref) − (1/β)r]`. Manufacture
`π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)` so that `−(1/β)r = −log(Z(x)·π*/π_ref)`; substitute and the
bracket becomes `log(π/π*) − log Z(x)`, so the objective is `min_π E_x[KL(π‖π*) − log Z(x)]`. `Z` does not
depend on π, and KL ≥ 0 with equality at π = π*, so the optimum is the exponential tilt
`π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)`. Tilt mass toward high-reward completions; β controls how
hard. This is famous — and famously unusable directly, because `Z(x) = Σ_{y'} π_ref(y')exp(r(y')/β)` sums
over all sequences, intractable. That intractable `Z` is the wall the reward-weighted-regression rung hit.

Here is the move. The form `π* = π_ref exp(r/β)/Z` I have been reading left-to-right ("given r, here is
π*"). Read it right-to-left: it is one equation relating r, π*, π_ref, Z — solve it for *r*. Take logs:
`r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)`. So *any* reward can be written as β times the log-ratio
of its own optimal policy to the reference, plus an x-only term. Parameterize the *policy* directly and read
off its implicit reward `r̂(x,y) = β log(π_θ/π_ref) + β log Z(x)`. `Z` is still there — but where does r
enter the data? Through Bradley-Terry, which depends only on the *difference* `r(y_w) − r(y_l)`. The
`β log Z(x)` term is a function of x only, the same for `y_w` and `y_l`, so in the difference it cancels.
Substitute and watch it die: the partition function factors out of numerator and both denominator terms of
the Bradley-Terry expression and cancels, leaving
`p*(y_w ≻ y_l|x) = σ(β log(π*(y_w)/π_ref(y_w)) − β log(π*(y_l)/π_ref(y_l)))`. No Z, no sum over sequences.
The thing that made the closed form unusable evaporated the moment I expressed preferences through it,
because preferences only ever see reward *differences* and Z was a reward *offset*. So the implicit reward
is just `r̂(x,y) = β log(π_θ(y|x)/π_ref(y|x))` — the policy is secretly a reward model, and the secret
reward is β times how much more likely the policy makes a completion relative to the reference.

Now flip the pipeline: write the preference NLL with r already replaced by the implicit reward, and fit π
directly:
`L_DPO = −E_{(x,y_w,y_l)}[ log σ( β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) ) ]`. One stage,
a supervised classification loss on the policy, no reward model, no critic, no sampling — `y_w`, `y_l` come
straight from the fixed dataset. And it has *both* ingredients the ladder told me I needed. The reference
is in every term — `β log(π_θ/π_ref)` is measured against `π_ref`, so this is the **anchor** ORPO's
MATH-500 regression showed was missing, the thing that keeps the chains I am not actively pushing from
eroding. And the gradient is the **self-pacing growth** IPO lacked. Differentiate: with
`s = r̂(y_w) − r̂(y_l)`, the per-example loss `−log σ(s)` has gradient
`∇L_DPO = −β·E[σ(r̂(y_l) − r̂(y_w))·(∇log π_θ(y_w) − ∇log π_θ(y_l))]`. The bracket *raises* the correct
chain's log-probability and *lowers* the wrong one — active growth on `y_w`. The scalar weight
`σ(r̂(y_l) − r̂(y_w))` is large precisely when the implicit reward orders the pair *wrong* and `→ 0` once
the pair is correctly ordered with margin — self-pacing, pouring gradient into the examples the model gets
wrong and stopping on the ones it has right. That weight is the crux that separates DPO from the naive
"just raise log p(y_w), lower log p(y_l)" objective (the unweighted unlikelihood baseline), which has no
brake on the likelihood *minimization* of `y_l` and degenerates. The σ weight, scaled by β and anchored to
`π_ref` through the log-ratio, is what prevents the collapse that ate SimPO — it pushes only until the pair
is ordered, and "lowering p(y_l)" is measured relative to the reference rather than in absolute terms.

So DPO is exactly the synthesis: IPO's reference anchor + ORPO's active, self-pacing growth, in a single
loss with a finite implicit-reward margin that the saturating σ stops paying out on once met. The only
cost is the one ORPO escaped and IPO paid — the frozen reference, four forwards per step. On this ladder
that cost buys precisely the thing ORPO's MATH-500 regression said was worth buying.

The substrate wiring is the one place DPO differs from the reference-free rungs. The `sigmoid` loss is
*not* in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` does **not** length-average — DPO sees
the **summed** sequence log-probs, which is correct: the Bradley-Terry derivation is written in
log-likelihoods, and the `β log Z` cancellation is a property of the summed log-prob, not a per-token
average. And `sigmoid` is *not* in the reference-free set in `finetuning_args.py`, so `use_ref_model` is
True, the frozen reference is loaded, and the loss routes through `self.dpo_loss`, which receives all four
log-probs. In TRL's `dpo_loss`, `loss_type="sigmoid"` forms `logits = (chosen_logps − rejected_logps) −
(reference_chosen_logps − reference_rejected_logps)` — that is `(r̂(y_w) − r̂(y_l))/β` — and returns
`−logsigmoid(β·logits)·(1 − label_smoothing) − logsigmoid(−β·logits)·label_smoothing` with
`label_smoothing = 0` here, plus the detached implicit rewards `β·(policy − reference)` for logging. So for
this named baseline my edit is the default dispatch into `dpo_loss` with `loss_type="sigmoid"`; the summed
log-probs and the reference are supplied by the frozen loop. (The full scaffold dispatch is in the answer.)

The falsifiable expectations against ORPO's 85.37 / 72.4 / 13.33. DPO restores the reference anchor ORPO
dropped while keeping a self-pacing growth term, so my specific prediction is: it *repairs MATH-500* back
toward the ~74 that SimPO and IPO held (the anchor stops the middle-benchmark erosion), while *holding AIME*
near ORPO's strong 13.33 (the self-pacing growth still cracks the long correct chains). GSM8K stays
saturated ~85–86. If both hold, DPO clears ORPO's 57.03 — MATH-500 up ~+1.5 to ~74 with AIME flat at 13.33
nets an average around 57.8, the best on the ladder, which would explain why plain DPO is the strongest
baseline despite being the one the others were all reacting to. The risk: DPO's summed-log-prob reward is
*not* length-normalized, so on math, where correct chains are long, the un-normalized gradient lets long
responses dominate a batch — the very length sensitivity SimPO was built to remove. If that bites, MATH-500
or AIME could come in below my prediction, and the signature would be a length-correlated error that the
finale must fix by re-normalizing or by directly protecting the long correct chain's likelihood. That is the
gap I will be reading at the top of the ladder.
