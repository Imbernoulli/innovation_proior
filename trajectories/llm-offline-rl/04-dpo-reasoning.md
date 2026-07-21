ORPO's number is the most interesting yet, because it broke the pattern in a way that tells me what
matters. Seed 42: GSM8K 85.37, MATH-500 72.4, AIME 13.33, average 57.03. Deltas from IPO's
85.9 / 74.4 / 6.67: GSM8K −0.53, MATH-500 −2.0, AIME +6.66, average
`(−0.53 − 2.0 + 6.66)/3 = 1.38` — the biggest single jump on the ladder. But look where it came from. AIME
leapt to 13.33, two more correct problems (four of thirty, where IPO had two and SimPO one) — the active
SFT growth on the long correct chains paying off as I bet. And MATH-500 *fell*, 74.4 → 72.4, the first time
a benchmark went backward. So ORPO is not a clean win; it is a trade. The reference-free fused objective
grows the correct chains hard enough to crack AIME, but the mildness of the odds-ratio penalty is not free:
without a reference holding the line, pushing SFT and contrast together costs accuracy on the broad middle,
the exact risk I flagged — nothing anchors the chains it is *not* actively growing.

Now the diagnosis sharpens into a synthesis problem, and the empty cell of a table is the answer. I have
been toggling two ingredients: *anchor* — a reference the correct chain's likelihood is measured against,
so chains I do not push cannot silently erode — and *growth* — a term that actively raises the correct
chain's likelihood rather than merely defending it. SimPO had neither and collapsed AIME. IPO had the
anchor but no growth (a conservative far target) and barely moved, +1.20, almost all one AIME problem. ORPO
had the growth but no anchor and cracked AIME but regressed MATH-500. Three of the four {anchor, growth}
cells are filled, each with a characteristic failure, and the pattern says the same thing three times: both
ingredients are necessary and I have never had them together. Every rung gave something back because none
held both. The empty cell is (anchor yes, growth yes), and the cleanest object that fills it is the one I
have been circling the whole ladder without running: DPO itself, the reference-based Bradley-Terry MLE
every other rung was a reaction to.

Before deriving it, rule out the cheaper way to fill the cell: I have IPO (anchor, no growth) and ORPO
(growth, no anchor), so why not literally add them — append ORPO's SFT term to IPO's regression? It would
nominally put both ingredients in one loss, but it is a two-term Frankenstein with two coefficients pulling
on different scales (a squared per-token residual against a summed NLL), so tuning them to cooperate is its
own project, and I would still pay for IPO's reference. The appeal of a *principled* single loss is that
both ingredients come from one term with one coefficient, self-consistently. That is what makes the DPO
derivation worth doing: if the anchor and the growth both fall out of a single Bradley-Terry loss, I get the
fourth cell with `β` as the only knob.

The objective everyone writes is `max_π E_{y∼π}[r(x,y)] − β·KL(π‖π_ref)`, with the reward from preferences
through Bradley-Terry. The painful part is maximizing `E_π[r]` over a discrete autoregressive LM — it needs
RL, a reward model, a critic, on-policy sampling. I want to collapse the two stages into one supervised
loss, so look at the closed-form optimum. Write the KL out, flip max to min:
`min_π E_{y∼π}[log(π/π_ref) − (1/β)r]`. Manufacture `π*(y|x) = (1/Z(x))·π_ref·exp(r/β)` so that
`−(1/β)r = −log(Z·π*/π_ref)`; substitute and the bracket becomes `log(π/π*) − log Z`, so the objective is
`min_π E_x[KL(π‖π*) − log Z]`, whose optimum is the exponential tilt `π* ∝ π_ref exp(r/β)`. Famous — and
famously unusable directly, because `Z(x) = Σ_{y'} π_ref(y')exp(r(y')/β)` sums over all sequences,
intractable. And read `β` in the tilt: as `β → ∞`, `π* → π_ref` (infinite regularization); as `β → 0`, all
mass on argmax-reward. So `β = 0.1` is a fairly cold leash, letting the policy move a real distance toward
high-reward completions — the setting I want on a strong SFT start.

Here is the move, a change of reading direction. Read `π* = π_ref exp(r/β)/Z` right-to-left: it is one
equation relating `r, π*, π_ref, Z` — solve it for *r*. Take logs:
`r(x,y) = β log(π*/π_ref) + β log Z(x)`. So any reward is `β` times the log-ratio of its own optimal policy
to the reference, plus an `x`-only term. Parameterize the *policy* directly and read off its implicit
reward `r̂ = β log(π_θ/π_ref) + β log Z(x)`. `Z` is still there — but `r` enters the data only through
Bradley-Terry, which depends on the *difference* `r(y_w) − r(y_l)`, and `β log Z(x)` is a function of `x`
only, identical for `y_w` and `y_l`, so it cancels:
`r̂(y_w) − r̂(y_l) = β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l))`. The thing that made the closed
form unusable evaporated the moment I expressed preferences through it, because preferences see only reward
*differences* and `Z` was a reward *offset*. So the implicit reward is `r̂(x,y) = β log(π_θ/π_ref)` — the
policy is secretly a reward model. And I can see the whole ladder folded into this: set `π_ref` uniform and
`r̂` collapses to `β log π_θ(y) + const`, so the DPO logit becomes `β·(log π_θ(y_w) − log π_θ(y_l))` —
reference-free Bradley-Terry on *summed* policy log-probs, which is exactly SimPO before its length-
normalization and margin. SimPO is the `π_ref`-uniform corner of this same object, and the reason it had no
anchor is literally that it set the reference to a constant.

Now flip the pipeline: write the preference NLL with `r` replaced by the implicit reward and fit `π`
directly,
`L_DPO = −E[ log σ( β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) ) ]`. One supervised stage, no
reward model, no critic, no sampling. Check it against the empty cell. The reference is in *every* term, so
this is the **anchor** ORPO's regression showed missing. And the gradient is the **self-pacing growth** IPO
lacked: with `s = r̂(y_w) − r̂(y_l)`, `∇L_DPO = −β·E[σ(r̂_l − r̂_w)·(∇log π_θ(y_w) − ∇log π_θ(y_l))]` — the
bracket raises the correct chain and lowers the wrong one (active growth on `y_w`, which IPO's far target
never delivered), and the scalar weight `σ(r̂_l − r̂_w)` is large precisely when the pair is ordered *wrong*
and `→ 0` once correctly ordered with margin. At initialization `π_θ = π_ref`, so `r̂_w = r̂_l = 0`, loss
`−log σ(0) = 0.69`, weight `σ(0) = 0.5` — nonzero, so DPO begins separating immediately from the cleanest
possible starting point. As the model separates a pair, the weight falls smoothly with the reward margin
(`σ(−2) = 0.12`, `σ(−4) = 0.018`) — the brake IPO's far target approximated crudely and SimPO's unbounded
sigmoid never had.

That weight is the crux separating DPO from the naive unlikelihood objective — "just raise log p(y_w),
lower log p(y_l)" — which has no brake on the minimization of `y_l` and degenerates, driving the rejected
probability to zero and, on near-duplicate pairs, the chosen with it. The `σ` weight stops pushing the
moment the pair is correctly ordered. And anchored to `π_ref` through the log-ratio, "lowering p(y_l)" is
measured relative to the reference, so DPO trims the wrong chain *toward the reference* rather than *toward
zero* — the mechanism that keeps the SimPO collapse at bay. So DPO is exactly the synthesis the table
demanded: IPO's anchor plus ORPO's self-pacing growth, in a single loss whose saturating `σ` stops paying
out once the margin is met. The only cost is the one ORPO escaped and IPO paid — the frozen reference,
~3 GB per GPU and four forwards per step — which on this ladder buys precisely the anchor ORPO's MATH-500
regression said was worth buying.

The substrate differs from the reference-free rungs in the length handling, and it is a real decision. The
`sigmoid` loss is *not* in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` does **not** length-
average — DPO sees the **summed** sequence log-probs. Is that right? The Bradley-Terry step and the
`β log Z` cancellation are written in `log π(y)`, the log-likelihood of the whole sequence, which for an
autoregressive model *is* the sum of per-token log-probs; the `log Z` offset that cancels is a property of
that summed quantity. So summed is what the math wrote, not a harness quirk. And `sigmoid` is not reference-
free, so `use_ref_model` is True, the reference is loaded, and the loss routes through `self.dpo_loss` with
all four log-probs; its `sigmoid` branch forms `logits = (chosen − rejected) − (ref_chosen − ref_rejected)`
— that is `(r̂_w − r̂_l)/β` — and returns `−logsigmoid(β·logits)` (`dpo_label_smoothing = 0`). So my edit is
the default dispatch. (The full dispatch is in the answer.) The `β = 0.1` here versus SimPO's 2.0 is the
same calibration, not an inconsistency: DPO's reward is a summed log-ratio, differences of order 1–20 nats
over hundreds of tokens, so `β = 0.1` brings `0.1 × [1..20]` back into the sigmoid's responsive region —
`β` ~20× smaller because the reward is ~20× larger than SimPO's per-token average.

Against ORPO's 85.37 / 72.4 / 13.33, DPO restores the anchor ORPO dropped while keeping a self-pacing
growth term, so I predict it *repairs MATH-500* back toward the ~74 that SimPO and IPO both held, while
*holding AIME* near ORPO's 13.33 (the self-pacing growth still cracking the long chains). GSM8K stays
saturated. If both hold, DPO clears ORPO's 57.03 and would be the best on the ladder — which would explain
why plain DPO is the strongest baseline despite being the one the others reacted to: it is the only one that
ever held both ingredients at once.

The risk to read against that: DPO's summed reward is *not* length-normalized, so the un-normalized gradient
`∇log π(y_w) − ∇log π(y_l)` lets long responses dominate a batch — a 300-token correct chain contributes 300
per-token terms while a 260-token wrong chain contributes 260, so long math pairs shout ~10× louder than a
30-token GSM8K pair, and the loudest are exactly the AIME/MATH-500 derivations where shared-token erosion
does the most damage. The DPO logit is also blind in a subtler way: it sees only the winner-minus-loser
*gap*, `β·(ρ(y_w) − ρ(y_l))` with `ρ(y) = log π_θ(y) − log π_ref(y)`, and cannot tell whether that gap was
won by raising the chosen chain or lowering the rejected. Two worlds give the identical logit `β·1.0`:
`ρ(y_w) = +0.5, ρ(y_l) = −0.5` (correct chain risen above reference, healthy) and
`ρ(y_w) = −1.0, ρ(y_l) = −2.0` (correct chain a full nat *below* reference, the loss just as content). On
near-duplicate math pairs, where lowering `ρ(y_l)` drags the shared-token `ρ(y_w)` down with it, world B is
the default way the optimizer wins the easy margin — and it is precisely a fallen `ρ(y_w)` that greedy
accuracy punishes. To be even-handed, the length choice is not purely a defect: on math the longer chain is
usually the *correct* one, so an un-normalized gradient pours more growth into the long correct chains,
which is arguably why DPO should crack AIME as well as ORPO did despite lacking the SFT term. The two
effects may partly cancel, and only the per-benchmark split will tell which dominates where. The `σ` brake
keeps the gap-blindness milder than SimPO's unbounded version — why I expect DPO to top the ladder rather
than collapse — but milder is not cured. That residual, gap-blindness underneath even the strongest
baseline, is the hole whatever comes after DPO would have to close.
