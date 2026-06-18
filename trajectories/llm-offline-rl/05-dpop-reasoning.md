DPO topped the ladder, and the way it topped it is precisely the opening for the next move. Seed 42:
GSM8K 85.9, MATH-500 74.2, AIME 13.33, average 57.81 — the best on the ladder, and exactly the synthesis I
predicted: it repaired MATH-500 back to ~74 where ORPO had regressed to 72.4 (the reference anchor doing
its job), while holding AIME at ORPO's strong 13.33 (the self-pacing growth still cracking the long correct
chains). So DPO confirms both ingredients matter and that having them together beats every rung that had
only one. But re-read the one risk I flagged at the close of the DPO derivation, because the number is
consistent with it: DPO's implicit reward `β log(π_θ/π_ref)` is a function of *summed* log-probs, and the
DPO loss is satisfied by making the *difference* `log(π_θ(y_w)/π_θ(y_l))` exceed the reference's difference.
On near-identical math pairs that condition can be met two ways — by raising the correct chain's likelihood,
or by *lowering the rejected chain's*. And lowering the rejected chain, which shares almost every token with
the correct one, drags the correct chain down too. DPO's σ self-pacing brake stops the push *once the pair
is ordered*, which is milder than SimPO's unbounded version — that is why DPO does not collapse. But "stop
once ordered" is not the same as "never let the correct chain's absolute probability fall." The DPO logit
only cares that the winner-minus-loser *gap* clears the reference's; it is blind to whether that gap was won
by pushing the chosen up or the rejected down. On a math benchmark scored by greedy correctness — which
depends on the *absolute* likelihood of the correct chain, not the gap — that blindness is the residual
failure mode sitting underneath DPO's 57.81.

This is a known, diagnosable pathology of the DPO objective: across the early-training updates, the
log-probability of the *chosen* response can *decrease* even as the chosen-minus-rejected margin grows,
because the optimizer is allowed to win the margin by collapsing the rejected response, and on
near-duplicate pairs that collapse pulls the chosen down with it. It is most acute exactly in the setting
this task lives in — math preference pairs from a step-level dataset, where chosen and rejected solutions are
edit-distance-tiny — and it is the same disease, viewed through DPO's reference-anchored lens, that ate
SimPO's AIME and nicked ORPO's MATH-500. So the natural move past the strongest baseline is not a new family;
it is to take DPO — which already has the right anchor and the right self-pacing — and *add a term that
forbids the correct chain's likelihood from falling below the reference's*. Keep everything that made DPO the
top of the ladder; close the one hole the leaderboard's shape implies is still open.

Let me derive the term from the failure, not bolt it on. The DPO logit is
`β·(log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)))`. Define the per-response log-ratio
`ρ(y) = log π_θ(y|x) − log π_ref(y|x)` — how much more (or less) likely the policy makes a response than the
reference does. The DPO logit is `β·(ρ(y_w) − ρ(y_l))`, and the loss is happy whenever this is large and
positive. The pathology is `ρ(y_w) < 0` — the policy has made the *correct* chain *less* likely than the
reference did — while `ρ(y_w) − ρ(y_l)` is still positive because `ρ(y_l)` fell even further. The fix has to
make the loss *unhappy* when `ρ(y_w) < 0`, regardless of the gap. The cleanest such term is a one-sided
penalty on the correct chain's reference-relative log-ratio: penalize `max(0, −ρ(y_w)) = max(0, log π_ref(y_w)
− log π_θ(y_w))`. It is exactly zero when the policy keeps the correct chain at least as likely as the
reference (`ρ(y_w) ≥ 0`, nothing to fix), and grows linearly as the correct chain falls below the reference
(`ρ(y_w) < 0`, the pathology). One-sided is the whole point: I do not want to *force* the correct chain
above the reference everywhere (that would just be SFT and would fight the contrast), I only want to *forbid*
it from dropping below — a floor, not a target.

Where does the penalty go? Inside the DPO logit, subtracted, scaled by the same β, so it lives in the same
units as the reward gap and the σ self-pacing applies to it too. The new logit is
`β·( ρ(y_w) − ρ(y_l) − λ·max(0, −ρ(y_w)) )` and the loss is the same Bradley-Terry NLL,
`L_DPOP = −E[ log σ( β·( ρ(y_w) − ρ(y_l) − λ·max(0, log(π_ref(y_w)/π_θ(y_w))) ) ) ]`. Read what this does to
the gradient. When `ρ(y_w) ≥ 0` the `max(0,·)` is zero and its gradient is zero — DPOP is *exactly DPO*, so
on the pairs where the correct chain is already at least reference-likely I keep DPO's behavior untouched,
losing nothing that made it the top baseline. When `ρ(y_w) < 0`, the penalty term `−λ·(−ρ(y_w)) = λ·ρ(y_w)`
is active and *negative*, which shrinks the logit, which raises the loss, which (through the same σ weight)
pours gradient into `+∇log π_θ(y_w)` — it actively pushes the correct chain back up above the reference. So
the penalty is a barrier: invisible while the correct chain stays anchored, sharply restoring once it slips.
This is precisely the "never let the correct chain's absolute probability fall" guarantee the DPO logit was
blind to, expressed in the reference-relative coordinate DPO already uses, so it composes with the anchor
and the self-pacing rather than fighting them.

Two design checks before I trust it. First, does it reduce to DPO cleanly? Set λ = 0 and the penalty
vanishes identically — DPOP is a strict generalization of DPO with one extra non-negative knob, so it can
never do worse than DPO on pairs that do not trigger the barrier, and the entire ladder's strongest result is
recoverable by construction. Second, why one-sided rather than a symmetric pull toward `ρ(y_w) = 0`? Because a
symmetric term would penalize `ρ(y_w) > 0` too — it would punish the model for making the correct chain
*more* likely than the reference, which is exactly the active growth that cracked AIME. The `max(0,·)` keeps
all of that upside and clips only the downside. That asymmetry is the difference between "anchor the correct
chain to a floor" and "regress it to a fixed value"; I want the floor.

Now the substrate, line by line, because the finale's edit is a real fill of `compute_preference_loss` —
this is the `custom` slot, the one the task reserves. `custom` is *not* in the `["ipo","orpo","simpo"]` set
in `concatenated_forward`, so the loss receives **summed** sequence log-probs — correct, because DPOP is
DPO's objective and `ρ(y) = log π_θ(y) − log π_ref(y)` is written in log-likelihoods, the same coordinate in
which the `β log Z` cancellation holds. And `custom` is *not* in the reference-free set in
`finetuning_args.py` (`use_ref_model = stage=="dpo" and pref_loss not in ["orpo","simpo"]`), so for
`pref_loss=custom`, `use_ref_model` is True, the frozen reference is loaded, and all four log-probs arrive at
`compute_preference_loss` — exactly what the penalty needs (`reference_chosen_logps` for the floor). So I
leave the `finetuning_args.py` line untouched and add an `elif self.loss_type == "custom"` branch in the
reference-based side. The branch computes the standard DPO log-ratios `pi_logratios − ref_logratios`, then the
chosen-side reference-relative log-ratio `ρ(y_w) = policy_chosen_logps − reference_chosen_logps`, forms the
one-sided penalty `relu(−ρ(y_w)) = relu(reference_chosen_logps − policy_chosen_logps)`, subtracts
`λ·penalty` inside the logits, and returns `−logsigmoid(β·logits)` with the same detached implicit rewards
DPO logs. λ is a new scalar hyperparameter; the method's own experiments used λ around 50 with a larger β on
big models, but those are 34B/72B numbers — for this β = 0.1, 1.5B math setting I will start λ small (on the
order of a few, e.g. λ = 5) so the barrier corrects the slip without overwhelming the contrast, and treat it
as the one knob to tune. (The full scaffold fill is in the answer.)

What must this clear, and what would I validate? The bar is DPO's 85.9 / 74.2 / 13.33, average 57.81. My
falsifiable claim is that DPOP holds DPO's gains everywhere the barrier is inactive (so GSM8K stays ~85–86
and the result can never regress below DPO by construction at λ = 0) and *adds* on the pairs where DPO was
silently letting the correct chain slip — the near-identical math pairs, which are densest on MATH-500 and
AIME. So I would expect MATH-500 to edge above 74.2 and AIME to hold or improve on 13.33, lifting the average
past 57.81. The cleanest validation is the diagnostic the penalty was built from: log the mean of `ρ(y_w)`
over training. Under DPO it should trend negative early (the chosen log-ratio decreasing — the pathology);
under DPOP that mean should be pinned at or above zero, and the benchmark gain should track exactly the
fraction of pairs where DPO's `ρ(y_w)` would have gone negative. If `ρ(y_w)` never goes negative under DPO on
this data, the barrier is inactive and DPOP collapses to DPO with no gain — that is the falsifier, and the
log of `ρ(y_w)` tells me immediately which world I am in. The risk in the other direction is λ too large: the
barrier then dominates and the loss becomes SFT-on-chosen with a vestigial contrast, which would push GSM8K
and MATH-500 around unpredictably and blunt the AIME growth — so λ is genuinely the knob to sweep, smallest
value that pins `ρ(y_w) ≥ 0` being the target.
