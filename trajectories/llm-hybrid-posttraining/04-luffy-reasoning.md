Pure GRPO settled the dissociation, more sharply than I expected. Seed 42: AIME24 0.087, AMC23 0.283,
MATH-500 0.419. MATH-500 is the breakthrough — up from the switch's 0.250 and the floor's 0.244, `+0.17` on
the broad split — confirming the diagnosis: the imitation tax *was* what capped MATH-500, and paying it zero
times recovered the mass. AIME24 nudged up 0.062 → 0.087, a little more than predicted; the model could
partly solve a few more AIME prompts than I thought. And the predicted cost landed: AMC23 fell 0.325 → 0.283,
pure GRPO giving back the small-split bootstrap. In problem-counts: MATH-500 `125 → 209.5`, `+84` out of 500
(I said the tax story needed tens of problems, not `+3`); AIME24 `+0.7` of a problem on 30, real but tiny;
AMC23 `−1.7` problems. The MATH-500 gain dwarfs both, so on this backbone killing the imitation tax dominates
the final score. Pure GRPO wins two of three splits including the largest — the strongest result so far.

But the *shape* of the win names the next move. GRPO is strong where the model has reward contrast and
structurally blind where it does not: on an all-wrong prompt the group advantage `(R − mean)/std` is
identically zero, so no gradient. AIME24's 0.087 is the symptom — at that rate the average AIME prompt fails
all eight rollouts with probability `0.913^8 ≈ 0.48`, so roughly half of AIME24 hands GRPO an all-equal group
and zero gradient every step. Pure RL sharpened the sampleable half from 0.062 to 0.087; the other half never
moved because there is no gradient to move it. The switch tried to fix this with SFT on stuck prompts and lost
MATH-500 to the tax; GRPO avoids the tax but abandons the stuck prompts. Neither approach gets *both* — that
is the gap.

So the question is precise: can I bootstrap the dead-gradient prompts the way the switch did, but through a
*reward* channel rather than imitation, so I do not re-import the MATH-500 tax? The switch's mistake was
routing the teacher trace to SFT — plain copy-the-teacher, which narrows and overfits. What if, on exactly
those stuck prompts, I inject the teacher trace as an *off-policy RL* sample and let the RL advantage
machinery handle it? The scaffold has that arm: the controller's `off_add_num < 0` branch adds an off-policy
sample with `whether_off=True`, consumed as off-policy RL.

Why injecting the teacher as an RL group member is right, and different from both SFT and pure GRPO: keep it
as one more member of the prompt's group and standardize the advantage over the *union*. On a stuck prompt,
seven on-policy rollouts wrong (reward 0) plus one correct teacher (reward 1), union `{0,0,0,0,0,0,0,1}`:
mean 0.125, std `√(0.875/7) = 0.354`, so the teacher gets `Â = (1 − 0.125)/0.354 = +2.47` and each failing
rollout `−0.354`. Where pure on-policy gave exactly zero for all eight, the union gives the teacher a large
positive push through the *advantage*, not through an imitation loss. Now the other end, which is what makes
it self-adjust: a prompt the model mostly solves, six of seven rollouts correct plus the teacher, union
`{1,1,1,1,1,1,0,1}`, mean 0.875, std 0.354 by symmetry — the teacher's advantage is now
`(1 − 0.875)/0.354 = +0.354`, an order of magnitude below the `+2.47` it got on the stuck prompt and no larger
than the model's own correct rollouts. So the teacher stops being special exactly when the model can carry the
prompt itself. The imitation-versus-exploration balance is not a gate threshold I tune — it falls out of the
group statistics, the competence signal the switch reached for with `on_solve_num`, but consumed as reward
instead of copying. This is the move that gets both: bootstrap the stuck prompts (fixing GRPO's AIME
blindness) without imitation (avoiding the switch's MATH-500 tax), because that `7×` swing in the teacher's
weight is a competence-dependent coefficient the SFT channel cannot express — SFT pulls up every teacher token
at coefficient `1` regardless, which is the tax I measured.

Two refinements the full off-policy method makes, and I have to be honest about which the harness lets me
realize. First, the off-policy member needs its own importance ratio with the teacher policy `π_φ`; to avoid
the teacher's per-token densities and tokenizer mismatch, set `π_φ = 1`, so the ratio collapses to the model's
own probability of the teacher token, and the clip is dropped on that branch (a `[1−ε,1+ε]` clip is ill-posed
for a raw probability). Second, the load-bearing one: that bare off-policy term *hacks* the objective. Its
per-logit gradient on the sampled teacher token scales as `π_θ(1 − π_θ)`, near-zero at both extremes — a
`π = 0.01` token gets roughly `25×` less gradient than a `π = 0.5` one. But the low-probability surprising
tokens are exactly the ones carrying the new reasoning the model does not yet have, so the cheapest way to
raise the objective is to push up the teacher tokens the model already agrees with, entropy collapses, and I
am back to a fast-converging imitator. The fix is policy shaping: replace the off-policy ratio `r̂` with
`f(r̂) = r̂/(r̂+γ)`, whose derivative `γ/(r̂+γ)²` is largest at small `r̂`. At `γ = 0.1` the amplification is
`≈ 8` near `π → 0`, crosses below `1` at `π ≈ 0.216`, and damps to `0.10×` at `π = 0.9` — surprising teacher
tokens get up to `≈ 1/γ = 10×` more gradient while already-likely ones are suppressed, pulling the optimizer
toward the deviating reasoning instead of easy agreement.

Now the harness boundary, which determines the actual edit. The editable surface is the trainer-side router;
the actor is fixed at `offline_loss_type=sft`, the advantage estimator at `grpo`, the rollout modules
read-only. So I control which prompts get an off-policy RL sample and the removal of the on-policy group, but
not the actor's off-policy loss form. When the controller returns `off_add_num = −1` on a stuck prompt, the
harness removes the eight dead on-policy rollouts and then adds the teacher tagged `whether_off=True`, and the
fixed `grpo` code standardizes over whatever group it is finally handed — so the exact `+2.47`/`−0.354` spread
I computed for the natural 7-on-plus-1-off group is *not* literally what the code sees once the on-policy
members are already gone; what it sees is the teacher entering the reward channel with a positive advantage, a
bootstrap on a correct trace where pure GRPO had exactly zero. That is the directional content of the union
idea, realized by the router. The piece I cannot reach is the loss form: the `π_φ = 1` unclipped, `f`-shaped
surrogate lives in the fixed actor. So the teacher enters through the reward channel, but the
entropy-preserving shaping that would emphasize its surprising tokens is withheld — I land the trainer-side
routing, not the full shaped objective.

The controller: the switch logic already has three routes — SFT (`+1`), off-policy RL (`−1`), keep-GRPO
(`0,0,0`). Keeping the on-policy group won where there is contrast, and the SFT route lost MATH-500, so make
the *off-policy-RL arm the default for stuck prompts* instead of SFT: when `on_solve_num ≤ switch_gate`, drop
the on-policy group and add the teacher as off-policy RL, `(rollout.n_verify, 0, −1)`; when `on_solve_num >
switch_gate`, keep the on-policy GRPO group untouched, `(0, 0, 0)`, preserving what won MATH-500. Why gate at
stuck-only rather than injecting the teacher everywhere? On a prompt the model handles, the teacher's union
advantage is only `+0.354`, so injecting it buys almost nothing and does cost, disturbing the clean on-policy
statistics that produced 0.419. The gate spends the off-policy machinery exactly where GRPO scored zero and
nowhere else.

The bar, against pure GRPO's real numbers — this is the endpoint, no feedback to lean on. The prediction on
AIME24, where GRPO is blind: routing the all-wrong prompts to off-policy RL gives them a positive gradient
through the union advantage, so the test is whether AIME24 lifts above 0.087. MATH-500 must hold near 0.419
and not regress toward 0.250 — the keep-GRPO route is unchanged on contrast prompts, and the stuck prompts now
go to RL rather than SFT, so the tax should not return; if MATH-500 falls back, the off-policy arm is behaving
like imitation in this fixed-actor harness and the shaping I cannot add was load-bearing. AMC23 should recover
toward the switch's 0.325, its stuck prompts getting a bootstrap again without the narrowing. The honest
caveat: without the actor-side shaping the off-policy term may still over-weight the teacher tokens the model
already agrees with, so the AIME24 lift may be smaller than the full method's — the claim is the direction,
not a number.
