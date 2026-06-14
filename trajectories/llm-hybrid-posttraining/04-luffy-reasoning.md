Pure GRPO's numbers settled the dissociation I set up, and they settled it more sharply than I expected.
Seed 42: AIME24 0.087, AMC23 0.283, MATH-500 0.419. MATH-500 is the breakthrough — 0.419, up from the
switch's 0.250 and the floor's 0.244, a +0.17 jump on the broad in-distribution split. That confirms the
diagnosis exactly: the imitation tax *was* what capped MATH-500, and paying it zero times — letting every
contrast prompt learn from its own graded rollouts instead of copying the teacher — recovered the entire
mass. AIME24 also nudged up, 0.062 → 0.087, a little more than I predicted; the model could partly solve
a few more AIME prompts than I thought, and pure RL sharpened them. And the predicted cost landed too:
AMC23 fell from the switch's 0.325 to 0.283 — pure GRPO gave back exactly the small-split bootstrap the
switch had won. So the picture is clean. On this 1.5B backbone over a broad OpenR1 mixture, killing the
imitation tax dominates the final score: pure GRPO wins on two of three splits, including the largest, and
it is the strongest rung on the board.

But read the *shape* of GRPO's win, because it names the next move. GRPO is strong precisely where the
model has reward contrast — where some of its eight rollouts pass — and it is *structurally blind* where
it does not. On an all-wrong prompt the group advantage `(R − mean)/std` is identically zero, so GRPO
contributes no gradient there at all. AIME24's 0.087 is the symptom: it improved over the switch, but it
is still low in absolute terms, because the hardest AIME problems are exactly the ones the 1.5B model
*never* samples a correct solution for — every rollout wrong, every advantage zero, no learning, ever.
Pure RL cannot bootstrap a capability the model never produces; it can only sharpen what the model can
already occasionally do. So GRPO's ceiling is the base model's own sampling distribution. The switch tried
to fix this with SFT on the stuck prompts and lost MATH-500 to the imitation tax in the process. GRPO
avoids the tax but abandons the stuck prompts. Neither rung gets *both* — and that is the gap.

So the question is precise: can I bootstrap the dead-gradient prompts the way the switch did, but through
a *reward* channel rather than an imitation channel, so I do not re-import the MATH-500 tax? The switch's
mistake was that on a stuck prompt it routed the teacher trace to SFT — plain copy-the-teacher, which
narrows and overfits. What if, on exactly those stuck prompts, I instead inject the teacher trace as an
*off-policy RL* sample and let the RL advantage machinery handle it? The scaffold has that arm: the
controller's `off_add_num < 0` branch adds an off-policy sample with `whether_off=True`, consumed as
off-policy RL rather than SFT. That is the trainer-side handle on the move I want.

Let me derive why injecting the teacher as an RL group member is the right thing, and what makes it
different from both SFT and pure GRPO. Keep the teacher trace as one more member of the prompt's rollout
group. The on-policy group is `G_on = {R(τ_i)}` (eight rollouts, all wrong on a stuck prompt), and the
off-policy member is `G_off = {R(τ_off)}` (the teacher, correct). Standardize the advantage over the
*union*: `Â_i = (R(τ_i) − mean(G_on ∪ G_off)) / std(G_on ∪ G_off)`. Watch the dead-gradient prompt come
alive: the on-policy rollouts are all 0, the teacher is 1, so the union has spread — nonzero mean, nonzero
std — the teacher gets a large positive advantage, the failing rollouts get a small negative one, and
there is a gradient again. The bootstrap happens *through the advantage*, not through an imitation loss. And
the beauty is it self-adjusts: on a prompt the model already solves, its own rollouts are mostly correct,
the teacher is no longer special, the on-policy rollouts dominate the group statistics, and the signal
stays self-driven. So the imitation-versus-exploration balance is not a gate threshold I tune — it falls
out of the group statistics, exactly the per-question competence signal the switch was reaching for with
`on_solve_num`, but consumed as reward instead of as copying. This is the move that gets *both*: it
bootstraps the stuck prompts (fixing GRPO's AIME blindness) without imitation (avoiding the switch's
MATH-500 tax), because the teacher trace is weighted by its advantage in the group, not pulled up
uniformly the way SFT pulls every teacher token.

Two refinements the full method makes, and I have to be honest about which ones this harness lets me
realize and which it does not. First, the off-policy member needs its own importance ratio, with the
teacher policy `π_φ` in the denominator. To avoid the teacher's per-token densities and any tokenizer
mismatch, the method sets `π_φ = 1`, so the off-policy ratio collapses to the model's own probability of
the teacher token, and the clip is dropped on that branch (a `[1−ε,1+ε]` clip is ill-posed for a raw
probability). Second — and this is the load-bearing refinement — that bare off-policy term *hacks* the
objective: its gradient scales with `π_θ`, so it pushes the teacher tokens the model already agrees with
and ignores the low-probability surprising tokens that carry the new reasoning, collapsing entropy. The
fix is policy shaping: replace the off-policy ratio `r̂` with `f(r̂) = r̂/(r̂+γ)`, whose derivative
`γ/(r̂+γ)²` is largest at small `r̂`, amplifying the gradient on exactly the low-probability crucial
tokens (by ≈`1/γ`, with `γ = 0.1`) and damping the easy agreement. The union-group advantage and the
shaping together are what make off-policy guidance bootstrap *beyond* the model's capability while keeping
exploration alive.

Here is the harness boundary, stated plainly, because it determines what the trajectory's edit actually
is. The editable surface is the *trainer-side router* — the `select_on_off_ada_balance` controller, the
per-question routing state, the off-policy sample construction, and the on-policy retention/deletion. The
*actor* is fixed at `offline_loss_type=sft`, the advantage estimator is fixed at `grpo`, and the rollout
modules are read-only. So I control *which* prompts get an off-policy RL sample injected, and I control
the removal of the on-policy group — but I do *not* control the actor's off-policy loss form. The
union-group advantage normalization is the part the fixed `grpo` advantage code computes over whatever
samples I assemble; the off-policy RL consumption is what the `whether_off=True` path triggers. What the
harness does *not* expose is LUFFY's policy-shaping refinement: `f(x)=x/(x+γ)` and the `π_φ=1` unclipped
off-policy ratio live inside the actor's off-policy surrogate, which is fixed here. So the lever I truly
control is the *routing* — inject the teacher as off-policy RL on the stuck prompts so it enters the
advantage group and bootstraps through reward, with the actor's fixed off-policy path doing the
consumption. I keep the method's natural form (the off-policy RL arm, the union over on/off samples) and
accept that the entropy-preserving shaping is the part this task's actor does not let me add — the
trajectory lands the trainer-side realization, not the full actor-side shaped objective.

Make the controller concrete. The switch logic already has the three routes I need: SFT (`off_add_num =
+1`) for the all-wrong band, off-policy RL (`off_add_num = −1`) for a middle band, keep-GRPO (`0,0,0`)
otherwise. The grpo rung showed the keep-GRPO route wins where there is contrast, and the switch rung
showed the SFT route loses MATH-500. So the LUFFY move is to make the *off-policy RL arm the default for
the stuck prompts* instead of SFT: when `on_solve_num ≤ switch_gate` (all-wrong, the dead-gradient case),
drop the on-policy group and add the teacher as an off-policy *RL* sample — `(on_remove_num, on_add_num,
off_add_num) = (rollout.n_verify, 0, −1)` — so the teacher enters the union advantage group through the
reward channel rather than as imitation; and when `on_solve_num > switch_gate` keep the on-policy GRPO
group untouched, `(0, 0, 0)`, preserving the pure-GRPO behavior that just won MATH-500. That is the whole
edit: GRPO everywhere the model has contrast, off-policy-RL guidance exactly where it is stuck — the
union-group bootstrap of the method, expressed entirely in the router the harness gives me.

Now the bar this has to clear, and what I would validate, against pure GRPO's real numbers — no invented
results here, this is the endpoint. The falsifiable claim is on AIME24, the split where GRPO is blind.
GRPO got 0.087 there because the hardest prompts are all-wrong and contribute no gradient; routing those
to off-policy RL should give them a *positive* gradient through the union advantage, so the test is whether
**AIME24 lifts above 0.087** — that is the one number that says the off-policy guidance bootstrapped what
pure RL could not. MATH-500 must *hold* near GRPO's 0.419 and not regress toward the switch's 0.250: the
keep-GRPO route is unchanged on the contrast prompts, and because the stuck prompts now go to off-policy
*RL* rather than SFT, the imitation tax that capped MATH-500 should not return — if MATH-500 falls back
toward 0.25, that would mean the off-policy RL arm is behaving like imitation in this fixed-actor harness
and the shaping I cannot add was load-bearing. AMC23 should recover toward or past the switch's 0.325,
since the small split's stuck prompts get a bootstrap again, now without the narrowing. The honest caveat
is the harness boundary: without the actor-side policy shaping, the off-policy RL term may still over-weight
the teacher tokens the model already agrees with and under-learn the surprising ones, so the AIME24 lift
may be smaller than the full method's — the trajectory's claim is the *direction* (off-policy-RL guidance
on stuck prompts beats both pure GRPO's AIME blindness and the switch's MATH-500 tax), not a specific
number. This rung is what the trajectory ends on: the union-group off-policy guidance whose reward-channel
bootstrap, routed only at the dead-gradient prompts, is designed to clear pure GRPO on AIME24 while
holding its MATH-500.

The causal chain in one breath: pure GRPO won MATH-500 (0.419) by killing the imitation tax but stays
blind on the all-wrong AIME prompts (0.087), where the group advantage is zero and pure RL cannot
bootstrap → bootstrap those prompts through a *reward* channel instead of SFT: inject the teacher trace as
an off-policy RL member of the group → standardize the advantage over the union `G_on ∪ G_off`, so the
teacher gets high advantage exactly when the model is stuck and low when it is competent — a per-question
self-adjusting mix, consumed as reward not imitation → the full method also sets `π_φ = 1` (unclipped
off-policy ratio) and shapes it by `f(x)=x/(x+γ)`, `γ=0.1`, to emphasize the low-probability surprising
teacher tokens, but that shaping is actor-side and this task's actor is fixed, so the harness exposes only
the *routing* → land it as the controller that routes stuck prompts (`on_solve_num ≤ switch_gate`) to the
off-policy RL arm (`n_verify, 0, −1`) and keeps GRPO (`0,0,0`) elsewhere → expecting AIME24 to lift above
GRPO's 0.087 (bootstrap) while MATH-500 holds near 0.419 (no imitation tax) — both, where each earlier
rung got only one.
