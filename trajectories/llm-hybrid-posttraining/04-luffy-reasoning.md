Pure GRPO's numbers settled the dissociation I set up, and they settled it more sharply than I expected.
Seed 42: AIME24 0.087, AMC23 0.283, MATH-500 0.419. MATH-500 is the breakthrough — 0.419, up from the
switch's 0.250 and the floor's 0.244, a +0.17 jump on the broad in-distribution split. That confirms the
diagnosis exactly: the imitation tax *was* what capped MATH-500, and paying it zero times — letting every
contrast prompt learn from its own graded rollouts instead of copying the teacher — recovered the entire
mass. AIME24 also nudged up, 0.062 → 0.087, a little more than I predicted; the model could partly solve
a few more AIME prompts than I thought, and pure RL sharpened them. And the predicted cost landed too:
AMC23 fell from the switch's 0.325 to 0.283 — pure GRPO gave back exactly the small-split bootstrap the
switch had won. Put all three deltas in problem-counts to weigh them against each other. MATH-500 `0.250 →
0.419` is `125 → 209.5` problem-equivalents, `+84` out of 500 — this is the jump I made falsifiable last
rung (I said the imitation-tax story needed tens of problems, not the switch's `+3`, and it delivered `+84`).
AIME24 `0.062 → 0.087` is `1.9 → 2.6`, `+0.7` of a problem on a 30-problem split — real but tiny, well
inside what a single problem flipping is worth. AMC23 `0.325 → 0.283` is `13.0 → 11.3`, `−1.7` problems, the
predicted small-split cost. So the picture is clean and the magnitudes are honest: the MATH-500 gain is an
order of magnitude larger than either the AIME nudge or the AMC give-back, so on this 1.5B backbone over a
broad OpenR1 mixture, killing the imitation tax dominates the final score. Pure GRPO wins on two of three
splits, including the largest, and it is the strongest rung on the board.

But read the *shape* of GRPO's win, because it names the next move. GRPO is strong precisely where the
model has reward contrast — where some of its eight rollouts pass — and it is *structurally blind* where
it does not. On an all-wrong prompt the group advantage `(R − mean)/std` is identically zero, so GRPO
contributes no gradient there at all. AIME24's 0.087 is the symptom: it improved over the switch, but it
is still low in absolute terms, because the hardest AIME problems are exactly the ones the 1.5B model
*never* samples a correct solution for — every rollout wrong, every advantage zero, no learning, ever. Size
the blind spot: at the measured `0.087`, the average AIME prompt has a per-rollout solve probability near
`0.087`, so it fails all eight rollouts with probability `(1 − 0.087)^8 ≈ 0.48`. Roughly half of AIME24 is
still handing GRPO an all-equal group and therefore zero gradient, every single step of the run. Pure RL
sharpened the sampleable half from `0.062` to `0.087`; the other half never moved because it *cannot* move —
there is no gradient to move it with.
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
alive with actual numbers. Take the worst case for pure GRPO: seven on-policy rollouts all wrong (reward 0)
and one teacher trace correct (reward 1), a union of `{0,0,0,0,0,0,0,1}`. The mean is `1/8 = 0.125` and the
sample std is `√((7·0.125² + 0.875²)/7) = √(0.875/7) = 0.354`. So the teacher gets `Â = (1 − 0.125)/0.354 =
+2.47` and each failing rollout gets `(0 − 0.125)/0.354 = −0.354`. Where pure on-policy gave *exactly zero*
for all eight, the union gives the teacher a large positive push and the failures a mild negative one — the
gradient is alive again and it points at the teacher. The bootstrap happens *through the advantage*, not
through an imitation loss. And now watch the other end, which is what makes this self-adjust rather than just
inject: a prompt the model already mostly solves, say six of seven on-policy rollouts correct plus the
teacher, union `{1,1,1,1,1,1,0,1}`, mean `0.875`, std `0.354` again by symmetry. The teacher's advantage is
now `(1 − 0.875)/0.354 = +0.354` — an order of magnitude below the `+2.47` it got on the stuck prompt, and no
larger than what the model's own correct rollouts receive. So the teacher stops being special exactly when
the model can carry the prompt itself; the on-policy rollouts dominate the group statistics and the signal
stays self-driven. So the imitation-versus-exploration balance is not a gate threshold I tune — it falls
out of the group statistics, exactly the per-question competence signal the switch was reaching for with
`on_solve_num`, but consumed as reward instead of as copying. This is the move that gets *both*: it
bootstraps the stuck prompts (fixing GRPO's AIME blindness) without imitation (avoiding the switch's
MATH-500 tax), because the teacher trace is weighted by its advantage in the group, not pulled up
uniformly the way SFT pulls every teacher token. Line the two channels up on the same two prompts to see
why this dodges the tax. Under the switch's SFT route the teacher gets gradient coefficient exactly `1` on
every token, on the stuck prompt *and* on the already-solved prompt alike — identical narrowing pressure
regardless of whether the model needs it, which is exactly the MATH-500 tax I measured (copying on prompts
the model already handles). Under the union-RL route the teacher gets `+2.47` on the stuck prompt and only
`+0.354` on the competent one — a `7×` difference the SFT channel cannot express, because SFT has no
competence-dependent coefficient at all. So on the broad split full of prompts the model can partly do, the
RL channel barely touches the teacher (`+0.354`, no larger than the model's own correct rollouts get), while
the SFT channel would hammer it at `1`. That is the mechanical reason routing through reward rather than
imitation should keep MATH-500's `0.419` intact where the switch bled it down to `0.250`.

Two refinements the full method makes, and I have to be honest about which ones this harness lets me
realize and which it does not. First, the off-policy member needs its own importance ratio, with the
teacher policy `π_φ` in the denominator. To avoid the teacher's per-token densities and any tokenizer
mismatch, the method sets `π_φ = 1`, so the off-policy ratio collapses to the model's own probability of
the teacher token, and the clip is dropped on that branch (a `[1−ε,1+ε]` clip is ill-posed for a raw
probability). Second — and this is the load-bearing refinement — that bare off-policy term *hacks* the
objective. Look at its gradient. The off-policy term is `Σ_t π_θ(τ_{j,t}) · Â_j`, and the per-logit gradient
on the sampled teacher token scales as `π_θ(1 − π_θ)` (from `∇π_θ = π_θ(1 − π_θ)∇logit`). Tabulate that
scale: `π = 0.01 → 0.0099`, `π = 0.1 → 0.09`, `π = 0.5 → 0.25`, `π = 0.9 → 0.09`. It peaks in the middle and
is near-zero at *both* extremes — and the extreme I care about, the low-probability surprising tokens that
carry the new reasoning the model does not yet have, is exactly where the gradient is smallest: a `π = 0.01`
token gets `≈ 25×` less gradient than a `π = 0.5` one. So the cheapest way for the optimizer to raise this
objective is to push up the teacher tokens the model *already* agrees with (mid-to-high `π`) and leave the
surprising ones untouched. The model converges fast onto the parts of the teacher trace it already produces
and never learns the parts that would expand its capability — entropy collapses, exploration dies, and I am
back to a fast-converging imitator. The fix is policy shaping: replace the off-policy ratio `r̂` with `f(r̂) = r̂/(r̂+γ)`, increasing and concave,
whose derivative `f'(r̂) = γ/(r̂+γ)²` is *largest* at small `r̂` and decays as `r̂` grows — the opposite of
the identity's flat `f' = 1`. Tabulate the amplification `shape/identity = γ/(π+γ)²` at `γ = 0.1` rather than
wave at "≈1/γ": at `π = 0.001` it is `9.80`, at `0.01` it is `8.26`, at `0.05` it is `4.44`, at `0.1` it is
`2.50`, at `0.3` it is `0.625`, at `0.9` it is `0.10`. So the amplification is `≈ 1/γ = 10×` only in the
`π → 0` limit, already down to `2.5×` by `π = 0.1`, and it crosses below `1` at `π = √γ − γ ≈ 0.216` — past
that point the shaping actively *damps* the gradient relative to the identity, down to `0.10×` at `π = 0.9`.
That is exactly the profile I wanted and it is more than a uniform rescale: low-probability surprising teacher
tokens get up to ten times more gradient while already-likely tokens get suppressed, so the optimizer is
pulled toward the deviating reasoning instead of the easy agreement. The union-group advantage and the
shaping together are what make off-policy guidance bootstrap *beyond* the model's capability while keeping
exploration alive.

Here is the harness boundary, stated plainly, because it determines what the trajectory's edit actually
is. The editable surface is the *trainer-side router* — the `select_on_off_ada_balance` controller, the
per-question routing state, the off-policy sample construction, and the on-policy retention/deletion. The
*actor* is fixed at `offline_loss_type=sft`, the advantage estimator is fixed at `grpo`, and the rollout
modules are read-only. So I control *which* prompts get an off-policy RL sample injected, and I control
the removal of the on-policy group — but I do *not* control the actor's off-policy loss form. The
union-group advantage normalization is the part the fixed `grpo` advantage code computes over whatever
samples I assemble; the off-policy RL consumption is what the `whether_off=True` path triggers. I should be exact about what
that produces, because the routing is a realization of the union-group idea and not a literal copy of it.
When the controller returns `off_add_num = −1` on a stuck prompt, the harness first *removes* the eight dead
on-policy rollouts and then adds the teacher trace tagged `whether_off=True`, and the fixed `grpo` advantage
code standardizes over the group it is finally handed. So the exact `+2.47`/`−0.354` spread I computed for
the method's natural 7-on-plus-1-off group is not literally what the frozen code sees on a fully-stuck
prompt, where the on-policy members are gone by advantage time; what the code sees is the teacher entering
the reward channel with a positive advantage, a bootstrap on a correct trace where pure GRPO had exactly
zero. That is the directional content of the union-group idea — stuck prompt goes from no gradient to a
positive push on a correct solution, *through reward rather than imitation* — realized by the router even
though the precise magnitude depends on how the harness assembles the post-removal group. The one thing I
cannot reach at all is the *loss form* the tagged sample is consumed under: LUFFY specifies the `π_φ = 1`
unclipped, `f`-shaped off-policy surrogate, and that lives in the actor, which is fixed at
`offline_loss_type=sft`. So the teacher enters through the reward channel, but the entropy-preserving shaping
that would emphasize its surprising tokens is the piece the harness withholds. What the
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
group untouched, `(0, 0, 0)`, preserving the pure-GRPO behavior that just won MATH-500. Why gate the
injection at stuck-only rather than dropping the teacher into *every* prompt's group the way the method's
plainest form does? The self-adjusting numbers answer it: on a prompt the model already handles, the
teacher's union advantage is only `+0.354`, no larger than what the model's own correct rollouts earn, so
injecting it there buys almost nothing — and it does buy a *cost*, because it disturbs the clean on-policy
group statistics that produced MATH-500's `0.419`, the one result I most want to protect. The gate spends the
off-policy machinery exactly where GRPO scored zero (the `≈ 0.48` of AIME prompts that are all-wrong) and
nowhere else, which keeps the two-of-three-split winner untouched on the splits it won. That is the whole
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
since the small split's stuck prompts get a bootstrap again, now without the narrowing: GRPO gave back
`1.7` problems (`13.0 → 11.3`) by abandoning the stuck AMC prompts, and re-bootstrapping them through the
reward channel should return most of that `+1.7` — a recovery toward `0.32`-ish — while not re-incurring the
narrowing that made the switch's version of that bootstrap expensive on MATH-500. The honest caveat
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
