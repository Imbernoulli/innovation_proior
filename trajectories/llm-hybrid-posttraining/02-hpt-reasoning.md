The pure-SFT floor told me what unconditional imitation costs, in numbers: AIME24 0.062, AMC23 0.217,
MATH-500 0.244. Read the structure. AMC23 (0.217) is the least embarrassing — the small 40-example split
where a handful of demonstration-shaped problems moves the score, as expected. But MATH-500, the broad
500-example in-distribution split where teacher traces should transfer *best*, sits at only 0.244 — barely
past the tiny AMC23 split in fractional terms — when in-distribution transfer should have opened a gap. In
problem-counts: `0.062 × 30 ≈ 1.9` on AIME24, `0.217 × 40 ≈ 8.7` on AMC23, `0.244 × 500 ≈ 122` on MATH-500.
The tell is that flatness on the largest, least-noisy split: one epoch of copying teacher traces on the
broad split buys almost nothing beyond what the same copying buys on 40 hand-shaped AMC problems. Not
because the teacher traces are bad — because copying them on *every* prompt, including the ones the model
already solves, narrows the policy to one solution style and never lets it practice its own reasoning, so
any test prompt whose shape differs is unsupported, and the model trains no recovery from its own
mid-solution mistakes (why the hard AIME problems collapse). The fix is to stop throwing the on-policy group
away where the model is already learning from its own rollouts, and reserve the demonstration for where it
is genuinely stuck — the route has to *condition* per-question.

But "combine SFT on some prompts with RL on others" forces the question the floor swept aside: why is it even
legitimate to mix an RL loss and an SFT loss? The floor already answered it — the two are the two halves of
the gradient of one KL-regularized objective, `∇J_μ = E_{τ∼π_θ}[r ∇log π_θ] + μ E_{τ∼π_β}[∇log π_θ]`,
differing only in which distribution the trajectory is sampled from. So the router is not a blend of two
competing forces; it is a per-prompt choice of how to sample and weight one estimator.

That is only useful if I can put both terms in one form and read off the knobs that distinguish them. A
change of measure to a common reference density `π_ref` (using `∇log π_θ = (1/π_θ)∇π_θ`) collapses both into
`grad_uni = 1_stable · (1/π_ref) · Â · ∇π_θ`. Four interchangeable components: the stabilization mask
`1_stable` (PPO's clip = a stop-gradient on unsafe samples), the reference denominator `1/π_ref` (SFT/REINFORCE
use `π_θ`; PPO/GRPO use `π_{θ_old}`; offline RL uses `π_ref ≡ 1`), the advantage `Â` (SFT `≡1`; REINFORCE
`±1`; GRPO group-normalized), and the shared `∇π_θ`. SFT sits at `(π_ref = π_θ, Â ≡ 1)`, giving `∇log π_θ` —
the demonstration half. GRPO sits at `(π_ref = π_{θ_old}, Â = group z-score)`, reconstructing the importance
ratio `π_θ/π_{θ_old}` with the group baseline in place of `r` — the reward half. The router is a per-prompt
choice of `(π_ref, Â)`, not a blend of rival objectives.

If they are all estimators of one gradient, the design question is how to weight them — and a *fixed* weighted
blend is wrong, because the bias and variance of each estimator are not fixed: they depend on the current
`π_θ` and the prompt. On a prompt the model already solves, the RL signal is informative and the SFT term is
pure overfitting pressure — the tax the floor's MATH-500 measured. On a prompt the model fails completely,
the RL term carries *no* information: when every rollout gets the same reward, the group-relative advantage
`(R − mean)/std` is identically zero. Run the standardization on `n_verify = 8` binary rewards. All-wrong
`[0,…,0]`: mean 0, std 0 (floored), `Â = 0`. All-correct `[1,…,1]`: mean 1, std 0, `Â = 0` again. Mixed, two
of eight `[1,1,0,0,0,0,0,0]`: mean 0.25, std `≈ 0.463`, the two correct get `Â ≈ +1.62` and the six wrong
`≈ −0.54` — a clean signed contrast. So the RL gradient does not fade gracefully as a prompt gets harder; it
*survives* on any mixed group and drops to exactly zero the instant the group goes all-equal. The two zeros
differ: the all-correct zero is saturation the model earned and can skip; the all-wrong zero is the dangerous
one — no gradient exactly where the model most needs to learn. And which prompt sits where *changes as the
model learns*. A single global mixing coefficient is structurally wrong the way one global learning rate is;
the weighting must be a function of how the model is currently doing on each prompt.

What measures that, per prompt? The loop already samples and verifies `n_verify = 8` rollouts, so the solve
count `on_solve_num = Σ_i v(τ_i) ∈ {0,…,8}` falls out for free — the model's current competence at no extra
cost. It is a coarse ruler, and worth knowing how coarse. Eight Bernoulli draws estimate the true solve
probability `p` with SE `√(p(1−p)/8)`, so a prompt at genuine `p = 0.1` produces all eight failures with
probability `0.9^8 ≈ 0.43`; at `p = 0.05`, `0.66`; at `p = 0.2`, `0.17`. So the `on_solve_num = 0` bucket is
not "prompts the model cannot solve" but "prompts it failed all eight this step," sweeping in many
rarely-solvable ones. That imprecision argues for a *hard* gate at exactly `on_solve_num = 0` rather than
higher: raising the gate only hands more borderline-solvable prompts to imitation, whereas `switch_gate = 0`
restricts the SFT route to prompts where the on-policy signal is *provably* dead (standardized advantage
exactly zero), the one case eight samples call with certainty.

The failure is a sharp cliff — the RL gradient dies at `on_solve_num = 0` — so the matching intervention is a
hard switch, not a smooth blend. Switch to SFT only when `on_solve_num ≤ switch_gate = 0`: drop the dead
group and add the demonstration, `(rollout.n_verify, 0, +1)` — the same SFT move as the floor, fired only on
stuck prompts. When there is reward contrast, keep the on-policy GRPO group untouched, `(0, 0, 0)`. That
repairs the floor's mistake directly: the MATH-500 narrowing came from copying the teacher on prompts the
model could already handle, and those now stay on RL.

The switch logic also exposes a middle band. When `switch_gate < on_solve_num ≤ switch_gate_off`, the
controller can drop the on-policy group and add an off-policy *RL* sample (`off_add_num = −1`,
`whether_off=True`) rather than SFT. Why is plain SFT still the right default for the *fully stuck* prompts?
Off-policy RL needs a reference policy for the teacher trajectory, which I do not have, forcing `π_ref ≡ 1` —
and in the unified template that turns the importance ratio into rejection sampling, valid only under a
uniform-coverage assumption that never holds, injecting heavy bias. Plain SFT (`Â ≡ 1`, `π_ref = π_θ`) has no
ill-posed ratio; it is the clean way to consume an offline trajectory, exactly the second half of the
gradient with `μ` folded into `sft_loss_coef`. So the all-wrong prompts use the SFT branch; the
off-policy-RL band stays in the controller as the method's full form, with `switch_gate_off` tuned so the
dominant routes are SFT-for-stuck and GRPO-for-contrast. No soft blend, and no contamination of the GRPO
statistics since the advantage code normalizes over on-policy samples only. (The full controller is in the
answer.)

Where the floor returned `(n_verify, 0, 1)` unconditionally, the switch reads `on_solve_num`, fires that SFT
triple only when `on_solve_num ≤ switch_gate`, routes the middle band to off-policy RL, and otherwise returns
`(0, 0, 0)`. The mixture self-adjusts: early the weak model fails many prompts, so more of the batch gets
demonstration gradients; as it strengthens, more prompts cross the gate and the batch tips toward RL. The
SFT-routed fraction is `(1−p)^8` per prompt of difficulty `p` — early, with `p` around `0.05`–`0.1`, that is
`0.66`–`0.43` routed to imitation; as training lifts `p` toward `0.2`–`0.3`, the same prompts fall to
`0.17`–`0.06`. The mixing ratio is read off the model's own live competence, not scheduled.

What I expect this to fix, and where I am unsure. The floor's MATH-500 narrowing came from imitating on
prompts the model could already handle, so conditioning the switch should lift MATH-500 off 0.244. AMC23
(0.217) should hold or improve, since demonstration-shaped problems the model fails still get the bootstrap.
AIME24 is the open question, and the risk is in the construction: on AIME competition problems the 1.5B model
may fail *all* eight rollouts on almost every prompt, so the gate fires SFT on nearly all of them and the
switch could collapse back toward the pure-SFT behavior I am trying to escape, leaving AIME24 near 0.062. If
that happens, the next question is whether keeping those stuck prompts on a *reward* channel does better on
the broad splits than routing them to imitation.
