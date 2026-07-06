The pure-SFT floor told me exactly what unconditional imitation costs, and it told me in numbers.
Seed 42 landed at AIME24 0.062, AMC23 0.217, MATH-500 0.244. Read the split structure: AMC23 (0.217) is
the *least* embarrassing — the small 40-example split is where a handful of demonstration-shaped problems
moves the score, exactly as I expected — but MATH-500, the broad 500-example in-distribution split where
teacher traces should transfer *best*, sits at only 0.244, and AIME24 is down at 0.062. Put the three
scores back into problem-counts to feel the shape: `0.062 × 30 ≈ 1.9` problem-equivalents on AIME24,
`0.217 × 40 ≈ 8.7` on AMC23, `0.244 × 500 ≈ 122` on MATH-500. The number that should jump is that MATH-500,
the split built to reward exactly the demonstration distribution, only edges past the tiny AMC23 split in
fractional terms — `0.244` versus `0.217`, a near-tie — when in-distribution transfer should have opened a
gap. On the broad split where teacher-style solutions are most abundant, one epoch of copying them buys
almost nothing beyond what the same copying buys on 40 hand-shaped AMC problems. That flatness is the tell,
and it is on the largest, least-noisy split, so it is a real signal and not a seed artifact. It is not that
the teacher traces are bad; it is that copying them on *every*
prompt — including the ones the 1.5B model already solves on its own — narrows the policy to one fixed
solution style and never lets it practice its own reasoning, so any test prompt whose solution shape
differs from the demonstrations is unsupported. The model also never trains recovery from its own
mid-solution mistakes, which is why the hard AIME24 problems collapse. So the failure is precise and it
is *not* "SFT is the wrong tool." SFT was the only thing that could bootstrap a stuck prompt. The
failure is that I applied it *unconditionally* — I paid the imitation tax on the prompts that did not
need it. The fix is to stop throwing the on-policy group away where the model is already learning from
its own rollouts, and reserve the demonstration for the prompts where it is genuinely stuck. That means
the route has to *condition* on something — and the something has to be per-question.

But the moment I say "combine SFT on some prompts with RL on others," I have to confront the thing the
floor swept under the rug: why is it even legitimate to mix an RL loss and an SFT loss? They look like
different objectives — one maximizes expected verifier reward under the model's own distribution, the
other minimizes cross-entropy against an external dataset. If they optimize different things, summing
them is just hoping two unrelated forces point somewhere useful. Before I design any switching rule I
need to know whether these are the same objective wearing two costumes, because if they are, then
combining them is not a hack — it is choosing how to estimate one gradient.

So write the most naive common objective I can. RL wants high expected verifier reward; SFT wants the
policy to stay close to the demonstration distribution `π_β`, and "stay close to a distribution" is a KL
penalty: `J_μ(θ) = E_{τ∼π_θ}[r(τ|q)] − μ·KL(π_β(·|q) ‖ π_θ(·|q))`, `μ ≥ 0`. Take its gradient term by
term. The reward term, by the score-function identity, is `E_{τ∼π_θ}[r ∇log π_θ]` — that is REINFORCE,
the on-policy reward policy gradient. The KL term I have to be careful with, because the direction is
exactly the kind of sign I would wave my hands through. `KL(π_β ‖ π_θ) = E_{τ∼π_β}[log π_β − log π_θ]`;
`π_β` is fixed in `θ` and the expectation is over `π_β`, also fixed, so I differentiate only the
`−log π_θ` inside, giving `∇KL = −E_{τ∼π_β}[∇log π_θ]`, and the contribution to `∇J_μ` is
`−μ·∇KL = +μ E_{τ∼π_β}[∇log π_θ]`. The two minus signs cancel to a plus, and `E_{τ∼π_β}[∇log π_θ]` is
exactly the maximum-likelihood gradient — push up the log-prob of the demonstration tokens. So

  ∇J_μ = E_{τ∼π_θ}[r ∇log π_θ]  +  μ E_{τ∼π_β}[∇log π_θ].

This is exactly the spot where a hand-waved sign would poison everything downstream, so I check it on a
toy I can compute in closed form. Take three trajectories, `π_θ = softmax(θ)` with `θ = (0.1, 0.4, −0.3)`,
so `π_θ = (0.331, 0.447, 0.222)`, and a fixed demonstration distribution `π_β = (0.6, 0.3, 0.1)`. For a
softmax, `∂ log π_θ(k)/∂θ_j = δ_{kj} − π_θ(j)`, so `E_{τ∼π_β}[∇log π_θ]_j = Σ_k π_β(k)(δ_{kj} − π_θ(j)) =
π_β(j) − π_θ(j)`, which comes out `(0.269, −0.147, −0.122)`. Then `∇KL(π_β ‖ π_θ) = −E_{π_β}[∇log π_θ] =
(−0.269, +0.147, +0.122)`. The demonstration term `+μ E_{π_β}[∇log π_θ] = μ(0.269, −0.147, −0.122)` pushes
mass *onto* trajectory 0 — the one `π_β` weights most heavily, `0.6` — and off trajectories 1 and 2. That
is the behavior-cloning direction: ascending the demonstration half moves `π_θ` toward `π_β`. Had I written
the KL the other way, `KL(π_θ ‖ π_β)`, the expectation would be over `π_θ` and I would pick up an extra
term from differentiating the sampling measure — not the clean SFT gradient. The direction I wrote is the
one that makes the second half fall out as plain behavior cloning, and the toy confirms both the sign and
the target.

There it is. The gradient of one objective is an on-policy reward term plus the SFT term. SFT and RL are
not two objectives; they are two halves of the gradient of a single objective, distinguished only by
which distribution the trajectory is sampled from and what scalar weights it. The pure-SFT floor was
keeping only the right half for every prompt; pure GRPO keeps only the left half. Adding them is not a
hack — it *is* the gradient. That reframes the whole router: I am not blending two competing forces, I
am choosing, per prompt, how to sample and weight one estimator.

Push on that, because I want both terms in one shared form so I can read off which knobs distinguish
them. Both are an expectation of (something) times `∇log π_θ`, sampled from different distributions. A
change of measure to a common reference density `π_ref` (using `∇log π_θ = (1/π_θ)∇π_θ`, which cancels
the explicit `π_θ`) collapses both into one estimator, `grad_uni = 1_stable · (1/π_ref) · Â · ∇π_θ`,
with `Â_uni = r + μ·π_β/π_θ`. Four interchangeable components: the stabilization mask `1_stable` (PPO's
clip = a stop-gradient on unsafe samples), the reference denominator `1/π_ref` (SFT/REINFORCE use `π_θ`;
PPO/GRPO use `π_{θ_old}`; offline RL uses `π_ref ≡ 1`), the advantage `Â` (SFT: `≡1`; REINFORCE: `±1`;
GRPO: group-normalized), and the shared likelihood gradient `∇π_θ`. SFT, REINFORCE, PPO, GRPO all fall
out as choices of `π_ref` and `Â`. So each is an estimator — sometimes biased — of the *same* underlying
policy gradient, with different bias-variance profiles.

Before I trust that template I want it to *reproduce* the two halves I already derived, not merely resemble
them, so let me instantiate the two dials I actually care about. SFT: set `π_ref = π_θ` and `Â ≡ 1`, and the
estimator is `(1/π_θ) · 1 · ∇π_θ = ∇π_θ/π_θ = ∇log π_θ` — the plain cross-entropy ascent on the demonstration
token, which is exactly the second half `E_{π_β}[∇log π_θ]` under its own sampling measure. GRPO: set
`π_ref = π_{θ_old}`, and the `1/π_ref` denominator paired with `∇π_θ` reconstructs the importance ratio
`π_θ/π_{θ_old}`, with `Â` the group z-score standing where the reward term's `r` sat — the first half, now
variance-reduced by the group baseline. Both halves I derived from `∇J_μ` are single points in this
`(π_ref, Â)` space, which is the concrete sense in which "SFT and RL are two halves of one gradient" is not a
slogan: they differ only in the reference denominator and the advantage, and nothing else in the estimator
moves. That is what licenses treating the router as a per-prompt choice of `(π_ref, Â)` rather than a blend of
two rival objectives.

That is the payoff for designing a router. If they are all estimators of one gradient, the design
question is how to *weight* the estimators — and the instinct from estimation theory is to average noisy
measurements, weighted toward the lower-variance ones. So maybe a fixed weighted blend of the SFT and RL
gradients? Wall. The bias and variance of each estimator are not fixed properties — they depend on the
*current* state of `π_θ` and on the prompt. On a prompt the model already solves, the on-policy RL signal
is informative and low-variance and the SFT term is pure overfitting pressure — exactly the imitation tax
the floor's MATH-500 number measured. On a prompt the model fails completely, the on-policy term carries
*no* information at all: when every rollout in the group gets the same reward, the group-relative
advantage `(R − mean)/std` is identically zero for that prompt, so GRPO contributes no gradient. Let me run
the standardization on the three cases that matter with `n_verify = 8` binary rewards rather than assert the
degeneracy. All-wrong, `R = [0,0,0,0,0,0,0,0]`: mean `0`, std `0` (floored), so `Â = [0,…,0]`. All-correct,
`R = [1,…,1]`: mean `1`, std `0` (floored), `Â = [0,…,0]` again. But a mixed group, say two of eight passing,
`R = [1,1,0,0,0,0,0,0]`: mean `0.25`, sample std `≈ 0.463`, so the two correct rollouts get `Â = (1−0.25)/0.463
≈ +1.62` and the six wrong get `(0−0.25)/0.463 ≈ −0.54` — a clean signed contrast. So the gradient does not
gracefully fade as a prompt gets harder; it *survives* on any group with a mix of passes and fails and drops to
*exactly zero* the instant the group goes all-equal. Two of the three cases vanish, but they are not the same
kind of vanish: the all-correct zero is a saturation the model has earned and can be skipped, while the
all-wrong zero is the dangerous one — no gradient on a prompt the model still cannot solve, which is precisely
where it most needs to learn. And which prompt sits in which case *changes as the model learns*. So a single global mixing coefficient — exactly what the floor's
unconditional rule and any fixed-blend method commits to — is structurally wrong, the same way one
global learning rate is wrong when different parameters need different scales. The "average the
estimators" idea is right in spirit; the constant weight was the mistake. The weighting must be a
function of how the model is *currently* doing on each prompt.

What signal do I have, per prompt, that measures that? The loop already samples `rollout.n_verify = 8`
on-policy rollouts per prompt and verifies them to compute the GRPO advantage. The solve count falls out
for free: `on_solve_num = Σ_i v(τ_i) ∈ {0,…,8}`, the number of the eight rollouts that passed the
verifier. That is exactly the model's current competence on this prompt, and it costs nothing extra —
the same verifier scores GRPO already consumes. So let the route be a function of `on_solve_num`.

I should be honest that `on_solve_num` is a *coarse* competence ruler, and it is worth knowing how coarse
before I lean the whole router on it. Eight Bernoulli draws estimate a prompt's true per-rollout solve
probability `p` with a standard error around `√(p(1−p)/8)`, so a prompt with genuine `p = 0.1` — one the model
can in fact solve one time in ten — produces all eight failures with probability `(1−p)^8 = 0.9^8 ≈ 0.43`. At
`p = 0.05` that is `0.95^8 ≈ 0.66`; at `p = 0.2`, `0.8^8 ≈ 0.17`. So the `on_solve_num = 0` bucket is not
"prompts the model cannot solve" — it is "prompts the model failed all eight times this step," which sweeps in
a large fraction of genuinely-but-rarely-solvable prompts. That imprecision is real, and it is an argument for a
*hard* gate at exactly `on_solve_num = 0` rather than a threshold higher up: raising the gate would only pull in
more of these borderline-solvable prompts and hand them to imitation, whereas `switch_gate = 0` restricts the
SFT route to the prompts where the on-policy signal is *provably* dead (the standardized advantage is exactly
zero, not merely small), which is the one case eight samples can call with certainty. The noise in `on_solve_num`
is a reason to gate conservatively, not a reason to abandon the signal.

Now, what is the simplest function that does the right thing? I could make it a smooth blend, but think
about what the degenerate-zero analysis demands. The failure I am targeting is itself a sharp cliff: the
RL gradient does not gracefully shrink as the prompt gets harder, it *dies* exactly at `on_solve_num = 0`
where all rewards are equal. The sharpest version is a hard switch at a gate. Set the gate so that a
prompt switches to SFT only when `on_solve_num ≤ switch_gate`. With `switch_gate = 0` that means switch
to SFT only when *every* rollout failed — precisely the all-wrong case where the on-policy advantage
vanishes before competence appears. There the route should drop the dead on-policy group and add the
demonstration: `(on_remove_num, on_add_num, off_add_num) = (rollout.n_verify, 0, +1)` — the same SFT move
as the floor, but now fired *only* on the stuck prompts instead of all of them. And when the prompt has
reward contrast (`on_solve_num` above the gate), keep the on-policy GRPO group untouched: `(0, 0, 0)` —
the pure-GRPO move. That is the minimal intervention: I inject the teacher exactly where RL has no
signal, and nowhere else, so I pay the imitation tax only on the prompts that earn it and preserve as
much exploration as possible everywhere else. It directly repairs the floor's mistake — the MATH-500
narrowing came from copying the teacher on prompts the model could already handle, and those prompts now
stay on RL.

The scaffold's switch logic also exposes a middle band — an off-policy *RL* arm — between the two gates.
When `switch_gate < on_solve_num ≤ switch_gate_off`, the controller can drop the on-policy group and add
an off-policy *RL* sample (`off_add_num = −1`, which sets `whether_off=True`) rather than an SFT sample.
This is the third route: keep the demonstration but score it as RL rather than as imitation. Why is plain
SFT, not this off-policy-RL arm, the right default for the *fully stuck* prompts? Because off-policy RL
needs a reference policy for the teacher trajectory, which I do not have, so it would force `π_ref ≡ 1` —
and in the unified template that turns the importance ratio into rejection sampling, valid only under a
uniform-coverage assumption that never holds, injecting heavy bias. Plain SFT (`Â ≡ 1`, `π_ref = π_θ`)
has no ill-posed ratio; it is the clean way to consume an offline trajectory, and it is exactly the
second half of the unified gradient with `μ` folded into `sft_loss_coef`. So for the all-wrong prompts I
use the SFT branch. The off-policy-RL band stays in the controller as the method's full form, but with
`switch_gate_off` tuned so the dominant routes are SFT-for-stuck and GRPO-for-contrast.

Notice what the harness exposes and what it does not. The route here is a hard, per-question switch keyed
on the live solve count, and it lands as exactly that — the literal controller compares `on_solve_num`
against two gate thresholds and returns one of three triples. What it does *not* expose is any softening:
there is no `soft` blend running here (that branch returns `(0,0,1)` for every prompt and is not the
configured strategy), no continuous `α(P)/β(P)` weighting, and no contamination of the GRPO group
statistics by the injected samples — the advantage code group-normalizes over on-policy samples only, so
the RL measurement stays clean. The router is the whole degree of freedom; the actor's mixed loss
(`pg_loss = sft_loss · sft_loss_coef + pg_loss`) and the advantage normalization are fixed downstream of
it. So my edit is the switch controller and nothing else.

So the delta from the floor is concrete: where pure-SFT returned `(n_verify, 0, 1)` unconditionally, I now
read `on_solve_num`, fire that *same* SFT triple only when `on_solve_num ≤ switch_gate`, route the middle
band to the off-policy-RL arm, and otherwise return `(0, 0, 0)` to keep the on-policy GRPO group. The
mixture self-adjusts: early in training the weak 1.5B model fails many prompts, so more of the batch gets
demonstration gradients; as it strengthens, more prompts cross the gate and the batch moves toward
on-policy RL. Put rough numbers on that drift. Of the 128 prompts in a step, the SFT-routed fraction is the
fraction with `on_solve_num = 0`, and by the binomial reading above a prompt of true difficulty `p` lands there
with probability `(1−p)^8`. Early on, when the model is weak and typical `p` is small — say `0.05`–`0.1` across
the harder mass — that is `0.66`–`0.43` of those prompts routed to imitation, so a large share of each batch is
demonstration gradient; as training lifts the typical `p` toward `0.2`–`0.3`, the same prompts fall to
`0.17`–`0.06`, and the batch tips toward on-policy RL without my touching the gate. The mixing ratio is not a
hyperparameter I schedule; it is read off the model's own live competence, step by step. (The full controller is
in the answer.)

Reading the floor's shape, here is what I expect this to fix and where I am unsure. The floor's MATH-500
narrowing came from imitating on prompts the model could already handle; conditioning the switch should
*lift MATH-500 off 0.244*, because the prompts with reward contrast now stay on RL and practice their own
reasoning rather than copying. AMC23, which was the floor's best split (0.217), should hold or improve,
since demonstration-shaped problems the model fails still get the bootstrap. AIME24 is the open question
and I can feel the risk in the construction: on AIME competition problems the 1.5B model may fail *all*
eight rollouts on almost every prompt, so the gate fires SFT on nearly all of them — and on that split
the hybrid switch could collapse back toward the pure-SFT behavior I am trying to escape, leaving AIME24
near the floor's 0.062. If that happens, the diagnosis for the next step is already written: when almost
every prompt is all-wrong, routing the stuck prompts to *SFT* re-imports the imitation tax, and the
question becomes whether keeping those prompts on a *reward* channel — never imitating, pure on-policy
GRPO, or off-policy RL guidance — does better on the broad splits than the hybrid switch does.

The causal chain in one breath: pure-SFT's low MATH-500 (0.244) is the imitation tax — copying the
teacher on prompts the model already solves narrows the policy and skips its own reasoning → SFT and RL
are two halves of one objective's gradient, so the route is a choice of estimator per prompt, not a blend
of rival forces → a fixed mixing weight is wrong because each estimator's bias-variance depends on the
model's current competence, which the free solve count `on_solve_num` measures → hard-switch on a gate:
SFT (`n_verify,0,1`) only at `on_solve_num ≤ switch_gate` (the all-wrong, dead-RL-gradient case),
GRPO (`0,0,0`) when there is contrast, off-policy RL in the middle band — plain SFT for the stuck case
because off-policy RL's `π_ref ≡ 1` is biased → expecting MATH-500 to lift off the floor and AMC23 to
hold, watching AIME24 for a collapse back toward pure-SFT when nearly every prompt is all-wrong.
