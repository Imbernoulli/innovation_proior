The pure-SFT floor told me exactly what unconditional imitation costs, and it told me in numbers.
Seed 42 landed at AIME24 0.062, AMC23 0.217, MATH-500 0.244. Read the split structure: AMC23 (0.217) is
the *least* embarrassing — the small 40-example split is where a handful of demonstration-shaped problems
moves the score, exactly as I expected — but MATH-500, the broad 500-example in-distribution split where
teacher traces should transfer *best*, sits at only 0.244, and AIME24 is down at 0.062. That low
MATH-500 is the tell. It is not that the teacher traces are bad; it is that copying them on *every*
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

That is the payoff for designing a router. If they are all estimators of one gradient, the design
question is how to *weight* the estimators — and the instinct from estimation theory is to average noisy
measurements, weighted toward the lower-variance ones. So maybe a fixed weighted blend of the SFT and RL
gradients? Wall. The bias and variance of each estimator are not fixed properties — they depend on the
*current* state of `π_θ` and on the prompt. On a prompt the model already solves, the on-policy RL signal
is informative and low-variance and the SFT term is pure overfitting pressure — exactly the imitation tax
the floor's MATH-500 number measured. On a prompt the model fails completely, the on-policy term carries
*no* information at all: when every rollout in the group gets the same reward, the group-relative
advantage `(R − mean)/std` is identically zero for that prompt, so GRPO contributes no gradient — and the
all-wrong case is the dangerous one, because the model still cannot solve the prompt. Which prompt is
which *changes as the model learns*. So a single global mixing coefficient — exactly what the floor's
unconditional rule and any fixed-blend method commits to — is structurally wrong, the same way one
global learning rate is wrong when different parameters need different scales. The "average the
estimators" idea is right in spirit; the constant weight was the mistake. The weighting must be a
function of how the model is *currently* doing on each prompt.

What signal do I have, per prompt, that measures that? The loop already samples `rollout.n_verify = 8`
on-policy rollouts per prompt and verifies them to compute the GRPO advantage. The solve count falls out
for free: `on_solve_num = Σ_i v(τ_i) ∈ {0,…,8}`, the number of the eight rollouts that passed the
verifier. That is exactly the model's current competence on this prompt, and it costs nothing extra —
the same verifier scores GRPO already consumes. So let the route be a function of `on_solve_num`.

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
on-policy RL. (The full controller is in the answer.)

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
