The router is the whole point, but it bolts onto a fixed training stack, and the cleanest floor to
start from is the degenerate routing rule that throws the on-policy half away entirely — because if I
cannot beat *that*, no clever routing is worth building. The scaffold gives me, per prompt, both an
on-policy GRPO rollout group and a teacher demonstration `τ★`. The default fill keeps every on-policy
group and never touches the demonstration — pure GRPO, the on-policy extreme. The opposite extreme is
pure SFT: for *every* question, drop the on-policy attempts and replace them with one off-policy SFT
demonstration. That is the floor by construction — not because SFT is bad, but because applying it
*unconditionally* discards exactly the signal the hybrid stack was built to use.

What is SFT actually doing to this model? The 1.5B backbone was pretrained by next-token prediction; its
native conditional is "continue the distribution I saw," not "given this math prompt, produce the worked
solution." SFT continues the same machinery on `(prompt → demonstration)` pairs, minimizing the
token-averaged NLL `L_SFT = −(1/|τ★|) Σ_t log π_θ(τ★_t | q, τ★_{<t})`. Two details are load-bearing and
the scaffold handles them: the loss is masked to the demonstration tokens (the model learns
`p(response | prompt)`, not `p(prompt)`), and the per-token losses are *averaged*, not summed —
otherwise a long teacher trace would dominate the batch by length alone, and on a competition mixture
trace lengths span an order of magnitude. `sft_loss_coef = 1.0` passes that averaged NLL in at unit
weight. This is maximum likelihood on expert demonstrations: every observed token is a positive target,
no comparison saying one answer beats another, no penalty on plausible answers absent from the
demonstrations. That is the strength and the ceiling at once.

The evaluation settings pin down how hard this floor pushes: `train_batch_size = 128 × total_training_steps
= 200 = 25600` prompt-visits, exactly the requested subset. So pure SFT here is *one epoch* of behavior
cloning — each teacher trace seen once, not hammered — which bounds both the damage and the benefit. It
also exposes the waste: the loop still draws `rollout.n_verify = 8` rollouts per prompt to compute the
solve count before the controller runs, so the run generates `128 × 8 × 200 ≈ 2×10⁵` rollouts, each up to
8192 tokens, and the controller throws all of them away. The floor pays the full sampling bill and consumes
none of it.

Why "positive targets only" is a ceiling: a demonstration says what a good solution looks like, nothing
about a bad one, and nothing about how good one candidate is relative to another. So SFT cannot rank and
cannot push *down* — a confidently wrong solution the model loves gets no negative gradient, because SFT
never evaluates the model's own outputs at all. Both failures are the same missing object: no signed,
graded credit on the model's own behaviors. Make it quantitative through the per-token gradient coefficient
on `∇log π_θ`. For SFT it is identically `1` on every demonstration token, independent of how the model is
doing: a final-answer token already at `π_θ = 0.9` still gets coefficient `1`, spending gradient to raise a
token that is essentially correct; a token wrong at `π_θ = 0.01` gets the *same* `1`. A signed, graded
signal would do the opposite — near-zero push on the already-correct token, a large push on the wrong one,
a *negative* push on a confidently-wrong token the teacher never wrote. The on-policy RL half of the stack
is exactly what supplies that, and pure SFT throws all of it away, for every prompt regardless of need.

A second cost bites specifically on long reasoning traces. Behavior cloning optimizes nothing about the
states the *model* visits after a mistake, so the moment a generated solution drifts off the teacher
manifold, the model is in a state the demonstration never covered, with no corrective gradient pulling it
back. Model it crudely: per-step probability `ε` of a step the teacher would not take, solution `L` steps
long, so staying on-manifold the whole way has probability `(1−ε)^L`. Short formulaic answer, `L ≈ 10`,
`ε ≈ 0.05`: `0.95^10 ≈ 0.60` — rarely bites. Competition-length chain, `L = 40`: `0.95^40 ≈ 0.13` at
`ε = 0.05`, `0.9^40 ≈ 0.015` at `ε = 0.1`. So on a long solution the chance of never diverging is near zero,
and a single early divergence cascades unrecovered. RL scores the model's own trajectories end to end, so it
trains recovery from the model's own error states; SFT has none of that, which is the failure I expect to
dominate where traces are longest.

There is a cleaner way to see why this rule sits at the floor, and it is the frame the whole ladder turns
on. Write one objective — maximize expected verifier reward while staying close to the demonstration policy
`π_β`: `J_μ(θ) = E_{τ∼π_θ}[r(τ|q)] − μ·KL(π_β ‖ π_θ)`. The reward term, by the score-function identity, is
`E_{τ∼π_θ}[r ∇log π_θ]` — REINFORCE, the on-policy reward gradient. For the KL term `π_β` is fixed in `θ`,
so `∇KL = −E_{τ∼π_β}[∇log π_θ]` and `−μ·∇KL = +μ E_{τ∼π_β}[∇log π_θ]` — exactly the maximum-likelihood
gradient, pushing up the demonstration tokens. So

  ∇J_μ = E_{τ∼π_θ}[r ∇log π_θ]  +  μ E_{τ∼π_β}[∇log π_θ].

The score has zero mean, so the reward term only *reshapes* probability by reward contrast among the model's
own samples — no net push. The demonstration term is the only one whose sampling measure does not depend on
`θ`, so it alone adds unconditional upward pressure on the teacher's tokens — the SFT gradient. SFT and RL
are not two objectives; they are the two halves of the gradient of one objective, differing only in which
distribution the trajectory comes from. Pure SFT keeps *only* the demonstration half and zeroes the reward
half for every prompt — a valid but deliberately one-sided ascent direction, which is why it is the floor.

The controller that realizes this returns `(on_remove_num, on_add_num, off_add_num)`. Drop all on-policy
attempts is `on_remove_num = rollout.n_verify`; generate no extra is `on_add_num = 0`; replace with one
off-policy SFT demonstration is `off_add_num = +1` (positive routes through the fixed `offline_loss_type=sft`
actor). Why `+1` not `+8` to match the eight removed? The demonstration is one trace and the SFT loss is a
token average — adding it eight times just multiplies its gradient by eight, i.e. `sft_loss_coef = 8` in
disguise, and I have no reason to upweight imitation eightfold. Why remove the on-policy group rather than
keep it alongside the demonstration? A blend is already a router with an opinion, not a floor; an endpoint
has to measure one signal in isolation. Once `on_remove_num = 8` fires everywhere, the GRPO advantage code
receives empty groups and contributes nothing, the actor sees only the demonstration as prefix-masked
token-NLL, and the hybrid stack collapses exactly to SFT on `τ★` at `sft_loss_coef = 1.0`. It ignores
`on_solve_num` entirely — every question, easy or hard, gets the identical treatment:
`del on_solve_num; return self.config.actor_rollout_ref.rollout.n_verify, 0, 1`.

The floor's one strength is bootstrapping: on a hard prompt where all eight rollouts fail, SFT still pulls up
the likelihood of a correct solution with no exploration — the only rule that can teach a capability the
model never samples, precisely the all-wrong case where on-policy RL has no signal. Everything else is
ceiling: the same copy-the-teacher update on *every* prompt, including the ones already solved
eight-of-eight, where the demonstration term is pure narrowing pressure. I can size the waste without
measurements: if the backbone can partly handle even a third of the mixture, that is `≈ 43` of 128 prompts
each step getting a redundant narrowing update; if half, `64`. Over one epoch, tens of thousands of
prompt-updates overwriting reasoning the model already had. The floor cannot tell these prompts from the
genuinely stuck ones, because that requires conditioning on `on_solve_num`, which this rule refuses to read.

That is what I expect to bite on the held-out splits, unevenly. MATH-500, the broad in-distribution split,
is where teacher traces should transfer best but also where pure imitation's narrowing hurts, since the
model never practices its own reasoning; being the largest split (500 examples) it is where I can read real
signal rather than seed noise — a point is five problems, versus AIME24 where one problem is `≈ 0.033`,
inside the noise. AIME24 is 30 hard problems where the 1.5B model is weak regardless; SFT's bootstrap keeps
it off the absolute floor, but copying a teacher it cannot match leaves it unable to recover from its own
mistakes, so I expect a handful of problems at most. AMC23, the small 40-example split, is where a few
demonstration-shaped problems move the score most, so unconditional imitation should look least bad there
relative to what it could be.

So the failure is *not* that SFT is the wrong tool — it is that applying it *unconditionally* is; the fix is
to keep the on-policy signal where there is reward contrast and reserve the demonstration for genuinely
stuck prompts. But conditioning needs both endpoints measured, so the next thing to run is the other
extreme: never switch, pure GRPO.
