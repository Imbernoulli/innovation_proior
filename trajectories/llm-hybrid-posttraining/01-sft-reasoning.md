The router is the whole point, but it bolts onto a fixed training stack, and the cleanest floor to
start from is the degenerate routing rule that throws the on-policy half of the stack away entirely —
because if I cannot beat *that*, no clever routing is worth building. The scaffold gives me, per prompt,
both an on-policy GRPO rollout group and a teacher demonstration `τ★`. The default fill keeps every
on-policy group and never touches the demonstration — that is pure GRPO, the on-policy extreme. The
opposite extreme is pure SFT: for *every* question, drop the on-policy attempts and replace them with
one off-policy SFT demonstration. That is the rule I start from, and it is the floor by construction —
not because SFT is bad, but because applying it *unconditionally* discards exactly the signal the
hybrid stack was built to use. I want two clean endpoints on the board before I let the router do
anything interesting in between, and this is the endpoint that uses the demonstration and nothing else.

Before I write the controller, let me be precise about what supervised fine-tuning is *doing* to this
model, because the whole ladder is a series of decisions about when to lean on it. The 1.5B backbone was
pretrained by next-token prediction on broad text; its native objective is "continue the distribution I
saw," which is the wrong conditional distribution for "given this math prompt, produce the worked
solution." SFT corrects that by continuing the same next-token machinery on `(prompt → expert
demonstration)` pairs. The likelihood of a demonstration under a causal LM factors autoregressively,
`π_θ(τ★ | q) = Π_t π_θ(τ★_t | q, τ★_{<t})`, and maximizing it is minimizing the token-averaged negative
log-likelihood `L_SFT = −(1/|τ★|) Σ_t log π_θ(τ★_t | q, τ★_{<t})`. Two details are load-bearing and the
scaffold already handles them: the prompt tokens are conditioning context, not targets — the loss is
masked to the demonstration tokens only, so the model learns `p(response | prompt)` and not `p(prompt)` —
and the per-token losses are *averaged* over valid demonstration tokens, not summed, so a long teacher
trace does not dominate the batch purely by length.

That averaging is worth making concrete, because on a competition-math mixture the trace lengths vary by
an order of magnitude and if I got this wrong the batch would silently become a length-weighted objective.
Take two demonstrations in the same batch, one a terse 200-token AMC-style answer and one an 1800-token
worked AIME solution. Under a *summed* loss each contributes `Σ_t (−log π_θ)`, so the long trace supplies
roughly `1800/200 = 9×` the gradient mass of the short one purely because it has nine times as many tokens
— the optimizer would spend most of its step imitating whichever demonstrations happen to be verbose. Under
the token *average* each contributes a per-token mean, so both land at the same scale regardless of length
and the batch is an unweighted average over demonstrations, which is what I want: every prompt's teacher
gets an equal vote. The scaffold's `sft_loss_coef = 1.0` then just passes that averaged NLL straight into
the update at unit weight. Statistically this is maximum likelihood on expert demonstrations: every observed
token is a positive target, there is no comparison saying one answer beats another, and there is no penalty
on plausible answers absent from the demonstrations. That is the shape of the strength and the ceiling at
once.

There is one more thing the evaluation settings pin down that I should read off before I judge how hard
this floor pushes. Training runs `train_batch_size = 128` for `total_training_steps = 200`, which is
`128 × 200 = 25600` prompt-visits — exactly the requested subset size. So pure SFT here is *one epoch* of
behavior cloning: each teacher trace is seen essentially once, not hammered for dozens of passes. That
bounds the damage in one direction (a single-epoch BC pass cannot memorize the demonstration set the way a
long SFT run would) and the benefit in the other (one exposure to `τ★` is a gentle nudge, not a deep
rewrite). It also exposes the raw waste of this rule: the loop still draws `rollout.n_verify = 8` on-policy
rollouts per prompt to compute the solve count before the controller runs, so across the run I generate
`128 × 8 × 200 ≈ 2.05 × 10⁵` rollouts — each up to `max_response_length = 8192` tokens — and then the
controller throws *all* of them away. The floor pays the full on-policy sampling bill and consumes none of
it. That is not an argument against the floor; it is the sharpest statement of what "unconditional" costs.

It is worth being concrete about *why* "positive targets only" is a ceiling and not just a description,
because the whole ladder is an argument against this rule. A demonstration tells the model what a good
solution looks like; it says nothing about what a *bad* one looks like, and nothing about how good one
candidate is relative to another. So SFT cannot rank: if the model, left to itself, would produce two
different solutions to a prompt — one elegant and correct, one clumsy and correct — SFT has no machinery
to prefer the first, because neither appears in the demonstration set and the only signal is "match
`τ★`." Worse, it cannot push *down*: a confidently wrong solution the model loves to produce gets no
negative gradient under SFT, because SFT never evaluates the model's own outputs at all — it only ever
raises the likelihood of the teacher's tokens. Put the two failures side by side and they are the same
missing object: SFT has no *signed, graded* credit on the model's own behaviors. The on-policy RL half of
the hybrid stack is exactly the thing that supplies that — a per-attempt advantage that is positive for the
good rollouts and negative for the bad — and pure SFT throws all of it away. That is the structural reason
this rung is the floor: not that the demonstrations are low quality, but that the rule consuming them is the
least informative one available — it uses one of the two signals the stack provides and discards the other
entirely, for every prompt regardless of need.

I can make the "least informative rule" claim quantitative by looking at what multiplies `∇log π_θ` on each
token — the per-token gradient coefficient. For SFT that coefficient is identically `1` on every demonstration
token: the update is `Σ_t 1 · ∇log π_θ(τ★_t)`, a uniform push-up with no dependence on how the model is already
doing. Watch what that costs on a prompt the model nearly solves. Suppose the key final-answer token of `τ★`
already has model probability `π_θ = 0.9`; SFT still applies coefficient `1` and spends gradient raising a token
that is essentially already correct. On the same prompt a token the model gets wrong sits at `π_θ = 0.01`, and
SFT applies the *same* coefficient `1` — no prioritization between the token that needs the push and the one that
does not. A signed, graded signal would do the opposite: near-zero push on the already-correct token, a large
push on the wrong one, and a *negative* push on a confidently-wrong token the model likes but the teacher never
wrote. SFT has none of that structure — coefficient `1` everywhere, sign always positive, magnitude never a
function of the model's state — which is the precise sense in which it is the least informative consumer of the
demonstration, and the precise reason a whole epoch of it on already-solved prompts is gradient spent on tokens
that did not need it.

There is a second, subtler cost that bites specifically on long reasoning traces, and I can put a number on
how fast it bites. Behavior cloning is trained on the teacher's *on-distribution* states — the states the
teacher visits along `τ★` — and it optimizes nothing about the states the *model* visits after it makes a
mistake. So the moment a generated solution drifts off the teacher manifold (a wrong intermediate step, a
misread of the problem), the model is in a state the demonstration never covered, has been trained to do
nothing sensible there, and has no corrective signal pulling it back. Model this crudely: say the cloned
policy has a per-step probability `ε` of taking a step the teacher would not, and a solution is `L` reasoning
steps long. The chance it stays on the teacher manifold the whole way is `(1−ε)^L`, and once it leaves,
there is no recovery gradient. For a short, formulaic answer — `L ≈ 10`, `ε ≈ 0.05` — that is `0.95^10 ≈
0.60`, so more than half the generations land on the demonstration manifold and the failure rarely bites.
But push to a competition-length chain: `L = 40` at the same `ε = 0.05` gives `0.95^40 ≈ 0.13`, and at
`ε = 0.1` it collapses to `0.9^40 ≈ 0.015`. So on a long solution the probability of never once diverging is
near zero — a single early divergence cascades unrecovered to a wrong final answer, and it happens on almost
every hard generation. RL, by contrast, scores the model's *own* trajectories end to end, so it naturally
trains recovery from the model's own error states. Pure SFT has none of that — which is precisely the kind of
failure I expect to dominate on the hardest split, where the traces are longest and `ε` is largest.

There is a cleaner way to see why this rule sits at the floor, and it comes from the unified view the
hybrid stack is built around. Write one common objective — maximize expected verifier reward while
staying close to the demonstration policy `π_β`, `J_μ(θ) = E_{τ∼π_θ}[r(τ|q)] − μ·KL(π_β ‖ π_θ)`, `μ ≥ 0`.
Take its gradient term by term, carefully, because the sign on the KL is exactly the kind of thing I would
wave through and get wrong. The reward term, by the score-function identity
`∇_θ E_{τ∼π_θ}[f] = E_{τ∼π_θ}[f ∇log π_θ]` (which follows from `∇π_θ = π_θ ∇log π_θ`), is
`E_{τ∼π_θ}[r ∇log π_θ]` — REINFORCE, the on-policy reward policy gradient. The KL term I expand as
`KL(π_β ‖ π_θ) = E_{τ∼π_β}[log π_β − log π_θ]`; `π_β` is fixed in `θ` and the expectation is over `π_β`,
also fixed, so I differentiate only the `−log π_θ` inside, giving `∇KL = −E_{τ∼π_β}[∇log π_θ]`, and the
contribution to `∇J_μ` is `−μ·∇KL = +μ E_{τ∼π_β}[∇log π_θ]`. The two minus signs cancel to a plus, and
`E_{τ∼π_β}[∇log π_θ]` is exactly the maximum-likelihood gradient — push up the log-prob of the demonstration
tokens. So

  ∇J_μ = E_{τ∼π_θ}[r ∇log π_θ]  +  μ E_{τ∼π_β}[∇log π_θ].

The one place I want a check rather than a claim is that this demonstration term really is unconditional
upward pressure on the teacher's tokens and not something the reward term already supplies. The corollary I
keep leaning on, `E_{τ∼π_θ}[∇log π_θ] = ∇_θ Σ π_θ = ∇_θ 1 = 0`, says the score has zero mean, so the reward
term can only *reshape* probability by reward contrast — it moves mass between the model's own samples and
adds no net push. The demonstration term is the only one with a fixed sampling measure `π_β` that does not
depend on `θ`, so it adds unconditional upward pressure on specific tokens — the teacher's — and that is
exactly the SFT gradient. So SFT and RL are not two objectives; they are the two halves of the gradient of
one objective, distinguished only by which distribution the trajectory is sampled from and what scalar weights
it. Pure SFT is the limit that keeps *only* the demonstration half and zeroes the reward half for every prompt.
It is a valid ascent direction on `J_μ` — but a deliberately one-sided one, which is why it is the floor: it
never uses the on-policy half the hybrid stack exists to exploit. (I keep "within `J_μ`" as the caveat — this
legitimacy is relative to the KL-regularized objective I just wrote; a different regularizer would give a
different split. *Which* estimator each half is, and how to weight them, is not something this rung needs; here
I only need that pure SFT is one half held alone.)

Now write the controller this corresponds to, and check that the harness actually collapses to plain SFT
when I do. The contract returns `(on_remove_num, on_add_num, off_add_num)`. "Drop all on-policy attempts"
means `on_remove_num = rollout.n_verify` — remove all eight rollouts this prompt generated. "Generate no
extra on-policy rollouts" means `on_add_num = 0`. "Replace them with one off-policy SFT demonstration" means
`off_add_num = +1`; positive routes the sample through the fixed `offline_loss_type=sft` actor, which scores
it as the token-NLL above. Why exactly `+1` and not, say, `+8` to match the eight rollouts I removed? Because
the demonstration is a single teacher trace and the SFT loss is a token *average* — adding the same trace
eight times would multiply its gradient by eight, which is just `sft_loss_coef = 8` in disguise, and I have no
reason to upweight imitation eightfold; one copy at unit coefficient is the clean floor. And why remove the
on-policy group rather than keep it *and* add the demonstration? Because keeping both would blend the two
signals on every prompt, and the whole purpose of an endpoint is to measure one signal in isolation — a blend
is not a floor, it is already a router with an opinion. So the endpoint has to be pure. Trace what the harness
then does: once `on_remove_num = 8` fires for every prompt, the GRPO advantage code — which group-normalizes
over on-policy samples only — receives an empty on-policy group, produces no advantages, and contributes no
policy-gradient term; the actor sees only the demonstration samples and scores each as the prefix-masked
token-NLL. The hybrid stack has collapsed, exactly, to supervised fine-tuning on `τ★` at `sft_loss_coef = 1.0`
— pure behavior cloning run inside the harness. And crucially this is unconditional: the controller ignores
`on_solve_num` entirely — every question, easy or hard, solved-eight-of-eight or solved-zero-of-eight, gets
the identical copy-the-teacher treatment. So the literal edit is the one-liner `del on_solve_num; return
self.config.actor_rollout_ref.rollout.n_verify, 0, 1`. There is no per-question state to maintain, no
off-policy RL arm, no regeneration — the simplest possible fill of the contract, and the only one that uses
none of `on_solve_num`'s information.

Reason about what this floor does end to end. The one strength is bootstrapping. On a hard prompt the 1.5B
model fails completely — all eight rollouts wrong — SFT still pulls up the likelihood of a correct solution
with no exploration required; it is the only rule on the ladder that can teach a capability the model never
samples, which is precisely the all-wrong case where on-policy RL has no learning signal at all. The ceiling
is everything else. SFT sees positive demonstrations only; it gets no signal about which of the model's *own*
rollouts were good, it cannot rank a better self-generated solution above a worse one, and because it is the
same copy-the-teacher update on *every* prompt — including the ones the model already solves eight-of-eight —
it spends its single epoch on prompts that no longer need it. Every one of those already-solved prompts is a
prompt where the demonstration term is pure overfitting pressure: it narrows the policy toward one fixed
solution style and never lets the model practice its own reasoning. Worse, behavior cloning is blind to the
model's own off-trajectory mistakes: it never trains the recovery a model needs when its mid-solution state
drifts off the demonstration manifold, so on long competition solutions a single early divergence — which the
`(1−ε)^L` estimate said is near-certain — has nothing pulling it back.

I can put a rough size on the wasted-epoch effect even without any measurements, just from how the batch is
built. Each step processes 128 prompts, and the fraction of them the model can already solve at least once in
its eight rollouts is exactly the fraction where the demonstration is redundant. I do not know that fraction —
that is what `on_solve_num` would tell me if I read it, which this rule refuses to do — but I can bracket it.
If the 1.5B math backbone can partly handle even a third of the broad mixture, that is `≈ 43` of the 128 prompts
each step receiving a copy-the-teacher update whose only effect is to narrow them toward the demonstration; if
it is half, `64` of 128. Over the one epoch, that is tens of thousands of prompt-updates spent overwriting
reasoning the model could already do with imitation it did not need. The floor cannot tell these prompts apart
from the genuinely stuck ones, because telling them apart requires conditioning on `on_solve_num`, and the whole
definition of this rung is that it does not. That single number — how many of the eight rollouts passed — is
sitting unused in the loop, and the entire argument of the next rung is that reading it is the fix.

That last point is what I expect to bite on the held-out splits, and it should bite unevenly. The three
benchmarks differ in how far the demonstration distribution sits from what the model must produce at test
time. MATH-500 is the broad, in-distribution split where the teacher traces should transfer best, but it is
also where pure imitation's narrowing hurts, because the model never practices its own reasoning — it only
learns to reproduce one fixed style of solution, and any test prompt whose solution shape differs from the
demonstrations is unsupported. Because MATH-500 is the largest split (500 examples), it is also where I can
read a real signal rather than seed noise: on the 30-example AIME24 split a single problem moving is worth
`1/30 ≈ 0.033`, so anything at that granularity is inside the noise, whereas on MATH-500 a point of score is
five solved problems and the number is stable. AIME24 is 30 hard competition problems where the 1.5B model is
weak regardless; here SFT's bootstrap should at least keep it off the absolute floor, but copying a teacher it
cannot match leaves it unable to recover from its own mistakes mid-solution, so I expect AIME24 to sit near the
bottom, a handful of problems' worth of score at most. AMC23 is the small 40-example split where a handful of
demonstration-shaped problems can move the score the most, so I expect this is where unconditional imitation
looks least bad relative to what it could be.

So pure-SFT is the weakest routing rule I can run by construction: it discards the on-policy signal entirely
and treats every question identically, so it can only ever imitate, never sharpen or explore. Whatever the
precise per-split numbers, the diagnosis is already pointed at the next step, and the diagnosis is sharp. The
failure here is *not* that SFT is the wrong tool — it is that applying it *unconditionally* is. The fix is to
stop throwing the on-policy group away on questions where the model is already learning from its own rollouts:
keep the on-policy GRPO signal where there is reward contrast, reserve the demonstration for the prompts where
the model is genuinely stuck. But before I add any conditioning, I need to measure the *opposite* extreme too —
what does keeping all the on-policy signal and never imitating buy me? — because the right router lives between
these two floors, and I cannot place it without both endpoints on the board. So the next rung is the other
extreme: never switch, pure GRPO.
