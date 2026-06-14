The router is the whole point, but it bolts onto a fixed training stack, and the cleanest floor to
start from is the degenerate routing rule that throws the on-policy half of the stack away entirely —
because if I cannot beat *that*, no clever routing is worth building. The scaffold gives me, per prompt,
both an on-policy GRPO rollout group and a teacher demonstration `τ★`. The default fill keeps every
on-policy group and never touches the demonstration — that is pure GRPO, the on-policy extreme. The
opposite extreme is pure SFT: for *every* question, drop the on-policy attempts and replace them with
one off-policy SFT demonstration. That is the rule I start from, and it is the floor by construction —
not because SFT is bad, but because applying it *unconditionally* discards exactly the signal the
hybrid stack was built to use.

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
trace does not dominate the batch purely by length. Statistically this is maximum likelihood on expert
demonstrations: every observed token is a positive target, there is no comparison saying one answer
beats another, and there is no penalty on plausible answers absent from the demonstrations. That is the
shape of the strength and the ceiling at once.

It is worth being concrete about *why* "positive targets only" is a ceiling and not just a description,
because the whole ladder is an argument against this rule. A demonstration tells the model what a good
solution looks like; it says nothing about what a *bad* one looks like, and nothing about how good one
candidate is relative to another. So SFT cannot rank: if the model, left to itself, would produce two
different solutions to a prompt — one elegant and correct, one clumsy and correct — SFT has no machinery
to prefer the first, because neither appears in the demonstration set and the only signal is "match
`τ★`." Worse, it cannot push *down*: a confidently wrong solution the model loves to produce gets no
negative gradient under SFT, because SFT never evaluates the model's own outputs at all — it only ever
raises the likelihood of the teacher's tokens. The on-policy RL half of the hybrid stack is exactly the
thing that supplies graded, signed credit on the model's own behaviors, and pure SFT throws all of it
away. That is the structural reason this rung is the floor: not that the demonstrations are low quality,
but that the rule consuming them is the least informative one available — it uses one of the two signals
the stack provides and discards the other entirely, for every prompt regardless of need.

There is a second, subtler cost that bites specifically on long reasoning traces. Behavior cloning is
trained on the teacher's *on-distribution* states — the states the teacher visits along `τ★` — and it
optimizes nothing about the states the *model* visits after it makes a mistake. So the moment a generated
solution drifts off the teacher manifold (a wrong intermediate step, a misread of the problem), the model
is in a state the demonstration never covered, has been trained to do nothing sensible there, and has no
corrective signal pulling it back. On a short, formulaic answer this rarely matters; on a multi-step
competition solution, a single early divergence cascades unrecovered to a wrong final answer. RL, by
contrast, scores the model's *own* trajectories end to end, so it naturally trains recovery from the
model's own error states. Pure SFT has none of that — which is precisely the kind of failure I expect to
dominate on the hardest split.

There is a cleaner way to see why this rule sits at the floor, and it comes from the unified view the
hybrid stack is built around. Write one common objective — maximize expected verifier reward while
staying close to the demonstration policy `π_β`, `J_μ(θ) = E_{τ∼π_θ}[r(τ|q)] − μ·KL(π_β ‖ π_θ)`. Its
gradient splits into an on-policy RL term, `E_{τ∼π_θ}[r ∇log π_θ]`, plus a demonstration term,
`μ·E_{τ∼π_β}[∇log π_θ]` — and that second term is exactly the SFT gradient (the two minus signs from the
KL direction cancel to a plus). So SFT and RL are not two objectives; they are the two halves of the
gradient of one objective, distinguished only by which distribution the trajectory is sampled from and
what scalar weights it. Pure SFT is the limit that keeps *only* the demonstration half and zeroes the
reward half for every prompt. It is a valid ascent direction on `J_μ` — but a deliberately one-sided one,
which is why it is the floor: it never uses the on-policy half the hybrid stack exists to exploit.

Now write the controller this corresponds to. The contract returns `(on_remove_num, on_add_num,
off_add_num)`. "Drop all on-policy attempts" means `on_remove_num = rollout.n_verify` — remove all
eight rollouts this prompt generated. "Generate no extra on-policy rollouts" means `on_add_num = 0`.
"Replace them with one off-policy SFT demonstration" means `off_add_num = +1`; positive routes the
sample through the fixed `offline_loss_type=sft` actor, which scores it as the token-NLL above. And
crucially, this is unconditional: the controller ignores `on_solve_num` entirely — every question, easy
or hard, solved-eight-of-eight or solved-zero-of-eight, gets the identical copy-the-teacher treatment.
So the literal edit is the one-liner `del on_solve_num; return self.config.actor_rollout_ref.rollout.
n_verify, 0, 1`. There is no per-question state to maintain, no off-policy RL arm, no regeneration —
the simplest possible fill of the contract, and the only one that uses none of `on_solve_num`'s
information.

Reason about what this floor does end to end. Once the on-policy group is removed for every prompt, the
GRPO advantage code has nothing on-policy to normalize, and the actor only ever sees the demonstration
samples. So the hybrid stack collapses to supervised fine-tuning on `τ★`, scaled by `sft_loss_coef =
1.0`: pure behavior cloning of the teacher, run inside the hybrid harness. The one strength is
bootstrapping. On a hard prompt the 1.5B model fails completely — all eight rollouts wrong — SFT still
pulls up the likelihood of a correct solution with no exploration required; it is the only rule on the
ladder that can teach a capability the model never samples, which is precisely the all-wrong case where
on-policy RL has no learning signal at all. The ceiling is everything else. SFT sees positive
demonstrations only; it gets no signal about which of the model's *own* rollouts were good, it cannot
rank a better self-generated solution above a worse one, and because it is the same copy-the-teacher
update on *every* prompt — including the ones the model already solves eight-of-eight — it spends most
of the run on prompts that no longer need it. Worse, behavior cloning is blind to the model's own
off-trajectory mistakes: it never trains the recovery a model needs when its mid-solution state drifts
off the demonstration manifold, so on long competition solutions a single early divergence has nothing
pulling it back.

That last point is what I expect to bite on the held-out splits, and it bites unevenly. The three
benchmarks differ in how far the demonstration distribution sits from what the model must produce at
test time. MATH-500 is the broad, in-distribution split where the teacher traces should transfer best,
but it is also where pure imitation's narrowing hurts, because the model never practices its own
reasoning — it only learns to reproduce one fixed style of solution, and any test prompt whose solution
shape differs from the demonstrations is unsupported. AIME24 is 30 hard competition problems where the
1.5B model is weak regardless; here SFT's bootstrap should at least keep it off the floor, but copying a
teacher it cannot match leaves it unable to recover from its own mistakes mid-solution. AMC23 is the
small 40-example split where a handful of demonstration-shaped problems can move the score the most, so
I expect this is where unconditional imitation looks least bad.

So pure-SFT is the weakest routing rule I can run by construction: it discards the on-policy signal
entirely and treats every question identically, so it can only ever imitate, never sharpen or explore.
Whatever the precise per-split numbers, the diagnosis is already pointed at the next step, and the
diagnosis is sharp. The failure here is *not* that SFT is the wrong tool — it is that applying it
*unconditionally* is. The fix is to stop throwing the on-policy group away on questions where the model
is already learning from its own rollouts: keep the on-policy GRPO signal where there is reward
contrast, reserve the demonstration for the prompts where the model is genuinely stuck. But before I add
any conditioning, I need to measure the *opposite* extreme too — what does keeping all the on-policy
signal and never imitating buy me? — because the right router lives between these two floors, and I
cannot place it without both endpoints on the board. So the next rung is the other extreme: never
switch, pure GRPO.
