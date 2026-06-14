The HPT switch moved exactly the split I predicted it would, and exactly *not* the one I worried about —
and that contrast is the whole diagnosis. Seed 42: AIME24 0.062, AMC23 0.325, MATH-500 0.250. AMC23 is
the clear win — 0.325, up from the floor's 0.217 and the best AMC23 on the board so far — because
conditioning the switch let the demonstration-shaped problems still get bootstrapped while the model
kept practicing the ones it could partly do. But the two numbers I was watching barely moved. MATH-500
went from 0.244 to 0.250, essentially flat — the lift "off the floor" I expected from keeping contrast
prompts on RL did *not* materialize at scale. And AIME24 stayed pinned at 0.062, exactly the pure-SFT
value, which is the tell I flagged: on hard competition problems the 1.5B model fails all eight rollouts
on nearly every prompt, so the gate fires SFT on almost all of them and the hybrid switch collapses back
toward the pure-SFT behavior I was trying to escape. So the switch traded the floor's AMC23 weakness for
an AMC23 win, but it did *not* fix the imitation tax on the broad MATH-500 split, and on AIME24 it bought
nothing.

That points somewhere specific, and it is worth saying plainly before I design the next rung: the switch
helped on the small split where a few demonstrations dominate and stalled on the broad split where the
model's *own* reasoning has to carry. The reason is structural. The switch still routes a large fraction
of prompts to SFT — every all-wrong prompt — and on this weak 1.5B backbone over a broad OpenR1 mixture,
that fraction is high. Each of those prompts pays the imitation tax: the model copies the teacher instead
of practicing, narrows toward the demonstration distribution, and learns no recovery from its own
mistakes. The hybrid switch reduced *how often* I pay the tax compared to pure SFT, but it did not
eliminate it, and on MATH-500 — where success depends on the model generalizing its own reasoning across
500 varied prompts rather than reproducing teacher traces — the residual tax is enough to keep the score
flat at 0.250. So the natural question for this rung is the opposite extreme from the floor: what does
*never* imitating buy me? If the imitation tax is the thing capping MATH-500, then the rule that pays it
*zero* times — pure on-policy RL, the demonstration never touched — is the experiment that isolates it.

So derive that rule from scratch, because it is the other endpoint I need on the board. The pure-RL move
is to keep every prompt's on-policy GRPO group and never switch, which collapses the hybrid stack to
plain GRPO. Let me be clear about what GRPO *is* and why it is the right on-policy learner here, not just
the default — because the reason it works is the reason it should beat the switch on MATH-500.

Start from what hurts in RL-fine-tuning one of these math models. The reward is the verifier: it reads
the *entire* solution and emits *one* number (correct/incorrect), effectively at the last token, with
nothing in between. The classical actor-critic recipe — PPO — would train a separate value network
`V_ψ`, policy-sized, to smear that terminal scalar backward into per-token credit, and feed it through
GAE: `δ_t = r_t + γV(s_{t+1}) − V(s_t)`, `Â_t = Σ_l (γλ)^l δ_{t+l}`. Stare at GAE and the dependence is
glaring — *every* term needs an accurate `V` at *every* token. But that is exactly the thing the
last-token-only reward makes hardest: I would be asking a 1.5B-sized critic to learn an accurate value
at every intermediate token of a long reasoning chain from a single end-of-sequence bit. When `V` is
badly fit, GAE feeds biased, noisy advantages straight into the policy. So I am paying for a policy-sized
network whose central job is precisely the job the reward makes impossible. That is the wall: I want
PPO's stable clipped update and a variance-reducing baseline, but not this learned per-token critic.

So the question sharpens: can I get a baseline — an approximation to "the default value of this state" —
without learning `V`? A baseline only has to be (i) independent of the action being scored and (ii) close
to the expected return from here, to cut variance; the subtraction is free in expectation
(`E_a[∇log π(a|s) · b(s)] = b(s) ∇_θ Σ_a π = 0`). What do I already have lying around? For each prompt I
do not sample one solution — I draw a *group* of `rollout.n_verify = 8` rollouts from the rollout policy.
Each gets a verifier reward `R_i`. The empirical mean `mean(R_1,…,R_8)` is exactly a Monte-Carlo estimate
of the question's response-level value under the rollout policy — the very quantity `V` was trying to
approximate, at the only granularity the reward actually exists. It costs nothing extra; the samples are
already drawn. Use the group mean as the baseline. And it is not just cheap — it is the *right* baseline,
because the reward here is comparative (correct relative to other attempts on the same question), so a
per-question relative baseline matches the signal's nature.

The spread of `R_i` differs wildly across questions: an easy prompt where all eight pass has rewards near
the top and tiny variance, a hard one splays them across {0,1}. Using raw `R_i − mean` would let the
easy and hard prompts contribute advantages of incomparable scale. So normalize by the group standard
deviation too: `Â_i = (R_i − mean(R)) / (std(R) + ε)`, a per-question z-score. Scale-invariant across
questions, controls the magnitude so one high-variance prompt cannot dominate, and keeps the comparative
reading clean. (The `ε` floors the denominator; a degenerate group where all eight rewards are equal has
`std = 0`, floored, so the centered reward is just zero — *no gradient*. That is the same degeneracy the
switch was built to catch, and here, with no switch, I accept it: an all-wrong or all-correct prompt
simply contributes nothing on-policy.) With a single terminal reward there is no within-sequence signal,
so I broadcast the scalar advantage to every token of the response: `Â_{i,t} = Â_i` — outcome
supervision, the simplest thing consistent with the information I have.

Keep PPO's clipped surrogate for the update — it is the stable part, no reason to abandon it. With the
per-token ratio `ρ_{i,t} = π_θ / π_{θ_old}`, the objective is the group-averaged
`min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t})`, a trust region as a flat spot in the loss. The
KL anchor to the reference is moved out of the reward and added directly to the loss as its own term
(folding it into the reward and then z-scoring would entangle the regularizer with the advantage), with
the k3 per-token estimator `π_ref/π_θ − log(π_ref/π_θ) − 1`, which is unbiased and always non-negative.
That is GRPO: PPO's machinery with a free group-relative baseline standing in for the GAE advantage, no
value network anywhere.

Now make it concrete in this task's edit surface. The advantage code, the clipped actor loss, and the
group normalization are all *fixed* downstream — `adv_estimator=grpo` is frozen. The only thing I control
is the router, and "pure GRPO" is the router that never switches: keep every on-policy group, add nothing
off-policy. So the controller ignores `on_solve_num` and returns `(on_remove_num, on_add_num, off_add_num)
= (0, 0, 0)` for every prompt. The literal edit is `del on_solve_num; return 0, 0, 0`. No per-question
state, no off-policy arm, no SFT — the demonstration `τ★` is never touched, and the imitation tax is paid
exactly zero times. Note what the harness then does and does not give me: it gives me the clean GRPO
update on every prompt's eight rollouts, and it takes away any bootstrap on the all-wrong prompts —
those contribute zero gradient and the model is left to discover those solutions on its own or not at
all. Pure on-policy means pure exploration-bounded-by-the-base-model.

So the delta from the switch is the cleanest possible: where HPT fired SFT on every all-wrong prompt, I
fire it on none. The trade is explicit and it is exactly the experiment I want. I *lose* the bootstrap on
the genuinely stuck prompts — the all-wrong AIME-style problems the model cannot sample a correct
solution for will get no signal, so I do not expect AIME24 to move off 0.062 either; pure RL cannot
bootstrap a capability the model never produces. But I *gain* the entire MATH-500 mass back: every prompt
with reward contrast — and on the broad in-distribution split there are many, since the 1.5B math model
can partly solve a large fraction of MATH-500 — now learns from its own graded, signed rollouts instead
of copying a teacher, with no narrowing and full practice of its own reasoning.

Reading the switch's shape, the falsifiable claims are sharp and they are on the broad split. If the
imitation tax is what capped MATH-500 at 0.250, then removing it entirely should lift MATH-500
*substantially* above 0.250 — this is the headline prediction and the largest split, so it is where the
signal is cleanest. AIME24 I expect to stay near 0.062, because pure RL has no bootstrap and the model
cannot sample correct AIME solutions; if AIME24 *rises* anyway, that would mean the model could partly
solve more AIME prompts than I think and the switch was wasting them on SFT. AMC23 is the genuine risk:
the switch's AMC23 win (0.325) came precisely from bootstrapping demonstration-shaped problems, so pure
GRPO might *give some of that back* — if AMC23 drops below 0.325, that is the cost of never imitating, and
it tells me the small split genuinely needs the teacher even though the broad split does not. So the
expectation is a clean dissociation: MATH-500 up, AIME24 flat, AMC23 possibly down — the broad-split gain
from killing the imitation tax bought at the price of the small-split bootstrap. The leaderboard
direction will say which of those two — broad reasoning practice or small-split imitation — dominates the
final score on this backbone.

The causal chain in one breath: the HPT switch won AMC23 (0.325) but left MATH-500 flat (0.250) and
AIME24 pinned (0.062), because it still routes every all-wrong prompt to SFT and pays the residual
imitation tax on the broad split → isolate the tax by paying it zero times: pure on-policy GRPO, never
switch → GRPO is the right on-policy learner because it replaces PPO's impossible last-token critic with
a free group-mean baseline, z-scored per question, broadcast to every token, under PPO's clipped surrogate
and a k3 KL-in-the-loss → the router that realizes it is `(0, 0, 0)` for every prompt, demonstration never
touched → expecting MATH-500 to lift substantially (tax removed), AIME24 to stay flat (no bootstrap), and
AMC23 to possibly fall back from 0.325 (the small split's lost bootstrap) — a dissociation that says
whether broad reasoning practice or small-split imitation wins the final score.
