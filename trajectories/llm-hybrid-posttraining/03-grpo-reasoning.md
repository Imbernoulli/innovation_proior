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

Convert the deltas to problem-counts to see how lopsided this is. AMC23 went `0.217 → 0.325`, which on 40
problems is `8.7 → 13.0`, a real `+4.3` problem-equivalents — a genuine, sizeable win. MATH-500 went
`0.244 → 0.250`, which on 500 problems is `122 → 125`, `+3` out of five hundred: that is a fraction of a
percent, indistinguishable from run-to-run wobble and emphatically not the "lift off the floor" I promised
the switch would deliver on the broad split. AIME24 is `0.062 → 0.062`, `1.9 → 1.9` problem-equivalents,
a dead flat two-of-thirty. So the entire measurable benefit of conditioning the route landed on the 40-problem
split and none of it reached the 500-problem split. That asymmetry is not noise — MATH-500 is the split where
a `+3` would be trivially inside the noise band but a `+40` would be unmistakable, and I got `+3`. Whatever the
switch fixed, it was the thing that helps 40 hand-shaped problems and not the thing that helps 500 varied ones.

That points somewhere specific, and it is worth saying plainly before I design the next rung: the switch
helped on the small split where a few demonstrations dominate and stalled on the broad split where the
model's *own* reasoning has to carry. The reason is structural. The switch still routes a large fraction
of prompts to SFT — every all-wrong prompt — and on this weak 1.5B backbone over a broad OpenR1 mixture,
that fraction is high. Estimate it: a prompt the model solves with true per-rollout probability `p` gets
routed to SFT with probability `(1 − p)^8`, and over a broad mixture full of prompts the 1.5B model handles
only weakly — `p` in the `0.05`–`0.1` range — that is `0.66`–`0.43` of them handed to imitation every step.
So even after conditioning, something like half the batch is still paying the imitation tax; the switch
turned "100% SFT" into "roughly half SFT," which is why it dented the tax without removing it. Each of those prompts pays the imitation tax: the model copies the teacher instead
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
at every intermediate token of a long reasoning chain from a single end-of-sequence bit. Put the numbers on
it. The critic is policy-sized, so a 1.5B critic shadowing the 1.5B actor roughly doubles the trainable
footprint on the same 4×H200 box, and it has to be optimized in lockstep with the actor. And the regression
it is being asked to solve is `V(s_t)` at up to `max_response_length = 8192` intermediate positions from a
single supervising bit at the end of the sequence — one label to pin eight thousand intermediate values. When
`V` is badly fit, GAE's `δ_t = r_t + γV(s_{t+1}) − V(s_t)` is a difference of two noisy value estimates at
adjacent tokens, so the noise does not average out; it feeds biased, high-variance advantages straight into
the policy. So I am paying for a policy-sized network, doubling the memory, whose central job — accurate
per-token values — is precisely the job the last-token-only reward makes impossible. That is the wall: I want
PPO's stable clipped update and a variance-reducing baseline, but not this learned per-token critic.

So the question sharpens: can I get a baseline — an approximation to "the default value of this state" —
without learning `V`? A baseline only has to be (i) independent of the action being scored and (ii) close
to the expected return from here, to cut variance; the subtraction is free in expectation
(`E_a[∇log π(a|s) · b(s)] = b(s) ∇_θ Σ_a π = 0`). What do I already have lying around? For each prompt I
do not sample one solution — I draw a *group* of `rollout.n_verify = 8` rollouts from the rollout policy.
Each gets a verifier reward `R_i`. The empirical mean `mean(R_1,…,R_8)` is exactly a Monte-Carlo estimate
of the question's response-level value under the rollout policy — the very quantity `V` was trying to
approximate, at the only granularity the reward actually exists. It costs nothing extra; the samples are
already drawn. Use the group mean as the baseline. There is one honesty check I owe before trusting it: a
baseline has to be *independent of the action being scored*, and `R_i` is one of the terms in its own mean,
so the group mean is not strictly independent. The clean version would be a leave-one-out mean, the average
of the other seven rewards. How much does using the full mean cost? A line of algebra gives
`m − m_{−i} = (R_i − m)/(G−1)`. Take the worst case for a group I actually care about: one of eight rollouts
correct, `R = [1,0,0,0,0,0,0,0]`, so `m = 0.125`; for the correct rollout the full mean is `0.125` and the
leave-one-out mean is `0/7 = 0`, a gap of `0.125 = (1−0.125)/7`, and for a failing rollout the gap is
`(0−0.125)/7 = −0.018`. So the self-inclusion bias is bounded by the reward range over `G−1` and shrinks like
`1/G`; at `G = 8` it is a fraction of the advantage scale, and it gets swallowed by the normalization I am
about to divide by anyway. The full mean is not a theorem-clean baseline, but the gap is small, controllable,
and depends only on the question and its sampled group — not on any token position inside `R_i`. And it is not
just cheap — it is the *right* baseline, because the reward here is comparative (correct relative to other
attempts on the same question), so a per-question relative baseline matches the signal's nature.

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
supervision, the simplest thing consistent with the information I have. The tempting alternative is to
allocate credit *within* a correct solution — reward the pivotal deduction more than the boilerplate — but
that needs a per-step signal I do not have: the verifier scores the whole response `{0,1}` and says nothing
about which token mattered. Inventing a within-sequence gradient from a single terminal bit is exactly the
critic's impossible regression I already rejected; doing it implicitly with a heuristic would be the same
fabrication in cheaper clothes. So the honest choice is to give every token of a response the same
response-level advantage and let many prompts and many rollouts average out which tokens actually carry the
credit. It also means the per-token gradient coefficient is uniform within a response and equal to `Â_i` — a
real-valued, signed number — which is exactly the graded reinforcement the demonstration-copying floor
lacked: the `+1.62` rollout is pushed harder than a `+0.5` one, and the `−0.54` rollouts are pushed *down*,
rather than everything getting the SFT coefficient `1`.

Keep PPO's clipped surrogate for the update — it is the stable part, no reason to abandon it. With the
per-token ratio `ρ_{i,t} = π_θ / π_{θ_old}`, the objective is the group-averaged
`min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t})`, a trust region as a flat spot in the loss. Trace
what the `min`-of-clip actually does so I know it is not silently discarding signal. For a positive-advantage
token the model wants to push `ρ` up; once `ρ` exceeds `1 + ε` the clipped branch `(1+ε)Â` is the smaller of
the two and its derivative in `ρ` is zero, so the update stops rewarding further increases past the trust
radius — a stop-gradient on an already-overshot token. For a negative-advantage token the model wants to push
`ρ` down; once `ρ` drops below `1 − ε` the clip flattens again. In both cases the clip does not change *which*
direction a token moves, only refuses to chase a token that has already left the trust region around
`π_{θ_old}` — which is exactly the stabilization I want and exactly why I keep it rather than bare REINFORCE. The
KL anchor to the reference is moved out of the reward and added directly to the loss as its own term
(folding it into the reward and then z-scoring would entangle the regularizer with the advantage), with
the k3 per-token estimator `π_ref/π_θ − log(π_ref/π_θ) − 1`. I claimed that is "unbiased and always
non-negative," and both properties matter enough that I should check them rather than assert, because the
obvious one-sample estimator `log(π_θ/π_ref)` fails the second. Take a three-symbol case, `π_θ = (0.5,0.3,0.2)`
and `π_ref = (0.2,0.3,0.5)`, true `KL(π_θ‖π_ref) = 0.5 ln 2.5 + 0.2 ln 0.4 = 0.2749`. The naive per-symbol
estimator `k1 = log(π_θ/π_ref) = (+0.916, 0, −0.916)` has a *negative* entry — a per-token "penalty" that
momentarily rewards divergence — even though the true KL is non-negative; its expectation under `π_θ` is
`0.5(0.916) + 0.2(−0.916) = 0.2749`, unbiased but signed. Now `k3` with `u = π_ref/π_θ = (0.4,1,2.5)`:
`(u − 1) − log u = (0.316, 0, 0.584)`, every entry non-negative (since `log u ≤ u − 1`), and its expectation
`0.5(0.316) + 0.2(0.584) = 0.2749` lands on the same true KL. So k3 is unbiased like the naive form but never
swings negative per token — a control-variate trade that also cuts variance, since it removes k1's large
negative excursion. That is the estimator I want anchoring the loss. That is GRPO: PPO's machinery with a free
group-relative baseline standing in for the GAE advantage, no value network anywhere.

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
solution for will get no signal. Quantify the loss where it bites hardest. On AIME24 the floor's `0.062`
translated to a per-rollout solve probability of roughly `0.06` on the average problem, and a prompt at
`p = 0.06` fails all eight rollouts with probability `(1 − 0.06)^8 ≈ 0.61` — so on the order of six in ten
AIME prompts hand GRPO an all-equal group and therefore *exactly zero* gradient, every step. Those prompts
are invisible to pure RL: it can only sharpen the minority where the model already samples a correct
solution occasionally. So I do not expect AIME24 to move off `0.062` by much; pure RL cannot bootstrap a
capability the model never produces, and more than half of AIME is exactly that. But I *gain* the entire MATH-500 mass back: every prompt
with reward contrast — and on the broad in-distribution split there are many, since the 1.5B math model
can partly solve a large fraction of MATH-500 — now learns from its own graded, signed rollouts instead
of copying a teacher, with no narrowing and full practice of its own reasoning.

Reading the switch's shape, the falsifiable claims are sharp and they are on the broad split. If the
imitation tax is what capped MATH-500 at 0.250, then removing it entirely should lift MATH-500
*substantially* above 0.250 — this is the headline prediction and the largest split, so it is where the
signal is cleanest. And "substantial" has a size: the switch moved MATH-500 by `+3` problem-equivalents,
which was noise; for the imitation-tax story to be right, killing the tax should move it by something an
order of magnitude larger — tens of problems out of 500, a jump of `0.10`–`0.15` in score, not a wobble.
If MATH-500 instead lands within a few problems of `0.250` again, the imitation tax was *not* what capped it
and my whole diagnosis of the switch is wrong. That is the falsifiable version. AIME24 I expect to stay near 0.062, because pure RL has no bootstrap and the model
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
