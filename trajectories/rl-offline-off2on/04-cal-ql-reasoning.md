IQL delivered exactly what I predicted, and reading its numbers tells me precisely what is left on the
table. Hammer-cloned climbed from AWAC's dead-flat 0.336 to a mean of 76.7 — and decisively, on *every*
seed (69.0 / 68.4 / 92.7), not one lucky run. The in-support expectile backup did the stitching AWAC
could not: the value function propagated the rare good fragments through the noise and the
advantage-weighted policy found the manipulation behavior. Pen-cloned rose to 103.0 (131.4 / 76.3 /
101.2), beating AWAC's 63.7, and hammer-expert held at 129.7, even edging AWAC's 126.8. So IQL strictly
dominates the other two baselines, and it does so with a *no-op* transition — no schedule to tune. It is
the right floor to build on. But two things in these numbers tell me IQL is leaving online improvement
unclaimed. First, the seed spread is still wide: pen-cloned ranges 76 to 131, hammer-cloned 68 to 93.
Second, and more fundamentally, IQL's whole design is *transition-agnostic by omission* — it handles the
offline→online handoff well precisely because it does nothing special there, which means it also does
nothing to *exploit* the transition. Its value function is whatever the expectile regression produced; it
is never deliberately *scaled* to make the first online updates cheap. The question I could not ask of
SPOT or AWAC — both of which were still fighting the dip — I can finally ask of IQL: given a stable
initializer, can I make the online phase climb *faster and higher* by engineering the value function to be
a good initializer on purpose?

That reframes the objective. SPOT, AWAC, IQL were all answering "how do I pretrain stably and not
collapse at the transition." IQL answers it well. The unclaimed axis is "what property must the offline
value function have so that online fine-tuning is not just stable but *fast and monotone* — so the online
budget is spent climbing, not repairing." To get there I have to go back to the one offline family the
ladder skipped: conservative value regularization (CQL), which the initial context flagged as the
strongest *offline* method but a poor *initializer*. IQL is safe because it never queries OOD actions at
all; CQL is safe because it pushes their value down. CQL's push-down is a more aggressive lever — it can
produce a sharper offline value — but it has the failure the context named: its `Q` is a correct lower
bound that is *uncalibrated in scale*, driven far below the true return. With IQL I have proof that a
stable transition is achievable; now I want to ask whether a *deliberately scaled* conservative value
beats IQL's incidental one on the online climb.

Run CQL's transition in my head, because it is the dip IQL avoided by accident and I want to avoid it on
purpose while keeping CQL's sharper push-down. Offline, CQL's regularizer
`α(E_{s~D}[log Σ_a exp Q(s,a)] − E_{(s,a)~D}[Q(s,a)])` drives `Q` down by a large, essentially arbitrary
margin — the values can sit far beneath, even far below zero, the true positive returns of these Adroit
tasks. Switch to online: the policy rolls out, real transitions arrive, and the TD targets now contain
*actual* sampled returns — large and positive. The TD error is enormous because the target towers over
the depressed `Q`, so the critic makes a big, fast correction to reach the real scale, and the greedy
policy, riding a `Q` that is mid-lurch and temporarily incoherent across actions, degrades. That is the
dip — and crucially it is caused not by the lower-bound *direction* of CQL's conservatism (which I want to
keep, for the sharp OOD suppression) but by the unbounded *magnitude* of it. IQL never had this problem
because its expectile value was never artificially depressed; but IQL also never got CQL's aggressive
push-down. If I can keep CQL's push-down direction and bound its magnitude, I get a value function that is
both sharply conservative *and* well-scaled — a strictly better initializer than IQL's incidental one.

So the fix, stated before I know its form: keep the conservative lower bound on the learned policy's value
(still suppress OOD over-estimation), but bound the conservatism *from below* so `Q` cannot fall
arbitrarily far. Pin it to a sane scale. The formal object is a *calibrated* value — a lower bound on the
policy value *and* an upper bound on a trusted reference: `V^μ(s) ≤ Q_offline(s, π(s)) ≤ Q^π(s)`. The
right inequality is CQL's lower bound; the left is the new constraint — do not let `Q` drop below the
reference. The reference must be computable from the offline dataset with no model and must be a
meaningful floor. The behavior policy is the natural choice — it generated the data, so its value is a
sane scale, and its returns are sitting in the dataset. For a transition at time `t`, the discounted sum
of the rewards that actually followed, `R_t = Σ_{k≥0} γ^k r_{t+k}`, is an unbiased Monte-Carlo sample of
the behavior policy's return-to-go — free, by one reverse scan over each trajectory. So the calibration
floor is the MC return-to-go: never let the conservative `Q` be pushed below `R_t`.

Where it enters is one line. CQL's push-down is the importance-sampled log-sum-exp over candidate actions
(random + current-policy + next-policy); minimizing it lowers `Q` on those candidates, and unchecked it is
what drives `Q` arbitrarily low. So before the log-sum-exp, floor the *policy-action* candidates by the
reference: `Q̃(s,a) = max(Q(s,a), R(s))`. Wherever the penalty would push a policy-action `Q` below the
behavior return, the `max` kills its gradient and the floor holds; above the floor CQL is unchanged. This
only ever *raises* clamped entries toward the truth, so `Q` stays a (looser) lower bound on `Q^π` — the
offline safety is preserved — while the floor enforces the `V^μ` upper bound. The value is boxed into
`[V^μ, Q^π]`, near the real scale, so when online data arrives the TD targets are already close to `Q`,
the early correction is small, and the dip that plagued raw CQL is gone — but I kept CQL's sharp push-down,
which IQL never had.

The transition handling is now the *active* ingredient, the thing IQL left blank. Offline, the floor gives
the good scale. Online, real feedback is teaching the critic the true value, and the behavior reference is
a *suboptimal* floor I now want to climb *above* — so at the handoff I **disable calibration** (drop the
`max`, set `α` to its online value) and let the value rise past the behavior return as online returns
warrant. The MC floor for newly collected online transitions is meaningless (shifting distribution) and is
set to zero. And to keep the critic anchored to the offline scale while it learns online, I sample a fixed
50/50 mixing ratio of offline and online transitions during fine-tuning. So unlike IQL's no-op
`on_online_start`, Cal-QL's transition is a deliberate switch — calibration did its one job, hand the
online phase a well-scaled initialization, and then steps aside.

In this task's harness the Cal-QL fill is specific and I want it exact, because the scaffold exposes some
of this and omits some. The MC returns are precomputed in `__init__` from the replay buffer by a reverse
scan that detects episode boundaries from terminals *and* from state discontinuities (`‖s_{t+1} − s'_t‖`),
because the harness's flat buffer has no explicit trajectory index — and the maximum episode length is
inferred from the data (Adroit = 200). The buffer's `sample` is then *monkey-patched* to return the MC
return as a 7th batch element and, when online, to draw the 50/50 offline/online mix; `add_transition` is
patched to write `mc = 0` for online entries. The actor is the template's 3×256 `TanhGaussianPolicy` but
with learnable `log_std` multiplier/offset `Scalar`s and orthogonal init, and it supports `repeat=n` to
emit `n` candidate actions; the critic is 3×256 with multi-action support (it reshapes `(B, n, act)`
inputs). Automatic entropy tuning uses target entropy `−dim(A)`, `policy_lr = 1e-4`, `qf_lr = 3e-4`. The
calibration is applied to the current- and next-policy candidate Q-values (`torch.maximum(..., lower)`),
exactly as derived; `cql_max_target_backup = True` takes the in-support max over `cql_n_actions = 10` next
actions; `cql_clip_diff_min = −200`. `on_online_start` flips `_calibration_enabled` off and sets
`cql_alpha = cql_alpha_online = 1.0`. `CONFIG_OVERRIDES = {"normalize": False}`. The one thing the harness
does *not* expose that the generic method would want is a separate, larger online replay or a per-task
`α` schedule — here `α` simply switches to its (equal) online value, and the mixing is the only online
data-balancing knob; I note this because it means the online improvement rests entirely on calibration +
mixing, with no extra online conservatism annealing. I am also at the parameter cap: the multi-action CQL
critics plus the entropy `Scalar`s sit just under the 1.2× budget, so there is no capacity to spare — the
gain has to be the calibration, which it is (calibration adds *zero* parameters; it is a `max`).

The bar this has to clear is IQL's real numbers, and here is what I would validate. The falsifiable claim
is that a *deliberately calibrated* conservative value is a better online initializer than IQL's
incidental expectile value, so I would expect Cal-QL to match-or-exceed IQL on the final online score on
all three tasks — concretely, to hold pen-cloned at or above ~103 and hammer-cloned at or above ~77 while
keeping hammer-expert near the ceiling (~130) — and, more pointedly, to show a *faster, dip-free* online
climb than CQL would (the whole reason calibration exists). The specific way this could fail to beat IQL:
on the heavily-noisy hammer-cloned mixture the MC return-to-go floor is computed over noisy trajectories,
so the behavior reference itself is low and ragged, and a low floor calibrates weakly — if hammer-cloned
lands *below* IQL's 76.7, the lesson is that calibration's benefit is bounded by the quality of the
reference return, and on cloned data the expectile in-support max (IQL) is the safer signal. The case
where Cal-QL clearly wins is hammer-expert and the dip-prone early online phase: a clean expert reference
gives a tight, high floor, so I expect the calibrated value to track the true return almost immediately
and the online curve to be monotone from step one — beating IQL's incidental scaling where IQL had to let
the expectile value drift up on its own. That is the endpoint I am betting on: keep IQL's stability,
add CQL's sharp push-down, and bound its magnitude with a free MC-return floor so the online phase climbs
on purpose rather than by omission. The full scaffold module is in the answer.
