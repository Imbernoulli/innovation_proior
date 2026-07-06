IQL delivered exactly what I predicted, and reading its numbers tells me precisely what is left on the
table. Hammer-cloned climbed from AWAC's dead-flat 0.336 to a mean of 76.7 — and decisively, on *every*
seed (69.0 / 68.4 / 92.7), not one lucky run the way SPOT's 55.8 was. That two-hundred-fold jump from
0.336 to 76.7, uniform across seeds, is the in-support expectile backup doing the stitching AWAC could
not: the value function propagated the rare good fragments through the noise and the advantage-weighted
policy found the manipulation behavior. Pen-cloned rose to 103.0 (131.4 / 76.3 / 101.2), beating AWAC's
63.7 by a factor of about 1.6, and hammer-expert held at 129.7 (129.5 / 130.1 / 129.7), even edging
AWAC's 126.8 and staying every bit as tight. So IQL strictly dominates the other two baselines on all
three tasks at once, and it does so with a *no-op* transition — no schedule to tune. It is the right floor
to build on. But two things in these numbers tell me IQL is leaving online improvement unclaimed. First,
the seed spread is still wide where the data is noisy: pen-cloned ranges 76 to 131, a 55-point band, and
hammer-cloned 68 to 93 — the stabilizers helped but did not close it. Second, and more fundamentally,
IQL's whole design is *transition-agnostic by omission* — it handles the offline→online handoff well
precisely because it does nothing special there, which means it also does nothing to *exploit* the
transition. Its value function is whatever the expectile regression produced; it is never deliberately
*scaled* to make the first online updates cheap. The question I could not ask of SPOT or AWAC — both of
which were still fighting the dip and the collapse — I can finally ask of IQL: given a stable initializer,
can I make the online phase climb *faster and higher* by engineering the value function to be a good
initializer on purpose?

That reframes the objective. SPOT, AWAC, IQL were all answering "how do I pretrain stably and not
collapse at the transition." IQL answers it well. The unclaimed axis is "what property must the offline
value function have so that online fine-tuning is not just stable but *fast and monotone* — so the online
budget is spent climbing, not repairing." To get there I have to go back to the one offline family the
ladder skipped: conservative value regularization (CQL), which the initial context flagged as the
strongest *offline* method but a poor *initializer*. IQL is safe because it never queries OOD actions at
all; CQL is safe because it pushes their value down. CQL's push-down is a more aggressive lever — it can
produce a sharper offline value, actively suppressing the OOD actions rather than merely declining to
evaluate them — but it has the failure the context named: its `Q` is a correct lower bound that is
*uncalibrated in scale*, driven far below the true return. With IQL I have proof that a stable transition
is achievable; now I want to ask whether a *deliberately scaled* conservative value beats IQL's incidental
one on the online climb. The reason to reach for CQL's sharper lever at all, rather than just accept IQL,
is that IQL's residual weakness is exactly on the noisy tasks where its expectile value is softest — the
55-point pen spread — and a sharper, better-scaled conservative value is a candidate cure for precisely
that.

Run CQL's transition in my head, because it is the dip IQL avoided by accident and I want to avoid it on
purpose while keeping CQL's sharper push-down. Offline, CQL's regularizer
`α(E_{s~D}[log Σ_a exp Q(s,a)] − E_{(s,a)~D}[Q(s,a)])` drives `Q` down by a large, essentially arbitrary
margin — the log-sum-exp term pushes down `Q` on the candidate (mostly OOD) actions while the second term
pulls it up on the dataset actions, and the *magnitude* of the net push is set by `α` times an OOD-vs-data
gap that has no reason to respect the true return scale. On these Adroit tasks the true returns are large
and positive (expert-normalized around 100–130, as the IQL numbers show), yet the CQL `Q` can sit far
beneath, even far below zero. Switch to online: the policy rolls out, real transitions arrive, and the TD
targets now contain *actual* sampled returns — large and positive. The TD error is enormous because the
target towers over the depressed `Q`: if offline `Q` sits near, say, a large negative value while the true
return is order +100, the first online backup faces a gap of well over a hundred, and at `γ = 0.99` that
gap is bootstrapped with a `1/(1−γ) = 100` fixed-point amplification, so the critic makes a big, fast,
system-wide correction to reach the real scale. While that correction is mid-flight the greedy policy
rides a `Q` that is temporarily incoherent across actions — the *relative* ordering of actions is
scrambled while the *absolute* level lurches — and it degrades. That is the dip. Crucially it is caused
not by the lower-bound *direction* of CQL's conservatism (which I want to keep, for the sharp OOD
suppression) but by the unbounded *magnitude* of it. IQL never had this problem because its expectile
value was never artificially depressed — it tracked the real return scale by construction — but IQL also
never got CQL's aggressive push-down. If I can keep CQL's push-down direction and bound its magnitude, I
get a value function that is both sharply conservative *and* well-scaled — a strictly better initializer
than IQL's incidental one.

So the fix, stated before I know its form: keep the conservative lower bound on the learned policy's value
(still suppress OOD over-estimation), but bound the conservatism *from below* so `Q` cannot fall
arbitrarily far. Pin it to a sane scale. The formal object is a *calibrated* value — a lower bound on the
policy value *and* an upper bound on a trusted reference: `V^μ(s) ≤ Q_offline(s, π(s)) ≤ Q^π(s)`. The
right inequality is CQL's lower bound; the left is the new constraint — do not let `Q` drop below the
reference. The reference must satisfy three things: computable from the offline dataset with no model, a
meaningful floor on the true scale, and cheap. The behavior policy is the natural choice — it generated
the data, so its value is a sane scale, and its returns are sitting in the dataset already. For a
transition at time `t`, the discounted sum of the rewards that actually followed,
`R_t = Σ_{k≥0} γ^k r_{t+k}`, is an unbiased Monte-Carlo sample of the behavior policy's return-to-go —
free, by one reverse scan over each trajectory, `R_t = r_t + γ R_{t+1}`. Note the horizon interacts with
the discount cleanly here: Adroit episodes run to 200 steps but `γ = 0.99` gives an effective horizon of
about `1/(1−γ) = 100`, so the reverse-accumulated `R_t` is dominated by the next ~100 steps of realized
reward and the tail contributes negligibly — the floor is a well-conditioned, bounded quantity, not a
sum that blows up over the episode. So the calibration floor is the MC return-to-go: never let the
conservative `Q` be pushed below `R_t`.

Where it enters is one line. CQL's push-down is the importance-sampled log-sum-exp over candidate actions
(random + current-policy + next-policy); minimizing it lowers `Q` on those candidates, and unchecked it is
what drives `Q` arbitrarily low. So before the log-sum-exp, floor the *policy-action* candidates by the
reference: `Q̃(s,a) = max(Q(s,a), R(s))`. Read what the `max` does to the gradient: wherever the penalty
would push a policy-action `Q` below the behavior return, the `max` selects `R(s)`, a constant, so its
gradient is zero and the push-down stops exactly at the floor; wherever `Q` is already above `R(s)`, the
`max` is transparent and CQL is unchanged. This only ever *raises* clamped entries toward the truth, so
`Q` stays a (looser) lower bound on `Q^π` — the offline safety is preserved, I never turn conservatism
into optimism — while the floor enforces the `V^μ` upper bound. The value is boxed into `[V^μ, Q^π]`,
near the real scale, so when online data arrives the TD targets are already close to `Q`, the early
correction is the small residual between the behavior floor and the true return rather than the whole
hundred-point gap, and the dip that plagued raw CQL is gone — but I kept CQL's sharp push-down, which IQL
never had.

Be precise about which candidates get floored and why not all of them, because the importance-sampled
log-sum-exp mixes three action sources — uniform-random, current-policy, next-policy — and the CQL term
subtracts each candidate's proposal log-density (`random_density = log(0.5^{action_dim}) = action_dim·
log 0.5`, which for Hammer's 26-dim action is about `26·(−0.693) ≈ −18`, a large negative offset that
correctly down-weights the uniform-random candidates relative to the policy ones). I floor only the
*current-policy and next-policy* candidate Q-values, not the random ones. The reason is that the
calibration argument is about the *learned policy's* value being a lower bound above the behavior
reference — `Q_offline(s, π(s)) ≥ V^μ(s)` — so the floor belongs on the actions the policy actually
proposes; the uniform-random candidates exist only to broaden the OOD push-down and have no claim to sit
above the behavior return. Flooring them would blunt exactly the OOD suppression I am trying to preserve.

Put the transition arithmetic on paper so I know the dip really shrinks and is not just relabelled. Take
hammer-expert, where the true return is around 130 and the behavior reference (clean expert demos) sits
close to it, say near 120–130. Raw CQL might depress `Q` to something like a large negative value offline;
the first online TD target, carrying a real sampled return near +130, then faces a gap on the order of
130 − (−X) — well over a hundred — and that gap is what the critic must traverse while the policy rides
the incoherent mid-lurch value. With the floor, the offline `Q` on policy actions cannot fall below the
~120–130 behavior return, so the first online target sits within a handful of points of `Q`, and the
correction is a small nudge rather than a system-wide rescaling. The `1/(1−γ) = 100` amplification that
turned CQL's offline depression into a horizon-long online repair now acts on a residual of a few points
instead of a hundred, so the compounded error is smaller by the same ratio the gap shrank. That is the
mechanism by which calibration converts a repair phase into a climb phase — it is a statement about the
*size* of the initial TD error, and the floor bounds that size directly.

The backbone under all this is CQL-on-SAC — a stochastic Tanh-Gaussian actor with automatic entropy
tuning — rather than the deterministic TD3 the first rung used, and the choice is forced by what the
push-down needs. CQL's regularizer requires sampling candidate actions from the current policy to form
the log-sum-exp, so the policy has to be a proper distribution I can draw `cql_n_actions = 10` samples
from, not a single deterministic point; a deterministic actor gives one candidate and the whole
importance-sampled OOD term collapses. The automatic entropy tuning solves the α-of-entropy the SAC actor
needs: it adjusts the entropy temperature so the policy's entropy tracks the target `−dim(A)` (one nat of
negative entropy per action dimension, the standard heuristic), which keeps the policy stochastic enough
to supply diverse candidates for the push-down but not so diffuse that it wanders OOD during action
selection. Note this is a *different* use of entropy than SAC's usual exploration bonus — here it is
mostly serving the conservative regularizer's need for candidate diversity, and I keep `backup_entropy =
False` so the entropy term does not leak into the bootstrap target and re-inflate it. The learnable
`log_std` multiplier and offset `Scalar`s give the policy a per-dimension handle on its spread that the
orthogonal-init base network can tune, which matters because the CQL candidates are only useful if their
spread matches the scale at which the critic's OOD errors actually live.

The transition handling is now the *active* ingredient, the thing IQL left blank. Offline, the floor gives
the good scale. Online, real feedback is teaching the critic the true value, and the behavior reference is
a *suboptimal* floor I now want to climb *above* — so at the handoff I **disable calibration** (drop the
`max`, set `α` to its online value) and let the value rise past the behavior return as online returns
warrant. Leaving the floor on would be actively harmful online: it would pin the value at the behavior
scale precisely when I want it to exceed the behavior scale, capping improvement at the level of the data
that generated the floor. The MC floor for newly collected online transitions is meaningless anyway (the
online distribution is shifting and non-stationary, so a reverse scan over a growing mixed buffer is not a
clean return-to-go) and is set to zero. And to keep the critic anchored to the offline scale while it
learns online, I sample a fixed 50/50 mixing ratio of offline and online transitions during fine-tuning —
half the batch is always the well-scaled offline data, so the critic cannot forget the calibration it was
handed even as it climbs on the online half. So unlike IQL's no-op `on_online_start`, Cal-QL's transition
is a deliberate switch — calibration did its one job, hand the online phase a well-scaled initialization,
and then step aside.

In this task's harness the Cal-QL fill is specific and I want it exact, because the scaffold exposes some
of this and omits some. The MC returns are precomputed in `__init__` from the replay buffer by a reverse
scan that detects episode boundaries from terminals *and* from state discontinuities (`‖s_{t+1} − s'_t‖`),
because the harness's flat buffer has no explicit trajectory index — the boundary test compares the next
row's state against this row's stored `next_state` and calls a mismatch a new episode — and the maximum
episode length is inferred from the data (Adroit = 200). The buffer's `sample` is then *monkey-patched* to
return the MC return as a 7th batch element and, when online, to draw the 50/50 offline/online mix by
index (`n_offline = int(batch·0.5)` from the offline range, the rest from the online range);
`add_transition` is patched to write `mc = 0` for online entries. The actor is the template's 3×256
`TanhGaussianPolicy` but with learnable `log_std` multiplier/offset `Scalar`s and orthogonal init, and it
supports `repeat=n` to emit `n` candidate actions; the critic is 3×256 with multi-action support (it
reshapes `(B, n, act)` inputs to a flat batch, evaluates, and reshapes back). Automatic entropy tuning
uses target entropy `−dim(A)`, `policy_lr = 1e-4`, `qf_lr = 3e-4`. The calibration is applied to the
current- and next-policy candidate Q-values (`torch.maximum(..., lower)`), exactly as derived;
`cql_max_target_backup = True` takes the in-support max over `cql_n_actions = 10` next actions — which is
itself a small piece of the same in-support-improvement idea, a max over ten policy samples rather than a
single one, tightening the backup target; `cql_clip_diff_min = −200` bounds the per-sample CQL penalty so
a single wild candidate cannot dominate. `on_online_start` flips `_calibration_enabled` off and sets
`cql_alpha = cql_alpha_online = 1.0`. `CONFIG_OVERRIDES = {"normalize": False}`. The one thing the harness
does *not* expose that the generic method would want is a separate, larger online replay or a per-task
`α` schedule — here `α` simply switches to its (equal) online value, and the 50/50 mixing is the only
online data-balancing knob; I note this because it means the online improvement rests entirely on
calibration + mixing, with no extra online conservatism annealing. I am also at the parameter cap: the
multi-action CQL critics plus the entropy `Scalar`s sit just under the 1.2× budget, so there is no
capacity to spare — and this is the elegant part of the bet, because calibration adds *zero* parameters.
It is a `max` against a precomputed vector; the entire gain, if it comes, comes from an arithmetic clamp,
not from any network the budget would have to pay for.

One design fork I should walk before committing, because there is a tempting alternative to the
disable-and-mix transition: I could instead *anneal* `α` downward over the online phase, the way the
first rung cooled its constraint coefficient, letting conservatism fade gradually rather than switching
the calibration off at a single instant. Walk it. Annealing `α` keeps the log-sum-exp push-down active at
diminishing strength, so the critic stays partly suppressed for a long online stretch — which reintroduces
exactly the scale mismatch calibration was built to remove, just spread out over time instead of
concentrated at step zero. And it adds a schedule with a rate I would have to tune per task, the very
kind of knob the middle rungs showed is fragile (SPOT's cooled `λ` was a suspect in its expert collapse).
The floor already did the scale job offline; once online feedback is teaching the true value there is no
reason to keep pushing down at all, so the clean switch dominates a gradual anneal — I disable calibration
outright and let the 50/50 mix, not a decaying penalty, be the thing that holds the scale. The mix is the
better anchor because it is *stationary*: half of every online batch is the well-scaled offline data, so
the critic is continuously re-shown the calibrated scale rather than being weaned off a schedule, and the
balance is a fixed 50/50 rather than a curve to mis-set. That is why the transition is a switch plus a
mix, not a cool.

The bar this has to clear is IQL's real numbers, and here is what I would validate. The falsifiable claim
is that a *deliberately calibrated* conservative value is a better online initializer than IQL's
incidental expectile value, so I would expect Cal-QL to match-or-exceed IQL on the final online score on
all three tasks — concretely, to hold pen-cloned at or above ~103 and hammer-cloned at or above ~77 while
keeping hammer-expert near the ceiling (~130) — and, more pointedly, to show a *faster, dip-free* online
climb than raw CQL would (the whole reason calibration exists). The specific way this could fail to beat
IQL: on the heavily-noisy hammer-cloned mixture the MC return-to-go floor is computed over noisy
trajectories, so the behavior reference itself is low and ragged, and a low floor calibrates weakly — if
hammer-cloned lands *below* IQL's 76.7, the lesson is that calibration's benefit is bounded by the quality
of the reference return, and on cloned data the expectile in-support max (IQL) is the safer signal. The
case where Cal-QL clearly wins is hammer-expert and the dip-prone early online phase: a clean expert
reference gives a tight, high floor near the ~130 ceiling IQL already reaches, so I expect the calibrated
value to track the true return almost immediately and the online curve to be monotone from step one —
beating IQL's incidental scaling, where IQL had to let the expectile value drift up on its own. That is
the endpoint I am betting on: keep IQL's stability, add CQL's sharp push-down, and bound its magnitude
with a free MC-return floor so the online phase climbs on purpose rather than by omission. The full
scaffold module is in the answer.
