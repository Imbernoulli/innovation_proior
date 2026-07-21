IQL gave me a clean read, and the numbers say exactly where its conservatism cost me. On HalfCheetah it
landed 48.10 with a standard deviation of 0.12 across the three seeds (48.094 / 48.230 / 47.982) — a
seed-to-seed spread of a quarter of one point on a scale where expert is 100, so it is *pinned* to the
behavior policy: medium HalfCheetah data is a single mediocre running policy, and IQL reproduced its
value almost exactly without divergence and without disagreement between seeds. That is the floor behaving
as designed — safe, in-sample, and not improving much. Walker2d was 80.46 but with std 5.9 (75.5 / 78.9 /
87.0), an 11.5-point swing from worst to best seed, about 14% of the mean, so there the expectile backup
did find some headroom but unevenly — different seeds climbed different amounts. The damning number is
Maze2d: 33.73 mean with std 4.6 (29.7 / 32.8 / 38.7), the worst of the three, a 9-point swing that is 27%
of the mean, and not close to the data's reachable ceiling. Let me line the three up as a ratio, because
the pattern is the diagnosis: HalfCheetah coefficient of variation ≈ 0.12/48.1 ≈ 0.2%, Walker ≈ 5.9/80.5
≈ 7%, Maze ≈ 4.6/33.7 ≈ 14%. Variability climbs monotonically as the task moves from "one coherent
policy to reproduce" toward "fragments to stitch," and the mean falls the same way. That is the stitching
failure I worried about made concrete. Maze2d's medium dataset is a heap of partial goal-reaching
fragments, and good performance needs the value of a goal-adjacent state to propagate backward across
transitions from *different* trajectories. IQL *can* do that in principle, but its propagation is
throttled by τ = 0.7: the expectile only weakly approximates the in-support max — on my earlier toy it
moved only about a third of the way from the mean toward the maximum — so the value signal that should
flow back through the maze stays weak, and the advantage-weighted extraction (which never pushes the
policy past dataset actions) caps how decisively the policy can commit to the goal-reaching fragments. The
diagnosis is sharp: IQL is *too hedged*. Its safety came from refusing to ever exploit the critic directly
(no argmax, no ∇_aQ ascent), and on the dense locomotion tasks that refusal cost little — HalfCheetah had
no headroom anyway — but on a task that needs aggressive value exploitation it left a lot on the floor,
and the 27%-of-mean per-seed spread on Maze2d says the weak signal is also fragile to seed.

So the move I want is the opposite stance: let the policy actually *ascend the critic* — real
deterministic-policy-gradient improvement, which can commit hard to high-value actions and propagate
value vigorously — but bolt on the minimum amount of behavior regularization needed to keep the OOD
overestimation from blowing up. IQL avoided the OOD query entirely; I now want to *allow* the query and
*tame* it, because allowing it is the only way to get improvement stronger than an in-sample expectile.
This is a different bet on the same disease, worth making precisely because IQL's weakness is on the task
where exploitation matters most. Pushing β higher within AWR will not do it — β only reweights *logged*
actions, so however sharp I make it the policy cannot propose the stitching action no single maze fragment
took; that ceiling is untunable. A conservative value penalty would allow the off-data query but adds an
offline-unvalidatable temperature and *courts* the OOD actions it must then suppress. So I take
deterministic policy gradient with the lightest possible leash: allow the ∇_aQ ascent AWR forbids, and add
exactly one term to keep it near the data — a single coefficient rather than a sampler plus a temperature,
the same parsimony lesson IQL taught.

Let me first rebuild the exploitation engine, because its own overestimation pathology is what the BC
term will have to counter. The base is a deterministic actor-critic: a deterministic actor a = π(s)
ascending a learned critic by ∇_φ J = E[∇_aQ(s,a)|_{a=π(s)} ∇_φπ(s)], with replay and target networks.
Even *online* this object overestimates, and I want the mechanism explicit because it carries over. The
critic has function-approximation error; the actor climbs wherever the critic bulges upward; the bulges
are disproportionately where the error is positive; so the actor selects for the upward errors and the
value inflates — the same selection-for-positive-error as a discrete max, just routed through the
gradient. This is the identical mechanism I quantified at the floor: an upper-order-statistic bias of
order σ per step, amplified by 1/(1−γ) = 100 with γ = 0.99. Offline it is catastrophic, because there are
no fresh transitions to correct an inflated pocket, but the structural cures are the same ones that
stabilize TD3 online, so I carry the full TD3 correction stack in.

First, clipped double-Q. The reason a single critic's target inflates is that the same estimator both
selects and evaluates the next action, so it evaluates its own optimistic pick. Keep two critics and,
where I need a single target value, take the *min* of the two target critics at the next action. The
asymmetry is the justification: an overestimated action gets selected by the actor and its inflation is
propagated through every policy update — overestimation is actively chased — whereas an underestimated
action is simply avoided, so it is not amplified. So biasing toward the smaller estimate is the safe
direction; the min caps the target at the worst case of a single critic and leans toward underestimation,
which this feedback loop tolerates far better than overestimation. Concretely, if the two critics
disagree by δ at the actor's chosen next action, the min pulls the target down by roughly δ/2 relative to
their average, and — crucially — it pulls down *most* exactly where the two critics disagree most, which
is exactly where extrapolation error is largest, so the correction is self-targeting. Second, target
policy smoothing. A deterministic actor will find and sit on a narrow spurious peak in the critic — an
action where approximation error happens to spike upward — and read the target off that knife-edge. So I
perturb the target action with clipped Gaussian noise, ã = clip(π_target(s') + clip(N(0,σ),−c,c),
−a_max, a_max), which makes the target reflect the value of a small *neighborhood* and averages the spike
away. In action units σ = 0.2·a_max and c = 0.5·a_max, so with a_max = 1 the noise has standard deviation
0.2 and is clipped to ±0.5 — the target action is jittered over a neighborhood that spans about a fifth of
the action range per dimension, wide enough to average across a spurious peak but clipped so a rare large
draw cannot fling the target action to the opposite corner of the box. Third, delayed policy updates and
Polyak targets: fit the critic for two steps per actor step (policy_freq = 2) so each policy move sees a
critic that has had time to drive its error down, and soft-update all targets with tau = 5e-3 so the
bootstrap chases a stationary objective rather than its own moving estimate. Over the 1e6-step budget,
policy_freq = 2 means about 500k actor updates against 1e6 critic updates — the critic gets twice the
gradient traffic, so the value it hands the policy gradient has settled between consecutive policy moves.
These three — min target for bias, smoothing for the deterministic-peak variance, delay plus soft targets
for accumulated-error variance — are one coordinated attack on function-approximation error, and they are
the only reason a DPG actor is stable enough to regularize.

There is a reason I keep the networks at 2×256 rather than reaching for the extra depth the 256-width cap
would still allow. TD3+BC's stability story rests entirely on the three function-approximation brakes
above, not on capacity, and a deeper unconstrained MLP critic is precisely the object that grows *more*
spurious peaks for the actor to climb — more parameters means a more flexible surface, and off the data
that flexibility is spent on extrapolation artifacts, not on real value. So at this rung, where the only
smoothing I have is the target noise, adding depth would be adding risk with no principled counterweight.
Two hidden layers of 256 is enough to represent an action-conditioned Q over a locomotion or a maze state,
and it keeps me comfortably inside the parameter cap with room to spare — the budget is not the binding
constraint here, the peakiness of a deep critic is, and I decline to spend capacity I cannot control.

I want to be concrete about the accumulated-error half of the variance story, because it is what
policy_freq and Polyak are for and it explains a specific feature of IQL's numbers. The bootstrap error
does not just bias the target upward; it also *moves*, because the target network is chasing the online
critic which is chasing a target built from itself. If I updated the actor every step against a critic
that was itself lurching, the policy gradient would be reading a value surface that changes underneath it,
and different seeds would catch that surface at different phases — which is exactly the kind of thing that
produces IQL's uneven Walker2d (75.5 / 78.9 / 87.0, an 11.5-point seed swing on a task with real
headroom): the value signal was there but arrived at different strengths on different seeds. Delaying the
actor to every second critic step and slowing the target with tau = 5e-3 turns that lurch into a slow
drift the policy gradient can track — a more stationary surface for the actor to exploit.

Now the offline piece, and I want it to be *minimal*, for a reason that IQL's experience reinforces: in
offline RL every extra knob is costly because I cannot validate it by interacting, so the more machinery
I add (generative behavior models, divergence estimators, conservative samplers), the more I am tuning
blind. IQL's whole appeal was that it added essentially one idea (the expectile) to in-sample value
learning; I want my exploitation-first method to be just as parsimonious. The minimal way to keep an
ascending actor near the data is to add a behavior-cloning term straight to the policy objective: while
the actor maximizes Q(s, π(s)), also penalize how far π(s) is from the dataset action a at that state.
The policy objective becomes maximize λ·Q(s, π(s)) − (π(s) − a)², an L2 pull toward the data. I pick L2
to the dataset action rather than a KL or MMD divergence for the same minimality reason — no behavior
model to fit, one line, deterministic policy — and there is no principled reason a particular divergence
wins here. This is the cheapest possible "stay near π_β," and it directly answers the OOD overestimation:
the actor can still ascend the critic, but it is anchored, so it cannot run off to the over-valued
out-of-distribution actions that have no correction.

The one subtlety that makes the BC term work across datasets is the coefficient — it is what lets a
*single* α serve three datasets with wildly different reward scales. The BC penalty (π − a)² is bounded — for tanh-squashed actions in [−1, 1] it is at most 4 per
dimension, so summed over an action_dim of about 6 it tops out near 24 and in practice, once the actor is
near the data, sits at O(0.1) — but Q scales with the reward scale, which differs across datasets by
orders of magnitude (a dense-reward locomotion return and a sparse maze return are simply not the same
size). A fixed λ would make the BC term dominate on small-Q tasks and vanish on large-Q ones. The fix is
to make λ a *normalizer*, not a fixed weight: λ = α / ((1/N) Σ |Q(s, a)|), with α = 2.5. Watch what that
does dimensionally: the actor's RL term is λ·Q = α·Q/mean|Q|, whose *typical magnitude* is α·1 = 2.5
regardless of whether mean|Q| is 3 or 300, because the reward scale cancels between numerator and
denominator. So the RL contribution to the actor gradient is held at a scale of about 2.5 while the BC
contribution sits at its own O(1)-per-dimension scale, and one α = 2.5 sets the same RL/BC balance on
HalfCheetah, Walker2d, and Maze2d alike. The mean |Q| must be detached so λ is a scale and not a gradient
path — otherwise the actor could cheat the balance by inflating its own critic readings. This is the
whole algorithm: TD3's stabilized critic, plus a reward-scale-normalized L2 pull toward the dataset action
in the actor loss. State normalization I get for free from the fixed loop (it computes exact dataset
mean/std and normalizes features), which matters because a deterministic actor is sensitive to feature
scaling — an unnormalized feature with a large dynamic range would dominate the first-layer activations
and warp the value surface the actor climbs; and as with IQL I leave the *rewards* unnormalized, so the
only offline-specific change versus online TD3 is the BC term and the λ normalizer.

Mapping to the edit surface: I build a `DeterministicActor` (2×256, tanh-squashed) and its target, twin
2×256 `Critic`s and their targets, three Adam optimizers at 3e-4. Every step I update both critics on the
min-of-target-critics smoothed target by MSE; every second step I update the actor on
−λ·Q1(s, π(s)) + MSE(π(s), a) with λ = α/|Q1|.mean().detach(), then Polyak the actor and both critic
targets. I ascend the *first* critic for the policy gradient (the DPG through Q1), exactly as the
stabilized base prescribes — the second critic exists only to supply the min in the target, so routing
the policy gradient through a single fixed critic keeps the actor from chasing whichever of the two
happens to be higher. Rewards and dones squeeze to (B,) so the min target matches each critic's output,
and in the actor step q.abs().mean() is the detached scalar normalizer against the scalar BC MSE.

What the BC term *is* implicitly tells me how it behaves across datasets. Maximizing λ·Q − (π − a)² is the
stationarity condition of a proximal step: the actor moves along ∇_aQ but is held by a quadratic well
centered on the logged action, giving an achieved displacement of roughly (λ/2)·∇_aQ. Because λ holds the
RL term at a fixed normalized scale the step size is dataset-independent, but the *improvement* it buys is
not — it depends on how much better ∇_aQ points than the logged action. Where the logged action is already
near-optimal (HalfCheetah) the step barely moves; where a goal-reaching action beats the meandering logged
one (Maze2d) it strains against the well and commits — which is exactly the improvement I am buying and,
against a seed-dependent critic peak, exactly the source of variance. One global α thus cannot separately
control how far each dataset moves.

Here is what I expect against IQL's numbers. On HalfCheetah I expect to roughly *match* IQL's 48.10:
medium HalfCheetah is a single coherent policy with little room above the data — IQL's 0.12 std says as
much — so "stay near π_β by expectile" and "stay near π_β by BC" converge to about the same near-data
value. The real test is Maze2d, where IQL's 33.7 exposed the stitching gap: the deterministic-policy-
gradient actor can commit hard to high-value goal-reaching actions instead of hedging, so I expect a
*substantially* higher Maze2d mean — but *high-variance across seeds*, because the value landscape is
multimodal and the BC anchor plus seed-dependent critic peaks will send different seeds to different
basins. A wide Maze2d spread is the predicted signature, and if it appears it is the next opening — a
method that does not let a single seed's critic peak dominate. On Walker2d I expect to clear IQL's 80.5,
and probably with a tighter seed spread, since a committed actor on a Polyak-slowed surface should not
disagree seed-to-seed as much as an expectile that climbed unevenly. If TD3+BC merely ties IQL everywhere,
the BC anchor is too tight and regularization, not exploitation, is the binding constraint.
