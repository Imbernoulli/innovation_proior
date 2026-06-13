IQL gave me a clean read, and the numbers say exactly where its conservatism cost me. On HalfCheetah it
landed 48.10 with a standard deviation of 0.12 across seeds — rock-steady, and essentially *pinned* to
the behavior policy: medium HalfCheetah data is a single mediocre running policy, and IQL reproduced its
value almost exactly without divergence. That is the floor behaving as designed — safe, in-sample, and
not improving much. Walker2d was 80.46 but with std 5.9 (75.5 / 78.9 / 87.0 across the three seeds), so
there the expectile backup did find some headroom but unevenly. The damning number is Maze2d: 33.73 mean
with std 4.6 (29.7 / 32.8 / 38.7), the worst of the three and not close to the data's reachable ceiling.
That is the stitching failure I worried about made concrete. Maze2d's medium dataset is a heap of
partial goal-reaching fragments, and good performance needs the value of a goal-adjacent state to
propagate backward across transitions from *different* trajectories. IQL *can* do that in principle, but
its propagation is throttled by τ = 0.7: the expectile only weakly approximates the in-support max, so
the value signal that should flow back through the maze stays weak, and the advantage-weighted extraction
— which never pushes the policy past dataset actions — caps how decisively the policy can commit to the
goal-reaching fragments. The diagnosis is sharp: IQL is *too hedged*. Its safety came from refusing to
ever exploit the critic directly (no argmax, no ∇_aQ ascent), and on the dense locomotion tasks that
refusal cost little, but on a task that needs aggressive value exploitation it left a lot on the floor,
and the per-seed spread on Maze2d (a 9-point swing) says the weak signal is also fragile to seed.

So the move I want is the opposite stance: let the policy actually *ascend the critic* — real
deterministic-policy-gradient improvement, which can commit hard to high-value actions and propagate
value vigorously — but bolt on the minimum amount of behavior regularization needed to keep the OOD
overestimation from blowing up. IQL avoided the OOD query entirely; I now want to *allow* the query and
*tame* it, because allowing it is the only way to get improvement stronger than an in-sample expectile.
This is a different bet on the same disease, and it is worth making precisely because IQL's weakness is
on the task where exploitation matters most.

Let me first rebuild the exploitation engine, because its own overestimation pathology is what the BC
term will have to counter. The base is a deterministic actor-critic: a deterministic actor a = π(s)
ascending a learned critic by ∇_φ J = E[∇_aQ(s,a)|_{a=π(s)} ∇_φπ(s)], with replay and target networks.
Even *online* this object overestimates, and I want the mechanism explicit because it carries over. The
critic has function-approximation error; the actor climbs wherever the critic bulges upward; the bulges
are disproportionately where the error is positive; so the actor selects for the upward errors and the
value inflates — the same selection-for-positive-error as a discrete max, just routed through the
gradient. Offline this is catastrophic, because there are no fresh transitions to correct an inflated
pocket, but the structural cures are the same ones that stabilize TD3 online, so I carry the full TD3
correction stack in.

First, clipped double-Q. The reason a single critic's target inflates is that the same estimator both
selects and evaluates the next action, so it evaluates its own optimistic pick. Keep two critics and,
where I need a single target value, take the *min* of the two target critics at the next action. The
asymmetry is the justification: an overestimated action gets selected by the actor and its inflation is
propagated through every policy update — overestimation is actively chased — whereas an underestimated
action is simply avoided, so it is not amplified. So biasing toward the smaller estimate is the safe
direction; the min caps the target at the worst case of a single critic and leans toward underestimation,
which this feedback loop tolerates far better than overestimation. Second, target policy smoothing. A
deterministic actor will find and sit on a narrow spurious peak in the critic — an action where
approximation error happens to spike upward — and read the target off that knife-edge. So I perturb the
target action with clipped Gaussian noise, ã = clip(π_target(s') + clip(N(0,σ),−c,c), −a_max, a_max),
which makes the target reflect the value of a small *neighborhood* and averages the spike away; σ = 0.2,
c = 0.5 in action units. Third, delayed policy updates and Polyak targets: fit the critic for two steps
per actor step (policy_freq = 2) so each policy move sees a critic that has had time to drive its error
down, and soft-update all targets with tau = 5e-3 so the bootstrap chases a stationary objective rather
than its own moving estimate. These three — min target for bias, smoothing for the deterministic-peak
variance, delay plus soft targets for accumulated-error variance — are one coordinated attack on
function-approximation error, and they are the only reason a DPG actor is stable enough to regularize.

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

The one subtlety that makes the BC term work across datasets is the coefficient. The BC penalty
(π − a)² is bounded — for actions in [−1, 1] it is at most about 4 per dimension — but Q scales with the
reward scale, which differs across datasets, so a fixed λ would make the BC term dominate on
small-reward tasks and vanish on large-reward ones. The fix is to make λ a *normalizer*, not a fixed
weight: λ = α / ((1/N) Σ |Q(s, a)|), with α = 2.5. Dividing by the mean absolute Q over the minibatch
rescales the Q term so the RL/BC balance is decoupled from the reward scale and one α works across all
three datasets; the mean |Q| must be detached so λ is a scale and not a gradient path. This is the whole
algorithm: TD3's stabilized critic, plus a reward-scale-normalized L2 pull toward the dataset action in
the actor loss. State normalization I get for free from the fixed loop (it computes exact dataset mean/std
and normalizes features), which matters because a deterministic actor is sensitive to feature scaling;
and as with IQL I leave the *rewards* unnormalized, so the only offline-specific change versus online TD3
is the BC term and the λ normalizer.

Mapping to the edit surface: I build a `DeterministicActor` (2×256, tanh-squashed) and its target, twin
2×256 `Critic`s and their targets, three Adam optimizers at 3e-4. Every step I update both critics on the
min-of-target-critics smoothed target by MSE; every second step I update the actor on
−λ·Q1(s, π(s)) + MSE(π(s), a) with λ = α/|Q1|.mean().detach(), then Polyak the actor and both critic
targets. I ascend the *first* critic for the policy gradient (the DPG through Q1), exactly as the
stabilized base prescribes. The full scaffold module is in the answer.

Here is what I expect against IQL's numbers, stated so it can be falsified. On HalfCheetah I expect to
roughly *match* IQL (~48): medium HalfCheetah is a single coherent policy with little room above the
data, so both "stay near π_β by expectile" and "stay near π_β by BC" should converge to about the same
near-data value — if anything, TD3+BC's direct critic ascent might edge it slightly. The real test is
Maze2d, where IQL's 33.7 exposed the stitching gap: the deterministic-policy-gradient actor can commit
hard to high-value goal-reaching actions instead of hedging, so I expect a *substantially* higher Maze2d
mean than IQL's 33.7 — but I also expect it to be *high-variance across seeds*, because Maze2d's value
landscape is multimodal (many goal approaches) and the BC anchor plus seed-dependent critic peaks will
send different seeds to different basins; a wide per-seed spread on Maze2d is the predicted signature, and
if it appears it is the next opening — a method that does not let a single seed's critic peak dominate.
On Walker2d I expect to clear IQL's 80.5, because the exploitation actor should extract more from the
walking data than the hedged expectile did, and probably with tighter seed spread than IQL's 5.9. If
instead TD3+BC merely ties IQL everywhere, the BC anchor is too tight and the lesson is that the
regularization, not the exploitation, is the binding constraint.
