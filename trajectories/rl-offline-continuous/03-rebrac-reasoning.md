TD3+BC confirmed the bet *and* exposed exactly the fragility I flagged. On HalfCheetah it landed 48.33
(std 0.42, per-seed 48.18 / 48.01 / 48.80) — a dead heat with IQL's 48.10, which is the expected result
on a single-policy dataset with no headroom: both "stay near π_β" recipes converge to the same near-data
value, and the proximal actor step I described barely moved because ∇_aQ at the logged action is small.
On Walker2d it cleared IQL, 85.14 versus 80.46, and far tighter (std 0.26 versus IQL's 5.9, per-seed
85.32 / 85.27 / 84.84 — a spread of half a point): letting the deterministic actor ascend the critic
extracted more from the walking data than the hedged expectile did, and the delayed-update, Polyak-slowed
surface made the exploitation seed-stable exactly as I predicted. So exploitation paid off where there was
room to exploit, and it paid off cleanly. But Maze2d is the story, and it is ugly in precisely the way I
predicted: mean 50.29, std **56.8**, with per-seed scores of 114.99, 27.20, and 8.69. Let me read those
three numbers rather than the mean, because the mean is a lie here. The coefficient of variation is
56.8/50.3 ≈ 113% — the spread is larger than the mean itself. One seed essentially *solved* the maze
(115 is above expert-normalized 100, so it found a genuinely good goal-reaching policy); one was mediocre
at 27; one, at 8.69, is barely above random and has essentially failed. This is not a distribution with a
center; it is a trimodal lottery. The mean of 50.29 is higher than IQL's 33.73, but a mean over a coin
flip is not progress I can trust — a single lucky seed dragging up two poor ones. This is the
multimodal-basin failure I called out: Maze2d's value landscape has many goal approaches, the actor
commits hard to whatever high-value pocket its critic happens to carve, and the two design choices that
let TD3+BC commit — a sharp deterministic actor ascending a critic with spurious peaks, and a *single,
fixed* BC coefficient α = 2.5 normalized only by mean|Q| — mean different seeds fall into different basins
and stay there. The 8.69 seed is a critic whose peak landed in a bad basin and an actor that BC-anchored
itself there, with no mechanism to ever escape because there is no fresh data to contradict it.

It is worth seeing what these numbers do to the task score, because the ladder is ranked on the geometric
mean across the three datasets and that changes where the leverage is. IQL's geometric mean is
(48.10·33.73·80.46)^(1/3) ≈ (1.31×10⁵)^(1/3) ≈ 50.7; TD3+BC's is (48.33·50.29·85.14)^(1/3) ≈
(2.07×10⁵)^(1/3) ≈ 59.1. So TD3+BC already moved the task score up about eight points, and almost all of
that came from Maze2d rising from 33.7 to 50.3 — the geometric mean is most sensitive to its *smallest*
factor, so the weak Maze2d term is the one dragging the whole product down, and lifting it is the single
biggest lever I have. But 50.3 there is a mean over 8.69 / 27.20 / 114.99, so the "gain" is riding on one
lucky seed; if I could make Maze2d reliably sit where its good seed already reached, the geometric mean
would jump far more than another few points on the two locomotion datasets ever could. That is the
quantitative reason my whole attention this rung is on Maze2d's *reliability*, not on squeezing HalfCheetah
or Walker: the smallest, flakiest factor governs the product.

So I have two distinct things to fix, and they correspond to the two roles the BC term is being asked to
play with one knob. First, the *variance*: the critic on Maze2d has sharp, seed-dependent peaks that the
deterministic actor overfits, and TD3's target smoothing alone is clearly not enough to flatten them —
the 56.8 std says so, and I can see why: the smoothing averages the target over a 0.2-std action
neighborhood, which erases knife-edges *at the next-state action* but does nothing about a peak the
*online* actor climbs into at the current state during the policy update. Second, the *coupling*: TD3+BC
uses one coefficient α to control both how much the *actor* clones the data and, implicitly, how
conservative the *critic target* is, when these are genuinely different jobs that want different strengths
on different datasets. HalfCheetah (a tight single policy) wants almost no BC pressure so the actor can
squeeze the last drop of value — its 48 with tie-level std says the anchor there could safely be loosened;
Walker2d wants a bit more to hold the gait that gave it the clean 85; Maze2d wants the *critic target*
pinned hard to in-data next-actions so it stops manufacturing the spurious high-value pockets in the first
place, which is a different job than pinning the *actor*. One global α cannot serve all three, and the
113%-CV Maze2d proves it cannot even serve one dataset reliably. The next rung is the minimal-but-decoupled
version of exactly this.

Start with the variance, because it is the cheaper and more general fix and it changes the architecture.
The critic peaks are a function-approximation pathology: an unconstrained MLP critic, fit by bootstrapped
regression on a fixed dataset, develops sharp ridges in action space where it extrapolates, and those
ridges are what the deterministic actor climbs into. The structural remedy is to make the critic
*smoother as a function of its input* by adding LayerNorm after each hidden activation. Here is the
mechanism concretely, because "LayerNorm helps" is not a reason. LayerNorm re-centers and re-scales each
layer's activation vector to unit statistics before the next linear map, which bounds the norm of the
signal flowing forward regardless of how large the pre-activations tried to grow. An OOD action produces
an extreme, out-of-family activation pattern; without normalization that extreme pattern passes through
the subsequent linear layers and gets *amplified* into a large output — the spurious peak — whereas
LayerNorm clamps the activation magnitude at each layer, so the same OOD input can no longer blow up
downstream. It regularizes the Lipschitz behavior of the network: the output cannot change arbitrarily
fast across nearby inputs, so the critic cannot present the actor with a knife-edge OOD pocket to overfit,
and the value extrapolation that drives offline overestimation is damped at the source, by the network's
geometry rather than by an explicit penalty. This is the single most important architectural change from
TD3+BC, and it is what lets me then *deepen* the networks safely: I move actor and critic from 2×256 to
3×256, because a deeper critic has more capacity to represent the maze's value structure, and the
LayerNorm is what keeps that extra capacity from turning into extra spurious peaks. (The actor stays
LayerNorm-free — it is the function I want sharp enough to commit, just anchored by BC; only the *critic*
needs the smoothing, because the critic is what the actor overfits.)

Let me do the budget arithmetic on that deepening, because the 1.05×-largest-baseline cap is a hard
constraint and I need to know 3×256 fits before I commit. Going from two to three hidden layers adds one
256×256 linear per network, which is 256·256 + 256 = 65,792 parameters, and the three LayerNorms add only
2·256 = 512 learnable affine parameters each, 1,536 per critic — negligible next to the 65k the extra
depth costs. A 3×256 critic over concatenated (s, a) is on the order of the first-layer term
(s_dim+a_dim)·256 plus two 65,792 hidden blocks plus the 3·512 LayerNorm affines plus the 256→1 head,
roughly 140k trainable parameters; two of them plus a 3×256 actor is a few hundred thousand — and
critically this is the *deepest* twin-critic-plus-actor stack that stays inside the cap, because a fourth
hidden layer would add another ~66k per net across three nets and I would be spending capacity I am
forbidden from spending on the ensembles that would otherwise be the obvious variance fix. So the budget
verdict is: 3×256 with LayerNorm is admissible and one layer more is not, which means the most capacity I
am allowed is best spent on a geometrically-regularized single deeper critic rather than on width or on
ensembles I cannot have.

Now the decoupling, which is the algorithmic core. TD3+BC put its BC penalty in *one* place — the actor
loss — and used it to do double duty. I split it into two penalties with two coefficients, each aimed at
one of the two ways OOD actions leak into the value. The first leak is the one TD3+BC handles: the
*actor* proposing OOD actions. I keep an actor BC penalty, actor_bc·(π(s) − a)², but now actor_bc is its
own knob, tunable per dataset to be tiny on HalfCheetah (where the actor should barely clone) and larger
on Walker2d (where it should stay closer to the gait data). The second leak is the one TD3+BC leaves
entirely unaddressed and which I now think is the real Maze2d culprit: the *critic target* itself. The TD3
target evaluates Q at π_target(s') + smoothing noise — an action the *policy* chose at the next state,
which can already be drifting OOD, and whose over-value then backs up through every Bellman step
regardless of what the actor does. This is the leak the target smoothing cannot reach, because smoothing
averages over a neighborhood of that drifted action but does not pull it back toward the data. So I add a
*second* BC penalty directly in the critic target: subtract critic_bc·(ã' − a'_data)² from the
bootstrapped value, where ã' is the (smoothed) policy next-action and a'_data is the *dataset's* recorded
next action at that transition — which the scaffold hands me directly as `next_actions` in the batch. This
pushes the critic's bootstrap to *prefer* targets whose next-action stays near what the behavior policy
actually did next, penalizing the value precisely when the target relies on a next-action that has drifted
away from the data. Its effect on the fixed point is direct: the value being bootstrapped is now
Q(s',ã') − critic_bc·‖ã' − a'_data‖², which is maximized (over the policy's freedom to choose ã') by
staying near a'_data, so the whole recursion is pulled toward valuing in-data continuations — conservatism
injected at the bootstrap, not just at the policy, and the thing that should stop Maze2d from ever forming
the spurious high-value pockets that one seed fell into and another could not find. Two decoupled BC
penalties — one on the actor, one on the critic target — is the whole decoupled-regularization idea, and
it is still minimal: no behavior model, no divergence estimator, just two L2 pulls toward dataset actions
with two coefficients.

Let me check this decoupled design degenerates correctly, because a good sanity test is that the new
method contains the old one as a special case. Set critic_bc = 0 and the critic target loses its
next-action penalty, collapsing back to the plain TD3 min-target; set actor_bc back to playing α's role
and drop the LayerNorm and the extra layer, and the actor loss is TD3+BC's again. So ReBRAC is TD3+BC
plus two orthogonal moves — a geometric one (LayerNorm depth) and an algorithmic one (the second BC
coefficient) — and at the zero-setting of both new knobs I recover the exact predecessor whose numbers I
am trying to beat. That is reassuring: I am not gambling on a wholesale redesign, I am adding two
independently-defensible terms to a base that already gave me a clean 85 on Walker and a tie on
HalfCheetah, and only the Maze2d lottery needs fixing.

I should also confront the tempting cheaper alternative and say why I reject it, because it would save me
the architecture change. Why not simply keep TD3+BC exactly and make its single α per-dataset — crank it
up on Maze2d to force the actor to clone harder? Walk it forward: a large actor-BC on Maze2d would indeed
pull the policy toward logged actions and cut the variance, but it would do so by *cloning the meandering
behavior data*, which is precisely the policy that scores in the 30s — I would be trading the 115 seed
away to lift the 8.69 seed, collapsing toward the behavior mean, not toward the goal. The lottery is not
that the actor commits; it is that the *critic* offers seed-dependent pockets to commit to. Cranking the
actor-BC treats the symptom (the actor's commitment) instead of the cause (the critic's peaks), so it
cannot both raise the mean and cut the variance — it can only cut the variance by lowering the mean. That
is the concrete reason the fix has to be at the critic (LayerNorm + critic-target BC), not a bigger dose
of the actor knob, and it is why per-dataset α on TD3+BC is a dead end rather than a shortcut.

The critic-target BC needs its plumbing checked, because it uses the sixth batch tensor the earlier rungs
ignored. The batch is [states, actions, rewards, next_states, dones, next_actions_data]; next_actions_data
is (B, a_dim), the dataset's recorded action at s' for each transition. The policy's smoothed next action
ã' = clip(π_target(s') + clip(noise), −a_max, a_max) is also (B, a_dim), so (ã' − next_actions_data)²
summed over the action axis is a per-transition (B,) penalty, aligned one-to-one with the (B,) min-target
Q — I subtract critic_bc·penalty elementwise before forming the (B,) bootstrap target, and everything is
inside the no-grad block so the penalty shapes the target value without back-propagating through the
target networks. Using a'_data (the *actual* logged next action) rather than the smoothed ã' as the anchor
is the point: ã' is what the policy *wants* to do next and may be OOD; a'_data is what the behavior policy
*did* do next and is in-support by definition, so penalizing their gap is exactly "charge the bootstrap
for relying on a drifted continuation." If I had anchored to the smoothed action itself the penalty would
be near zero always and buy nothing.

I keep the rest of the stabilized TD3 stack unchanged because it is still doing its job: twin critics
with a min target, clipped target-policy smoothing (noise 0.2, clip 0.5), delayed actor updates
(policy_freq = 2), Polyak targets. The actor loss keeps the reward-scale normalization from TD3+BC — I
scale the Q term by λ = 1/(|Q|.mean() + ε) so the per-dataset BC coefficients are not also fighting the
reward scale — and the actor ascends the first critic. The per-dataset hyperparameters are the price of
decoupling and they are read from the environment name, and I want to justify the *pattern* of the table
rather than just assert it. HalfCheetah takes actor_bc = 0.001, critic_bc = 0.01, lr = 1e-3: almost no
actor cloning because its 48-with-tiny-std told me there is no OOD trouble to guard against and the
binding constraint was the anchor being too tight, light critic conservatism, and a fast learning rate
because the single-policy data is easy to fit. Walker2d takes actor_bc = 0.05, critic_bc = 0.1, lr = 1e-3:
roughly fifty times HalfCheetah's actor cloning, because holding the gait mattered — its clean 85 came
from staying close to the walking data — and heavier critic BC to match. Maze2d takes actor_bc = 0.003,
critic_bc = 0.001, lr = 3e-4: light actor cloning so the actor stays *free to commit* to goal directions
(the thing that let one seed hit 115), *very* light critic-target BC, and a slow, careful learning rate —
because the maze's fix is supposed to come from the LayerNorm critic geometrically refusing to form the
spurious pockets, not from a heavy BC brute-forcing the value down, and a slow lr gives the LayerNorm
critic time to settle before the actor commits. Note the maze critic_bc is a *tenth* of HalfCheetah's and
a hundredth of Walker's: I am deliberately not leaning on the critic-target BC there, betting the
LayerNorm does the smoothing. The learning rates are applied directly to the Adam optimizers in the
trainer; the batch size stays the loop's default. The full scaffold module — the LayerNorm 3×256 critic,
the LayerNorm-free 3×256 actor, the per-env hyperparameter table, and the train step with both BC
penalties — is in the answer.

Let me trace the actor loss to confirm the normalizer and the new BC coefficient compose the way I
intend. The delayed actor update forms pi = actor(states), shape (B, a_dim); q = critic_1(states, pi),
shape (B,); the actor BC is ((pi − actions)²).sum(−1), a per-sample (B,) L2 to the logged action; and the
reward-scale normalizer is lmbda = 1/(|q|.mean().detach() + 1e-8), a scalar. The loss is
(actor_bc·bc_mse − lmbda·q).mean(): the RL term −lmbda·q has typical magnitude 1 (since lmbda·|q| ≈ 1 by
construction, the reward scale cancelling just as it did at the previous rung, only here the target
magnitude is 1 rather than α = 2.5 because ReBRAC folds the strength into the per-dataset actor_bc rather
than a global α), and the BC term is charged at the dataset's own actor_bc. So on HalfCheetah the actor
feels an RL pull of scale ≈ 1 against a BC pull of scale 0.001·‖pi − a‖² — the RL term dominates by three
orders of magnitude, which is exactly the "let the actor exploit the single-policy data" I wanted; on
Walker the BC coefficient 0.05 brings the anchor back to within a factor of ~20 of the RL term, holding
the gait. The +1e-8 in the denominator is the only guard against a degenerate all-zero critic early in
training and never otherwise matters. This confirms the two knobs are doing independent jobs: lmbda fixes
the RL scale across datasets, and actor_bc sets, per dataset, how hard to lean against it.

Here is the falsifiable bar against the prior numbers. On HalfCheetah I expect a clear jump over both
predecessors' ~48: the near-zero actor BC (0.001, essentially releasing the anchor that TD3+BC's α held)
plus a deeper LayerNorm critic should let the actor exploit the single-policy data harder without the
overestimation that capped it before — I am predicting the low 60s, and if HalfCheetah stays at 48 the
decoupling bought nothing there and the binding constraint was never the BC coupling. On Walker2d I expect
to hold or modestly beat TD3+BC's 85.14 with comparable tightness — the heavier per-dataset cloning (0.05)
should keep the gait stable and not disturb the clean half-point spread it already had. The decisive test
is Maze2d. TD3+BC's 50.29 was a 56.8-std coin flip spanning 8.69 to 114.99; if the LayerNorm critic and
the critic-target BC actually kill the spurious pockets, I expect Maze2d to come up *and* its seed variance
to come *down* relative to TD3+BC — a mean well above 50 with the per-seed spread no longer spanning 8 to
115, ideally with the failing 8.69-style seed lifted off the floor. That is the claim that separates this
rung from the last: not just a higher Maze2d mean, but a Maze2d that is high *and* no longer a lottery. If
the mean rises but the variance stays enormous, then the LayerNorm did not smooth the right thing and the
critic-target BC was too weak, and the next move would have to attack the multimodality directly. But if
the variance collapses while the mean climbs across all three datasets, the decoupled,
geometrically-regularized version has done what the single-knob TD3+BC could not, and this is the top of
the ladder I can reach inside the parameter budget.
