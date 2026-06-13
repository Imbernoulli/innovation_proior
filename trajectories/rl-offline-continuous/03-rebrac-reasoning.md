TD3+BC confirmed the bet *and* exposed exactly the fragility I flagged. On HalfCheetah it landed 48.33
(std 0.42) — a dead heat with IQL's 48.10, which is the expected result on a single-policy dataset with
no headroom: both "stay near π_β" recipes converge to the same near-data value. On Walker2d it cleared
IQL, 85.14 versus 80.46, and far tighter (std 0.26 versus IQL's 5.9): letting the deterministic actor
ascend the critic extracted more from the walking data than the hedged expectile did, and did it
stably. So exploitation paid off where there was room to exploit. But Maze2d is the story, and it is
ugly in precisely the way I predicted: mean 50.29, std **56.8**, with per-seed scores of 114.99, 27.20,
and 8.69. One seed essentially *solved* the maze; one was mediocre; one nearly failed. The mean is
higher than IQL's 33.73, but it is a mean over a coin flip — a single lucky seed dragging up two poor
ones. This is the multimodal-basin failure I called out: Maze2d's value landscape has many goal
approaches, the actor commits hard to whatever high-value pocket its critic happens to carve, and the
two design choices that let TD3+BC commit — a sharp deterministic actor ascending a critic with
spurious peaks, and a *single, fixed* BC coefficient α = 2.5 normalized only by mean|Q| — mean different
seeds fall into different basins and stay there. The 8.69 seed is a critic whose peak landed in a bad
basin and an actor that BC-anchored itself there.

So I have two distinct things to fix, and they correspond to the two roles the BC term is being asked to
play with one knob. First, the *variance*: the critic on Maze2d has sharp, seed-dependent peaks that the
deterministic actor overfits, and TD3's target smoothing alone is clearly not enough to flatten them —
the 56.8 std says so. Second, the *coupling*: TD3+BC uses one coefficient α to control both how much the
*actor* clones the data and, implicitly, how conservative the *critic target* is, when these are
genuinely different jobs that want different strengths on different datasets. HalfCheetah (a tight
single policy) wants almost no BC pressure so the actor can squeeze the last drop of value;
Walker2d wants a bit more; Maze2d wants the critic target pinned hard to in-data next-actions so it
stops manufacturing the spurious high-value pockets in the first place. One global α cannot serve all
three. The next rung is the minimal-but-decoupled version of exactly this.

Start with the variance, because it is the cheaper and more general fix and it changes the architecture.
The critic peaks are a function-approximation pathology: an unconstrained MLP critic, fit by bootstrapped
regression on a fixed dataset, develops sharp ridges in action space where it extrapolates, and those
ridges are what the deterministic actor climbs into. The structural remedy is to make the critic
*smoother as a function of its input* by adding LayerNorm after each hidden activation. LayerNorm
normalizes the pre-activation statistics layer by layer, which bounds how fast the critic's output can
change across nearby inputs — it regularizes the Lipschitz behavior of the network. A critic that cannot
spike arbitrarily across nearby (s, a) cannot present the actor with a knife-edge OOD pocket to overfit,
so the value extrapolation that drives offline overestimation is damped at the source, by the network's
geometry rather than by an explicit penalty. This is the single most important architectural change from
TD3+BC, and it is what lets me then *deepen* the networks safely: I move actor and critic from 2×256 to
3×256, because a deeper critic has more capacity to represent the maze's value structure, and the
LayerNorm is what keeps that extra capacity from turning into extra spurious peaks. (The actor stays
LayerNorm-free — it is the function I want sharp enough to commit, just anchored by BC; only the *critic*
needs the smoothing, because the critic is what the actor overfits.) The 256-width cap and the parameter
budget still bind: 3×256 with LayerNorm is the deepest twin-critic-plus-actor that stays inside the
1.05×-largest-baseline cap, so this is the most capacity I am allowed to spend, and I spend it on a
geometrically-regularized critic rather than on ensembles I am not permitted.

Now the decoupling, which is the algorithmic core. TD3+BC put its BC penalty in *one* place — the actor
loss — and used it to do double duty. I split it into two penalties with two coefficients, each aimed at
one of the two ways OOD actions leak into the value. The first leak is the one TD3+BC handles: the
*actor* proposing OOD actions. I keep an actor BC penalty, β₁·(π(s) − a)², but now β₁ is its own knob,
tunable per dataset to be tiny on HalfCheetah (where the actor should barely clone) and large on Walker2d
(where it should stay closer to the gait data). The second leak is the one TD3+BC leaves entirely
unaddressed and which I now think is the real Maze2d culprit: the *critic target* itself. The TD3 target
evaluates Q at π_target(s') + smoothing noise — an action the *policy* chose at the next state, which can
already be drifting OOD, and whose over-value then backs up through every Bellman step regardless of what
the actor does. So I add a *second* BC penalty directly in the critic target: subtract
β₂·(ã' − a'_data)² from the bootstrapped value, where ã' is the (smoothed) policy next-action and a'_data
is the *dataset's* recorded next action at that transition — which the scaffold hands me directly as
`next_actions` in the batch. This pushes the critic's bootstrap to *prefer* targets whose next-action
stays near what the behavior policy actually did next, penalizing the value precisely when the target
relies on a next-action that has drifted away from the data. It is conservatism injected at the
bootstrap, not just at the policy, and it is the thing that should stop Maze2d from ever forming the
spurious high-value pockets that one seed fell into and another could not find. Two decoupled BC
penalties — one on the actor, one on the critic target — is the whole decoupled-regularization idea, and
it is still minimal: no behavior model, no divergence estimator, just two L2 pulls toward dataset
actions with two coefficients.

I keep the rest of the stabilized TD3 stack unchanged because it is still doing its job: twin critics
with a min target, clipped target-policy smoothing (noise 0.2, clip 0.5), delayed actor updates
(policy_freq = 2), Polyak targets. The actor loss keeps the reward-scale normalization from TD3+BC — I
scale the Q term by λ = 1/(|Q|.mean() + ε) so the per-dataset BC coefficients are not also fighting the
reward scale — and the actor ascends the first critic. The per-dataset hyperparameters are the price of
decoupling and they are read from the environment name: HalfCheetah takes β₁ = 0.001, β₂ = 0.01, lr =
1e-3 (almost no actor cloning, light critic conservatism, fast learning on the easy single-policy data);
Walker2d takes β₁ = 0.05, β₂ = 0.1, lr = 1e-3 (heavier cloning to hold the gait); Maze2d takes β₁ =
0.003, β₂ = 0.001, lr = 3e-4 (light cloning, *very* light critic-target BC, slow careful learning —
because the maze needs the actor free to commit to goal directions while the LayerNorm critic, not a
heavy BC, supplies the smoothing). The learning rates are applied directly to the Adam optimizers in the
trainer; the batch size stays the loop's default. The full scaffold module — the LayerNorm 3×256 critic,
the LayerNorm-free 3×256 actor, the per-env hyperparameter table, and the train step with both BC
penalties — is in the answer.

Here is the falsifiable bar against the prior numbers. On HalfCheetah I expect a clear jump over both
predecessors' ~48: the near-zero actor BC plus a deeper LayerNorm critic should let the actor exploit the
single-policy data harder without the overestimation that capped it before — I am predicting the low
60s, and if HalfCheetah stays at 48 the decoupling bought nothing there and the binding constraint was
never the BC coupling. On Walker2d I expect to hold or modestly beat TD3+BC's 85.14 with comparable
tightness — the heavier per-dataset cloning should keep the gait stable. The decisive test is Maze2d. TD3+BC's
50.29 was a 56.8-std coin flip; if the LayerNorm critic and the critic-target BC actually kill the
spurious pockets, I expect Maze2d to come up *and* its seed variance to come *down* relative to TD3+BC —
a mean well above 50 with the per-seed spread no longer spanning 8 to 115. That is the claim that
separates this rung from the last: not just a higher Maze2d mean, but a Maze2d that is high *and* no
longer a lottery. If the mean rises but the variance stays enormous, then the LayerNorm did not smooth
the right thing and the critic-target BC was too weak, and the next move would have to attack the
multimodality directly. But if the variance collapses while the mean climbs across all three datasets,
the decoupled, geometrically-regularized version has done what the single-knob TD3+BC could not, and this
is the top of the ladder I can reach inside the parameter budget.
