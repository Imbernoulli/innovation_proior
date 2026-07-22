OK, let me think this through from scratch. I have a robot arm in simulation with several joints, and at every timestep I have to emit a vector of torques, real numbers, one per joint. Same story for a car with steering and throttle, or for a walking body with many actuators. I want to learn a controller from reward alone, model-free, and I want the action choice to be cheap enough to run online. So the question is: what learning algorithm gives me a good continuous-action policy without solving a control problem inside every environment step?

The obvious thing to reach for is the deep Q-network recipe. It learns an action-value function `Q(s,a|θ)` with a neural net, stores transitions in replay, and uses a target network so the Bellman target does not move on the same step as the regressor. But its control rule is greedy: `a = argmax_a Q(s,a)`. Its target is greedy too: `y = r + γ max_{a'} Q(s',a')`. On a small discrete action set, the network emits one value per action and the max is just a reduction over a short vector.

Now put continuous torques in there. `a ∈ R^m`. A neural `Q(s,a)` does not hand me an `argmax`. I could run gradient ascent over `a`, or random shooting, or another optimizer, but then I am solving a non-convex inner problem for every action and for every bootstrap target in a minibatch. That defeats the point of a real-time controller.

Discretization looks tempting until I count. Seven joints with only `{−k, 0, k}` per joint already gives `3^7 = 2187` actions, before any fine torque resolution. The count grows exponentially with the number of actuators, and the grid also throws away metric structure: `0.30` and `0.31` are nearby torques, but two action labels do not know that unless the value function relearns it. Discretization is a dead end.

The max over actions is the thing I need to remove. It appears because the value function is being asked to produce both an evaluation and the greedy action. I can split those jobs. Keep a critic `Q(s,a)`, but introduce another differentiable function `μ(s)` whose job is to emit the action I will actually use. Then acting is one forward pass. The training question becomes: how do I move the parameters of `μ` so that `Q(s, μ(s))` increases?

The critic is differentiable in the action, and the actor is differentiable in its parameters. Using the policy-Jacobian-then-action-gradient convention, the local chain rule gives

`∇_θ Q(s, μ_θ(s)) = [∇_θ μ_θ(s)] [∇_a Q(s,a)]_{a=μ_θ(s)}`.

The action-gradient says which direction in action space would increase value at this state; the policy Jacobian pulls that direction back into parameter space. So the inner action optimizer is being amortized: instead of solving `argmax_a Q(s,a)` at every timestep, I train a network whose output moves toward actions the critic rates highly.

I still need to check that this is not just a heuristic for a fixed batch of states. Start from the deterministic value functions:

`V^μ(s) = Q^μ(s, μ_θ(s))`

and

`Q^μ(s,a) = E_{r,s'}[r + γ V^μ(s') | s,a]`.

Differentiate `V^μ(s)` with respect to `θ`. There are two pieces. The current action changes because `μ_θ(s)` changes, and future values change because all future actions are also produced by `μ_θ`. Holding the current action slot of `Q^μ(s,a)` open for the first piece gives

`∇_θ V^μ(s) = [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} + [∇_θ Q^μ(s,a)]_{a=μ_θ(s)}`.

The second term follows only the downstream dependence of `Q^μ` on the policy parameters while holding the current action fixed:

`[∇_θ Q^μ(s,a)]_{a=μ_θ(s)} = γ E_{s'~p(·|s,μ_θ(s))}[∇_θ V^μ(s')]`.

That is the useful recursion:

`∇_θ V^μ(s) = [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} + γ E_{s'}[∇_θ V^μ(s')]`.

The derivative of how the next-state distribution changes with the current action is not missing; it is inside `∇_a Q^μ(s,a)`, because `Q^μ` is the whole action-value function. The remaining recursive term is only the downstream dependence of future actions on `θ`. Unroll it:

`∇_θ V^μ(s_0) = E[Σ_{t≥0} γ^t [∇_θ μ_θ(s_t)] [∇_a Q^μ(s_t,a)]_{a=μ_θ(s_t)} | s_0]`.

I want to be sure I have the bookkeeping right before I trust this, because there is an easy way to double-count. The recursion produces one summand per timestep, each weighted by `γ^t`, and inside each summand the action-gradient `[∇_a Q^μ(s_t,a)]` differentiates *only the explicit action slot at step `t`*. My first instinct was to say "the action at step 0 sets up the whole discounted tail, so its gradient should carry the tail's discount too" — but that is exactly the trap, because the tail's dependence on `θ` is already accounted for by the later summands. Let me make a tiny problem and check by brute force. Take a single scalar action with `Q(s,a) = -(a-2)^2` (peak at `a=2`, so `r'(a) = -2(a-2)`) and a two-step episode where both steps play the deterministic action `θ`, `γ=0.9`. Then the true objective is `J(θ) = r(θ) + γ r(θ) = (1+γ)r(θ)`, whose derivative is just `(1+γ)r'(θ)`.

If I wrongly fold the tail into the step-0 term, I get `(1+γ)r'(θ)` from step 0 *plus* `γ r'(θ)` from step 1 — too big. The correct unrolling assigns `γ^0 r'(θ)` to step 0 (its own action slot only) and `γ^1 r'(θ)` to step 1, summing to `(1+γ)r'(θ)`. Numerically at `θ=0`: finite differences give `+7.600`, the wrong fold gives `+11.2`, and the correct sum `r'(0) + γ r'(0) = 4.0 + 3.6 = +7.600` matches. Checking `θ ∈ {1, 1.5, 3}` the correct sum reproduces the finite-difference gradient to four decimals every time (`+3.800`, `+1.900`, `−3.800`), while the wrong fold is off by a factor each time. So the rule is: one `γ^t`-weighted term per visited state, each with the action-gradient at that state's own action. Good — the unrolled form above is right.

Integrating over the start-state distribution gives

`∇_θ J = ∫ ρ^μ(s) [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} ds`,

where `ρ^μ(s)=Σ_{t≥0} γ^t P(s_t=s | μ)` is the unnormalized discounted visitation measure under the start-state distribution. If I instead sample from the normalized distribution `d^μ=(1-γ)ρ^μ`, the same expression has a constant `1/(1-γ)` in front, and that constant is absorbed by the step size. What I notice across both conventions is that the integral runs over states only — there is no integral over actions anywhere in the gradient.

That absence of an action integral matters for off-policy learning. A stochastic policy-gradient estimator has a sampled action and a score term, so off-policy data normally brings in action likelihood ratios. Here the action in the target policy is just `μ(s)`. If I collect data with a noisy behavior policy `β`, the critic can still learn `Q^μ` from transitions by using the target action `μ(s')` in the Bellman backup; the environment transition `(s,a,r,s')` is valid regardless of why action `a` was chosen. For the actor update, sampling states from replay changes the state weighting, and the formal off-policy actor-critic step drops a downstream term involving `∇_θ Q^μ`. So this is a behavior-weighted approximation, not a magic recovery of the original start-state gradient. But there is still no action-space importance ratio to pay. I need coverage of the states the learned policy will care about, not a high-variance correction over continuous actions.

I keep an actor `μ(s|θ^μ)` and a critic `Q(s,a|θ^Q)`. The deterministic Bellman equation has no action expectation at the next state:

`Q^μ(s,a) = E_{r,s'}[r + γ Q^μ(s', μ(s'))]`.

For a sampled transition with terminal flag `d`, the regression target has to stop bootstrapping at true terminals:

`y = r + γ(1-d) Q(s', μ(s'))`.

Let me sanity-check the `(1-d)` factor does what I mean on two concrete transitions, say `r=1`, `γ=0.99`, and a target critic that reports `Q(s',μ(s'))=50`. For a non-terminal step `d=0`: `y = 1 + 0.99·1·50 = 50.5`, the reward plus the discounted continuation. For a true terminal `d=1`: `y = 1 + 0.99·0·50 = 1`, exactly the reward with no future value leaking in. That is the behavior I want — at a terminal there is no `s'` to bootstrap from, and the mask zeroes the continuation cleanly. The critic minimizes mean-squared error to that target. The actor takes the chain-rule step that increases `Q(s, μ(s))`. In code, because optimizers minimize losses, the actor loss must be the negative value estimate: `loss_pi = -Q(s, μ(s)).mean()`. Minimizing that is gradient ascent on the critic.

A naive neural actor-critic still worries me. The critic target contains the same kind of self-reference that made deep Q-learning unstable: if I compute `y` with the live critic, every critic step moves both the prediction and the target. Replay fixes the data correlation, but the target can still chase itself. The discrete-action value recipe already has the right repair: compute the bootstrap with a target network. Here the target depends on two learned functions, the critic and the actor, so both need target copies. The stable target becomes

`y_i = r_i + γ(1-d_i) Q_targ(s'_i, μ_targ(s'_i))`.

Hard-copying target parameters every so often would work in principle, but the actor and critic are coupled: the critic target uses the target actor, and the actor update uses the live critic. Smooth tracking makes the regression target drift instead of jump. I use Polyak averaging with `ρ` close to one:

`θ_targ ← ρ θ_targ + (1-ρ) θ`, with `ρ=0.995`.

In code the natural way to write this in place is to multiply the target tensor by `polyak` and then add `(1-polyak)` times the live tensor. I should check this two-step in-place form actually equals the formula rather than something subtly different, because `mul_` then `add_` mutates the same tensor I am reading. Trace it on a scalar: target currently `10`, live parameter has moved to `0`, `ρ=0.995`. Step one, `t ← 10·0.995 = 9.95`. Step two, `t ← 9.95 + (1−0.995)·0 = 9.95`. The direct formula gives `0.995·10 + 0.005·0 = 9.95` — same value, so the in-place ordering is safe here because the live parameter is read from a separate tensor, not the one being mutated. And the per-update displacement is `9.95 − 10 = −0.05`, which is `(1−ρ)` of the gap `(0 − 10) = −10`; so the target closes a fraction `τ = 1−ρ = 0.005` of the distance to the live value each update, a roughly 200-update time constant. That slow tracking is the point.

Exploration is separate from the deterministic target policy because learning is off-policy. During data collection I can use `β(s)=μ(s)+noise`. In heavy physical systems I might choose a temporally correlated Ornstein-Uhlenbeck perturbation; with zero mean and unit time step its discretization is `x ← x + θ_ou(-x) + σξ`, which is the sign I want because the drift pulls the noise state back toward zero. For this PyTorch learner, I keep the behavior simpler: a long initial phase of uniform random actions, then independent Gaussian action noise, `a ← μ(s) + noise_scale * randn(act_dim)`, clipped to the symmetric action bounds. That is faithful to the off-policy argument: the noise only changes which transitions enter replay, while the critic target and actor loss remain about the clean deterministic policy.

The actor must output actions inside the environment bounds, so its final nonlinearity is `tanh`, scaled by `act_limit` under the usual symmetric-bound assumption; hidden layers use ReLU MLP blocks. The critic estimates a scalar for a state-action pair, so it concatenates observation and action and sends the combined vector through an MLP ending in one output, squeezed to shape `(batch,)`. The target networks are deep copies of the live actor-critic and have `requires_grad=False` because they move only by Polyak averaging. Adam updates both live networks. The code defaults are `hidden_sizes=(256,256)`, `pi_lr=q_lr=1e-3`, replay size `10^6`, minibatch size `100`, `start_steps=10000`, `update_after=1000`, `update_every=50`, `act_noise=0.1`, `gamma=0.99`, and `polyak=0.995`.

I seed the RNGs, build the environment and the actor-critic, copy it into targets, and store every transition in a FIFO replay buffer. While `t <= start_steps`, I act uniformly at random so the buffer is not a narrow slice of state space. After that, I act with `μ(s)` plus Gaussian noise. Once at least `update_after` interactions exist, every `update_every` environment steps, I run `update_every` gradient steps so the environment-step to gradient-step ratio stays one. The critic step minimizes `((Q(s,a)-backup)**2).mean()` with `backup = r + γ(1-d)Q_targ(s', μ_targ(s'))` computed under `torch.no_grad()`. The actor step freezes the critic parameters for efficiency and minimizes `-Q(s, μ(s)).mean()`. Then every target parameter gets the Polyak update.

At this point each piece has earned its place by a different argument: the deterministic actor exists because the greedy max over continuous actions was intractable; the actor update direction is the chain-rule gradient I just checked against finite differences; off-policy replay is admissible because the gradient carries no action integral and the Bellman backup is valid for any action that was actually taken; the terminal mask and the Polyak coefficient I traced numerically. None of it leans on the others being correct in a way I have not separately checked, so I can commit it to a compact PyTorch learner.
