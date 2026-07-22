Let me start from what actually bothers me about how I train continuous-control agents. I optimize expected return, `max_π Σ_t E_{(s,a)~ρ_π}[ r(s,a) ]`, and under full observability the optimum of that is a deterministic policy — there's a single best action at every state, so the policy degenerates to a delta. Stochasticity, when I have any, is something I bolt on: I add noise to the actions, or I start a Gaussian policy with high variance and watch it shrink. The randomness is a means to an end (exploration), and the end is deterministic.

But there are situations where I genuinely want the *solution* to be stochastic, and multimodal at that. Picture a reward landscape with several roughly-equally-good behaviors — four symmetric goals around the agent, say. A unimodal policy has to pick one early; if it's slightly wrong about which is best, it commits and never recovers. A policy that keeps all the good modes alive explores far better, because it's still sampling every option until it's sure. And there's a second payoff: if I learn *all* the ways to move forward, that policy is a much better starting point for finetuning into "run forward fast" or "bound" than a near-deterministic expert that only knows one gait. Near-optimal policies under the standard objective are near-deterministic, which makes them terrible initializers.

So I want two things that pull against the standard setup. First, an objective that *rewards* being stochastic instead of treating entropy as something to be annealed away. Second, a representation for the policy that's expressive enough to actually be multimodal in a continuous high-dimensional action space — and that I can still sample from quickly enough to act online.

The objective part has a known shape. If I cast control as inference — put the reward into a graphical model as a factor and infer the distribution over actions — the optimal policy comes out stochastic, and what it's optimizing is reward *plus* entropy. Concretely:

`π*_MaxEnt = argmax_π Σ_t E_{(s,a)~ρ_π}[ r(s,a) + α H(π(·|s)) ]`,

where `H` is the policy's entropy at each state and `α` weights entropy against reward. There's a tempting cheaper alternative that is *not* this: Boltzmann/greedy exploration, which makes the action distribution at the current step proportional to `exp(Q)`, maximizing entropy *now*. That's myopic — it adds randomness this step but doesn't value *getting to states where I'll have lots of good options later*. The objective above is over the whole trajectory's entropy, so it actively plans toward future high-entropy states. That distinction is the whole point; I have to keep the entropy *inside* the long-horizon sum, not just at the leaf.

(One nuisance I'll note and set aside: with a discount, the precise objective is subtle, because in practice we discount rewards but not the state visitation. The thing that's actually optimized under a discount is `argmax_π Σ_t E_{(s,a)~ρ_π}[ Σ_{l=t}^∞ γ^{l-t} E[ r + α H | s_t,a_t ] ]` — discounted reward-plus-entropy accumulated from each state-action, weighted by how often the current policy visits it. I'll just carry `γ` through and use it; writing the exact discounted objective out is unwieldy and doesn't change the derivations.)

Now the representation. Prior maxent work used either tabular policies (fine in a grid, hopeless in continuous space) or a simple parametric family — a conditional Gaussian, a multinomial. And here's the worry: even if I make a neural net output the *parameters* of a Gaussian, the policy is still a Gaussian, and a Gaussian is unimodal. No amount of network capacity in the mean and covariance head changes the *shape* of `N(μ(s), Σ(s))`. But is that actually a problem, or does a sufficiently wide Gaussian quietly cover both modes anyway? Take a single state where the value landscape is genuinely bimodal — two equally good action regions. I'll model the long-horizon value as `Q(a)` with two equal bumps at `a=−1` and `a=+1`, and ask what the entropy-optimal policy `∝ exp(Q/α)` looks like versus the best Gaussian I could fit to it. Discretizing the action on a fine grid and computing the normalized `π ∝ exp(Q)` (temperature `α=1`), the policy has two local maxima, at `a≈−1` and `a≈+1` — it *is* bimodal. Its mean is `0` by symmetry. Now compare against the single Gaussian with that same mean and variance: the Gaussian's *mode* sits at `a=0`, which is the valley *between* the two good actions. Numerically the ratio `π(0)/π(1) ≈ 0.05`: the right policy is ~20× more likely to act at a true mode than at the spot where the Gaussian is most confident, and `KL(π‖Gaussian) ≈ 0.73` nats. So the moment-matched Gaussian doesn't "smear and survive" — it concentrates probability exactly where it shouldn't. That settles it: the expressiveness has to live in the *shape* of the distribution, not in a map to a fixed family's parameters.

So what distribution family can be arbitrarily multimodal? The check I just did points the way: I want something whose log-density can have several peaks of its own choosing. The energy-based form does exactly that — write `π(a|s) ∝ exp(−E(s,a))` for an energy `E` represented by a neural net. With a universal approximator for `E`, this is *any* conditional density: I match a multimodal shape by sculpting `E` to dip wherever the policy should peak, which is precisely what failed for the fixed Gaussian shape. The catch, which I'll have to deal with later, is the usual EBM catch: the normalizer `∫ exp(−E) da` is intractable, so I can't evaluate the density and I can't trivially sample.

Now I have to decide what `E` should be. I could treat the energy as a free function and learn it by some generic density-matching loss, but that throws away the structure of the problem — `E` ought to be tied to value, since high-value actions should be high-probability. The question is whether that connection is forced or whether I'm free to pick it. I derive what the optimal maxent policy looks like, with no prior commitment about its form, and see what energy falls out.

Set `α = 1` for the derivation (I can always divide rewards by `α` to recover the general case). Define the soft Q-value of a policy `π` as the entropy-augmented return from taking `a` in `s`:

`Q^π_soft(s,a) = r + E_{τ~π, s_0=s,a_0=a}[ Σ_{t=1}^∞ γ^t ( r_t + H(π(·|s_t)) ) ]`.

Now I want to *improve* a policy. With ordinary returns, the policy-improvement move is "act greedily w.r.t. Q." What's the analogue when there's an entropy bonus? I should pick the new policy at each state to maximize *entropy plus expected value*, one step of lookahead:

`maximize over ρ:  H(ρ(·|s)) + E_{a~ρ}[ Q^π_soft(s,a) ]`.

This is a constrained optimization over a distribution `ρ`. Let me just solve it. The thing to recognize is that `H(ρ) + E_ρ[Q] = E_ρ[ Q − log ρ ]`, and if I define `π̃(a|s) ∝ exp(Q^π_soft(s,a))` with normalizer `Z(s) = ∫ exp(Q^π_soft(s,a)) da`, then

`H(ρ) + E_{a~ρ}[Q^π_soft] = E_ρ[ Q − log ρ ] = E_ρ[ log(exp(Q)/ρ) ] = E_ρ[ log( Z · π̃ / ρ ) ] = −KL(ρ ‖ π̃) + log Z(s)`.

So maximizing entropy-plus-value over `ρ` is the same as *minimizing* `KL(ρ ‖ π̃)`, whose minimum is at `ρ = π̃`. The improving policy is exactly `π̃(a|s) ∝ exp(Q^π_soft(s,a))`. So the energy-based form is not a modeling choice I'm imposing — it falls out of the one-step maxent improvement, with the (soft) Q-function sitting as the negative energy. The earlier question — free function or forced? — answers itself: I don't get to pick the energy; the entropy bonus pins it to `−Q_soft`.

And it genuinely improves: define `π̃(·|s) ∝ exp(Q^π_soft(s,·))` and I claim `Q^{π̃}_soft ≥ Q^π_soft` everywhere. From the inequality I just derived, `H(π(·|s)) + E_{a~π}[Q^π] ≤ H(π̃(·|s)) + E_{a~π̃}[Q^π]`. Expand `Q^π_soft` one step and substitute the better one-step policy at the next state, then again, then again — telescoping the entropy-plus-value bound forward:

`Q^π_soft(s,a) = E_{s_1}[ r_0 + γ( H(π(·|s_1)) + E_{a_1~π}[Q^π(s_1,a_1)] ) ]`
`             ≤ E_{s_1}[ r_0 + γ( H(π̃(·|s_1)) + E_{a_1~π̃}[Q^π(s_1,a_1)] ) ]`,

and recursing on the inner `Q^π(s_1,a_1)` the same way, I keep replacing `π` by `π̃` at each future step, accumulating discounted reward-plus-entropy under `π̃` the whole way down, until in the limit the right side is exactly `Q^{π̃}_soft(s,a)`. So `Q^π_soft ≤ Q^{π̃}_soft`. Iterating `π_{i+1} ∝ exp(Q^{π_i}_soft)` improves monotonically and converges to a `π_∞` that satisfies `π_∞(a|s) ∝ exp(Q^{π_∞}_soft(s,a))`. Since any non-optimal policy can be strictly improved by this step, the *optimal* maxent policy is itself a fixed point of it, i.e. it has the energy-based form. The Gaussian I ruled out numerically wasn't just inexpressive; it's the wrong functional class — the maxent optimum is Boltzmann in the soft Q.

Now I need the normalizer, because `π` is only defined up to it. Define the soft value as exactly the log-normalizer:

`V_soft(s) = log ∫_A exp( Q_soft(s,a') ) da'`,

so that `π(a|s) = exp( Q_soft(s,a) − V_soft(s) )` is properly normalized. Restoring `α`: `V_soft(s) = α log ∫ exp(Q_soft(s,a')/α) da'` and `π(a|s) = exp( (Q_soft(s,a) − V_soft(s))/α )`. `V` is a log-sum-exp over actions, a *soft* maximum. It should become the ordinary hard max `max_a Q` as `α → 0`, but the prefactor `α` and the `1/α` inside fight each other, so I compute the sweep directly. Take the same bimodal `Q` from before (two bumps of height `3` over a bounded action interval) and compute `V_soft(α)` for a sweep of temperatures by Riemann sum:

```
max_a Q = 3.000
α=3.00   V_soft=5.103
α=1.00   V_soft=2.777
α=0.30   V_soft=2.669
α=0.10   V_soft=2.832
α=0.03   V_soft=2.931
α=0.01   V_soft=2.972
```

So at small `α` the soft value does march up toward `3.000` (the hard max), confirming the zero-temperature limit — but the approach is *not* monotone, and at large `α` (e.g. `5.1` at `α=3`) it sits well *above* the max. That makes sense once I read it correctly: `V_soft = α log∫exp(Q/α) ≈ max_a Q + α·log(volume of near-optimal actions)`, so at high temperature the entropy/volume bonus dominates and inflates `V` above the greedy value, while as `α→0` the bonus vanishes and only the peak survives. So `V_soft` is the greedy value *plus an explicit entropy bonus*, collapsing to the hard value at zero temperature — exactly the behavior the maxent objective should produce.

What's the Bellman equation here? I have `π(a|s) = exp(Q^π_soft − V^π_soft)`. Plug it into the definition of `Q^π_soft`. The entropy term at the next state is `H(π(·|s')) = −E_{a'~π}[ log π(a'|s') ] = −E_{a'~π}[ Q^π_soft(s',a') − V^π_soft(s') ] = V^π_soft(s') − E_{a'~π}[Q^π_soft(s',a')]`. So

`Q^π_soft(s,a) = r + γ E_{s'~p}[ H(π(·|s')) + E_{a'~π}[Q^π_soft(s',a')] ] = r + γ E_{s'~p}[ V^π_soft(s') ]`.

The entropy term and the expected-Q term combine into exactly the log-normalizer `V_soft`. So the soft Bellman equation is clean:

`Q_soft(s,a) = r + γ E_{s'~p}[ V_soft(s') ]`,  with  `V_soft(s) = α log ∫ exp(Q_soft(s,a')/α) da'`.

It's the ordinary Bellman equation with the hard `max` over next actions replaced by a log-sum-exp. Now: can I solve this by iteration the way I solve ordinary Q-learning? Define the soft backup operator `T Q(s,a) = r + γ E_{s'}[ α log ∫ exp(Q(s',a')/α) da' ]`. I need this to be a contraction or the iteration won't converge. In the sup-norm: let `ε = ‖Q_1 − Q_2‖_∞ = max_{s,a}|Q_1 − Q_2|`. Then `Q_1 ≤ Q_2 + ε` pointwise, so

`α log ∫ exp(Q_1(s',a')/α) da' ≤ α log ∫ exp((Q_2(s',a') + ε)/α) da' = ε + α log ∫ exp(Q_2(s',a')/α) da'`,

and symmetrically it is at least `−ε + α log ∫ exp(Q_2/α)`. The `α` outside cancels the `1/α` inside the exponent — the cancellation is exact, independent of `α`'s actual value — so the soft values differ by at most `ε` everywhere. After the `γ` discount, `‖T Q_1 − T Q_2‖_∞ ≤ γ ε = γ ‖Q_1 − Q_2‖_∞`. It's a `γ`-contraction, so there's a unique fixed point and `Q ← TQ` converges to it — call this soft Q-iteration. (This contraction is the same one Fox et al. 2015 used for G-learning.)

So in tabular form I'd be done: iterate

`Q(s,a) ← r + γ E_{s'~p}[ V(s') ]`,  `V(s) ← α log ∫ exp(Q(s,a')/α) da'`,

and read off `π ∝ exp(Q/α)`. But I'm in continuous, high-dimensional space, with a neural-net `Q_θ`, and two things are intractable: the integral inside `V`, and sampling from `π ∝ exp(Q/α)`. I resolve them one at a time.

The `V` integral first. `V_soft(s) = α log ∫ exp(Q(s,a')/α) da'` — an integral over all actions. The move for any intractable integral is to make it an expectation under something I can sample. Introduce an arbitrary positive proposal `q_{a'}(a')` and importance-weight:

`V^θ_soft(s) = α log E_{a'~q_{a'}}[ exp(Q^θ_soft(s,a')/α) / q_{a'}(a') ]`.

Now I can estimate `V` from samples. The proposal is free; a uniform `q_{a'}` is simple but scales badly with dimension, while using the current policy is unbiased and concentrates samples where they matter. Either works.

Next, the fixed-point iteration is "make `Q = TQ` at *all* `(s,a)`," which is an infinite set of equality constraints — not something I can SGD. I want to turn "`Q(s,a) = target(s,a)` for all `(s,a)`" into a single scalar loss. The trick is the identity: for two functions `g_1, g_2`, `g_1(x) = g_2(x) ∀x ⟺ E_{x~q}[ (g_1(x) − g_2(x))² ] = 0` for *any* strictly positive density `q`. So the fixed point is equivalent to driving a squared-error expectation to zero, and I can minimize it by sampling:

`J_Q(θ) = E_{s~q_s, a~q_a}[ ½ ( Q̂^{θ̄}_soft(s,a) − Q^θ_soft(s,a) )² ]`,

with target `Q̂^{θ̄}_soft(s,a) = r + γ E_{s'~p}[ V^{θ̄}_soft(s') ]`. The `θ̄` are *delayed target parameters* — same reason as in deep Q-learning, a bootstrapped regression target that moves with the network you're training is unstable, so I freeze a copy and refresh it periodically. `q_s, q_a` just need to be positive over their spaces; in practice I use real transitions from rollouts of the current policy. So the critic is just bootstrapped value regression with the log-sum-exp soft value plugged into the target. That half is now a standard SGD loss.

The hard half is sampling from `π ∝ exp(Q^θ/α)` — I need it both to *act* (pick `a` in the environment) and, if I want, to generate the action samples for the `V` estimate. The energy is a general neural net, so this is the full EBM sampling problem. MCMC would do it, but I need samples *online*, while executing the policy in real time, and running a Markov chain to convergence at every timestep is hopeless. The alternative from the EBM literature is to train a *separate sampling network* that emits approximate draws in one forward pass.

So: learn `a = f^φ(ξ; s)` with `ξ ~ N(0,I)`, a state-conditioned net mapping noise to actions, whose induced distribution `π^φ(·|s)` should match the target EBM. The natural objective is

`J_π(φ; s) = KL( π^φ(·|s) ‖ exp( (Q^θ_soft(s,·) − V^θ_soft)/α ) )`.

How do I minimize a KL to an *unnormalized* target that I can only evaluate (up to the normalizer) through `Q`? This is exactly the problem Stein variational gradient descent solves. SVGD takes a set of particles and asks: what's the perturbation `a ← a + Δ(a)` that most reduces `KL(q ‖ p)`, if I restrict `Δ` to the unit ball of a reproducing-kernel Hilbert space with kernel `κ`? The answer (Liu & Wang 2016) is a functional of the *score* `∇ log p`:

`φ*(·) = E_{a~q}[ κ(a, ·) ∇_a log p(a) + ∇_a κ(a, ·) ]`.

Look at the two terms. The first is `κ`-weighted score: each particle drags its neighbors (weighted by kernel proximity) toward higher `log p`. Here `log p(a) = Q^θ_soft(s,a)/α` up to a constant, so `∇_a log p = (1/α) ∇_a Q`. The second term `∇_a κ` is a *repulsion*: it pushes nearby particles apart so they don't all pile onto the same mode. That repulsive term is exactly what I need for multimodality — without it the particles collapse to the single highest-`Q` point, the MAP action. With it, the cloud spreads to cover all the modes of `exp(Q/α)`. Writing the energy in terms of `Q` directly (rather than `Q/α`), the `1/α` rides on the score term and the descent direction for a particle `a = f^φ(ξ; s)` is

`Δf^φ(·;s) = E_{a~π^φ}[ κ(a, f^φ(·;s)) ∇_{a'} Q^θ_soft(s,a')|_{a'=a} + α ∇_{a'} κ(a', f^φ(·;s))|_{a'=a} ]`,

with `α` now multiplying the repulsion. The balance is right: more temperature `α` → stronger repulsion → broader, higher-entropy spread, which is exactly what a larger entropy weight should do.

But `Δf^φ` perturbs the particle *positions*; I have a *network* `f^φ` whose weights I want to train. That's the amortized step (Feng, Wang & Liu): choose a surrogate whose gradient update moves the network output along the SVGD direction, equivalently using `-Δf^φ` as the loss gradient for gradient descent or maximizing a surrogate with gradient `Δf^φ`. The chain rule gives the update direction

`∂J_π(φ;s)/∂φ ∝ E_ξ[ Δf^φ(ξ;s) · ∂f^φ(ξ;s)/∂φ ]`.

So I get a feed-forward sampler I can query in O(1), trained by backpropagating the Stein direction. And now something clicks: this sampler `f^φ` is an *actor*. It takes a state, emits an action; it's trained by backpropagating a gradient of the critic `Q` (the score term) into its weights. That's precisely the structure of DDPG's actor — DDPG backprops `∇_a Q` into a deterministic policy net to chase the argmax. The *only* difference is the `α ∇κ` repulsion term. Drop it, and my actor estimates the MAP action — it *becomes* DDPG (with a soft critic). Keep it, and the actor captures the entire multimodal EBM instead of one mode. So an entropy-regularized actor-critic is just approximate soft Q-learning, the actor playing the role of an approximate sampler from `exp(Q/α)`; and DDPG is the degenerate single-particle MAP case. That also explains why DDPG works off-policy: as an approximate Q-learning maximizer, it doesn't need on-policy data.

Soft Q-learning was just derived as value iteration, but there's a completely separate lineage — entropy-regularized *policy gradient* — that people already use. If the critic update above is really the right thing, the two should land on the same parameter update, though they needn't a priori. Suppose I parametrize *any* policy in Boltzmann form `π^φ(a|s) = exp( E^φ(s,a) − Ē^φ(s) )` with `Ē^φ(s) = log ∫ exp E^φ(s,a) da` the log-partition. The entropy-regularized policy gradient is

`∇_φ J = E_{(s,a)~ρ}[ ∇_φ log π^φ ( Q̂ + b(s) ) ] + ∇_φ E_{s}[ H(π^φ(·|s)) ]`.

The entropy gradient is `∇_φ H = −∇_φ E[ E_{a~π}[ log π^φ ] ] = −E_{(s,a)~ρ}[ ∇_φ log π^φ ( 1 + log π^φ ) ]` (the `1` comes from differentiating through the sampling distribution). Substitute `log π^φ = E^φ − Ē^φ`, pick the baseline `b(s) = Ē^φ(s) + 1` to cancel the stray constants, and the whole thing collapses to

`∇_φ J = E_{(s,a)~ρ}[ ( ∇_φ E^φ(s,a) − ∇_φ Ē^φ(s) ) ( Q̂ − E^φ(s,a) ) ]`.

Now put this next to the gradient of my soft Bellman error. Set `E^φ = Q_soft` and write the empirical target as `Q̂ = Â_soft + V_soft` (an advantage carrying no gradient, plus the value). Then `∇_θ J_Q = E[ (∇ Q − ∇ V)( Q̂ − Q ) ]`, and `∇ V = ∇ Ē` for the Boltzmann energy. Match this term by term against the policy-gradient expression above: `∇_φ E^φ = ∇ Q`, `∇_φ Ē^φ = ∇ V`, and the multiplier `(Q̂ − E^φ) = (Q̂ − Q)` is the same residual. So with `E^φ = Q_soft` and `ρ_{π^φ} = q_s q_a` the two expressions are the same gradient — they didn't have to be, but the algebra makes every factor line up. Entropy-regularized policy gradient and soft Q-learning turn out to be one update viewed two ways. (One thing the matching exposes: the advantage-form target makes `J_Q` independent of `V`, so `Q_soft` is pinned only up to an additive constant — harmless, since the Boltzmann policy is invariant to a constant shift in the energy.)

Now the practical machinery. The action space is bounded — `[-1,1]` per dimension — but SVGD lives on `R^d` and would saturate particles at the boundary. So I let `f^φ` emit *unbounded* raw values and squash with `tanh`, and the Q-network sees the squashed action. When I evaluate the target log-density for the SVGD score, I have to account for the squash: the change of variables for `a = tanh(u)` contributes `Σ_i log(1 − a_i²)` to `log π`, so the log-density I differentiate is `Q_soft(s,a) + Σ_i log(1 − a_i² + ε)`. (The `ε` is just numerical safety as `a → ±1`.)

The `V` estimate, concretely, with a *uniform* proposal on `[-1,1]^d`: `q_{a'} = (1/2)^d`, so `V(s) = α log [ (1/N) Σ_{j=1}^N exp(Q(s,a_j)/α) · 2^d ] = α( logsumexp_j Q(s,a_j)/α − log N + d log 2 )`. With `α=1` (folded into the reward scale), that's `logsumexp` over the `N` uniform samples, minus `log N`, plus `d · log 2`. So `reward_scale` is just the temperature knob — scaling the reward scales `1/α`.

The kernel: a radial basis function `κ(a,a') = exp( −‖a − a'‖² / h )` with a *median heuristic* bandwidth `h`. The idea (Liu & Wang) is to set `h` so that each particle's kernel-weighted neighborhood holds about the right number of other particles — use `h = (median of pairwise squared distances) / log K` where `K` is the particle count, clamped to a floor. With this `h`, `∇_a κ = −2 (a − a')/h · κ`. The bandwidth adapts per state, which matters because the action cloud's scale changes as training progresses.

For the SVGD update itself I need two expectations — one over the score/kernel particles and one over the particles whose network outputs I am moving. So I draw a single set of particles per state and split it: `n_fixed` fixed particles `a_i` (gradient stopped, used for the score and the first kernel argument) and `n_updated` particles `ã_j` (the sampler outputs that receive the backpropagated direction). The empirical Stein direction at state `s` is

`∇̂_φ J_π(φ;s) = (1/KM) Σ_j Σ_i ( κ(a_i, ã_j) ∇_{a'} Q(s,a')|_{a_i} + ∇_{a'} κ(a', ã_j)|_{a_i} ) ∇_φ f^φ(ξ̃_j; s)`,

averaged over states in the minibatch.

Concretely, the actor/sampler concatenates the observation with Gaussian noise, runs an MLP, and squashes:

```python
import tensorflow as tf

class StochasticNNPolicy:
    """The sampler f^φ: maps (state, gaussian noise ξ) → action. This is the actor."""
    def __init__(self, env_spec, hidden_layer_sizes, squash=True, name='policy'):
        self._action_dim = flat_dim(env_spec.action_space)
        self._observation_dim = flat_dim(env_spec.observation_space)
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples, self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)
        latents = tf.random_normal(latent_shape)        # ξ ~ N(0, I)
        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(              # f^φ before squashing
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)
        # tanh squash → keep particles inside the bounded action space
        return tf.tanh(raw_actions) if self._squash else raw_actions
```

The RBF kernel with the median-heuristic bandwidth and its gradient:

```python
import numpy as np, tensorflow as tf

def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    leading_shape = tf.shape(xs)[:-2]
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)      # ... x Kx x Ky x D
    dist_sq = tf.reduce_sum(diff**2, axis=-1)                   # ... x Kx x Ky
    # median of the pairwise squared distances (the median heuristic)
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(tf.reshape(dist_sq, input_shape),
                            k=(Kx * Ky // 2 + 1), sorted=True)
    medians_sq = values[..., -1]
    h = medians_sq / np.log(Kx)                                 # bandwidth h
    h = tf.maximum(h, h_min)
    h = tf.stop_gradient(h)
    h = tf.expand_dims(tf.expand_dims(h, -1), -1)
    kappa = tf.exp(-dist_sq / h)                                # κ = exp(-||a-a'||² / h)
    kappa_grad = -2 * diff / tf.expand_dims(h, -1) * tf.expand_dims(kappa, -1)  # ∇κ
    return {"output": kappa, "gradient": kappa_grad}
```

And the algorithm — the soft critic update, then the amortized-SVGD actor update:

```python
EPS = 1e-6

class SQL:
    def _create_td_update(self):
        # soft value of the next state via importance sampling with a UNIFORM proposal
        with tf.variable_scope('target'):
            target_actions = tf.random_uniform(
                (1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)
        self._q_values = self.qf.output_for(
            self._observations_ph, self._actions_pl, reuse=True)

        # V(s') = α[ logsumexp_a Q(s',a)/α  − log N  + d·log 2 ];  α folded into reward_scale
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))   # − log N
        next_value += self._action_dim * np.log(2)                          # + d log 2 (uniform)

        # soft Bellman target  Q̂ = r + γ V(s'),  with delayed target params (stop gradient)
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl
                              + (1 - self._terminals_pl) * self._discount * next_value)
        # soft Bellman error  J_Q = ½ E[(Q̂ − Q)²]
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)
        if self._train_qf:
            self._training_ops.append(
                tf.train.AdamOptimizer(self._qf_lr).minimize(
                    bellman_residual, var_list=self.qf.get_params_internal()))

    def _create_target_ops(self):
        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')
        self._target_ops = [tf.assign(tgt, src) for tgt, src in zip(target_params, source_params)]

    def _create_svgd_update(self):
        actions = self.policy.actions_for(self._observations_ph,
                                          n_action_samples=self._kernel_n_particles, reuse=True)
        # split particles: fixed a_i for score/kernel expectation, updated a~_j receive gradients
        n_updated = int(self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed = self._kernel_n_particles - n_updated
        fixed_actions, updated_actions = tf.split(actions, [n_fixed, n_updated], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)

        # log-density of the target EBM: Q_soft + tanh change-of-variables correction
        svgd_target_values = self.qf.output_for(
            self._observations_ph[:, None, :], fixed_actions, reuse=True)
        squash_correction = tf.reduce_sum(tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction
        grad_log_p = tf.gradients(log_p, fixed_actions)[0]      # ∇_a Q_soft (the score)
        grad_log_p = tf.stop_gradient(tf.expand_dims(grad_log_p, axis=2))

        kernel = self._kernel_fn(xs=fixed_actions, ys=updated_actions)
        kappa = tf.expand_dims(kernel["output"], dim=3)

        # Stein direction: κ·(score)  +  ∇κ (repulsion), averaged over the fixed particles
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel["gradient"], reduction_indices=1)

        # amortized SVGD: backprop the Stein direction into φ via grad_ys
        gradients = tf.gradients(updated_actions, self.policy.get_params_internal(),
                                 grad_ys=action_gradients)
        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)])
        if self._train_policy:
            self._training_ops.append(
                tf.train.AdamOptimizer(self._policy_lr).minimize(
                    -surrogate_loss, var_list=self.policy.get_params_internal()))
```

The training loop alternates exactly as the pieces dictate: act with `f^φ`, store `(s,a,r,s')` in a replay buffer, draw a minibatch, run the soft-Bellman critic update and the amortized-SVGD actor update, and periodically hard-copy the online params into the delayed target params.
