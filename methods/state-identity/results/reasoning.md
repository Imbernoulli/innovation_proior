Let me start from what a value function actually buys me and where it stops paying. I have a single function `V(s; theta)`, a linear model or a neural net, that estimates the long-term return from state `s`, and the whole reason it earns its keep is generalization: I fit it on the states I've visited, and because it's a smooth function of `s`, it gives me sensible values for states I haven't visited but that look similar. I can learn it from partial trajectories, off-policy, by bootstrapping `V(s)` toward `r + gamma V(s')`. That one idea — cache the utility of every state in one shared function and let approximation fill in the gaps — is the engine of the whole field. But there's a silent assumption baked into "the utility of a state": utility *toward what*. The value of a state is only defined relative to the reward, i.e. relative to the goal. The instant I change the goal — a new waypoint, a different thing to reach — the number `V(s)` cached for the old goal is just wrong, and I have to refit from scratch. Everything `theta` learned for the old goal is discarded.

And that's painful in a way that should be avoidable, because goals aren't independent of each other. Picture a maze and plot the optimal value function for one goal: the values are smooth inside a room and fall off a cliff at every wall. Now plot it for a *different* goal in the same maze. The numbers inside each room are different — you're heading somewhere else — but the cliffs sit at exactly the same walls, because the walls are a property of the *environment*, not of the goal. So the value landscapes for two related goals agree on their whole skeleton and differ only in the goal-dependent detail near where the reward sits. That's Foster and Dayan's observation, and it's a fact about the world that exists before I build anything: there is just as much structure in the space of goals as in the space of states. My single-goal approximator exploits the first kind of structure (generalize over `s` for one fixed goal) and throws the second kind away completely. Refitting per goal is paying full price for something I've mostly already learned.

So who has tried to share across goals, and where does each stop short? The most literal "many goals at once" architecture I know is Horde. The idea there is to keep a whole army of independent value functions — "demons" — each answering one goal-directed or predictive question, each with its own target policy, its own pseudo-reward, its own pseudo-discount that also encodes termination, all learned in parallel and off-policy from one shared stream of experience with gradient-TD. Formally each demon is a general value function `q(s,a; pi, gamma, r, z) = E[G_t | S_t=s, A_t=a, A_{t+1:T-1} ~ pi, T ~ gamma]`, approximated linearly as `q_hat(s,a;theta) = theta^T phi(s,a)`. I want to keep two things from this. First, the *vocabulary*: a goal can be specified as a pseudo-reward function plus a pseudo-discount that goes to zero at termination, so "reach `g`" is just a particular `(r, gamma)` and the value of a state for that goal is a perfectly well-defined GVF. Second, the demonstration that you really can learn many goal-directed value functions simultaneously, off-policy, from arbitrary behavior. But here's exactly where it stalls: the demons are *enumerated and learned separately*. Knowledge is a discrete bag of functions, one per question, and unless two demons happen to share parameters there's no transfer between them. If I want a value for a goal I didn't put a demon on, I'm stuck — there's nothing to evaluate. And the cost grows with the number of demons, not with how complex the environment actually is. Enumerating doesn't generalize; it just lists.

Foster and Dayan go further toward sharing, and theirs is the closest attempt. They treat the *set* of optimal value functions `{V^g(.)}` as data and fit an unsupervised mixture to it. Each state's value under a goal is a mixture of constants, `V_bar^g(x) = sum_i pi_i(x) e_i^g`, where the fragment-membership weights `pi_i(x) = softmax_i(a_i(x))` are goal-*independent* — that's the shared skeleton, the rooms of the maze, fit by EM across all goals — and the per-fragment constants `e_i^g` are goal-*dependent*. This is genuinely the right instinct: factor the value into a part that depends only on where you are and a part that depends on the goal. And it works — the fragments come out as rooms, and an actor-critic trained on a state representation augmented with fragment membership learns faster. But the representation is a mixture of *constants*: within a fragment, for a given goal, the value is a single number `e_i^g`. That's a very coarse model of the joint value surface `V(s,g)`; in their own experiments it has to be propped up by a tabular state-and-goal representation. It captures that structure *exists* across goals but doesn't give me a flexible, scalable approximator of value as a smooth function of both `s` and `g`.

OK. So line up what I want against what I have. Single-goal FA generalizes beautifully over states but takes no goal argument. Horde takes many goals but only by listing them, with no generalization to a new one. Foster–Dayan factors value into shared-plus-goal-specific but caps the goal-specific part at a constant per fragment and leans on a table. The thing none of them is doing is the obvious extrapolation of the one idea that already works. Function approximation took "a value per state" and made it "a smooth function of `s`" so it could generalize over unvisited states. The exact same move applied to the goal would be: take "a value function per goal" and make it "a smooth function of `g`" so it generalizes over goals I never trained on. Not a value function indexed by `g`, a value function *of* `g`. One approximator, `V(s, g; theta)`, that eats the goal as a second argument the same way it eats the state, and is fit jointly over `(s, g)` pairs so that nearby goals — by whatever similarity the network can learn — get consistent values, including goals never seen during training. Call it universal, in the sense that the one function covers a whole space `G` of goals rather than a single one.

Let me make the object precise before I worry about the network, using the GVF machinery from Horde. For a goal `g` I attach a pseudo-reward `R_g(s,a,s')` and a pseudo-discount `gamma_g(s)` that doubles as soft termination: it is zero at the goal and otherwise follows the environment's continuation discount. Then the general value function for goal `g` under policy `pi` is
```
V_{g,pi}(s) := E[ sum_{t=0}^inf (prod_{k=1}^{t} gamma_g(s_k)) * R_g(s_t, a_t, s_{t+1}) | s_0 = s ],
```
where the empty product for `t = 0` is `1`, actions come from `pi`, and the corresponding action-value is the one-step Bellman form
```
Q_{g,pi}(s,a) := E_{s'}[ R_g(s,a,s') + gamma_g(s') * V_{g,pi}(s') ].
```
The optimal policy for `g` is `pi_g*(s) = argmax_a Q_{g,pi}(s,a)`, with optimal value `V_g* := V_{g,pi_g*}` and `Q_g* := Q_{g,pi_g*}`. What I'm after are approximators `V(s,g;theta) ~= V_g*(s)` and `Q(s,a,g;theta) ~= Q_g*(s,a)` that hold across a large or even infinite goal set `G`.

Now, what is `g` concretely, and how does it enter the network? The cleanest and most common case — and I'll sidestep the thorny question of where goals come from — is that a goal *is a state*: `G ⊂ S`, and reaching the goal state is what's rewarded. Then the pseudo-reward and pseudo-discount specialize to
```
R_g(s,a,s') = 1   if s' = g and gamma_ext(s) != 0,   else 0
gamma_g(x)  = 0   if x = g,                            else gamma_ext(x)
```
with `gamma_ext` the external discount and external termination signal. Under that, `V_g*(s)` is the expected discounted first-hit reward for reaching `g` from `s` — a clean, well-posed target. And `G ⊂ S` is a gift for the architecture, because now `g` is *the same kind of object as* `s`: a state observation vector. I don't need to invent an encoding for goals at all. I can hand the network the raw goal observation exactly as I hand it the raw state observation. The most direct realization is to concatenate them, `F: S x G -> R`, stack `[s; g]` into one input vector, and run it through an MLP. That's it — that's the most flexible thing I can write down, because the MLP can model arbitrary interactions between `s` and `g` with no assumption about how value factors. Let me call this the concatenated form. It is the baseline against which anything fancier has to justify itself, and notice it costs me almost nothing over ordinary FA: same net, same training, one extra block of inputs. The "goal representation" here is the identity — `g` goes in as itself.

Stare at that for a second though, because the very directness that makes the concatenated form attractive is also what it gives up. By feeding `[s; g]` to one monolithic MLP, I've thrown away the structure I started this whole derivation from — the fact that the value surface factors into a goal-independent skeleton (the environment) and goal-dependent detail. The MLP *could* discover that factorization internally, but nothing in its shape encourages it, and I have no separate handle on "the part that depends only on the state." If I believe Foster and Dayan's picture — and the maze plots are pretty convincing — I should be able to do better by building the factorization into the architecture instead of hoping it emerges.

So suppose the value really does have low-rank-ish structure across `(s, g)`. Then instead of one joint net I want two streams: a state map `phi: S -> R^n` producing a state embedding, a goal map `psi: G -> R^n` producing a goal embedding, both into the same `n`-dimensional vector space, and a simple combiner `h: R^n x R^n -> R` that turns a pair of embeddings into the scalar value, `V(s,g;theta) = h(phi(s), psi(g))`. Make `phi` and `psi` general approximators (MLPs) and let `h` be something simple like an inner product. Why is this better than the concatenated MLP, when the concatenated MLP is strictly more expressive? Because expressiveness isn't the bottleneck — sample efficiency and the right inductive bias are. The two-stream form *commits* to the hypothesis that value factors through a shared low-dimensional code for states and a shared low-dimensional code for goals, which is exactly the structure I argued exists. With `G ⊂ S`, goals and states are the same kind of object, so `phi` and `psi` can even share their first layers — common low-level features for "what a state looks like" feed both the state-embedding and goal-embedding heads. That sharing is free transfer: features learned from seeing states as starting points help represent them as goals too.

And there's a special case worth pulling out. In a reversible environment the optimal value is symmetric, `V_g*(s) = V_s*(g)` — the cost of getting from `s` to `g` equals the cost from `g` to `s`. If I know that, I should *enforce* it, not relearn it: use a single network for both streams, `phi = psi`, with a symmetric combiner `h`. The natural symmetric `h` when `G = S` is a distance, `h(phi(s), psi(g)) = -||phi(s) - phi(g)||`, so two states that are mutually easy to reach get nearby embeddings. That's a beautiful consequence: train this and the embedding space becomes a *metric* on the environment, where small embedding distance means "few steps apart," which is a genuinely useful state representation in its own right. So I have a family — fully concatenated, two-stream asymmetric, two-stream symmetric/distance-based — trading expressiveness for structure, and the concatenated identity-goal form sits at the most-flexible, least-assumed end.

Now, how do I actually fit any of these? The most obvious route is supervised and end-to-end: I'm given (or can compute, in a small domain) the true values `V_g*(s)` for a set of observed `(s, g)` pairs, and I minimize the mean-squared error `E[ (V_g*(s) - V(s,g;theta))^2 ]` by SGD, backpropagating through whichever architecture I chose. This works for the concatenated MLP and for the two-stream nets alike, and it's the honest baseline for training.

But let me think about what end-to-end regression is really being asked to do for the two-stream case, because there's a hidden difficulty. Lay the data out as a big table `M`: one row per observed state `s`, one column per observed goal `g`, entry `M_{s,g} = V_g*(s)`. The table is sparse, noisy, and possibly huge. End-to-end I'm asking two coupled nonlinear networks to jointly reconstruct this table from scratch -- `phi` and `psi` have to simultaneously discover good embeddings *and* the mapping from raw `s`/`g` to those embeddings, all tangled together through the regression loss. That's a hard, slow optimization. But look at the structure of what I want: if `h` is a dot product, then `V(s,g) = phi(s)^T psi(g)`, which means each entry should look like `M_{s,g} ~= phi_hat_s^T psi_hat_g`, a low-rank factorization of the table. And low-rank factorization of a sparse, noisy matrix is a thoroughly solved problem -- matrix completion, with mature algorithms such as OptSpace for the sparse batch case and SVD when the table is dense and the combiner is a dot product. So I can split the hard joint problem into two easy stages. Stage one: ignore the raw inputs entirely, treat `M` as a matrix, and find row and column factors so that `M_{s,g} ~= phi_hat_s^T psi_hat_g`, which hands me *idealized target embeddings* -- a target vector `phi_hat_s` for each observed state row and `psi_hat_g` for each observed goal column. Stage two: now it's two *separate, standard* supervised regressions -- train `phi` to map raw `s` to its target `phi_hat_s`, and independently train `psi` to map raw `g` to its target `psi_hat_g`. The factorization figured out *what* embeddings achieve the values; the two regressions only have to learn *how to compute them from raw inputs*. Decoupling like this is dramatically faster than fighting the tangled end-to-end loss, and if I want, I can bolt on an optional third stage that fine-tunes `phi`, `psi`, and `h` together end-to-end.

Let me sanity-check that this family contains ordinary value-function approximation as its degenerate limit, because if it doesn't, I've over-engineered. If there is only one goal, then the goal stream has nothing meaningful to vary over: `psi(g)` is just a constant multiplier, and `phi(s)` carries the state-dependent value information. The two-stream dot product then collapses to an ordinary single-goal function approximator, just written in factored coordinates. Good: I haven't replaced the old tool, I've generalized it. And the embedding dimension `n` is now an interpretable knob -- how many shared factors of variation the value surface has across goals. Acting only needs the *argmax* structure, not the exact numbers, so the control-relevant structure may be lower-dimensional than a precise value reconstruction.

That handles the supervised setting, but the real point is reinforcement learning, where nobody hands me `V_g*(s)` -- I only have a stream of states, actions, and rewards, and I have to discover the values myself. Two ways to get there. The first reuses everything above: run a *finite* Horde of demons that learn `Q_g` for the goals in my training set, off-policy, in parallel -- that's exactly what Horde is for -- then build the table `M` from the demons' own estimates, `M_{t,g} = Q_g(s_t, a_t)`, factorize it as `M_{t,g} ~= phi_hat_t^T psi_hat_g`, and train `phi, psi` by regression toward the resulting target embeddings. The payoff is the whole reason for the exercise: the Horde only learned the *training* goals, but the factorized `phi, psi` generalize the value to goals *no demon was ever trained on*. The list-based method seeds a function-based one that goes beyond the list. Writing it as a procedure: initialize a transition history `H`; collect transitions into it; for each training goal, run its demon to learn `Q_g` off-policy from `H`; build `M` with rows indexed by transitions and columns by training goals, `M_{t,g} = Q_g(s_t, a_t)`; compute the rank-`n` factorization; initialize `phi, psi`; and finally do regression updates pushing the state-action map `phi(s_t, a_t) -> phi_hat_t` and the goal map `psi(g) -> psi_hat_g`, returning `Q(s,a,g) := h(phi(s,a), psi(g))`.

The second RL route skips the Horde detour and bootstraps the universal function *directly*, with a goal-conditioned Q-learning update. At a randomly sampled transition `(s_t, a_t, s_{t+1})` *and* a randomly sampled goal `g`,
```
target_g = r_g + gamma_g(s_{t+1}) * max_{a'} Q(s_{t+1}, a', g)
Q(s_t, a_t, g) <- Q(s_t, a_t, g) + alpha * (target_g - Q(s_t, a_t, g)),
```
where `r_g = R_g(s_t, a_t, s_{t+1})` is the pseudo-reward for that goal. The crucial twist over ordinary Q-learning is that I sample the goal too: every real transition becomes a training signal for *many* goals at once, not just the one I'm currently pursuing. Now I have to be honest about a wall here. Bootstrapping with function approximation is already the shakiest part of RL — the deadly triad — and I'm now doing it *while simultaneously generalizing over goals*, which makes the moving target move in two directions at once and the process is prone to go unstable. Two things calm it. The blunt one: much smaller learning rates (and accept slower convergence). The structural one: choose a *well-behaved* combiner `h`. My pseudo-rewards live in `[0, 1]` by construction (1 at the goal, 0 elsewhere), so if I pick an `h` whose outputs are bounded to match — a distance-based `h(phi(s), psi(g)) = gamma^{||phi(s) - psi(g)||_2}`, which is naturally in `(0, 1]` and decays with embedding distance — the values can't blow up past what the rewards justify, and the bootstrapping is far more stable. The cost is generality: committing to a distance-based bounded `h` is a stronger assumption than a free-form combiner. That's the trade — stability bought with structure — and it's worth it for the direct-bootstrapping regime.

Let me step back and read off which member of this family the simplest, most-robust deployment actually wants, because that's the form I'll ship as the baseline. When the goal is a raw state observation and I want the least-assumption, most-direct thing — no factorization hypothesis, no symmetry assumption, no bounded-`h` commitment — I take the concatenated form with the *identity* goal representation: feed the raw goal observation straight to the value and policy networks alongside the state. The "goal encoder" is literally the identity map, `phi(g) = g`, and there is no auxiliary representation loss because there's no separate representation to train — the value and actor networks themselves absorb whatever interaction between `s` and `g` matters, learned end-to-end by the agent's own value loss (a TD-style regression toward `r_g + gamma_g(s') * V(s', g)`) and actor loss. Everything I derived about two-stream factorization, matrix-completion two-stage training, symmetric distance embeddings, and direct bootstrapping is the richer end of the same family; the identity-goal concatenated value function is its plainest member, and it's the one a goal-conditioned agent reaches for first because it inherits the existing machinery unchanged and adds no trainable representation path to destabilize.

So let me write that plainest member as the code that actually drops into the harness. The harness expects a goal-encoding module exposing `encode_goal` and `compute_rep_loss`; the value and actor networks consume whatever `encode_goal` returns and add `compute_rep_loss` to the total objective. For the identity goal representation, `encode_goal` returns the goal unchanged, and `compute_rep_loss` returns zero -- the goal flows through raw, and the agent's own value/actor losses do all the learning. The one mechanical condition is that the downstream goal width must equal the raw goal-observation width, so the baseline runner auto-sets `rep_dim` to `obs_dim`.

```python
from typing import Sequence

import flax.linen as nn


class GoalRepresentation(nn.Module):
    """Identity goal representation: the raw state IS the goal vector.

    The value function is V(s, g) with g entering as the raw goal observation
    (UVFA concatenated form); no separate goal encoder is learned.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        # No parameters: the goal-to-vector map is the identity.
        pass

    def encode_goal(self, goals):
        # phi(g) = g.
        return goals

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        # No auxiliary representation loss: the agent's own value/actor losses
        # train V(s, g) and pi(a | s, g) end-to-end on the raw goal.
        return 0.0, {}

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)
```

And for completeness the computational core of the concatenated member -- the universal value/Q net and the direct goal-conditioned bootstrapping target -- is what that identity encoder feeds into:

```python
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class UVFAConcat(nn.Module):
    """Concatenated UVFA: V(s,g;theta) or Q(s,.,g;theta) = MLP([s ; g])."""

    hidden_dims: Sequence[int] = (256, 256)
    act_dim: Optional[int] = None

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        out_dim = 1 if self.act_dim is None else self.act_dim
        return nn.Dense(out_dim)(x)


def goal_as_state_terms(next_observations, goals, gamma_ext_t, gamma_ext_next):
    """R_g(s,a,s') and gamma_g(s') for the goal-as-state case."""
    reached = jnp.all(next_observations == goals, axis=-1)
    continuing = gamma_ext_t != 0
    rewards = (reached & continuing).astype(jnp.float32)
    gamma_next = jnp.where(reached, 0.0, gamma_ext_next)
    return rewards, gamma_next


def tabular_uvfa_q_update(q_sa, next_q_values, rewards, gamma_next, alpha):
    """Exact tabular update: Q <- Q + alpha * (target - Q)."""
    td_target = rewards + gamma_next * jnp.max(next_q_values, axis=-1)
    return q_sa + alpha * (td_target - q_sa)


def uvfa_td_loss(q_values, next_q_values, actions, rewards, gamma_next):
    """Differentiable fitted-Q loss toward r + gamma_g(s') max_a' Q(s',a',g)."""
    q_sa = jnp.take_along_axis(q_values, actions[..., None], axis=-1).squeeze(-1)
    target = rewards + gamma_next * jnp.max(next_q_values, axis=-1)
    target = jax.lax.stop_gradient(target)
    return jnp.mean((target - q_sa) ** 2)
```

Let me trace the causal chain once more. A value function caches the utility of each state and generalizes over states by approximation, but its utility is defined only relative to a fixed goal, so changing the goal discards everything learned — even though related goals share their entire value skeleton (the environment) and differ only in goal-dependent detail. Horde tackles many goals by enumerating independent demons, but enumeration can't answer a query for an un-enumerated goal and scales with the list. Foster and Dayan factor value into a goal-independent skeleton plus goal-dependent constants, the right instinct, but cap the goal-dependent part at a per-fragment constant and lean on a table. The missing move is the one that already worked once: as function approximation turned "a value per state" into "a smooth function of `s`," turn "a value function per goal" into a smooth function of `g` — a single `V(s, g; theta)`, universal over the goal space, generalizing to unseen goals. With `G ⊂ S`, a goal is a state, so it enters the network as a raw observation; concatenating `[s; g]` into one MLP is the most direct, least-assumed realization, and its goal encoder is simply the identity. Believing the value factors, a two-stream `phi, psi` with a combiner `h` builds that structure in, which (with a dot-product `h`) turns end-to-end regression into a low-rank matrix completion — letting me factorize the value table first and then learn `phi, psi` by two separate regressions — and which collapses to conventional FA when the goal set has only one member. In RL the universal function is learned either by seeding it from a finite Horde and factorizing the demons' estimates so it generalizes past the trained goals, or by directly bootstrapping a goal-conditioned Q-learning update that samples a goal alongside each transition and uses `gamma_g(s')` to stop bootstrapping at the reached goal. The plainest, most-robust member — the one a goal-conditioned agent deploys first — is the concatenated identity-goal value function: the raw state is the goal vector, no separate encoder, no auxiliary loss, the agent's own value and actor losses learning `V(s, g)` and `pi(a | s, g)` end-to-end.
