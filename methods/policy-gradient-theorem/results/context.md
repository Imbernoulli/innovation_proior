# Context: learning to act well by directly adjusting a stochastic policy

## Research question

We have an agent interacting with a Markov decision process: at each time step it is in a
state, picks an action, receives a scalar reward, and transitions to a new state according to
the environment's (unknown) dynamics. We want it to learn behavior that maximizes its
long-run reward — either the average reward per step (continuing tasks) or the discounted
return from a designated start state (episodic tasks).

The hard part is *generalization*. Real problems have enormous or continuous state spaces, so
the agent cannot keep a separate number for every state; it must use a function approximator —
a neural network, a decision tree, a linear-in-features model — that shares parameters across
states. The question is: **how do we train such an approximator so that the agent's behavior
provably improves, and ideally converges?** Concretely, suppose the agent's behavior is
controlled by a parameter vector θ. We would like to do gradient ascent on the performance
measure ρ(θ): nudge θ a little in the direction that most increases long-run reward. But the
performance depends on θ in a tangled way — θ changes which actions are taken, which changes
which states are visited, which changes the rewards, and the distribution over states the agent
spends its time in is itself a function of θ and of the unknown environment. If computing the
gradient required knowing how that state distribution shifts when θ changes, gradient ascent
would be hopeless to estimate from sampled experience. A solution would have to deliver a
gradient *estimable from trajectories the agent generates by acting*, with no model of the
environment and no derivative of the unknown state-visitation distribution.

## Background

**The MDP framework.** State s_t, action a_t, reward r_t. Transition probabilities
P^a_{ss'} = Pr{s_{t+1}=s' | s_t=s, a_t=a} and expected rewards R^a_s = E{r_{t+1} | s_t=s, a_t=a}.
A policy assigns to each state a distribution over actions. Two standard performance measures:
the *average reward per step* ρ = lim_{n→∞}(1/n)E{r_1+…+r_n}, under which the agent settles into
a stationary distribution d(s) over states; and the *discounted return* from a start state s_0,
ρ = E{Σ_{t=1}^∞ γ^{t-1} r_t | s_0}, with discount γ∈[0,1].

**Value functions and the Bellman equations.** The state-value V(s) is the expected return from
s; the action-value Q(s,a) is the expected return from taking a in s and following the policy
thereafter. They satisfy the Bellman consistency relations: in the discounted case
Q(s,a) = R^a_s + Σ_{s'} γ P^a_{ss'} V(s') and V(s) = Σ_a π(s,a) Q(s,a); in the average-reward
case the differential form Q(s,a) = Σ_{t=1}^∞ E{r_t − ρ}, equivalently
Q(s,a) = R^a_s − ρ + Σ_{s'} P^a_{ss'} V(s'). These recursions are the backbone of dynamic
programming and of temporal-difference learning.

**The prevailing wisdom: estimate a value function, act greedily.** For roughly a decade the
dominant approach put all approximation effort into estimating a value function and represented
the policy only implicitly, as the greedy policy that picks the highest-valued action in each
state. This worked in many applications.

**The diagnostic findings that set up the problem.** Two facts about the value-function-plus-
greedy approach are by now well documented. First, it is oriented toward *deterministic*
policies — the greedy rule always commits to one action — whereas the optimal policy can be
*stochastic* (e.g. Singh, Jaakkola & Jordan 1994 show that with state aliasing / partial
observability the best stationary policy may have to randomize). Second, and more corrosive:
an arbitrarily small change in an estimated value can flip which action is greedy, so the
induced policy is a *discontinuous* function of the value estimates. This discontinuity has
been identified as the key obstacle to convergence guarantees under function approximation
(Bertsekas & Tsitsiklis 1996). It is not just a theoretical worry: Q-learning, Sarsa, and
approximate dynamic programming have all been shown to *fail to converge to any policy* on
simple MDPs with simple function approximators (Baird 1995; Gordon 1995, 1996; Tsitsiklis &
Van Roy 1996; Bertsekas & Tsitsiklis 1996) — and this can happen even when, at each step, the
best value approximation in the mean-squared (or TD, or residual-gradient) sense is found
before the policy is changed.

**The score-function / likelihood-ratio idea.** Outside RL, a standard trick estimates the
gradient of an expectation taken over a parameterized distribution: ∇_θ ∫ P_θ(x) f(x) dx
= ∫ P_θ(x) ∇_θ log P_θ(x) f(x) dx, because ∇_θ P_θ = P_θ ∇_θ log P_θ. The gradient becomes an
expectation under the *same* distribution, so it can be estimated by sampling. A companion
fact: E_{x∼P_θ}[∇_θ log P_θ(x)] = 0, since ∫ P_θ = 1 has zero gradient. This is the lever that,
applied to a policy, might let one differentiate performance without differentiating the
environment.

## Baselines

**Value-function methods with function approximation (Q-learning, Sarsa, fitted value
iteration).** Maintain an approximate Q̂(s,a) or V̂(s); update it toward a Bellman target; act
greedily (or ε-greedily). *Core idea:* solve the prediction problem and read off control.
*Gap it leaves:* the greedy map is discontinuous in the estimates, so small estimation errors
can cause large policy swings; with function approximation these methods can oscillate or
diverge and carry no guarantee of converging to a good — or any — policy (Baird 1995; Gordon
1995). Gordon's (1995) fitted value iteration is provably stable but is not guaranteed to find
a *locally optimal policy*.

**REINFORCE (Williams 1988, 1992).** Represent the policy by its own parameters and update each
parameter by Δw_ij = α_ij (r − b_ij) e_ij, where e_ij = ∂ ln g_i / ∂w_ij is the *characteristic
eligibility* (the score of the unit's output distribution) and b_ij is a *reinforcement
baseline*. *Core result (Williams' Theorem 1):* the expected update lies along the true
gradient of expected reward, E{ΔW}·∇_W E{r} ≥ 0, and equals it when the learning rate is shared
— so (r − b_ij) e_ij is an *unbiased* estimate of ∂E{r}/∂w_ij, **for any baseline** b_ij that is
conditionally independent of the unit's output. The reason the baseline is free: summing the
output-distribution's derivative over all outputs gives zero (Σ_ξ ∂g_i/∂w_ij = 0, because the
probabilities sum to one), so the baseline term contributes nothing in expectation. Williams
also shows this extends to episodic, delayed-reward tasks by unfolding the recurrent network in
time. *Gap it leaves:* REINFORCE uses the actual return as its reward signal and learns without
any learned value function, so its estimates are high-variance and it learns slowly; it has
received relatively little attention as a result.

**Actor–critic / policy-iteration architectures (Barto, Sutton & Anderson 1983; Sutton 1984;
Kimura & Kobayashi 1998).** Keep a separately parameterized *actor* (the policy) and *critic*
(a value estimate); the critic's prediction error supplies a low-variance reinforcement signal
that drives the actor, and a running value estimate serves as a *reinforcement comparison*
baseline (Sutton 1984; Dayan 1991). *Core idea:* use a learned value function to reduce the
variance of the policy update — the empirically essential ingredient REINFORCE lacks. *Gap it
leaves:* despite long empirical success, there was no proof that an actor–critic update with
general function approximation climbs the true performance gradient or converges to a locally
optimal policy; the interaction between an approximate critic and the policy update was not
understood well enough to guarantee improvement.

**Related gradient expressions (Jaakkola, Singh & Jordan 1995; Cao & Chen 1997; Marbach &
Tsitsiklis 1998).** For the average-reward setting, and for the special case of tabular
partially-observable problems, expressions for the performance gradient in terms of the value
function were derived, and for tabular POMDPs one could guarantee a *positive inner product*
with the true gradient (enough to ensure improvement). *Gap:* these results were tied to
special cases (average reward only; tabular POMDP function approximation) and did not establish
*equality* with the gradient for *general differentiable* approximators, nor cover the
start-state discounted formulation.

## Evaluation settings

The natural yardsticks at the time are standard control MDPs solved from sampled interaction:
small gridworlds and corridor tasks (including ones with state aliasing, where a stochastic
policy is necessary), pole-balancing / cart-pole control, and the kind of simple MDPs on which
divergence of value-based methods had been demonstrated. The agent learns purely from sampled
trajectories — states, actions, and rewards generated by acting in the environment — with no
access to the transition model. The relevant comparison is against value-function methods
(Q-learning, Sarsa, fitted value iteration) and against plain REINFORCE, measured by long-run
average or discounted return achieved and by whether learning converges. A policy is represented
by a differentiable approximator (e.g. a soft-max over linear-in-features action preferences, or
a neural network outputting action probabilities); a value estimate, where used, is a second
differentiable approximator trained from the same trajectories.

## Code framework

The primitives that already exist: a simulator exposing `step`, a differentiable policy module
producing action probabilities, an optimizer doing parameter updates, and a trajectory-sampling
loop. What does *not* yet exist is the rule that turns sampled experience into a parameter
update with a guarantee behind it — that is the empty slot below.

```python
class Policy:
    """Differentiable stochastic policy a ~ pi(.|s; theta)."""
    def __init__(self, params):
        self.theta = params
    def action_probs(self, s):        # forward pass -> distribution over actions
        raise NotImplementedError
    def sample(self, s):              # draw a ~ pi(.|s)
        raise NotImplementedError
    def grad_log_prob(self, s, a):    # d/dtheta log pi(a|s) = grad pi / pi
        raise NotImplementedError

class ValueEstimate:
    """Optional second approximator trained from the same trajectories."""
    def __init__(self, params):
        self.w = params
    def value(self, s, a):            # f_w(s, a) (or a state value)
        raise NotImplementedError
    def grad_w(self, s, a):           # d/dw f_w(s, a)
        raise NotImplementedError

def collect_trajectory(env, policy):
    """Act in the env; return the (s, a, r) sequence. The visited states
       follow the agent's own on-policy dynamics -- no model needed."""
    traj = []
    s = env.reset()
    done = False
    while not done:
        a = policy.sample(s)
        s2, r, done = env.step(a)
        traj.append((s, a, r))
        s = s2
    return traj

def policy_performance_gradient(traj, policy, value=None):
    """Estimate d rho / d theta from sampled (s, a, r) alone: the rule
       that makes gradient ascent on long-run reward feasible, and the
       role (if any) of the value estimate.

       Open questions this slot must resolve:
         - does the estimate need the derivative of the state-visitation
           distribution (which we cannot sample)?            # TODO
         - what return / value signal multiplies grad_log_prob?  # TODO
         - can a state-dependent baseline be subtracted for free?  # TODO
    """
    raise NotImplementedError  # TODO

def train(env, policy, value=None, alpha=1e-2):
    while True:
        traj = collect_trajectory(env, policy)
        g = policy_performance_gradient(traj, policy, value)   # the empty slot
        policy.theta = policy.theta + alpha * g                # ascent on rho
        # TODO: if a value estimate is used, update its parameters too
```
