# Context: provably efficient learning in an unknown MDP

## Research question

An agent is dropped into an environment it does not understand. The environment is a Markov decision process (MDP): a finite set of states, a finite set of actions, and at each step the agent picks an action, receives a bounded reward, and is carried to a next state drawn from a transition distribution it does not know. The agent's job is to accumulate as much reward as possible. It cannot consult an oracle; it must learn how the world works *and* how to act well in it, purely from the stream of experience it generates by acting.

The question is: **is there a learning algorithm that, in a general unknown MDP, is guaranteed to act near-optimally after only a polynomial amount of experience — polynomial in the number of states, the number of actions, the desired accuracy, the desired confidence, and the inherent time-scale of the problem?** Every step the agent spends trying an action just to find out where it leads is a step it is not spending collecting good reward. A solution must navigate this tension and come with a *finite, provable* resource bound, not merely a promise of optimality "in the limit."

## Background

**The MDP and its planning theory.** A (discounted) MDP is solved, when *known*, by dynamic programming. The optimal value function V\* satisfies the Bellman optimality equation V\*(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V\*(s') ], and value iteration — repeatedly applying the Bellman operator — converges to V\* because that operator is a γ-contraction in the max norm; after O( (1/(1−γ)) log(1/(ε(1−γ))) ) sweeps the greedy policy is ε-optimal, each sweep costing O(N²A) (Bellman 1957; Puterman 1994; Bertsekas & Tsitsiklis 1989, 1996). In the undiscounted average-reward setting one competes against the long-run average return; finite-time statements then require a notion of how fast a policy settles to its average. So *if the agent had the model*, near-optimal planning is a solved, polynomial problem. The entire difficulty is that the model is unknown.

**The exploration–exploitation dilemma.** Because the model is unknown, every action plays a double role: it earns (or fails to earn) reward, and it reveals (or fails to reveal) information. Exploiting current knowledge maximizes immediate expected return; exploring tries under-sampled actions to improve the model. This trade-off (Bellman 1987; Kumar & Varaiya 1986; Thrun 1992) is the central obstacle. In the special case of a single state with several actions — a multi-armed bandit — the Bayesian-optimal exploration strategy is given by the Gittins index, but that decomposition is special to bandits and does not extend to general MDPs.

**Asymptotic learning algorithms.** Two families of learning algorithms existed, both *asymptotic*:

- *Model-based (indirect).* Maintain empirical estimates of the transition probabilities and rewards from observed frequencies; run value iteration on the estimated model; act greedily (Jalali & Ferguson 1989; Gullapalli & Barto 1994). With infinite exploration the estimates converge and so does the policy.
- *Model-free (direct).* Watkins's Q-learning (Watkins 1989; Watkins & Dayan 1992) updates Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ] and provably converges to Q\* — *provided every state-action pair is visited infinitely often* (Jaakkola, Jordan & Singh 1994; Tsitsiklis 1994). SARSA and related on-policy variants share this profile.

**Optimism as a heuristic.** Practitioners had noticed that *optimistic* initialization helps: initialize value estimates high, and an agent acting greedily is automatically pulled toward under-explored options because their inflated values have not yet been corrected downward. This appears as Kaelbling's interval-estimation method (Kaelbling 1993), the exploration bonus in Dyna (Sutton 1990), Schmidhuber's curiosity-driven exploration (Schmidhuber 1991), and the exploration mechanism in prioritized sweeping (Moore & Atkeson 1993); Sutton & Barto (1998) catalog "optimistic initial values" as a simple trick effective on stationary problems.

**Finite-time results under restrictions.** Saul & Singh (1996) gave learning curves for a special sub-class of MDPs. Fiechter (1994) gave the first polynomial-time approximate-learning result, but in a weakened protocol: his agent has a *reset* action returning it to a distinguished start state at will. Schapire & Warmuth (1994) and Singh & Dayan (1998) gave non-asymptotic results for *prediction* in uncontrolled Markov processes.

**Why the very target must depend on a mixing/horizon time.** In the undiscounted average-reward setting, finite-time competition forces a time-scale into the statement. Consider two states, one action: state 0 with payoff 0, self-loop probability 1−ρ and probability ρ of moving to an absorbing state 1 with large payoff R. The asymptotic return is R, but reaching it from state 0 takes on the order of 1/ρ steps. No algorithm can approach R in many fewer than 1/ρ steps — the *policy itself* needs that long. So the strongest reasonable goal is to compete, in time polynomial in T, with the best policy whose own return settles within T steps; T is a genuine lower bound, not a defect of the algorithm. In the discounted setting the analogous quantity is the horizon time (1/(1−γ)) log(R_max/(ε(1−γ))), beyond which discounting has made the tail negligible.

## Baselines

- **Q-learning (Watkins 1989; Watkins & Dayan 1992).** Model-free stochastic approximation of the Bellman optimality equation. *Core math:* Q(s,a) ← (1−α) Q(s,a) + α [ r + γ max_{a'} Q(s',a') ]. *Guarantee:* converges to Q\* with probability 1 under standard step-size conditions if every (s,a) is visited infinitely often.

- **Model-based value iteration on empirical estimates (Jalali & Ferguson 1989; Gullapalli & Barto 1994).** Estimate P̂, R̂ from frequencies; plan by value iteration; act greedily. Convergence requires infinite exploration.

- **Optimistic-initialization heuristics (Kaelbling 1993; Sutton 1990; Schmidhuber 1991; Moore & Atkeson 1993).** Inflate value estimates or add exploration bonuses so greedy behavior explores under-sampled actions.

- **Fiechter (1994), efficient RL with reset.** First polynomial-time approximate-learning result for the discounted case. *Core idea:* model-based learning exploiting a reset-to-start action to repeatedly sample from a fixed start distribution.

- **Gittins-index / Bayesian-optimal exploration (Gittins).** Optimal exploration for a known prior over MDPs. *Core idea:* index policy decoupling the arms of a bandit; the general-MDP version is intractable and presumes a known prior.

## Evaluation settings

The natural yardstick is a finite, general MDP given by N states and k actions per state, with rewards in [0, R_max] (or normalized to [0,1]). The clean transition-sampling bound treats the immediate reward attached to a slot as a deterministic payoff learned when the slot is first sampled; if rewards are stochastic means instead of slot payoffs, a parallel Hoeffding budget for reward estimates has to be added. A learnable slot is a state-action pair; in the zero-sum stochastic-game generalization it is a state plus joint action, giving Nk² slots when each player has k actions. Two return criteria are standard: the **discounted** infinite-horizon return Σ_t γ^{t−1} r_t with discount γ ∈ [0,1), and the **undiscounted average** return lim_T (1/T) Σ_{t≤T} r_t. Performance is measured against the optimal value V\* (discounted) or against opt(Π^{T,ε}), the best asymptotic average return achievable by any policy whose ε-return mixing time is at most T (undiscounted). Accuracy ε and confidence δ are inputs. The resources to be bounded are three: number of actions taken (sample/learning complexity), per-step computation, and memory; the demand is that each be polynomial in N, k, 1/ε, 1/δ, R_max, and the mixing time T (undiscounted) or horizon time 1/(1−γ) (discounted), with the guarantee holding with probability at least 1−δ. The protocol is online and continuous: a single trajectory, no resets, the agent told its current state at each step. Allowed concentration tooling: Hoeffding's inequality for Bernoulli sample means, Pr(|n^{-1}Σ_i X_i − E X| > b) ≤ 2 exp(−2nb²), its bounded zero-mean variant, and the Pigeonhole Principle for counting visits.

## Code framework

The generic harness for a model-based RL algorithm has three pieces: a representation of the empirical model, an MDP planner (value iteration, which already exists), and the online loop. The unresolved part is what the agent should do about the slots it has not yet pinned down.

```python
import numpy as np

class EmpiricalModel:
    """Running sufficient statistics from observed transitions/rewards."""
    def __init__(self, n_states, n_actions):
        self.S, self.A = n_states, n_actions
        self.counts = np.zeros((n_states, n_actions, n_states))  # n(s,a,s')
        self.n_sa   = np.zeros((n_states, n_actions))            # n(s,a)
        self.reward_seen = np.zeros((n_states, n_actions), dtype=bool)
        self.reward_value = np.zeros((n_states, n_actions))      # learned slot payoff

    def record(self, s, a, r, s_next):
        if not self.reward_seen[s, a]:
            self.reward_value[s, a] = r
            self.reward_seen[s, a] = True
        self.counts[s, a, s_next] += 1
        self.n_sa[s, a] += 1

    def empirical_P(self, s, a):
        n = self.n_sa[s, a]
        return self.counts[s, a] / n if n > 0 else None

    def empirical_R(self, s, a):
        return self.reward_value[s, a] if self.reward_seen[s, a] else None


def value_iteration(P, R, gamma, horizon):
    """Standard MDP planner — pre-existing dynamic programming.
    Returns a (possibly non-stationary) near-optimal policy and Q."""
    # TODO: backward induction / contraction sweeps on the SUPPLIED model.
    pass


def build_planning_model(emp: EmpiricalModel, params):
    """Given running statistics, produce the transition/reward model
    handed to the planner each round."""
    # TODO: turn the running statistics into the model the planner sees.
    pass


def agent_loop(env, n_states, n_actions, gamma, horizon, params):
    emp = EmpiricalModel(n_states, n_actions)
    s = env.current_state()
    while True:
        P, R = build_planning_model(emp, params)   # TODO above
        policy, Q = value_iteration(P, R, gamma, horizon)
        a = policy(s)
        r, s_next = env.step(a)
        emp.record(s, a, r, s_next)
        s = s_next
```
