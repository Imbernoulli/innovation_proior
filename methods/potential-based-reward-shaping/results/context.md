# Context: reward shaping and the question of policy invariance

## Research question

In a sequential decision problem the entire "task" is encoded in one object: the reward function. Fix a Markov decision process and its reward, and the optimal policy is pinned down. So a sharp, elementary question arises the moment one starts engineering rewards by hand: **what freedom do we have to modify the reward function without changing which policy is optimal?**

This is not idle. Reinforcement-learning practitioners routinely add *extra* "shaping" rewards on top of the task's own reward, to guide and speed up learning — and this is often the difference between a problem being tractable and being hopeless. A simple pattern of helpful extra rewards can make an otherwise intractable problem easy. But the extra rewards change the objective the agent is actually optimizing. The agent maximizes (task reward + shaping reward), not the task reward. So the working question is the precise one: **for which shaping-reward functions F is every policy that is optimal under the shaped reward still optimal under the original task reward?** A solution would have to characterize exactly that class of F — ideally without assuming the agent (or designer) knows the transition dynamics, since in practice we usually do not.

Why it matters: if we cannot answer this, then every time shaping "works" we are gambling that we did not silently move the optimum. And the related inverse problem — recovering a reward (or model) from observed optimal behavior, as in structural estimation of MDPs and inverse reinforcement learning — is exactly governed by which reward transformations leave behavior invariant: that set is what is *unrecoverable*.

## Background

**MDPs and optimality.** A finite Markov decision process is a tuple $M=(S,A,T,\gamma,R)$: states $S$, a set of $k\ge 2$ actions $A$, transition probabilities $T=\{P_{sa}(\cdot)\}$ with $P_{sa}(s')$ the probability of moving to $s'$ on action $a$ in $s$, a discount $\gamma\in(0,1]$, and a (bounded, here deterministic) reward $R$. Rewards are sometimes written $R(s,a)$ but more generally $R(s,a,s')$ — the reward for taking $a$ in $s$ and landing in $s'$. A policy $\pi:S\to A$ has value $V^\pi_M(s)=\mathbb E[r_1+\gamma r_2+\gamma^2 r_3+\cdots\mid\pi,s]$. The optimal value is $V^*_M(s)=\sup_\pi V^\pi_M(s)$, the optimal action-value is

$$Q^*_M(s,a)=\mathbb E_{s'\sim P_{sa}(\cdot)}\!\big[R(s,a,s')+\gamma V^*_M(s')\big],$$

and the optimal policy is $\pi^*_M(s)\in\arg\max_{a}Q^*_M(s,a)$. $Q^*_M$ is the unique solution of the Bellman optimality equations

$$Q^*_M(s,a)=\mathbb E_{s'}\!\Big[R(s,a,s')+\gamma\max_{a'}Q^*_M(s',a')\Big],$$

which is the fixed point that value iteration and Q-learning converge to. Note that a policy is defined over *states*, so the *same* policy object can be applied to two MDPs that share $S$ and $A$ but differ in their rewards — which is exactly the comparison we will need.

**Regularity for the undiscounted case.** When $\gamma=1$ one assumes a distinguished absorbing state $s_0$ (the process stops on entering it, with no further reward) and that all policies are *proper*: from any state, any policy reaches $s_0$ with probability 1. This is a condition on $T$. Discounted MDPs ($\gamma<1$) have no absorbing state and are infinite-horizon; for them one writes $S\setminus\{s_0\}=S$.

**The single-step precedent.** Utility theory studies one-shot decisions and already answers the analogous invariance question there: for decisions *without* uncertainty any monotonic transformation of utilities preserves the optimal choice; *with* uncertainty (expected-utility maximization) only **positive linear transformations** are allowed [von Neumann and Morgenstern, 1944]. These facts underwrite how one designs evaluation functions in games and elicits utilities from people. The sequential, multi-step version of "which reward transformations preserve the optimum" had not been worked out.

**Shaping and its documented failure modes.** Shaping — supplying helpful auxiliary rewards to accelerate learning — has a substantial empirical track record [Dorigo and Colombetti, 1994; Mataric, 1994; Randløv and Alstrøm, 1998], with antecedents in the animal-training literature [Saksida et al., 1997]. The diagnostic facts that frame the problem are its *bugs*, all observed on existing systems:

- A bicycle agent rewarded for making progress toward a goal but **not** penalized for moving away learned to **ride in tiny circles** near the start, collecting the progress reward each time it happened to point goalward [Randløv and Alstrøm, 1998].
- A soccer robot given reward for **touching the ball** learned to sit next to the ball and **"vibrate,"** touching it as often as possible (David Andre and Astro Teller, reported in the same line of work).

In both, the shaping reward, not the task, dictated the learned behavior, and the result is clearly suboptimal for the original objective. The common structure: there is a closed loop of states the agent can traverse repeatedly and collect **net positive** shaping reward each lap — a positive-reward cycle that "distracts" the agent from the real task.

**Approximate invariance.** It was known that if rewards are perturbed by at most $\varepsilon$, the new policy's value is within $2\varepsilon/(1-\gamma)$ of optimal [Singh and Yee, 1994; Williams and Baird, 1994] — a robustness bound, but not a characterization of *exact* policy-preserving transformations.

## Baselines

The "prior art" here is the set of shaping heuristics people actually used, together with the theoretical tools they leaned on.

- **Distance-to-goal / progress shaping** [Randløv and Alstrøm, 1998]. Core idea: give a positive reward $r$ on any transition that moves the agent closer to the goal, zero otherwise — $F(s,a,s')=r$ if $s'$ is closer to the goal than $s$. Math/algorithm: add $F$ to the environment reward and run ordinary TD learning (e.g. Sarsa$(\lambda)$). **Gap:** it is not symmetric — moving away is not penalized — so a short there-and-back loop nets positive reward; the agent learns to farm the loop (the tiny-circles bug). Even a symmetric "closer/farther" version is an *ad hoc* state-pair reward with no guarantee that it does not shift the optimum.

- **Subgoal / event shaping** [Mataric, 1994; Dorigo and Colombetti, 1994]. Core idea: hand out reward for reaching designated subgoals or for performing useful primitive events (touching the ball, picking up a flag) — $F(s,a,s')=r$ when $a=a_1$ for $s$ in some set $S_0$, else $0$. **Gap:** an "event" the agent can repeat — touch, release, touch again — is again a positive-reward cycle (the vibration bug). Subgoal bonuses with the wrong magnitudes can also make a detour-through-a-subgoal beat the truly optimal path.

- **The general additive form.** For a fixed MDP and additive, memoryless shaping, the most general shaping is $R'=R+F$ with $F:S\times A\times S\to\mathbb R$ bounded. This is implementable even model-free: with only sample access to $M$ (take action, observe $s'$ and $R$), one simulates access to $M'$ by simply reporting $R(s,a,s')+F(s,a,s')$ on the same transition, since $M$ and $M'$ share $S,A,T$. So online/offline, model-based/model-free algorithms transfer unchanged from $M$ to $M'$. **Gap:** this is the whole problem — the form is far too permissive; without a constraint on $F$ it admits exactly the cycle-farming pathologies above.

- **The value/Q-learning machinery itself** [Sutton and Barto, 1998; Bertsekas and Tsitsiklis, 1996]. Q-learning estimates $Q^*$ by sampling the Bellman optimality backup; the optimal policy is the greedy argmax over $Q^*$. This is the tool a designer is trying to accelerate with shaping, and — crucially — it is *greedy over $Q$ at each state*, so anything that perturbs the *relative ordering* of $Q(s,\cdot)$ across actions at a state changes the policy. That observation is what a characterization will have to exploit.

## Evaluation settings

The natural yardstick is small, fully understood domains where the true optimum is known and "steps to goal" can be plotted against learning trials. The standard settings of the time:

- **Shortest-path grid worlds.** A 10×10 (and a larger 50×50) grid, start and goal in opposite corners, $-1$ reward per step (so the optimal policy is the shortest stochastic path), no discounting. Four compass actions; the intended move succeeds 80% of the time and is replaced by a uniformly random direction 20% of the time; bumping a wall keeps the agent in place. Metric: number of steps taken to reach the goal as a function of trial number, averaged over independent runs.
- **Grid world with ordered subgoals.** A 5×5 grid in which the agent must collect flags/subgoals in a fixed order $1,2,3,4,G$ before finishing; the state is augmented with which flags have been collected. Same step cost and stochastic dynamics. Metric: again steps-to-goal vs. trial.
- **Learning algorithm and protocol.** Sarsa (and Sarsa$(\lambda)$), $\varepsilon$-greedy exploration ($\varepsilon\approx0.1$), small learning rate, results averaged over many runs. These domains, the per-step-cost convention, and TD control were standard practice [Sutton and Barto, 1998].

## Code framework

A generic tabular TD-control loop can add an auxiliary reward from a placeholder function before the ordinary TD backup.

```python
import numpy as np

# --- environment: a generic MDP we can only sample (model-free access) ---
class MDP:
    def __init__(self, states, actions, gamma):
        self.states, self.actions, self.gamma = states, actions, gamma
    def reset(self): ...                 # return start state s
    def step(self, s, a): ...            # return (s_next, r_env, done) by sampling T, R

# --- auxiliary reward supplied by the designer ----------------------------
def shaping_reward(s, a, s_next, gamma):
    # Return the bounded additive term F(s, a, s_next).
    pass

# --- a standard tabular Q-learning / Sarsa control loop ------------------
def learn(mdp, episodes, alpha=0.02, eps=0.1):
    Q = {(s, a): 0.0 for s in mdp.states for a in mdp.actions}
    for _ in range(episodes):
        s = mdp.reset(); done = False
        while not done:
            a = eps_greedy(Q, s, mdp.actions, eps)
            s2, r_env, done = mdp.step(s, a)
            F = shaping_reward(s, a, s2, mdp.gamma)   # add auxiliary reward
            r = r_env + F
            target = r + (0.0 if done else mdp.gamma * max(Q[(s2, b)] for b in mdp.actions))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = s2
    return Q

def eps_greedy(Q, s, actions, eps):
    if np.random.rand() < eps:
        return actions[np.random.randint(len(actions))]
    return max(actions, key=lambda b: Q[(s, b)])
```
