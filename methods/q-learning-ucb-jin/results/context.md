# Context

## Research question

Most of the reinforcement-learning methods that work in practice are *model-free*: Q-learning, and its deep descendants (DQN, A3C, policy-gradient methods like TRPO) directly update a value function or a policy without ever building an explicit model of the environment's transition dynamics. They are online, they store only a value table (or a network), and they are more flexible than committing to a particular parametric model of the world. Against this, the prevailing belief — backed by empirical comparisons — is that model-free methods pay for these advantages with a *higher sample complexity*: they need more interaction with the environment to reach a given level of performance than model-based methods that first estimate the dynamics and then plan.

The question is whether that gap is fundamental or merely a gap in our understanding. Concretely, in the cleanest possible setting — a finite, tabular, episodic Markov decision process with $S$ states, $A$ actions, and horizon $H$ per episode, where the agent must *explore on its own* (no simulator that lets it query an arbitrary state-action pair, and no resetting in the middle of an episode) — can a model-free algorithm achieve regret that grows like $\sqrt{T}$ in the total number of steps $T$? That $\sqrt{T}$ rate is what the best model-based methods achieve and is information-theoretically optimal in $T$.

## Background

**The episodic MDP and the Bellman backbone.** A tabular episodic MDP is a tuple $(\mathcal S,\mathcal A,H,\mathbb P,r)$: at each step $h\in[H]$ the agent in state $x$ takes action $a$, gets reward $r_h(x,a)\in[0,1]$, and transitions by $\mathbb P_h(\cdot|x,a)$; after $H$ steps the episode restarts. The transition kernels $\mathbb P_1,\dots,\mathbb P_H$ may differ across steps. Value functions $V_h^\pi,Q_h^\pi$ and their optima $V_h^\star,Q_h^\star$ obey the Bellman equations $Q_h^\star(x,a)=(r_h+\mathbb P_h V_{h+1}^\star)(x,a)$, $V_h^\star(x)=\max_a Q_h^\star(x,a)$, $V_{H+1}^\star\equiv 0$. Performance over $K$ episodes ($T=KH$) is measured by regret $\mathrm{Regret}(K)=\sum_{k}[V_1^\star(x_1^k)-V_1^{\pi_k}(x_1^k)]$, with the starting state $x_1^k$ chosen adversarially.

**Tabular Q-learning and its convergence theory.** The classic algorithm (Watkins 1989; Watkins & Dayan 1992) maintains a table $Q(x,a)$ and, on observing a transition $(x,a,r,x')$, performs the incremental update $Q(x,a)\leftarrow(1-\alpha_t)Q(x,a)+\alpha_t[r+\gamma\max_{a'}Q(x',a')]$. It is off-policy and converges to $Q^\star$ almost surely under the Robbins–Monro conditions on the step sizes: $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$, with every pair visited infinitely often (Tsitsiklis 1994; Jaakkola, Jordan & Singh 1994). These are *asymptotic* guarantees and say nothing about how the rate of convergence or the regret depends on the horizon.

**The learning-rate dichotomy (Even-Dar & Mansour 2003).** This is the load-bearing prior result. Studying convergence *rates* for Q-learning, they prove a sharp dependence on the *form* of the step size. With a *linear* rate $\alpha_t=1/t$, the partial sums behave like $\sum_{t=1}^T\alpha_t=O(\ln T)$, and the time to converge is *exponential* in $1/(1-\gamma)$ (the effective horizon) — and they exhibit a hard MDP where this exponential behaviour is unavoidable for the linear rate. With a *polynomial* rate $\alpha_t=1/t^\omega$, $\omega\in(1/2,1)$, the partial sums grow like $O(T^{1-\omega})$ and convergence becomes *polynomial* in the horizon. The intuition they isolate is that $1/t$ spreads weight uniformly over all past samples, so any value can only be reached after $\sum\alpha_t$ has grown large — which for $1/t$ takes exponentially long in the horizon; a rate that forgets old samples faster lets recent information dominate.

**Optimism and exploration bonuses.** In bandits and in model-based RL, the principle that solves exploration is optimism in the face of uncertainty (OFU): inflate each estimate by a confidence width so that the imagined value is as high as is statistically plausible, act greedily under that optimistic view, and let the inflation shrink as visit counts grow. UCRL2 (Jaksch, Ortner & Auer 2010) builds confidence sets around the *transition model* and achieves $\tilde O(\sqrt{DSAT})$ regret with a matching $\Omega(\sqrt{DSAT})$ lower bound, where $D$ is the MDP diameter (the analogue of $H$ in the episodic case). UCB-VI (Azar, Osband & Munos 2017) sharpens this by adding the confidence bonus *directly to the value estimates* rather than to the transition matrix, and by using a *Bernstein* bonus scaled by the empirical variance of the next-state value; with the further observation that the *total* variance of the value along an episode is $O(H^2)$ rather than the naive $O(H^3)$ (a law-of-total-variance argument), it reaches $\tilde O(\sqrt{H^2SAT})$, matching the information-theoretic lower bound up to logs. All of these, however, store and update the full transition model.

## Baselines

- **Q-learning with $\varepsilon$-greedy (Watkins 1989).** Incremental tabular updates with dithering exploration.

- **Delayed Q-learning (Strehl, Li, Wiewiora, Langford & Littman 2006).** A model-free PAC algorithm: each $(x,a)$'s Q-value is updated only once per $m=\tilde O(1/\varepsilon^2)$ visits, replacing the old value by the batch average, with an optimism-style attempted-update rule. Translated to the episodic regret setting it gives $\tilde O(T^{4/5})$ regret.

- **UCRL2 (Jaksch, Ortner & Auer 2010).** Model-based: maintains confidence intervals on the empirical transition probabilities and rewards, and runs optimistic value iteration over the confidence set. Achieves $\tilde O(\sqrt{DSAT})$ regret.

- **Optimistic posterior sampling (Agrawal & Jia 2017).** Model-based, posterior-sampling flavour with an optimistic correction; $\tilde O(\sqrt{H^3S^2AT})$ in the episodic reduction.

- **UCB-VI / vUCQ (Azar, Osband & Munos 2017; Kakade, Wang & Yang 2018).** Model-based: bonus added to value estimates, Bernstein/variance-reduction refinements, reaching $\tilde O(\sqrt{H^2SAT})$ and meeting the lower bound up to logs (under equal transition matrices $\mathbb P_1=\dots=\mathbb P_H$; distinct kernels cost a $\sqrt H$). These methods estimate and store the full transition matrix.

- **Information-theoretic limit.** For the episodic setting with possibly distinct $\mathbb P_h$, the regret of *any* algorithm is at least $\Omega(\sqrt{H^2SAT})$ (a series of $H$ hard JAO-style two-state MDPs). This is the target to approach.

## Evaluation settings

The natural yardstick is *regret* on a finite tabular episodic MDP, $\mathrm{Regret}(K)=\sum_{k=1}^K[V_1^\star(x_1^k)-V_1^{\pi_k}(x_1^k)]$ with $T=KH$, reported as a function of $S$, $A$, $H$, $T$ and a failure probability $p$ (with $\iota=\log(SAT/p)$ the log factor); and the equivalent PAC sample-complexity question — number of episodes to return an $\varepsilon$-optimal policy $V_1^\star(x_1)-V_1^\pi(x_1)\le\varepsilon$ when $x_1$ comes from a fixed distribution. Alongside statistical efficiency, the relevant axes are *time* and *space* complexity, since model-free methods store only a Q-table ($O(SAH)$ space) instead of an $S^2A$ transition model. Hard instances for stress-testing exploration are the combination-lock MDP and the JAO two-state chains put in series. The transition kernels may be step-dependent ($\mathbb P_1,\dots,\mathbb P_H$ distinct), which is strictly harder than the equal-kernel assumption used by some model-based analyses.

## Code framework

The pieces that exist before the method does: an episodic-MDP rollout loop, a Q-table indexed by $(h,x,a)$, greedy action selection, and an incremental value update with open slots to be filled in.

```python
import numpy as np

class EpisodicMDP:
    """Environment: S states, A actions, horizon H, step-dependent kernels."""
    def __init__(self, S, A, H, P, r):
        self.S, self.A, self.H, self.P, self.r = S, A, H, P, r  # P[h]:(S,A,S), r[h]:(S,A)->[0,1]
    def reset(self): ...                      # returns initial state x_1 (adversary may choose)
    def step(self, h, x, a):                  # returns (reward, next_state)
        return self.r[h][x, a], np.random.choice(self.S, p=self.P[h][x, a])

class TabularValueLearner:
    """Online model-free learner: keeps a Q-table and visit counts, nothing else."""
    def __init__(self, S, A, H):
        self.S, self.A, self.H = S, A, H
        self.Q = np.zeros((H, S, A))          # initial value -> TODO set below
        self.V = np.zeros((H + 1, S))
        self.N = np.zeros((H, S, A), dtype=int)
        self._init_values()

    def _init_values(self):
        # TODO: choose the initial Q (and V) values.
        pass

    def step_size(self, t):
        # TODO: the learning rate as a function of visit count t.
        raise NotImplementedError

    def exploration_bonus(self, t):
        # TODO: the term added to the Bellman target after t visits.
        raise NotImplementedError

    def update(self, h, x, a, reward, x_next):
        self.N[h, x, a] += 1
        t = self.N[h, x, a]
        alpha = self.step_size(t)
        b = self.exploration_bonus(t)
        target = reward + self.V[h + 1, x_next] + b
        self.Q[h, x, a] = (1 - alpha) * self.Q[h, x, a] + alpha * target
        self.V[h, x] = min(self.H, self.Q[h, x].max())   # value clipped to its valid range [0,H]

    def act(self, h, x):
        return int(self.Q[h, x].argmax())

def run(env, learner, K):
    for k in range(K):
        x = env.reset()
        for h in range(env.H):
            a = learner.act(h, x)
            reward, x_next = env.step(h, x, a)
            learner.update(h, x, a, reward, x_next)
            x = x_next
```
