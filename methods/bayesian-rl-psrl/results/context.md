# Context: efficient exploration in unknown Markov decision processes

## Research question

An agent is dropped into an unknown environment modeled as a finite-horizon Markov decision process and must act to maximize the reward it accumulates over time. The environment is a tuple $M = (\mathcal{S}, \mathcal{A}, R^M, P^M, \tau, \rho)$: a state space $\mathcal{S}$ of size $S$, an action space $\mathcal{A}$ of size $A$, a reward distribution $R^M_a(s)$ supported on $[0,1]$, transition kernel $P^M_a(s'\mid s)$, a known horizon $\tau$, and an initial-state distribution $\rho$. Interaction is episodic: the agent lives through repeated episodes of length $\tau$, beginning at times $t_k = (k-1)\tau + 1$.

The agent does **not** know $R^M$ and $P^M$. It must learn them from the rewards and transitions it observes, and this is where the difficulty lives. To learn that a state-action pair is good, the agent must visit it; but visiting a pair it suspects is bad costs reward now. This is the exploration-exploitation tradeoff. The precise object to control is **regret**: against the (unknown) optimal policy $\mu^*$ for the true MDP $M^*$,

$$\text{Regret}(T,\pi) = \sum_{k=1}^{\lceil T/\tau\rceil}\Delta_k,\qquad \Delta_k = \sum_{s}\rho(s)\big(V^{M^*}_{\mu^*,1}(s)-V^{M^*}_{\mu_k,1}(s)\big),$$

where $V^{M}_{\mu,i}(s)=\mathbb{E}_{M,\mu}\big[\sum_{j=i}^\tau \overline R^M_{a_j}(s_j)\mid s_i=s\big]$ is the value of running policy $\mu$ in MDP $M$ from step $i$. A solution must achieve sublinear regret — ideally $\tilde O(\sqrt{T})$, the best achievable rate — while remaining computationally cheap and able to absorb whatever prior knowledge an agent has about its environment. The pain point that makes this hard: a naive agent that plugs a point estimate of the unknown MDP into a planner and acts greedily systematically *overstates its own knowledge*, explores too little, and can lock onto a suboptimal policy permanently.

## Background

The dominant paradigm for provably-efficient exploration is **optimism in the face of uncertainty** (OFU), traceable to Lai and Robbins's 1985 bandit work. The recipe: from observed data, build a high-probability confidence region around the unknown rewards and transitions; then inflate the agent's beliefs to the most favorable values inside that region; then plan and act optimally against this optimistic model. Poorly-understood state-action pairs get the largest optimism bonus, so the agent is drawn to visit them; as data accumulates the confidence region shrinks, the bonus fades, and behavior converges to optimal. Essentially every algorithm with polynomial sample-complexity or sublinear regret guarantees — $E^3$ (Kearns & Singh 2002), R-max (Brafman & Tennenholtz 2003), MBIE (Strehl & Littman 2008), UCRL2 (Jaksch, Ortner & Auer 2010), REGAL (Bartlett & Tewari 2009) — is built on optimism.

Two costs of optimism are well understood and motivate looking elsewhere. First, OFU is **statistically conservative**: to guarantee the optimistic model dominates the truth, the confidence sets must hold a worst-case mis-estimation *simultaneously in every state-action pair*, which is a far more demanding event than the truth being well-described overall — so the agent over-explores. Second, OFU is **computationally heavy**: planning is no longer "solve one MDP" but "optimize jointly over a whole family of plausible MDPs" (UCRL2's extended value iteration; REGAL has no tractable implementation at all). And the confidence sets themselves must be hand-designed per problem; for rich models, designing and computing with them is the bottleneck.

There is a separate, older idea. In 1933, William R. Thompson, studying how to allocate a scarce population between two medical treatments with unknown success probabilities, proposed **probability matching**: if $P$ is the posterior probability that treatment 1 is better, assign a fraction $f(p)=P$ of subjects to it. His own calculation shows the expected number sacrificed to the inferior treatment is then proportional to $2PQ < 1$ (with $Q=1-P$), strictly better than committing immediately to one treatment. The randomness is keyed to the *posterior uncertainty itself*: as evidence mounts and $P\to 1$, allocation concentrates on the better treatment automatically, with no externally-tuned exploration knob. This idea — sample an action according to the posterior probability it is optimal — lay largely dormant in the bandit literature for decades. Empirical studies (Scott 2010; Chapelle & Li 2011) then showed it matches or beats state-of-the-art bandit methods, prompting a wave of theory (Agrawal & Goyal 2012; Kaufmann, Korda & Munos 2012). Most decisively for what follows, Russo & Van Roy (2013) proved a *Bayesian* regret bound for posterior sampling in bandits by showing its regret decomposes exactly like a UCB algorithm's — for *any* confidence sequence at once — without the algorithm ever constructing a confidence set.

In reinforcement learning, the Bayesian view was present but under-analyzed. Strens (2000) proposed "Bayesian Dynamic Programming": maintain a prior over MDPs, and at the start of each episode sample one MDP and act optimally for it. It was offered as a heuristic, with no guarantees. Kolter and Ng (2009), surveying such methods, wrote that "little is known about these algorithms from a theoretical perspective, and it is unclear what (if any) formal guarantees can be made." The Bayesian-RL algorithms that *did* carry guarantees fell back on optimism: BOSS (Wang et al. 2005) samples *many* MDPs and merges them into an optimistic composite; BEB (Kolter & Ng 2009) adds a count-based exploration bonus. A motivating empirical observation about the failure mode of naive exploration is the *RiverSwim* chain (Strehl & Littman 2008): six states in a line, a small reward at the near (left) end, a large reward at the far (right) end reachable only by repeatedly swimming against a current that usually pushes you back. A greedy agent grabs the small near reward and never discovers the large one; efficient exploration is essential. A second, subtler empirical fact about Bayesian exploration in MDPs: if one samples a fresh model (and policy) at *every timestep*, exploration collapses — on a long chain where only the far ends are informative, an agent that re-randomizes each step is exponentially unlikely to commit to the $N$-step walk needed to reach either end, so it never learns.

## Baselines

**UCRL2** (Jaksch, Ortner & Auer 2010). The canonical optimistic RL algorithm and the natural yardstick. It proceeds in episodes whose boundaries are *adaptive* — a new episode starts when the visit count of some state-action pair doubles. At each episode it forms confidence sets: an $L^1$ ball on each transition row, $\lVert \hat P_a(\cdot\mid s)-P_a^M(\cdot\mid s)\rVert_1 \le \sqrt{14 S\log(2SAt_k/\delta)/\max\{1,N_{t_k}(s,a)\}}$, and an interval on each mean reward, $\lvert \hat R_a(s)-R^M_a(s)\rvert \le \sqrt{7\log(2SAt_k/\delta)/\max\{1,N_{t_k}(s,a)\}}$. It then runs *extended value iteration* to find the optimal policy of the most favorable MDP in this whole family, and executes it. Its regret is $\tilde O(D S\sqrt{AT})$, where the diameter $D=\max_{s'\ne s}\min_\pi \mathbb{E}[T(s'\mid M,\pi,s)]$ measures how long it takes to travel between states. Gap left open: extended value iteration optimizes over a *family* of MDPs (expensive, and the conservative simultaneous-worst-case sets cause over-exploration), and the bound is in terms of $D$, which can be large and is not tailored to a known episode length $\tau$.

**REGAL** (Bartlett & Tewari 2009). Tightens UCRL2's diameter $D$ to the span $\Psi \le D$ of the optimal bias/value function, giving $\tilde O(\Psi S\sqrt{AT})$. Strictly stronger bound, but no computationally tractable implementation is known — it requires solving a regularized optimization over the MDP family that nobody knows how to do efficiently.

**Bayesian Dynamic Programming / Strens (2000).** Maintain a prior over MDPs; each episode, sample one MDP from the posterior and follow its optimal policy. Conceptually simple, computationally light (one MDP solve per episode), naturally prior-driven. Gap: presented purely as a heuristic — no regret or sample-complexity guarantee existed.

**BOSS** (Wang et al. 2005) and **BEB** (Kolter & Ng 2009). Bayesian algorithms that *do* carry guarantees, but only by reintroducing optimism: BOSS samples several MDPs and stitches them into an optimistic merged model; BEB adds an explicit count-based bonus. They show guarantees were thought to *require* an optimistic construction layered on top of the Bayesian belief. Gap: extra construction (and in BOSS, extra computation) whose necessity was unquestioned.

**Russo & Van Roy (2013)**, bandit posterior sampling. Not an RL algorithm, but the load-bearing analytical antecedent. It proves: for posterior sampling in bandits, $\text{BayesRegret}(T,\pi^{PS}) = \mathbb{E}\sum_t[U_t(A_t)-f_\theta(A_t)] + \mathbb{E}\sum_t[f_\theta(A^*_t)-U_t(A^*_t)]$ for *any* upper-confidence-bound sequence $U_t$, because conditioned on the history the sampled action $A_t$ and the optimal action $A^*_t$ are identically distributed (and $U_t$ is a deterministic function of the history). This converts any UCB regret bound into a Bayesian regret bound for posterior sampling, with no confidence sets in the algorithm. It is established only for the one-step bandit, where the unknown is a single optimal action; the multi-step MDP — with delayed consequences and an unobserved optimal *policy* — lies outside its scope.

## Evaluation settings

The natural testbeds for an exploration algorithm in tabular MDPs, all available beforehand: the **RiverSwim** chain (Strehl & Littman 2008) — six states, a current that resists rightward motion, a small left reward and a large right reward — engineered so that only an agent that explores against the current can find the optimal policy; and **randomly generated finite MDPs**, e.g. drawn with $S=10$ states and $A=5$ actions from a prior. Protocol: run episodes of fixed length (e.g. $\tau=20$) with a periodic reset, and also the no-reset infinite-horizon variant where episodes are triggered by the visit-count-doubling rule. Beliefs are expressed through conjugate priors — Dirichlet over each transition row (conjugate to the multinomial, updated by incrementing pseudo-counts) and normal-gamma over rewards (conjugate to the normal) — with a diffuse setting such as Dirichlet concentration $\alpha=1/S$, reward prior mean/variance $\mu=\sigma^2=1$, pseudocount $n=1$. The performance metric is cumulative regret as a function of time, averaged over Monte-Carlo runs, with UCRL2 (and its KL-confidence-set variant) as the comparison.

## Code framework

A scaffold for episodic learning in a tabular finite-horizon MDP. The pieces that already exist: a tabular MDP simulator, finite-horizon planning by backward induction (Bellman's dynamic programming, known since the 1950s), conjugate-belief bookkeeping over an unknown model, and an episodic interaction loop. The one empty slot is the exploration strategy: given the history, how do we choose the policy to run this episode?

```python
import numpy as np

class TabularBelief:
    """Belief over an unknown finite-horizon MDP's rewards and transitions."""
    def __init__(self, S, A, horizon):
        self.S, self.A, self.H = S, A, horizon
        # TODO: a representation of uncertainty over R and P per (s,a),
        # updatable from observed (s,a,r,s') tuples.
        pass

    def update(self, s, a, r, s_next):
        # TODO: incorporate one observed transition into the belief.
        pass

    def choose_episode_policy(self):
        """Return a policy to follow for this episode."""
        # TODO: this is the whole question: how the belief drives exploration.
        pass


def finite_horizon_optimal_policy(R, P, S, A, horizon):
    """Backward induction: optimal policy of a *known* MDP (R, P)."""
    V = np.zeros((horizon + 1, S))
    mu = np.zeros((horizon, S), dtype=int)
    for i in reversed(range(horizon)):
        Q = R + P @ V[i + 1]              # Q[s,a]
        mu[i] = Q.argmax(axis=1)
        V[i] = Q.max(axis=1)
    return mu


def run(env, belief, n_episodes, horizon):
    for k in range(n_episodes):
        mu = belief.choose_episode_policy()            # the slot to fill
        s = env.reset()
        for i in range(horizon):
            a = mu[i][s]
            r, s_next = env.step(s, a)
            belief.update(s, a, r, s_next)
            s = s_next
```
