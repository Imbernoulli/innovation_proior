# Context

## Research question

Two or more agents acting in a shared, dynamic environment, each optimizing its own
reward, is the natural model for a competitive interaction — a board game, a pursuit, a
market. The cleanest competitive case is a **two-player zero-sum Markov game**: a shared
state evolves stochastically, both players act simultaneously at each step, they receive a
single reward that one maximizes and the other minimizes, and the episode runs for a fixed
horizon. The solution concept is the **Nash equilibrium / minimax value** of the game: a pair
of policies neither of which can profitably deviate.

Recent practice had produced a striking empirical fact: agents trained purely by **self-play**
— repeatedly playing their current policies against *themselves*, with no human expert and no
fixed expert opponent — reached super-human strength at GO, Starcraft, and Dota 2. This
contradicts the older intuition that a strong opponent is required to learn a strong policy.
The question is whether self-play is *provably* sound and *provably* sample-efficient: starting
from no knowledge of the transitions or rewards, and *exploring* the game by playing it for a
bounded number of episodes, how many episodes are needed to find an approximate Nash
equilibrium — and what is the unavoidable minimum?

A solution must do three things at once that single-agent learning does not. It must measure
performance against the *best response to whatever policy is currently deployed* (an adaptive
opponent), not against a fixed target; it must handle the fact that the opponent's action
affects both the reward *and* the next state, so the opponent shapes exploration itself; and it
must produce policies that are good *against their best responses*, not merely policies whose
*value estimate* is near the equilibrium value. The goal is a self-play algorithm with a
sublinear, ideally $\tilde O(\sqrt{T})$, regret against best responses, and a matching
sample-complexity (PAC) guarantee for $\epsilon$-approximate Nash, under no structural
assumption on the game and no access to a simulator.

## Background

**Zero-sum Markov games (stochastic games), Shapley 1953.** Shapley introduced the model:
$N+1$ states (one terminal/absorbing), a payoff matrix $A^k$ and transition matrices $P^{kl}$
for each non-terminal state $k$, two players who at each state reveal mixed strategies
simultaneously over the matrix $A^k$, and a strictly positive stopping probability so the game
ends almost surely. His central object is the per-state **value operator**: for stationary
strategies, define $\mathrm{val}[\,B\,] = \min_y \max_x x^\top B y$ on a matrix $B$, and form the
auxiliary matrix $A^k(\delta)$ whose entries are $a^k_{ij} + \sum_l p^{kl}_{ij}\delta^l$ — the
immediate matrix payoff plus the discounted continuation value $\delta$. The map
$T(\delta)^k = \mathrm{val}[A^k(\delta)]$ is a contraction in $\ell_\infty$ with modulus
$1-s$ (where $s$ is the minimum stopping probability), using the elementary bound
$|\mathrm{val}[B]-\mathrm{val}[C]| \le \max_{ij}|b_{ij}-c_{ij}|$. Its unique fixed point is the
vector of game values, and the per-state minimax strategies at that fixed point are stationary
optimal. The lasting message: a multi-step zero-sum game is solved state by state, each state a
matrix game, glued by a Bellman/contraction backup.

**Matrix games and minimax, von Neumann 1928; Nash 1951.** For a single zero-sum matrix $B$,
von Neumann's minimax theorem gives $\max_x \min_y x^\top B y = \min_y \max_x x^\top B y$, and
the equilibrium is computable by linear programming. For a **general-sum** matrix game (two
*different* payoff matrices), Nash 1951 guarantees an equilibrium exists, but its computation is
not an LP. These are the one-step primitives; a Markov game is their multi-step lift.

**Classical Markov-game algorithms.** A long line (Littman 1994 minimax-Q; Littman 2001;
Hu & Wellman 2003 Nash-Q; Hansen et al. 2013; Filar & Vrieze) solves Markov games either with
the model *known*, or in the *asymptotic* limit of infinite data. Nash-Q in particular performs
the model-free update
$Q_h(s,a,b)\leftarrow(1-\alpha)Q_h(s,a,b)+\alpha(r_h+V_{h+1}(s_{h+1}))$ and then sets $V$ by the
*Nash value* of the per-state matrix game. These give no finite-sample, exploration-aware
guarantee: they assume the data already covers the state space.

**Non-asymptotic Markov-game results, under restrictions.** Wei et al. 2017 give a self-play
upper-confidence method but assume that, whatever one player does, the other can reach every
state — a reachability assumption that defuses the exploration problem. Jia et al. 2019 and
Sidford et al. 2019 give near-optimal sample complexity for *turn-based* games but assume a
**simulator** (generative model) that returns a sample transition for any queried
$(s,a,b)$; with a simulator, uniform querying already explores. R-max (Brafman & Tennenholtz
2002) makes no such assumption and has guarantees against adversarial opponents, but its
guarantee compares the *deployed* pair to the *Nash value* — a weaker statement than regret
against the per-episode best response, and it does not yield a self-play regret bound in this
setting.

**Single-agent optimistic RL — the template.** In a single-agent episodic MDP, the
exploration-exploitation problem is solved by *optimism in the face of uncertainty*. UCBVI
(Azar, Osband & Munos 2017) runs value iteration on the empirical model with an added
exploration bonus, maintaining an optimistic upper estimate $Q^{\mathrm{up}}$ that provably
stays above the optimal $Q^\star$ throughout; greedily following $Q^{\mathrm{up}}$ yields
$\tilde O(\sqrt{H^2 S A T})$ regret, matching the lower bound $\Omega(\sqrt{H^2 S A T})$. A key
analytic move is to concentrate the next-state *value function* as a whole (an $\epsilon$-net /
union-bound over the value class), rather than the transition probabilities, to control the
$S$-dependence. Q-learning with a UCB bonus (Jin, Allen-Zhu, Bubeck & Jordan 2018) achieves
$\tilde O(\sqrt{H^3 S A T})$ *model-free*: its online incremental update with learning rate
$\alpha_t=(H+1)/(H+t)$ uses each fresh sample exactly once, against a continuation value that is
statistically independent of that sample — the property that keeps its state-dependence
*linear* in $S$.

**Adversarial-MDP / no-regret learning.** A separate line learns against adversarial *reward*
sequences in an MDP (Zimin & Neu 2013; Rosenberg & Mansour 2019; Jin et al. 2019). In a Markov
game the opponent perturbs the *transition* as well as the reward, so these do not transfer; and
they learn a best response to the adversary, not a Nash equilibrium. The natural idea —
run two no-regret learners against each other and average — is on the table as an approach a new
method would have to either use or rule out.

## Baselines

- **UCBVI (Azar et al. 2017)** — single-agent optimistic value iteration; bonus-augmented
  empirical Bellman backup; $Q^{\mathrm{up}}\ge Q^\star$ invariant; $\tilde O(\sqrt{H^2SAT})$.
  *Gap*: one optimistic estimate and a single greedy policy; there is no second agent, no notion
  of optimism against an opponent's best response, and "greedy" is well-defined only because the
  per-state object is a vector, not a matrix.
- **Q-learning + UCB (Jin et al. 2018)** — model-free, online, $\alpha_t=(H+1)/(H+t)$,
  bonus $\beta_t=c\sqrt{H^3\iota/t}$, $\tilde O(\sqrt{H^3SAT})$, $S$-linear via sample
  independence. *Gap*: single agent; produces a Markov greedy policy with a *value* guarantee,
  which in a game would not certify performance against a best response.
- **Nash-Q / minimax-Q (Hu & Wellman 2003; Littman 1994)** — model-free Markov-game updates
  that set $V$ to the per-state Nash/minimax value. *Gap*: asymptotic / data-already-covers-state
  assumptions; no exploration bonus, no finite-sample regret, and a single $Q$ cannot
  simultaneously serve a maximizer and a minimizer who need optimism in *opposite* directions.
- **Wei et al. 2017; Jia et al. 2019 / Sidford et al. 2019** — finite-sample self-play / turn-based
  results. *Gap*: a global reachability assumption, or a simulator — both remove the
  exploration challenge that the general setting forces.
- **R-max (Brafman & Tennenholtz 2002)** — no structural assumption, guarantees vs adversarial
  opponents. *Gap*: bounds the deployed pair against the *Nash value*, not against the
  per-episode best response, so it does not give a $\sqrt{T}$ self-play regret here.
- **Two no-regret learners against each other** — the obvious decoupled approach: average the
  iterates to get a near-Nash pair. *Gap (to be probed)*: requires each learner to achieve
  no-regret against an *adversarial* opponent, whose policy can be non-Markovian.

## Evaluation settings

The yardstick is analytical, not a benchmark dataset. The model is the tabular episodic
zero-sum Markov game $\mathrm{MG}(H,\mathcal S,\mathcal A,\mathcal B,\mathbb P,r)$ with
$|\mathcal S|\le S$, $|\mathcal A|\le A$, $|\mathcal B|\le B$, horizon $H$, deterministic
reward $r_h:\mathcal S\times\mathcal A\times\mathcal B\to[0,1]$, unknown $\mathbb P$ and $r$, no
simulator, and a possibly adversarial initial state each episode. Performance is reported as:
(i) **regret** over $K$ episodes ($T=KH$ steps), where each episode's deployed pair
$(\mu^k,\nu^k)$ is scored against its own best responses,
$\mathrm{Regret}(K)=\sum_k [V^{\dagger,\nu^k}_1(s_1^k)-V^{\mu^k,\dagger}_1(s_1^k)]$; and
(ii) the **PAC** sample complexity — the number of episodes to output an
$\epsilon$-approximate Nash pair, $V^{\dagger,\hat\nu}_1(s_1)-V^{\hat\mu,\dagger}_1(s_1)\le\epsilon$.
The reference points are the single-agent minimax bounds $\Omega(\sqrt{H^2SAT})$ and the
matching UCBVI/Q-learning rates; success is measured by how close a two-player bound comes to
the natural two-player lower bound $\Omega(\sqrt{H^2 S(A+B) T})$ obtained by collapsing the game
to a one-player MDP. Runtime is also a yardstick: an algorithm is "efficient" only if its
per-episode computation is polynomial in $S,A,B,H$.

## Code framework

The scaffold is a generic tabular optimistic-RL harness for a two-player game: empirical-model
bookkeeping, an optimistic Bellman backup with a confidence bonus, and a placeholder for the
per-state rule that turns value estimates into the two players' actions and state values.

```python
import numpy as np

# ----- empirical model from counts -----
class EmpiricalModel:
    def __init__(self, S, A, B, H):
        self.S, self.A, self.B, self.H = S, A, B, H
        self.N   = np.zeros((H, S, A, B))          # visit counts
        self.Nss = np.zeros((H, S, A, B, S))       # transition counts
        self.Rsum = np.zeros((H, S, A, B))         # reward sums

    def update(self, h, s, a, b, r, s_next):
        self.N[h, s, a, b]            += 1
        self.Nss[h, s, a, b, s_next]  += 1
        self.Rsum[h, s, a, b]         += r

    def Phat(self, h, s, a, b):
        n = max(self.N[h, s, a, b], 1)
        return self.Nss[h, s, a, b] / n

    def rhat(self, h, s, a, b):
        n = max(self.N[h, s, a, b], 1)
        return self.Rsum[h, s, a, b] / n

def bonus(n, S, H, iota):
    # confidence radius for the empirical backup
    return np.sqrt(S * H**2 * iota / max(n, 1))

# ----- per-state play rule -----
def per_state_play(value_estimates, s, h):
    """Given the agents' value estimate(s) at state s, step h, decide what each
    player plays here, and read off the state value(s) used by the backup.
    The final algorithm fixes the number of estimates and the per-state
    equilibrium notion."""
    # TODO: implement the per-state play rule.
    raise NotImplementedError

def backup(model, value_estimates, h, S, A, B, H, iota):
    """One optimistic Bellman sweep at step h. Fills value_estimates in place."""
    for s in range(S):
        for a in range(A):
            for b in range(B):
                n  = model.N[h, s, a, b]
                # TODO: optimistic Q backup(s) — how many, and signed how,
                #       depends on per_state_play's requirements.
                pass
        per_state_play(value_estimates, s, h)   # set policy & state value(s)

def run(model, S, A, B, H, K, iota):
    value_estimates = None   # TODO: initialize the value estimate(s) the method needs
    for k in range(K):
        # backward sweep computes policies; forward sweep plays one episode
        for h in reversed(range(H)):
            backup(model, value_estimates, h, S, A, B, H, iota)
        # play episode with the per-state policies, update model ... (standard)
        pass
    return value_estimates
```
