# Context: planning in a Markov decision process by mathematical programming

## Research question

Given a finite Markov decision process — a state set $S$, an action set $A$, transition kernel $P(s'\mid s,a)$, a one-step reward (or cost) $r(s,a)$, an initial-state distribution $\mu$, and a discount factor $\gamma\in[0,1)$ — find a policy $\pi$ that maximizes the expected discounted return

$$J(\pi)=\mathbb{E}\Big[\textstyle\sum_{t=0}^{\infty}\gamma^t\,r(s_t,a_t)\;\Big|\;s_0\sim\mu,\ a_t\sim\pi(\cdot\mid s_t),\ s_{t+1}\sim P(\cdot\mid s_t,a_t)\Big],$$

or, in the average-cost reading of the same model, minimize the long-run expected cost per step under statistical equilibrium.

The pain point is the *shape* of this problem. The standard route — dynamic programming — characterizes the optimum through a fixed-point equation involving a *maximization* over actions, so the operator one must invert is nonlinear; and the objective $\max_\pi J(\pi)$, viewed directly as a function of the policy $\pi$, is non-convex, because $\pi$ enters the return through an infinite geometric series in the policy-induced transition matrix. A solution worth having would re-express "find the optimal policy" as a *convex* optimization problem — ideally a genuine **linear program** — so that the optimum is pinned by a single numerical objective rather than an iterative fixed-point computation, and so that the entire apparatus of linear programming (duality, complementary slackness, the simplex and interior-point algorithms, sensitivity analysis) becomes available for planning. Whether such a reformulation exists, and what its variables and constraints are, is the question.

## Background

**The Markov decision model and its return.** A stationary stochastic policy is a map $\pi:S\to\Delta(A)$. Running it from $s_0\sim\mu$ generates a Markov chain on $S$ (and on $S\times A$). Two summaries of "how good a state is" organize the theory: the state-value $V^\pi(s)=\mathbb{E}[\sum_t\gamma^t r(s_t,a_t)\mid s_0=s,\pi]$ and the state-action value $Q^\pi(s,a)=\mathbb{E}[\sum_t\gamma^t r\mid s_0=s,a_0=a,\pi]$. The return decomposes as $J(\pi)=\mathbb{E}_{s\sim\mu}[V^\pi(s)]=\mathbb{E}_{s\sim\mu,a\sim\pi}[Q^\pi(s,a)]$. With bounded rewards $|r|\le R_{\max}$, the discounted return is bounded by $R_{\max}/(1-\gamma)$.

**The Bellman equations.** $V^\pi$ and $Q^\pi$ satisfy the expectation (backup) recursions
$V^\pi(s)=\mathbb{E}_{a\sim\pi,\,s'\sim P}[r(s,a)+\gamma V^\pi(s')]$ and
$Q^\pi(s,a)=\mathbb{E}_{s'\sim P,\,a'\sim\pi}[r(s,a)+\gamma Q^\pi(s',a')]$.
The optimal value $V^\star(s)=\max_\pi V^\pi(s)$ satisfies the **Bellman optimality equation**
$$V^\star(s)=\max_{a\in A}\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\star(s')\Big]=:(TV^\star)(s).$$
$T$ is a $\gamma$-contraction in the $\ell_\infty$ norm ($\|TV'-TV\|_\infty\le\gamma\|V'-V\|_\infty$) and is monotone ($V'\le V\Rightarrow TV'\le TV$); hence $V^\star$ is its unique fixed point. The load-bearing facts here are exactly these two: monotonicity and contraction. The difficulty is the $\max_a$ inside $T$ — it makes the optimality equation nonlinear, which is why value iteration and policy iteration solve it by iteration rather than in closed form.

**Stationary distributions of Markov chains.** Fixing a stationary policy $\pi$ turns the MDP into an uncontrolled Markov chain with transition matrix $P_\pi$. For such a chain the stationary distribution $d$ obeys the balance equations $d^\top=d^\top P_\pi$ — at equilibrium the probability mass flowing *into* each state equals the mass flowing *out*. This is standard Markov-chain theory: a policy induces, in addition to its value functions, a distribution over the states (and over $S\times A$) that the chain visits in the long run.

**The structure of the optimality operator.** What makes $T$ nonlinear is the single $\max_a$; everything inside the bracket, $r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')$, is linear in $V$, one such affine expression per state–action pair. The two load-bearing properties of $T$ noted above — monotonicity and the $\gamma$-contraction in $\ell_\infty$ — are the only handles available for reasoning about where $V^\star$ sits relative to other value vectors. Whether those handles can be turned into something an LP solver can consume, and what the variables and objective would be, is open.

**Linear-programming duality.** Every linear program has a dual; at optimum the two share a value (strong duality), and complementary slackness ties active primal constraints to nonzero dual variables. A primal with one inequality constraint per state–action pair has a dual with one nonnegative variable per state–action pair.

## Baselines

**Value iteration.** Iterate $V_{k+1}=TV_k$. Each sweep applies the nonlinear Bellman optimality operator; convergence is geometric at rate $\gamma$, so the number of sweeps grows with $1/(1-\gamma)$. Core idea: exploit the contraction. Gap it leaves: it never exposes the problem's convex/duality structure, and its iteration count degrades badly for long-horizon (near-1 $\gamma$) problems; it produces $V^\star$, from which a greedy policy is then read off.

**Policy iteration.** Alternate policy evaluation (solve the *linear* system $V^\pi=r_\pi+\gamma P_\pi V^\pi$ for the current $\pi$) with greedy policy improvement. Converges in finitely many policy changes. Core idea: each improvement step is a one-step lookahead. Gap: still an iterative scheme with an outer nonlinear (greedy) step; it does not by itself present planning as a single mathematical program with a clean numerical objective and dual prices.

**Bellman/dynamic-programming functional equations (Bellman 1957).** Solve $V^\star=TV^\star$ directly as the defining equation of the optimum. This is the prevailing wisdom of the period and the thing every alternative is measured against. Its limitation, in the eyes of someone with an operations-research toolkit, is precisely that it is a nonlinear functional equation: it does not connect to the mature theory of linear programming, its solvers, or its economic (dual-price) interpretation.

**Production/inventory mathematical-programming models of the time.** Programming-under-uncertainty formulations (Dantzig's linear programming under uncertainty; Radner's team-decision LPs) handle *finite-horizon* stochastic decision problems by carrying random outcomes explicitly. Their gap, for the present purpose, is the horizon: they are built for finitely many stages, not for an indefinitely long (infinite-horizon, stationary) decision process. The decision-rule models of the era that *do* aim at long-run cost (e.g. the linear-decision-rule production-smoothing line of Holt–Modigliani–Simon) buy tractability by forcing all cost functions to be quadratic/convex — a serious modeling restriction one would like to drop.

## Evaluation settings

The natural yardstick is a small, fully-specified finite MDP where the optimum can be checked by hand and against dynamic programming. The canonical instance is a single-item inventory control problem: the state $i$ is the stock on hand at the start of a period, the action $j$ is the quantity produced (bounded by a capacity), demand $n$ is a serially-independent random variable with known distribution $p_n$, and the period cost is the sum of an inventory-holding cost $c_1(i)$, a production cost $c_2(j)$, and a shortage cost $c_3(n-k)$ on unmet demand, with available stock $k=i+j$ and terminal stock $t=\max(0,k-n)$. Integer state/action ranges are capped ($0\le i\le K$, $i+j=k\le K$) so the program is finite. The metric is the long-run expected cost per period (average-cost reading) or the discounted return $J(\pi)$ (discounted reading); the comparison policies are those produced by the dynamic-programming functional equation on the same instance. No convexity is assumed of the cost components — the formulation must accommodate arbitrary $c_1,c_2,c_3$.

## Code framework

A planning harness can encode the MDP as arrays, evaluate a fixed policy, and leave one empty slot where "compute an optimal policy" will go. Only the MDP data structures and a generic linear-program solver are assumed to exist.

```python
import numpy as np
from scipy.optimize import linprog   # generic LP solver: min c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq

# ---- MDP specification ----
class MDP:
    def __init__(self, P, r, mu, gamma):
        # P[s, a, s'] = Pr(s' | s, a); r[s, a]; mu[s] = initial-state dist; gamma in [0,1)
        self.P, self.r, self.mu, self.gamma = P, r, mu, gamma
        self.nS, self.nA = r.shape

    def policy_return(self, pi):
        # J(pi) = E_mu[ V^pi ], via the linear evaluation system V = r_pi + gamma P_pi V
        nS = self.nS
        P_pi = np.einsum('sa,sat->st', pi, self.P)         # induced transition matrix
        r_pi = np.einsum('sa,sa->s', pi, self.r)           # induced reward
        V = np.linalg.solve(np.eye(nS) - self.gamma * P_pi, r_pi)
        return float(self.mu @ V)


def solve_optimal_policy(mdp: MDP):
    """Return an optimal policy pi[s, a] for the MDP.
    # TODO: cast 'find the best policy' as a mathematical program over the right
    # variables and solve it; then read off a policy. The variables, the
    # objective, the constraints, and the read-off rule are exactly what the
    # derivation has to discover.
    """
    pass


def greedy_policy_from_values(mdp: MDP, V):
    # one-step lookahead read-off, used once an optimal value function is available
    Q = mdp.r + mdp.gamma * np.einsum('sat,t->sa', mdp.P, V)
    pi = np.zeros_like(mdp.r)
    pi[np.arange(mdp.nS), Q.argmax(axis=1)] = 1.0
    return pi
```
