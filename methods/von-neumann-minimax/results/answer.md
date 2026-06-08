# The Minimax Theorem for finite zero-sum games

## Problem

A finite two-person zero-sum game is a real $m\times n$ payoff matrix $A=[g(i,j)]$: player 1 picks a row, player 2 a column, simultaneously; player 1 receives $g(i,j)$ and player 2 receives $-g(i,j)$. "Play optimally" is ill-defined because each player's best choice depends on the other's. The two well-defined quantities are what player 1 can guarantee, $\max_x\min_y g$, and what player 2 can hold him to, $\min_y\max_x g$. In **pure** strategies these can differ (e.g. Matching Pennies: $-1$ vs $+1$), so the game has no determinate value. The question: is there always a value once randomization is allowed?

## Key idea

Let players choose **mixed strategies** — probability vectors $\xi\in\Delta_m$ (over rows), $\eta\in\Delta_n$ (over columns) — and evaluate by **expected payoff** $\xi^\top A\eta$. Randomizing defeats the only real advantage in a zero-sum game, being *found out*; and crucially it replaces the arbitrary function $g$ by the **bilinear** form $\xi^\top A\eta$, whose convexity makes the value exist. Optimal play is then to randomize so as to make the opponent **indifferent** across the pure strategies he actually uses.

## Theorem

For every real $m\times n$ matrix $A$,
$$
\max_{\xi\in\Delta_m}\ \min_{\eta\in\Delta_n}\ \xi^\top A\eta
\;=\;
\min_{\eta\in\Delta_n}\ \max_{\xi\in\Delta_m}\ \xi^\top A\eta
\;=:\; v.
$$
The common number $v$ is the **value** of the game, and there exist optimal mixed strategies $\xi^\*,\eta^\*$ forming a **saddle point**:
$$
\xi^\top A\eta^\* \;\le\; v \;=\; \xi^{*\top}A\eta^\* \;\le\; \xi^{*\top}A\eta
\qquad\text{for all }\xi\in\Delta_m,\ \eta\in\Delta_n.
$$
Player 1 can guarantee at least $v$ ($\xi^{*\top}A\ge v\mathbf 1$) and player 2 can hold him to at most $v$ ($A\eta^\*\le v\mathbf 1$), each irrespective of the opponent.

## Proof (separating hyperplane / theorem of the alternative)

**Step 0 — the easy direction.** Define $a(\xi)=\min_{\eta'}\xi^\top A\eta'$ and $b(\eta)=\max_{\xi'}\xi'^\top A\eta$. For every pair $\xi,\eta$,
$$
a(\xi)\le \xi^\top A\eta\le b(\eta).
$$
Thus every $b(\eta)$ is an upper bound on every $a(\xi)$, so $\max_\xi a(\xi)\le b(\eta)$ for every $\eta$, and then
$$
\max_\xi\min_{\eta'}\xi^\top A\eta'\le \min_\eta\max_{\xi'}\xi'^\top A\eta.
$$
This gives $v_1'\le v_2'$ for any $A$.

Also, since $\xi^\top A\eta$ is linear in $\eta$, the inner min over $\Delta_n$ is attained at a vertex: $\min_\eta\xi^\top A\eta=\min_j(\xi^\top A)_j$. So $v_1'=\max_\xi\min_j(\xi^\top A)_j$ and $v_2'=\min_\eta\max_i(A\eta)_i$ — only one side need be mixed.

**Step 1 — supporting hyperplane.** Let $C\subseteq\mathbb R^N$ be closed convex and $y\notin C$. Let $z\in C$ minimize $\|z-y\|$. For $u\in C$, $t\in[0,1]$, convexity gives $tu+(1-t)z\in C$, so $\|(z-y)+t(u-z)\|^2\ge\|z-y\|^2$; expanding, dividing by $t>0$, and letting $t\to0^+$ yields $(z-y)\cdot(u-z)\ge0$, i.e. with $a=z-y\neq0$, $b=a\cdot z$:
$$
a\cdot u\ge b\ \ \forall u\in C,\qquad a\cdot y=b-\|a\|^2<b.
$$
A closed convex set and an outside point are separated by a hyperplane.

**Step 2 — theorem of the alternative.** Let $C=\operatorname{conv}\{A_{\cdot1},\dots,A_{\cdot n},e_1,\dots,e_m\}\subset\mathbb R^m$. If $0\in C$, then
$$
\sum_j\alpha_j A_{\cdot j}+\sum_i\beta_i e_i=0,\qquad
\alpha_j,\beta_i\ge0,\qquad \sum_j\alpha_j+\sum_i\beta_i=1.
$$
The column weight $\alpha=\sum_j\alpha_j$ is positive, since a nonzero convex combination of coordinate vectors cannot be $0$. With $\eta_j=\alpha_j/\alpha$,
$$
A\eta=-\alpha^{-1}\sum_i\beta_i e_i\le0,
$$
so $v_2'\le0$. If $0\notin C$, separation gives $a\cdot u\ge b$ for all $u\in C$ and $0=a\cdot0<b$. Thus $b>0$, hence $a_i=a\cdot e_i\ge b>0$ and $a^\top A_{\cdot j}\ge b>0$ for every $j$. With $\xi_i=a_i/\sum_k a_k$,
$$
\xi^\top A\ge0,
$$
so $v_1'\ge0$. Therefore
$$
\textbf{never } v_1'<0<v_2'. \tag{$\ast$}
$$

**Step 3 — shift by $w$.** For any constant $w$, subtract $w$ from every entry of $A$. Since $\sum_{i,j}\xi_i\eta_j=1$, the shifted payoff is $\xi^\top A\eta-w$ for all $\xi,\eta$; both $v_1',v_2'$ shift by $-w$. Applying $(\ast)$ to the shifted game: **never $v_1'<w<v_2'$**, for *every* real $w$. If $v_1'<v_2'$ we could insert $w$ strictly between — contradiction. Hence $v_1'=v_2'=:v$. $\qquad\blacksquare$

## Characterization of optimal strategies (indifference / complementary slackness)

At a saddle $(\xi^\*,\eta^\*)$, since $v=\xi^{*\top}(A\eta^\*)$ is an average of row-payoffs $(A\eta^\*)_i\le v$:
$$
\xi^*_i>0\ \Longrightarrow\ (A\eta^\*)_i=v,\qquad\qquad \eta^*_j>0\ \Longrightarrow\ (\xi^{*\top}A)_j=v.
$$
Every pure strategy in the support yields exactly $v$; a player randomizes precisely so as to make the opponent indifferent across the pure strategies the opponent uses.

## LP-duality equivalence

$v_1'=\max_\xi\min_j(\xi^\top A)_j$ is the LP
$$
\max\, t \ \ \text{s.t.}\ \ A^\top\xi\ge t\mathbf 1,\ \ \xi\in\Delta_m,
$$
Its dual signs can be checked without guessing. For any feasible $(\xi,t)$ and any $\eta\in\Delta_n$,
$$
t\le \sum_j\eta_j(\xi^\top A)_j=\xi^\top A\eta\le \max_i(A\eta)_i.
$$
Thus any feasible column-side bound $u$ with $A\eta\le u\mathbf1$ is an upper bound on the primal objective $t$. The tightest such bound is the LP
$$
\min\, u \ \ \text{s.t.}\ \ A\eta\le u\mathbf 1,\ \ \eta\in\Delta_n,
$$
which is exactly $v_2'=\min_\eta\max_i(A\eta)_i$. These are the formal primal-dual LP pair: the free scalar $t$ produces $\sum_j\eta_j=1$ in the dual, and the nonnegative variables $\xi_i$ produce the row constraints $A\eta\le u\mathbf1$. The minimax theorem is **strong LP duality** for this pair. The support conditions are the complementary-slackness equations
$$
\eta_j^\*((\xi^{*\top}A)_j-v)=0,\qquad
\xi_i^\*(v-(A\eta^\*)_i)=0,
$$
with $A^\top\xi^\*\ge v\mathbf1$ and $A\eta^\*\le v\mathbf1$. Separation, the alternative, game value, and LP duality are one fact.

## Computing the value and optimal strategies

```python
import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_game(A):
    """Value v and optimal mixed strategies (xi, eta) of the finite zero-sum game A.
    Player 1 maximizes xi^T A eta; player 2 minimizes."""
    A = np.asarray(A, dtype=float); m, n = A.shape
    shift = A.min() - 1.0; B = A - shift          # shift to a positive game; value shifts by 'shift'

    # Player 1:  max v  s.t.  xi^T B >= v 1,  xi in simplex.  Vars [xi, v], maximize v.
    c = np.r_[np.zeros(m), -1.0]
    A_ub = np.hstack([-B.T, np.ones((n, 1))]); b_ub = np.zeros(n)        # v - (xi^T B)_j <= 0
    A_eq = np.hstack([np.ones((1, m)), [[0.0]]]); b_eq = [1.0]
    r1 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=[(0, None)]*m + [(None, None)], method="highs")
    if not r1.success:
        raise RuntimeError(f"row-player LP failed: {r1.message}")
    xi, v = r1.x[:m], r1.x[-1]

    # Player 2 (dual):  min u  s.t.  B eta <= u 1,  eta in simplex.
    c2 = np.r_[np.zeros(n), 1.0]
    A_ub2 = np.hstack([B, -np.ones((m, 1))]); b_ub2 = np.zeros(m)        # (B eta)_i - u <= 0
    A_eq2 = np.hstack([np.ones((1, n)), [[0.0]]]); b_eq2 = [1.0]
    r2 = linprog(c2, A_ub2, b_ub2, A_eq2, b_eq2, bounds=[(0, None)]*n + [(None, None)], method="highs")
    if not r2.success:
        raise RuntimeError(f"column-player LP failed: {r2.message}")
    eta = r2.x[:n]
    return v + shift, xi, eta

if __name__ == "__main__":
    RPS = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
    v, xi, eta = solve_zero_sum_game(RPS)
    assert abs(v) < 1e-9 and np.allclose(xi, [1/3, 1/3, 1/3], atol=1e-6)
    assert np.min(xi @ RPS) >= v - 1e-9 and np.max(RPS @ eta) <= v + 1e-9   # saddle guarantees
```

## Why each choice

- **Mixed strategies, not cleverer pure rules:** the gap $\max\min<\min\max$ is exactly the value of being found out; only randomness hides intent, even from oneself.
- **Expected value as the criterion:** it is the unique *linear* functional of the mix; linearity converts the arbitrary $g$ into the bilinear $\xi^\top A\eta$, whose convexity is what makes the value exist (and Borel justified it directly as ex-ante win probability).
- **Separation / alternative argument:** "either I guarantee $\ge0$ or you hold me to $\le0$" is a statement about two dual linear systems; a closed convex set vs. the origin is separated by a hyperplane (Farkas). The shift-by-$w$ upgrades the zero threshold to all thresholds.
- **Against Borel's conjecture:** Borel guessed minimax fails for $n>3$, conflating "$\exists$ mix with payoffs $\ge0$" with "$\exists$ full-support mix with payoff $\equiv0$" (a kernel vector of skew-symmetric $A$, generically nonexistent for even $n$). Demanding only nonnegativity — which the alternative always provides — closes the gap at every size.
