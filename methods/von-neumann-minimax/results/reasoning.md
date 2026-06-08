Two people, a table, a game where every coin one wins the other loses. I want to say something definite about how to play it, and the trouble is that the usual sentence — "play as advantageously as possible" — dissolves the moment I look at it. I, player 1, want $g(x,y)$ large. But I do not own $g$. I choose the row $x$; the column $y$ belongs to my opponent, and his interests are exactly opposite to mine, coin for coin. So the most advantageous result *for me* depends on what he does, and what he does depends on what I do, and that depends back on him. It is circular, and a sentence that points in a circle isn't a definition.

Let me strip it down to what a single person can actually control and actually face. I control $x$. Whatever I pick, the payoff still hangs on his $y$. The honest thing to ask is: against the worst he can do to my choice, what do I come away with? If I commit to $x$, and he is fully against me, he will push the payoff down to $\min_y g(x,y)$. That number is mine to choose among, by choosing $x$. So the most I can *guarantee*, no matter how clever or hostile he is, is

$$v_{\text{low}} \;=\; \max_x \min_y g(x,y).$$

That is a clean, computable thing — a floor I can stand on. And by the mirror argument from his side, he can hold me to no more than

$$v_{\text{up}} \;=\; \min_y \max_x g(x,y),$$

a ceiling he can press down on my head. These two I can compute from the matrix alone, with no psychology in them at all. The whole theory, if there is one, lives in the relation between $v_{\text{low}}$ and $v_{\text{up}}$.

So how do they compare? Take any $x$ and any $y$. Certainly $\min_{y'} g(x,y') \le g(x,y) \le \max_{x'} g(x',y)$ — the left because the min is over all columns including $y$, the right because the max is over all rows including $x$. Now fix $y$ and let $x$ range over the outer max on the left piece: $\max_x \min_{y'} g(x,y') \le \max_x g(x,y)$, and this holds for *every* $y$, so I may take the min over $y$ on the right, which only lowers it:

$$\max_x \min_y g(x,y) \;\le\; \min_y \max_x g(x,y).$$

So $v_{\text{low}} \le v_{\text{up}}$, always, for any matrix whatsoever. The floor never rises above the ceiling. Good — and intuitive: whoever has to reveal his hand first is at a disadvantage; the second mover, the one who "finds the other out," is never worse off, and the gap $v_{\text{up}}-v_{\text{low}}\ge 0$ is precisely the worth of finding out.

The dream would be $v_{\text{low}}=v_{\text{up}}$. Then there is a single number $M$ — I can guarantee at least $M$, he can hold me to at most $M$, so the play *will* sit at $M$, and "the value of the game" means something. When does that happen? It happens exactly when there is a pair $(x_0,y_0)$ with $g(x,y_0)\le g(x_0,y_0)\le g(x_0,y)$ for all $x,y$: a saddle, a seat from which neither of us wants to rise. Then my best move is the fixed row $x_0$, his is the fixed column $y_0$, the matter is settled and deterministic.

But does a saddle always exist? Let me just try the smallest interesting matrix. Matching pennies:

$$\begin{pmatrix} +1 & -1 \\ -1 & +1 \end{pmatrix}.$$

Row 1: $\min$ over columns is $-1$. Row 2: $\min$ is $-1$. So $v_{\text{low}}=-1$. Column 1: $\max$ over rows is $+1$. Column 2: $\max$ is $+1$. So $v_{\text{up}}=+1$. The floor is $-1$, the ceiling is $+1$, and the gap is wide open. There is no value. And I feel exactly *why* there is no value when I imagine playing it: nothing about row 1 is better than row 2 in itself — the entire game is "did he guess which one I'd take?" Whoever reads the other wins. The rules explicitly forbid me from seeing his choice, and a theory that says "it depends entirely on the unobtainable thing" is no theory.

It is not a fluke of $2\times 2$. Stone, paper, scissors, written as a payoff matrix, is

$$\begin{pmatrix} 0 & -1 & +1 \\ +1 & 0 & -1 \\ -1 & +1 & 0 \end{pmatrix},$$

and again every row's minimum is $-1$, every column's maximum is $+1$, so $v_{\text{low}}=-1<+1=v_{\text{up}}$. The same hole, wider. In pure strategies the generic game is undetermined, and the strictly-determined ones — the ones with a saddle — are the exception, not the rule. I cannot build the theory on them.

What is the actual enemy here? It is *being found out*. In matching pennies, if there were any regularity in how I pick, he would learn it and beat me. The defense, then, is not to pick more cleverly — any deterministic cleverness is itself a pattern to be read — but to pick *unpredictably*. To make my own move something that even I cannot foretell until it happens. That means I should not choose a row at all; I should choose *odds*. I name a probability vector $\xi=(\xi_1,\dots,\xi_m)$, with $\xi_i\ge 0$ and $\sum_i\xi_i=1$, and then let a chance device draw the actual row for me. This is not a restriction on my freedom: if I genuinely want a definite row $x$, I set $\xi_x=1$ and the rest to zero. But if I set $\xi_1=\xi_2=\tfrac12$ in matching pennies, then *nobody* — not even I, before the draw — can know whether I will play 1 or 2. He cannot find out what does not yet exist. (Borel saw this much in 1921: against an opponent who profits by predicting you, a rational player should make himself unpredictable, and the way to do it is to randomize over your plans.) My opponent does the same, naming $\eta=(\eta_1,\dots,\eta_n)$ over his columns.

Now what is the payoff? Each of us has put randomness on the table, so the outcome is a random variable, and the natural single number to attach to it is its expected value. (Borel justified the expectation outright in his setting: take the payoff to *be* the probability of winning, and the expectation of a randomized strategy is exactly your ex-ante chance of winning — no extra assumption about preferences needed.) With my odds $\xi$ and his $\eta$, the expected payoff to me is

$$h(\xi,\eta) \;=\; \sum_{i}\sum_{j} g(i,j)\,\xi_i\,\eta_j \;=\; \xi^{\top} A\, \eta,$$

writing $A=[g(i,j)]$. And here I should pause, because something has quietly changed under my feet. The old $g(x,y)$ was an *arbitrary* function on a finite grid — it could be anything, and that arbitrariness is exactly why I couldn't pin it down. But $h(\xi,\eta)=\xi^\top A\eta$ is not arbitrary. It is *bilinear* — linear in $\xi$ when $\eta$ is fixed, linear in $\eta$ when $\xi$ is fixed. By enlarging the players' freedom (point masses to full distributions) I have, paradoxically, made the payoff function vastly more *special*. It now has structure I can grip: linearity, hence convexity. That is the lever. The arbitrary function admitted no proof of equality; the bilinear form might.

The same two quantities reappear, now over the simplices. What I can guarantee is

$$v_1' \;=\; \max_{\xi\in\Delta_m}\;\min_{\eta\in\Delta_n}\; \xi^\top A\eta,$$

and what he can hold me to is

$$v_2' \;=\; \min_{\eta\in\Delta_n}\;\max_{\xi\in\Delta_m}\; \xi^\top A\eta,$$

and the very same one-line argument as before gives $v_1'\le v_2'$. The question — *the* question — is whether equality now holds for *every* matrix $A$. Borel doubted it; he guessed it fails once each player has more than three strategies. So this is genuinely open, and I should not assume the answer.

Before I attack equality, let me simplify the two quantities, because the bilinearity hands me a gift. Fix my $\xi$. The inner $\min_{\eta}\xi^\top A\eta$ is a minimum of a *linear* function of $\eta$ over the simplex. A linear function on a simplex attains its minimum at a vertex — at a pure column. So

$$\min_{\eta\in\Delta_n}\xi^\top A\eta \;=\; \min_{j}\,(\xi^\top A)_j.$$

The opponent, against a mix, never needs to mix back; a pure best response suffices. So my guarantee is $v_1'=\max_{\xi}\min_j(\xi^\top A)_j$, and symmetrically $v_2'=\min_{\eta}\max_i(A\eta)_i$. Only *one* side ever needs to randomize at a time. That already trims the problem.

Now, the equality. Let me first understand Borel's wall, because if he is right I am chasing a phantom. What was he really demanding? In a symmetric game $A$ is skew-symmetric, $A^\top=-A$, and the value, if it exists, must be $0$ (interchanging the players negates everything). For $n=3$ — rock-paper-scissors with possibly uneven stakes, the non-transitive cycle — he found a full-support mix that scores exactly $0$ against *every* opponent strategy, and called it fair. Then he extrapolated: a fair strategy means a $\xi$ with $\xi^\top A = 0$ as a vector, a $\xi$ in the kernel of $A$. But a skew-symmetric matrix of *even* order is generically nonsingular — its kernel is $\{0\}$ — so for $n=4$ there is generically no such constant-zero $\xi$, and he concluded that no fair strategy exists, that the games are unsolvable, that minimax fails. (He even built a little budget-allocation game on three battlefields to "show" it.)

But look at what he asked for versus what I actually need. I do not need a $\xi$ whose payoff is *identically zero* against everything. I need a $\xi$ whose payoff is *non-negative* against everything — $\xi^\top A \ge 0$ componentwise — because that already guarantees me at least $0$. Non-negativity is a far weaker, one-sided demand than the two-sided "equals zero." Borel conflated "I can guarantee $\ge 0$" with "I have a strategy that ties everything dead," and the second really does fail generically. The first need not. His confident pessimism was an over-reading of $n=3$, where the only fair strategy happened to be the kernel one. So the conjecture is not a theorem against me; it is a warning to ask for nonnegativity, not constancy.

That reframing tells me exactly what to prove. Strip away the symmetry for a moment and state the cleanest possible target: for any matrix $A$, *either I can guarantee at least zero, or he can hold me to at most zero.* In symbols: either there is $\xi\in\Delta$ with $(\xi^\top A)_j\ge 0$ for all $j$ (so $v_1'\ge 0$), or there is $\eta\in\Delta$ with $(A\eta)_i\le 0$ for all $i$ (so $v_2'\le 0$). If I can show one of these always holds, I will have ruled out the worst case $v_1'<0<v_2'$ — and I will leverage that into full equality.

This "either … or" about two systems of linear inequalities — one weighting the columns, one weighting the rows, with opposite signs — has the unmistakable shape of an alternative theorem. And alternative theorems come from convexity, from separating a point from a convex set. Let me build the separation I need from the ground, because I want to be sure of it.

Take a closed convex set $C$ in $\mathbb{R}^N$ and a point $y$ not in it. Because $C$ is closed, there is a nearest point $z\in C$ to $y$; the distance $\|z-y\|$ is minimized there. Now take any other $u\in C$. By convexity the whole segment $t u+(1-t)z$, $t\in[0,1]$, lies in $C$, so by the minimality of $z$,

$$\|t u+(1-t)z - y\|^2 \;\ge\; \|z-y\|^2.$$

Write the left side as $\|(z-y)+t(u-z)\|^2 = \|z-y\|^2 + 2t\,(z-y)\cdot(u-z) + t^2\|u-z\|^2$. Subtract $\|z-y\|^2$, divide by $t>0$:

$$2(z-y)\cdot(u-z) + t\,\|u-z\|^2 \;\ge\; 0.$$

Let $t\to 0^+$: $(z-y)\cdot(u-z)\ge 0$, i.e. $(z-y)\cdot u \ge (z-y)\cdot z$. Set $a=z-y$ (not zero, since $z\ne y$) and $b=a\cdot z$. Then $a\cdot u \ge b$ for every $u\in C$, while $a\cdot y = a\cdot z - \|a\|^2 = b - \|a\|^2 < b$. So the hyperplane $\{x:a\cdot x=b\}$ has the entire set $C$ on the side $a\cdot x\ge b$ and the point $y$ strictly on the other side. A closed convex set and an outside point are cleanly separated by a hyperplane. That is the one fact I will lean on.

Now I deploy it to get the alternative for the matrix. Consider the columns of $A$ as vectors $A_{\cdot 1},\dots,A_{\cdot n}$ in $\mathbb{R}^m$, and let $C$ be the convex hull of these columns together with the coordinate vectors $e_1,\dots,e_m$. The coordinate vectors are slack: they let an equality with $0$ become a componentwise inequality. Ask: does the origin $0$ lie in $C$?

If $0\in C$, then there are numbers $\alpha_j\ge 0$ and $\beta_i\ge 0$, not all zero and with $\sum_j\alpha_j+\sum_i\beta_i=1$, such that

$$\sum_j \alpha_j A_{\cdot j}+\sum_i \beta_i e_i=0.$$

The $\alpha_j$ cannot all vanish, because a nonzero convex combination of the $e_i$ has nonnegative coordinates and positive coordinate sum, never the zero vector. So $\alpha=\sum_j\alpha_j>0$. Normalize $\eta_j=\alpha_j/\alpha$. Then

$$A\eta=\sum_j \eta_j A_{\cdot j}=-\frac{1}{\alpha}\sum_i \beta_i e_i\le 0$$

componentwise. With this $\eta$, no matter what row I play I get $\le 0$ in expectation, so $\max_i(A\eta)_i\le 0$, hence $v_2'\le 0$.

If $0\notin C$, the separation I just proved, with $y=0$, gives a vector $a$ and a number $b$ such that $a\cdot u\ge b$ for every $u\in C$ while $a\cdot 0=0<b$. Thus $b>0$. Since each $e_i$ lies in $C$, I get $a_i=a\cdot e_i\ge b>0$ for every $i$. Since each column $A_{\cdot j}$ lies in $C$, I also get

$$a^\top A_{\cdot j}=\sum_i a_i g(i,j)\ge b>0\qquad\text{for every }j.$$

Normalize $\xi_i=a_i/\sum_k a_k$ to a distribution. Then

$$ (\xi^\top A)_j=\frac{a^\top A_{\cdot j}}{\sum_k a_k}\ge 0 $$

for all $j$, so $\min_j(\xi^\top A)_j\ge 0$, hence $v_1'\ge 0$.

So in *every* case, either $v_1'\ge 0$ or $v_2'\le 0$. Equivalently:

$$\text{never } v_1' < 0 < v_2'.$$

This is not saying that exactly one side must hold in a weak equality case; both certificates may touch at zero. The content I need is narrower and sharper: the bad straddle $v_1'<0<v_2'$ is impossible. That kills Borel's pessimism at the zero threshold. But I want more than "the gap doesn't straddle zero" — I want $v_1'=v_2'$ outright. The zero is just an accident of where I put the origin.

So move the origin. Pick any real number $w$ and replace the matrix $A$ by $A-w$ — subtract $w$ from every single entry. What does that do to the two values? Because $\xi$ and $\eta$ are distributions, $\sum_{i,j}\xi_i\eta_j=1$, so $\xi^\top(A-w)\eta = \xi^\top A\eta - w$ for *every* pair $(\xi,\eta)$. The whole payoff surface drops by the constant $w$, and therefore both one-sided optima drop by exactly $w$: the new $v_1'$ is the old $v_1'-w$, the new $v_2'$ is the old $v_2'-w$. Now apply what I just proved — "never (new $v_1'$) $< 0 <$ (new $v_2'$)" — to the shifted game:

$$\text{never } v_1' - w < 0 < v_2' - w, \qquad\text{i.e.}\qquad \text{never } v_1' < w < v_2'.$$

And $w$ was *arbitrary*. If it were true that $v_1' < v_2'$, I could choose a $w$ strictly between them, $v_1'<w<v_2'$ — which I have just shown is impossible. So $v_1'<v_2'$ cannot hold. Combined with $v_1'\le v_2'$, which always holds, I am forced to

$$\boxed{\;v_1' \;=\; v_2'\;:\qquad \max_{\xi\in\Delta_m}\min_{\eta\in\Delta_n}\xi^\top A\eta \;=\; \min_{\eta\in\Delta_n}\max_{\xi\in\Delta_m}\xi^\top A\eta \;=\; v.\;}$$

There it is. For *every* matrix, once mixing is allowed, the floor meets the ceiling. The game has a value $v$. Borel's kernel obstruction was too strong; the theorem holds at all sizes. The arbitrary function had no value; the bilinear form always does — and the single fact that turned the impossible into the certain was the separating hyperplane, the same convexity that underlies the alternative for linear inequalities.

Let me make sure I have not just an equality of numbers but the *strategies*. From $v_1'=v$ there is, by definition of the max, a $\xi^\*$ achieving $\min_j(\xi^{*\top}A)_j = v$, i.e. $\xi^{*\top}A\ge v\cdot\mathbf{1}$: playing $\xi^\*$, I get at least $v$ against every column, hence at least $v$ against any mix $\eta$. From $v_2'=v$ there is an $\eta^\*$ with $A\eta^\*\le v\cdot\mathbf{1}$: playing $\eta^\*$, my opponent holds me to at most $v$ against every row, hence against any $\xi$. Put these together:

$$\xi^\top A\eta^\* \;\le\; v \;\le\; \xi^{*\top}A\eta \qquad\text{for all }\xi,\eta,$$

and in particular $\xi^{*\top}A\eta^\* = v$. So $(\xi^\*,\eta^\*)$ is a genuine saddle point of $\xi^\top A\eta$: neither of us can profit by unilaterally changing his odds. The circular "each best against the other" condition has a consistent solution after all, and $v$ is its value. I keep the separation argument in front because it also tells me how to compute.

Now the question that decides actual play: *which* pure strategies do the optimal mixes use, and why those? I have a saddle $(\xi^\*,\eta^\*)$. Consider my opponent's side. Playing $\eta^\*$, the vector of my row-payoffs is $A\eta^\*$, with $\max_i(A\eta^\*)_i = v$. My optimal mix is $\xi^\*$, and $\xi^{*\top}A\eta^\* = v$. But $\xi^{*\top}(A\eta^\*) = \sum_i \xi^*_i (A\eta^\*)_i$ is a weighted average of the row-payoffs $(A\eta^\*)_i$, each of which is $\le v$, and this average equals $v$. An average of quantities all $\le v$ can equal $v$ only if every quantity *carrying positive weight* is itself exactly $v$. Therefore

$$\xi^*_i > 0 \;\Longrightarrow\; (A\eta^\*)_i = v.$$

I put weight only on rows that are *best responses* to his $\eta^\*$ — rows tied for the maximum. Equivalently, any row $i$ that does strictly worse than $v$ against $\eta^\*$ gets weight zero. By the mirror argument on his side,

$$\eta^*_j > 0 \;\Longrightarrow\; (\xi^{*\top}A)_j = v,$$

he puts weight only on columns that hold me to exactly $v$. Read it as instructions: *every pure strategy in the support of my optimal mix yields exactly the value; every one outside it would do no better.* I am indifferent across the rows I actually use — and so the precise way I randomize is chosen not to help me directly (all my used rows pay the same $v$) but to make *him* indifferent: my $\xi^\*$ is engineered so that all of *his* used columns pay him the same, leaving him no lever to pull. Optimal play in a zero-sum game is to randomize so as to make the opponent indifferent. That is the whole secret of bluffing made exact, and it is exactly the complementary-slackness condition: support of one player ↔ tight inequalities for the other.

And that last observation tells me these are not two separate facts but one. Look again at the value as I'd compute it. To find $v_1'=\max_\xi\min_j(\xi^\top A)_j$ is to solve: maximize a scalar $t$ subject to $(\xi^\top A)_j \ge t$ for all $j$, with $\xi\ge 0$, $\sum_i\xi_i=1$. That is a linear program — a linear objective $t$ over a polyhedron of $(\xi,t)$.

I want the dual without losing a sign. If $(\xi,t)$ is feasible and $\eta$ is any distribution over columns, then the constraints give $t\le(\xi^\top A)_j$ for each $j$, so after multiplying by $\eta_j$ and summing,

$$t\le\sum_j\eta_j(\xi^\top A)_j=\xi^\top A\eta\le\max_i(A\eta)_i.$$

Thus every $\eta$ gives an upper bound on every feasible $t$, and the best such upper bound is obtained by minimizing $\max_i(A\eta)_i$. Introducing a scalar $u$ for that maximum gives

$$\min u\quad\text{subject to}\quad (A\eta)_i\le u\ \text{for all }i,\qquad \eta\ge0,\quad \sum_j\eta_j=1,$$

which is exactly $v_2'=\min_\eta\max_i(A\eta)_i$. So the minimax theorem, $v_1'=v_2'$, is strong duality for this primal-dual LP pair. The same slack equations I just derived as indifference are complementary slackness: $\eta_j^*((\xi^{*\top}A)_j-v)=0$ for the column constraints and $\xi_i^*(v-(A\eta^\*)_i)=0$ for the row constraints. The inequalities face the right way: player 1's optimal mix makes every column payoff at least $v$, while player 2's optimal mix makes every row payoff at most $v$, and positive support can appear only where the relevant inequality is tight. The separating hyperplane, the theorem of the alternative, the existence of a game value, and strong LP duality are four faces of one convexity fact. So I can compute the value and the optimal mixes by solving a linear program — but the *reason* it works is the saddle I just proved must exist.

Let me write down what the theorem actually buys me, as a procedure that takes a matrix and returns its value and a saddle pair.

```python
import numpy as np
from scipy.optimize import linprog

# Value of a finite zero-sum game given payoff matrix A (player 1 maximizes ξᵀAη, player 2 minimizes).
# The two LPs encode the saddle inequalities ξᵀA ≥ v·1 and Aη ≤ v·1.

def solve_zero_sum_game(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    shift = A.min() - 1.0
    B = A - shift                      # make B > 0 so the game value is positive; undo the shift at the end
                                       # (subtracting a constant shifts the value by that constant — the
                                       #  same "shift-by-w" invariance used to prove v1' = v2')

    # PLAYER 1's LP:  maximize v  s.t.  ξᵀB ≥ v·1,  ξ ≥ 0,  Σξ = 1.
    # Variables [ξ_1..ξ_m, v]; maximize v ⇔ minimize -v.
    c = np.zeros(m + 1); c[-1] = -1.0
    # -ξᵀB + v·1 ≤ 0   (i.e. for each column j:  v - Σ_i ξ_i B[i,j] ≤ 0)
    A_ub = np.hstack([-B.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))]); b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res1.success:
        raise RuntimeError(f"row-player LP failed: {res1.message}")
    xi = res1.x[:m]; v = res1.x[-1]

    # PLAYER 2's LP is the dual:  minimize u  s.t.  Bη ≤ u·1,  η ≥ 0,  Ση = 1.
    c2 = np.zeros(n + 1); c2[-1] = 1.0
    A_ub2 = np.hstack([B, -np.ones((m, 1))])     # Σ_j B[i,j] η_j - u ≤ 0
    b_ub2 = np.zeros(m)
    A_eq2 = np.hstack([np.ones((1, n)), np.zeros((1, 1))]); b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * n + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res2.success:
        raise RuntimeError(f"column-player LP failed: {res2.message}")
    eta = res2.x[:n]

    return v + shift, xi, eta            # undo the shift to recover the true value of A

# Checks the theorem on its own terms.
if __name__ == "__main__":
    RPS = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])   # value 0, uniform mix
    v, xi, eta = solve_zero_sum_game(RPS)
    assert abs(v) < 1e-9                                    # symmetric ⇒ value 0
    assert np.allclose(xi, [1/3, 1/3, 1/3], atol=1e-6)      # make opponent indifferent ⇒ uniform
    # saddle: ξ guarantees ≥ v against every column, η holds player 1 to ≤ v against every row
    assert np.min(xi @ RPS) >= v - 1e-9
    assert np.max(RPS @ eta) <= v + 1e-9
```

The chain, start to finish: the opposition is circular, so I replace "play best" with "guarantee the most against the worst case," getting a floor $v_{\text{low}}$ and a ceiling $v_{\text{up}}$ with $v_{\text{low}}\le v_{\text{up}}$ always; pure strategies leave a strict gap because the real enemy is being found out, which only randomness defeats; randomizing replaces the arbitrary $g$ by the bilinear $\xi^\top A\eta$, whose convex structure is the whole opening; a separating-hyperplane argument yields the theorem of the alternative — either I guarantee $\ge 0$ or he holds me to $\le 0$, never a straddle of zero — and shifting the matrix by a constant $w$ slides that statement to every threshold, forcing floor and ceiling to coincide at a single value $v$ with a saddle pair $(\xi^\*,\eta^\*)$; the saddle's support obeys complementary slackness, so optimal play is to randomize precisely to make the opponent indifferent across his live options; and the whole thing is strong LP duality, so a linear program computes the value and the optimal mixes.
