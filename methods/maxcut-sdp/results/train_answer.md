Given an undirected graph $G=(V,E)$ on $n$ vertices with nonnegative weights $w_{ij}=w_{ji}$, I want to split the vertices into two sides $S$ and $\bar S$ so that the crossing weight $w(S,\bar S)=\sum_{i\in S,\,j\notin S} w_{ij}$ is as large as possible. Exact MAX CUT is NP-complete — it is on Karp's original list and stays hard even with unit weights — so on the worst graph I cannot hope for the optimum in polynomial time. The realistic target is a $\rho$-approximation: a polynomial-time algorithm whose returned cut has (expected) weight at least $\rho\cdot\mathrm{OPT}$ on every input, with $\rho$ as large as I can *prove*. The dispiriting fact is that for nearly twenty years $\rho$ was stuck at $1/2$. The simplest algorithm — flip an independent fair coin per vertex, or the Sahni–Gonzales greedy derandomization of it — cuts each edge with probability exactly $1/2$, so its expected cut is $\tfrac12\sum_{i<j} w_{ij}\ge \tfrac12\,\mathrm{OPT}$. Every later refinement only added a vanishing $1/\mathrm{poly}(n,m)$ term on top of $1/2$.

The reason the constant will not move is worth stating precisely, because it tells me exactly what to fix. All of these analyses certify the cut as a *fraction of the total edge weight* $\sum_{i<j} w_{ij}$, then close the argument with $\sum w_{ij}\ge\mathrm{OPT}$. To beat $1/2$ this way I would need a universal lower bound $c\cdot\sum w_{ij}$ with $c>1/2$, and that simply cannot hold: in the complete graph $K_n$ the total weight is $n(n-1)/2$ while the maximum cut has $\lfloor n^2/4\rfloor$ edges, so $\mathrm{OPT}/\sum w_{ij}\to 1/2$. The denominator is the problem. The total weight is a far too generous upper bound on $\mathrm{OPT}$, and against it no fraction above $1/2$ is certifiable. So what I actually need is a *tighter, polynomial-time-computable upper bound on $\mathrm{OPT}$* and a cut measured against that. The Delorme–Poljak eigenvalue bound is exactly such a quantity — empirically its worst gap to $\mathrm{OPT}$ is the $5$-cycle at $32/(25+5\sqrt5)\approx 0.88445$ — but it supplies only a number, not a partition, and the best worst-case ratio anyone could prove for it was the trivial $0.5$. That dangling gap between an $0.88$-tight bound and an unproven guarantee is the target.

I propose the method that is now called Goemans–Williamson MAX CUT: a semidefinite relaxation rounded by a random hyperplane. The starting point is to write the cut algebraically. Give vertex $i$ a variable $y_i\in\{-1,+1\}$ with $S=\{i:y_i=+1\}$; then $\tfrac12(1-y_iy_j)$ is the indicator that edge $(i,j)$ is cut, and MAX CUT is the integer quadratic program $\max_{y\in\{-1,+1\}^n}\sum_{i<j}\tfrac12 w_{ij}(1-y_iy_j)$. The same data has a spectral face through the Laplacian $L$ ($L_{ij}=-w_{ij}$ off-diagonal, $L_{ii}=\sum_j w_{ij}$), with the identity $x^TLx=\sum_{i<j}w_{ij}(x_i-x_j)^2$; on $x_S\in\{-1,+1\}^n$ the term $(x_i-x_j)^2$ is $4$ on cut edges and $0$ otherwise, so $x_S^TLx_S=4\,w(S,\bar S)$. The crucial observation about why a tight bound exists is that I can perturb $L$ by a diagonal that is invisible on the cube: for any correcting vector $u$ with $\sum_i u_i=0$ and $U=\mathrm{diag}(u)$, every $x_S$ satisfies $x_S^TUx_S=\sum_i u_i x_{S,i}^2=\sum_i u_i=0$, so $4\,\mathrm{mc}(G)=x_S^T(L+U)x_S\le n\,\lambda_{\max}(L+U)$ by Rayleigh. Minimizing the right side over correcting vectors gives the bound $\tfrac n4\min_{\sum u_i=0}\lambda_{\max}(L+\mathrm{diag}(u))$, which is convex in $u$ and computable to arbitrary precision.

The genius is to recover this bound in a form that also yields a *cut*. The constraint $y_i\in\{-1,+1\}$ says $y_i$ is a unit vector in $\mathbb R^1$ and $y_iy_j$ is their inner product. I relax each scalar to a unit vector $u_i\in\mathbb R^n$ and each product to $u_i\cdot u_j$, giving $\max\sum_{i<j}\tfrac12 w_{ij}(1-u_i\cdot u_j)$ subject to $\|u_i\|=1$. This is a genuine relaxation — collinear unit vectors reproduce any $\pm1$ assignment — so its value $Z^*_P\ge\mathrm{OPT}$. It looks nonconvex, but in the Gram matrix $Y_{ij}=u_i\cdot u_j$ it becomes linear: a matrix is a Gram matrix iff it is symmetric PSD, and $\|u_i\|=1$ is just $Y_{ii}=1$. So the problem is the semidefinite program

$$\max\ \sum_{i<j}\tfrac12 w_{ij}(1-Y_{ij})\quad\text{s.t.}\quad Y_{ii}=1,\ \ Y\succeq 0,$$

solvable to additive $\varepsilon$ in time polynomial in the input and $\log(1/\varepsilon)$. Its dual is exactly the Delorme–Poljak eigenvalue bound — putting dual variables $z_i$ on the diagonal constraints forces $\mathrm{diag}(z)-L/4\succeq 0$ and minimizes $\sum_i z_i$, and the substitution $z_i=(\lambda-u_i)/4$ shows $\min\sum_i z_i=\tfrac n4\lambda_{\max}(L+\mathrm{diag}(u))$ over correcting vectors. So this relaxation is not a new loose thing; it *is* the tight $0.88$-ish bound, now wearing a form I can round.

The rounding is where the geometry pays off. After factoring $Y=UU^T$ (eigendecomposition $Y=Q\Lambda Q^T$, $U=Q\Lambda^{1/2}$), row $i$ of $U$ is the unit vector $u_i$ on the sphere. The relaxation's objective depends only on the *angles* between vectors, so it is rotation-invariant, and any rounding I pick should be too — otherwise I privilege a coordinate frame the problem ignores. The rotation-invariant way to assign a sign to a point on the sphere is to cut with a hyperplane through the origin whose normal $r$ is drawn uniformly at random, setting $S=\{i:u_i\cdot r\ge 0\}$. A uniform $r$ comes free from a spherical Gaussian — draw each coordinate i.i.d. $N(0,1)$ — and since only the sign of $u_i\cdot r$ matters, I needn't even normalize $r$. The payoff: for an edge at angle $\theta_{ij}=\arccos(u_i\cdot u_j)$, only the component of $r$ in the plane of $u_i,u_j$ affects the two signs, and that component points uniformly around the circle, so the pair lands on opposite sides exactly when the random line falls into one of two opposite wedges of width $\theta_{ij}$ out of $2\pi$. Hence

$$\Pr[(i,j)\text{ cut}]=\frac{\theta_{ij}}{\pi}=\frac{\arccos(u_i\cdot u_j)}{\pi},$$

probability proportional to precisely the angle the relaxation was pushing to enlarge, so $E[W]=\sum_{i<j}w_{ij}\,\theta_{ij}/\pi$.

Comparing this edge by edge against each edge's relaxation credit $\tfrac12 w_{ij}(1-\cos\theta_{ij})$ gives the guarantee. If $\theta/\pi\ge\alpha\cdot\tfrac12(1-\cos\theta)$ for all $\theta\in[0,\pi]$ with some constant $\alpha$, then since all weights are nonnegative, $E[W]\ge\alpha Z^*_P\ge\alpha\,\mathrm{OPT}$. So

$$\alpha=\min_{0<\theta\le\pi}\frac{\theta/\pi}{\tfrac12(1-\cos\theta)}=\min_{0<\theta\le\pi}\frac{2}{\pi}\cdot\frac{\theta}{1-\cos\theta}.$$

For small $\theta$, $1-\cos\theta\approx\theta^2/2$ makes the ratio blow up like $(4/\pi)/\theta$, and on $(0,\pi/2]$ one checks $2\theta\ge\pi(1-\cos\theta)$ so the ratio stays above $1$; the minimum therefore lives in $(\pi/2,\pi]$. Setting $\tfrac{d}{d\theta}[\theta/(1-\cos\theta)]=0$ gives $1-\cos\theta-\theta\sin\theta=0$, i.e. $\cos\theta+\theta\sin\theta=1$, whose nonzero root is $\theta^*=2.331122$ rad ($\approx133.6^\circ$). To pin the value cleanly: $1-\cos\theta$ is concave on $[\pi/2,\pi]$, so anchoring its tangent at $\theta^*$ where $1-\cos\theta^*-\theta^*\sin\theta^*=0$ gives $1-\cos\theta\le\theta\sin\theta^*$, hence $g(\theta)\ge\tfrac2\pi\cdot\theta/(\theta\sin\theta^*)=2/(\pi\sin\theta^*)\approx 0.87856$. So $\alpha=0.87856\ldots$, and accounting honestly for solving the SDP only to additive $\varepsilon$, this is a $(0.87856-\varepsilon)$-approximation for any $\varepsilon>0$ — the first jump past $1/2$ in twenty years, and the first use of semidefinite programming in approximation-algorithm design. The loss concentrates entirely at the one obtuse worst angle $\theta^*$; tiny angles (edges that don't want to be cut) and angles near $\pi$ (antipodal, almost surely cut) are handled near-perfectly, and the $5$-cycle's empirical $0.88446$ sits just above $\alpha$, so the analysis is essentially tight for this rounding.

```python
import numpy as np
import networkx as nx
import cvxpy as cvx

def max_cut(graph):
    """Returns (colors in {-1,+1}^n, cut weight of this coloring, SDP upper bound)."""
    # cut weight = x^T (L/4) x for x in {-1,+1}^n, since x^T L x = 4 * cut
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # SDP relaxation: Y = Gram matrix of unit vectors u_i (Y >= 0, diag = 1)
    X = cvx.Variable(laplacian.shape, PSD=True)
    objective = cvx.Maximize(cvx.trace(laplacian @ X))        # linear relaxation objective
    constraints = [cvx.diag(X) == 1]                          # ||u_i|| = 1
    cvx.Problem(objective, constraints).solve()

    bound = float(np.trace(laplacian @ X.value))              # Z*_P >= OPT

    # recover vectors u_i by eigendecomposing the Gram matrix
    gram = np.array(X.value, dtype=float)
    gram = 0.5 * (gram + gram.T)
    evals, evects = np.linalg.eigh(gram)
    keep = evals > 1e-6
    sdp_vectors = evects[:, keep] @ np.diag(np.sqrt(evals[keep]))

    # random-hyperplane rounding: r uniform on the sphere via a Gaussian
    r = np.random.randn(sdp_vectors.shape[1])
    r /= np.linalg.norm(r)                                    # normalization optional; only sign matters
    colors = np.sign(sdp_vectors @ r)                        # S = {i : u_i . r >= 0}
    colors[colors == 0] = 1

    score = float(colors @ laplacian @ colors.T)              # weight of the produced cut
    return colors, score, bound
```

Because the algorithm is randomized, repeat the rounding step several times with fresh $r$ and keep the best cut; each draw has expected weight at least $0.878\cdot\mathrm{OPT}$, and keeping the best observed cut can only improve the returned score.
