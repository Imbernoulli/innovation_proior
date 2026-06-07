# The max-entropy algorithm: a $(3/2-\varepsilon)$ approximation for metric TSP

## Problem

Metric TSP: given $n$ cities with symmetric distances obeying the triangle inequality, find a minimum-cost tour. NP-hard to approximate within $123/122$; the Christofides–Serdyukov $\tfrac32$ approximation stood as the best general bound for over four decades. The max-entropy algorithm beats it: a randomized tour of expected cost at most $(\tfrac32-3\cdot10^{-36})\mathrm{OPT}$.

## Key idea

Christofides' $\tfrac32$ is tight because a single deterministic spanning tree can be forced to have all vertices odd, making its matching cost $\tfrac12\mathrm{OPT}$. No instance can force *many* trees to be simultaneously bad. So replace the minimum spanning tree by a *random* tree sampled from the **maximum-entropy distribution with marginals equal to a subtour-LP optimum** $x$, then run Christofides' matching step unchanged.

- The marginals make the tree free: $\mathbb{E}[c(T)]=c(x)\le\mathrm{OPT}$.
- The max-entropy distribution is $\lambda$-uniform ($\Pr[T]\propto\prod_{e\in T}\lambda_e$), hence **strongly Rayleigh** — its generating polynomial is real stable. This yields negative dependence, log-concave degree distributions (each vertex is even with probability at least $\tfrac12(1-e^{-2})\approx0.43$), and the fact that conditioning on a tight cut being a tree splits the sample into two independent trees.
- The minimum O-join (= minimum matching on the odd set $O$) only binds on cuts crossed an **odd** number of times by $T$. Setting $y_e=x_e/2$ overpays on every cut that came out even. A random **slack vector** $s$ reclaims that overpayment — reduce $s_e$ to $-\beta x_e$ when an edge's nearby cuts are even, increase it to repair odd cuts — with negative dependence guaranteeing $\mathbb{E}[s_e]<0$, so $\mathbb{E}[c(\text{matching})]\le(\tfrac12-\varepsilon)\mathrm{OPT}$.

## Algorithm

1. Solve the subtour LP $\min\sum_e c_e x_e$ s.t. $x(\delta(v))=2$, $x(\delta(S))\ge2$, $x\ge0$; split a vertex across a weight-1 cost-0 edge $e_0$ so $x$ restricted to $E$ is in the spanning-tree polytope.
2. Find $\lambda:E\to\mathbb{R}_{\ge0}$ so the $\lambda$-uniform distribution has marginals $x_e(1\pm2^{-n})$.
3. Sample $T\sim\mu_\lambda$; let $O$ be its odd-degree vertices; let $M$ be the minimum-cost perfect matching on $O$; output $T\cup M$ shortcut to a tour.

## Why the matching is cheap (the analysis, in brief)

Build the O-join on $z=(x+\mathrm{OPT})/2$, with $y_e=x_e/4+s_e$ and $y^*_{e^*}=1/4+s^*_{e^*}$ on $\mathrm{OPT}$ edges.

- **Only near-min cuts matter.** With $\beta\le\eta/4.1$, any cut of value $>2(1+\eta)$ stays satisfied even if every crossing edge is reduced. The work is confined to the near-minimum cut regime.
- **Structure via averaging with OPT.** Every $\eta$-near-min cut of $z$ is a contiguous interval of the $\mathrm{OPT}$ cycle. Cuts crossed on both sides are odd with probability $O(\eta)$ and are charged to $\mathrm{OPT}$ edges ($\mathbb{E}[s^*_{e^*}]=O(\eta^2)$). Cuts crossed on one side form **polygons** that look like cycles ($x(E(a_i,a_{i+1}))\approx1$, $x(\delta(a_i))\approx2$), each collapsible to a **triangle**. The result is a laminar **hierarchy** $\mathcal H$ of degree cuts and triangle cuts; a triangle is *happy* (all its cuts even) when both sides $A_T,B_T$ are odd.
- **Slack on the hierarchy.** Reduce a top edge bundle when its top cuts are even; compensate when odd. Using $x(\delta(S))=2$ (independent trees inside/outside $S$), $\Pr[\delta(u)_T\text{ odd}\mid g\text{ reduced}]\le1-\Omega(\varepsilon)$, giving $\mathbb{E}[s_{\mathbf e}]\le-\varepsilon p\eta x_{\mathbf e}$.
- **Bad edges = half-edges.** Edges whose two top cuts are never simultaneously even must have $x_{\mathbf e}\approx\tfrac12$; they are never reduced. At least $\tfrac34$ of the mass in any cut is good, and of two half-edges sharing an endpoint at least one is good.
- **Bottom edges.** Reduced by $\beta$ vs top edges by $\tau=0.571\beta$, and reduced only when $A_T=B_T=1$ *with marginals preserved* (a max-flow event of probability $\Omega(\alpha^2)$). This makes bottom reductions $\infty$-ideal; e.g. the bottom-bottom case gives $\mathbb{E}[s_{\mathbf f}]=p\beta(-1+\tfrac8{27}+0.571)=-0.13\,p\beta<0$.

Setting $218\eta=\tfrac16\varepsilon_P$ with $\varepsilon_P=3.12\cdot10^{-16}$ and $\beta=\eta/4.1=\varepsilon_P/5362.8$ yields expected O-join cost $(\tfrac12-\tfrac16\varepsilon_P\beta)\mathrm{OPT}$, hence approximation ratio $\tfrac32-3\cdot10^{-36}$. Because the construction uses $\mathrm{OPT}$ edges, this bounds the algorithm, not the LP's integrality gap.

## Code

```python
def choose_tree(graph, cost, x, lam):
    # The tree slot is filled by one lambda-uniform maximum-entropy sample.
    # Its marginals are x, so E[c(T)] = c(x) <= OPT; the strong-Rayleigh
    # structure is what the slack-vector proof uses to make the matching cheap.
    return sample_spanning_tree(graph, lam)

def tsp_tour(graph, cost):
    x = held_karp_lp(graph, cost)
    lam = lambda_for_marginals(graph, x)
    T = choose_tree(graph, cost, x, lam)
    O = odd_degree_vertices(T)
    M = min_perfect_matching(O, cost)
    return eulerian_shortcut(T, M)  # expected cost <= (3/2 - 3e-36) OPT
```
