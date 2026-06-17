# The Crossing Number Inequality

## Statement

For a simple graph $G$ with $n$ vertices and $m$ edges, the **crossing number** $\operatorname{cr}(G)$ — the minimum number of edge crossings over all drawings of $G$ in the plane — satisfies, whenever $m \ge 4n$,

$$\operatorname{cr}(G) \ge \frac{m^3}{64\,n^2}.$$

The bound is sharp up to the constant: for dense graphs ($m$ of order $n^2$, e.g. the complete graph) it matches the trivial upper bound $\operatorname{cr}(G) = O(m^2)$.

## Key idea

A single application of Euler's formula gives only the weak bound $\operatorname{cr}(G) \ge m - 3n$, which is linear in $m$ and useless for dense graphs. The cube law is obtained by **amplifying** this bound on a random induced subgraph: each vertex is kept independently with probability $p$, which suppresses edges by $p^2$ but crossings by $p^4$. Applying the weak bound in expectation and optimizing over $p$ converts the additive bound into the multiplicative cube $m^3/n^2$.

## Proof

**Base bound (Euler).** A simple planar graph on $N$ vertices has at most $3N$ edges. For a connected planar graph with $N \ge 3$, add noncrossing edges until it is maximal planar; then every face is triangular, $3F = 2E$, and Euler's formula $V - E + F = 2$ gives $E = 3V - 6$ for the maximal graph, hence $E \le 3V - 6 \le 3V$ for the original graph. Disconnected components can first be joined in the plane without crossings, and the cases $N < 3$ are immediate under the weaker $3N$ bound. Now take an optimal drawing of $G$ with exactly $\operatorname{cr}(G)$ crossings. Deleting one edge from each crossing removes all crossings, leaving a planar graph on $n$ vertices with at least $m - \operatorname{cr}(G)$ edges. Hence $m - \operatorname{cr}(G) \le 3n$, i.e.

$$\operatorname{cr}(G) \ge m - 3n. \tag{$\ast$}$$

**Probabilistic amplification.** Fix $p \in (0,1]$. Form a random induced subgraph $H$ by keeping each vertex of $G$ independently with probability $p$ (and keeping an edge iff both its endpoints are kept). Let $n_H, m_H, \operatorname{cr}(H)$ be the vertex count, edge count, and crossing number of $H$. The bound $(\ast)$ holds for every graph, so $\operatorname{cr}(H) \ge m_H - 3 n_H$ for every realization.

To take expectations without confronting the minimum-over-drawings in $\operatorname{cr}(H)$, fix an optimal drawing $D$ of $G$ and let $X$ be the number of its crossings that survive into $H$ (i.e. all four involved vertices are kept). The restriction of $D$ to $H$ is a drawing of $H$ with $X$ crossings, so $\operatorname{cr}(H) \le X$, and therefore $X \ge m_H - 3 n_H$ pointwise. Taking expectations (linearity of expectation, valid without independence of the survival events):

- $\mathbb{E}[n_H] = p\,n$ — each vertex survives with probability $p$.
- $\mathbb{E}[m_H] = p^2 m$ — an edge survives iff both endpoints survive.
- $\mathbb{E}[X] = p^4 \operatorname{cr}(G)$ — in the **optimal** drawing $D$ no two crossing edges share a vertex (if they did, swapping the two arcs between the shared vertex and the crossing point would remove a crossing without adding any, contradicting optimality), so each crossing involves four distinct vertices and survives with probability $p^4$.

Hence $\mathbb{E}[X] \ge \mathbb{E}[m_H] - 3\,\mathbb{E}[n_H]$ becomes

$$p^4 \operatorname{cr}(G) \ge p^2 m - 3 p n.$$

**Optimization.** Dividing by $p^4 > 0$,

$$\operatorname{cr}(G) \ge \frac{m}{p^2} - \frac{3n}{p^3}.$$

The two terms balance near $p \approx 3n/m$; choosing the clean value

$$\boxed{p = \frac{4n}{m}}$$

(legal as a probability precisely because $m \ge 4n$ gives $p \le 1$), substitute:

$$\frac{m}{p^2} = \frac{m^3}{16 n^2}, \qquad \frac{3n}{p^3} = \frac{3 m^3}{64 n^2},$$

$$\operatorname{cr}(G) \ge \frac{m^3}{16 n^2} - \frac{3 m^3}{64 n^2} = \frac{4 m^3 - 3 m^3}{64 n^2} = \frac{m^3}{64\, n^2}. \qquad \blacksquare$$

## Remarks

- The threshold $m \ge 4n$ is exactly the condition that makes the chosen probability $p = 4n/m$ valid. At the boundary $m=4n$ this gives $p=1$, so the argument recovers the base bound value $m-3n=n=m^3/(64n^2)$.
- The exponents are forced by the survival arithmetic: vertices $\to p$, edges $\to p^2$, crossings $\to p^4$. Carrying the highest power on the crossings (the quantity bounded below) against a lower power on the edges (the positive driver), then optimizing $p \sim n/m$, is what turns the additive bound $m - 3n$ into the cube $m^3/n^2$.
