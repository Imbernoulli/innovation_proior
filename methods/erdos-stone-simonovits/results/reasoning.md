Start with the extremal question in its most concrete form. I fix a graph $H$ and want as many
edges as possible on $n$ vertices without containing $H$. If $H$ is a clique, Turan already gives
the answer: forbid $K_{r+1}$ and the balanced complete $r$-partite graph is optimal, with edge
density $1-1/r$.

The surprise is that the same density should govern every fixed graph with chromatic number
$r+1$, not only the clique $K_{r+1}$. The lower bound is immediate once I look through coloring.
Let $\chi(H)=r+1$. The Turan graph $T_r(n)$ is $r$-colorable, so it cannot contain $H$ at all.
Thus

$$
\mathrm{ex}(n,H)\ge e(T_r(n))=\left(1-\frac1r+o(1)\right)\frac{n^2}{2}.
$$

That part says chromatic number is at least a plausible obstruction. The hard part is showing it
is the only first-order obstruction.

If I try to force $H$ directly, the graph-specific details get in the way. An odd cycle, a sparse
high-chromatic graph, and a clique have very different local forms. But they share one feature:
each needs $r+1$ color classes. So I should force a universal host for all graphs with $r+1$ color
classes, rather than force $H$ itself.

That universal host is a blow-up of a clique. Let $K_{r+1}(t)$ be the complete $(r+1)$-partite
graph with $t$ vertices in each part. Any fixed graph $H$ with $\chi(H)=r+1$ embeds into
$K_{r+1}(t)$ for some fixed $t$: color $H$ with $r+1$ colors, put each color class in one part,
choose $t$ at least as large as the largest color class, and note that the complete multipartite
host has every cross-edge that $H$ could need. It may have extra edges, but containment only needs
a subgraph.

So the real forcing statement becomes: once an $n$-vertex graph has edge density more than
$1-1/r$ by a fixed amount, it must contain $K_{r+1}(t)$ for every fixed $t$, provided $n$ is large
enough. This is the Erdos-Stone core. Simonovits's contribution completes the general fixed-graph
form and the now-standard statement of the theorem: the extremal number depends asymptotically on
$\chi(H)$.

Why should the blow-up be forced? A graph above the Turan density cannot be organized as an
essentially $r$-partite object. The excess edges create enough dense interaction among
$r+1$ large vertex blocks to find many choices in each block with all cross-edges present. In
modern language this is often proved through regularity or dependent-random-choice style
reasoning; historically the point is the same: positive density above the Turan barrier upgrades
one forbidden clique into a complete multipartite pattern of fixed size.

Once $K_{r+1}(t)$ is forced, the original graph $H$ follows automatically. Therefore, for every
fixed $\epsilon>0$ and every fixed $H$ with $\chi(H)=r+1$, all sufficiently large graphs with

$$
e(G)>\left(1-\frac1r+\epsilon\right)\frac{n^2}{2}
$$

contain $H$. Equivalently,

$$
\mathrm{ex}(n,H)\le \left(1-\frac1r+o(1)\right)\frac{n^2}{2}.
$$

Together with the Turan lower bound, this gives

$$
\mathrm{ex}(n,H)=\left(1-\frac1r+o(1)\right)\frac{n^2}{2}
\quad\text{where }r=\chi(H)-1.
$$

This explains exactly what disappears and what remains. The detailed shape of $H$ determines how
large a blow-up $K_{r+1}(t)$ is needed, and it can determine exact finite-$n$ behavior or
subquadratic behavior in the bipartite case. But for fixed non-bipartite $H$, those details are
lower-order data. The first-order term sees only whether an $r$-partite Turan graph can avoid the
forbidden pattern. If it can, that gives the lower bound; if I add any fixed density beyond it, the
blow-up theorem forces every graph needing $r+1$ colors.

That is why ESS is a change in viewpoint. It is not another solution to one forbidden-subgraph
problem. It turns a family of apparently individual extremal questions into a universal structural
law: dense graphs just above the $r$-partite Turan barrier contain bounded complete
$(r+1)$-partite blow-ups, and fixed forbidden graphs are classified at first order by the number
of colors needed to embed them into such a blow-up.
