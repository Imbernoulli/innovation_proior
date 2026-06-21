## Core statement

Let $H$ be a fixed graph with chromatic number $\chi(H)=r+1\ge 2$. Then

$$
\mathrm{ex}(n,H)=\left(1-\frac1r+o(1)\right)\frac{n^2}{2}.
$$

Equivalently,

$$
\mathrm{ex}(n,H)=\left(1-\frac{1}{\chi(H)-1}+o(1)\right)\frac{n^2}{2}.
$$

For bipartite $H$ this says only $\mathrm{ex}(n,H)=o(n^2)$; the precise subquadratic order is a
different, graph-specific problem.

## Why this is the insight

The lower bound is the Turan obstruction. If $\chi(H)=r+1$, the balanced complete $r$-partite
graph $T_r(n)$ is $H$-free because it is only $r$-colorable, and it has

$$
\left(1-\frac1r+o(1)\right)\frac{n^2}{2}
$$

edges.

The upper bound is the blow-up mechanism. The Erdos-Stone theorem says that any fixed positive
excess over the Turan density forces a complete $(r+1)$-partite blow-up $K_{r+1}(t)$ for every
fixed $t$ once $n$ is large. Every fixed graph $H$ with $\chi(H)=r+1$ embeds into some such
blow-up: color $H$ with $r+1$ colors and place each color class in a part.

So the detailed forbidden graph disappears from the leading term. Its vertices and adjacencies
only determine the bounded blow-up size needed to host it. The first-order extremal density is
controlled by the color-count barrier: an $r$-partite Turan graph avoids every graph needing
$r+1$ colors, and any fixed density beyond that barrier forces a universal multipartite host for
all such graphs.

This is why Erdos-Stone-Simonovits is a shift from individual forbidden-subgraph arguments to a
general structural theorem. It classifies all fixed forbidden graphs, at the $n^2$ scale, by
chromatic number.
