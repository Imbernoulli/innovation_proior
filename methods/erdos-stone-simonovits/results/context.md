## Research question

Fix a finite graph $H$ and ask for $\mathrm{ex}(n,H)$: the maximum number of edges in an
$n$-vertex graph containing no copy of $H$. For a single forbidden graph this looks, at first,
like a question about the detailed shape of $H$: its cycles, local degrees, clique number, and
which vertices are adjacent inside it.

The question here is what determines the leading term of $\mathrm{ex}(n,H)$ as $n$ grows, for a
general fixed graph $H$. Turan's theorem settles the case where $H$ is a clique; the task is to
say what the first-order edge density is when $H$ is an arbitrary fixed graph.

## Background

Turan's theorem gives the exact extremal answer for forbidding a clique. If $K_{r+1}$ is
forbidden, the best construction is the balanced complete $r$-partite graph $T_r(n)$, with

$$
e(T_r(n))=\left(1-\frac1r+o(1)\right)\frac{n^2}{2}.
$$

The reason is coloring: $T_r(n)$ is $r$-colorable, so it cannot contain any graph whose
chromatic number is $r+1$.

For a general fixed graph $H$ with $\chi(H)=r+1$, this same Turan graph is automatically
$H$-free, since $H$ needs $r+1$ colors and $T_r(n)$ provides only $r$. That gives the lower bound

$$
\mathrm{ex}(n,H)\ge e(T_r(n)).
$$

A useful object on the construction side is the blow-up. The blow-up $K_{r+1}(t)$ is the complete
$(r+1)$-partite graph with $t$ vertices in each part. Every fixed $(r+1)$-chromatic graph $H$
embeds into some $K_{r+1}(t)$: take a proper $(r+1)$-coloring of $H$, put each color class into
one part, and add all missing cross-edges.

## Baselines

- **Turan for cliques.** Exact and structural for $K_{r+1}$: the extremal graph is the balanced
  $r$-partite $T_r(n)$.

- **Clique-number intuition.** One natural guess is that the largest clique inside $H$ controls
  the answer.

- **Local-count intuition.** Degrees, cycle lengths, and special substructures of $H$ are the
  quantities that govern exact and lower-order extremal questions.

- **Bipartite case.** When $\chi(H)=2$, $H$ embeds in a complete bipartite graph, a setting whose
  extremal counts are studied separately and are subquadratic in $n$.

## Evaluation settings

The target is the first-order asymptotic density at fixed $H$ as $n\to\infty$, not the exact
finite-$n$ extremal number and not the second-order term.

The natural stress tests are graphs with the same chromatic number but very different internal
form: $K_{r+1}$, odd cycles for $r=2$, complete multipartite graphs, sparse high-chromatic
graphs, and graphs whose color classes are very unbalanced.

The construction side must produce many $H$-free edges. The balanced Turan graph $T_r(n)$ does so
because it is $r$-colorable. The matching upper-bound side must show that a positive excess over
the Turan density forces a copy of $H$.

## Proof artifact

The final artifact should state the extremal density for a general fixed $H$ and supply the two
matching directions:

1. Lower bound: a Turan-type construction that is $H$-free and attains the claimed density.
2. Upper bound: an argument that exceeding that density by any fixed $\epsilon n^2$ forces a copy
   of $H$ once $n$ is large enough.

The construction relies on coloring; the embedding step links a fixed $H$ to the complete
multipartite blow-up $K_{r+1}(t)$ that hosts it.
