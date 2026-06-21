## Research question

Fix a finite graph $H$ and ask for $\mathrm{ex}(n,H)$: the maximum number of edges in an
$n$-vertex graph containing no copy of $H$. For a single forbidden graph this looks, at first,
like a question about the detailed shape of $H$: its cycles, local degrees, clique number, and
which vertices are adjacent inside it.

The Erdos-Stone-Simonovits question is whether those details control the leading term of
$\mathrm{ex}(n,H)$ as $n$ grows. The answer is strikingly coarse:

$$
\mathrm{ex}(n,H)=\left(1-\frac{1}{\chi(H)-1}+o(1)\right)\frac{n^2}{2}
$$

for every fixed graph $H$ with $\chi(H)\ge 2$, interpreting the bipartite case
$\chi(H)=2$ as $\mathrm{ex}(n,H)=o(n^2)$. The first-order edge density is governed by the
chromatic number alone.

The central conceptual move is to stop treating every forbidden subgraph as a bespoke object.
Instead, reduce the problem to the smallest number of color classes that $H$ needs. In the dense
limit, forbidding $H$ is asymptotically the same as forbidding the complete graph with
$\chi(H)$ colors expanded into bounded-size parts.

## Background

Turan's theorem gives the exact extremal answer for forbidding a clique. If $K_{r+1}$ is
forbidden, the best construction is the balanced complete $r$-partite graph $T_r(n)$, with

$$
e(T_r(n))=\left(1-\frac1r+o(1)\right)\frac{n^2}{2}.
$$

The reason is coloring: $T_r(n)$ is $r$-colorable, so it cannot contain any graph whose
chromatic number is $r+1$.

For a general fixed graph $H$ with $\chi(H)=r+1$, this same Turan graph is automatically
$H$-free. That gives the lower bound

$$
\mathrm{ex}(n,H)\ge e(T_r(n)).
$$

The hard direction is the converse: why should every graph with density just above the Turan
density contain $H$, even if $H$ is far from a clique? ESS answers by finding not merely a
$K_{r+1}$, but a bounded blow-up of it.

A blow-up $K_{r+1}(t)$ is the complete $(r+1)$-partite graph with $t$ vertices in each part.
Every fixed $(r+1)$-chromatic graph $H$ embeds into some $K_{r+1}(t)$: take a proper
$(r+1)$-coloring of $H$, put each color class into one part, and add all missing cross-edges.
So if a dense graph is forced to contain $K_{r+1}(t)$ for sufficiently large fixed $t$, it is
also forced to contain $H$.

## Baselines

- **Turan for cliques.** Exact and structural for $K_{r+1}$: the extremal graph is
  $r$-partite. Gap: most forbidden graphs are not cliques, so the theorem alone does not say why
  their internal details should vanish from the leading term.

- **Clique-number intuition.** One might guess that the largest clique in $H$ controls the answer.
  Gap: odd cycles have clique number $2$ but chromatic number $3$, and their extremal density is
  the same first-order density as forbidding $K_3$, not the same as forbidding an edge.

- **Local-count intuition.** Degrees, cycle lengths, and special substructures of $H$ matter for
  exact and lower-order extremal questions. Gap: these data do not control the $n^2$ coefficient
  for fixed non-bipartite $H$.

- **Bipartite exceptional regime.** If $\chi(H)=2$, the formula gives coefficient $0$:
  $\mathrm{ex}(n,H)=o(n^2)$. Gap: the theorem deliberately does not determine the correct
  subquadratic order; there the detailed shape of $H$ matters again.

## Evaluation settings

The theorem is evaluated at fixed $H$ and $n\to\infty$. The target is the first-order asymptotic
density, not the exact finite-$n$ extremal number and not the second-order term.

The natural stress tests are graphs with the same chromatic number but very different internal
form: $K_{r+1}$, odd cycles for $r=2$, complete multipartite graphs, sparse high-chromatic graphs,
and graphs whose color classes are very unbalanced. ESS predicts that all of them share the same
leading coefficient once $\chi(H)$ is fixed.

The construction side must produce many $H$-free edges. The balanced Turan graph $T_r(n)$ does so
because it is $r$-colorable. The forcing side must show that any fixed positive excess over the
Turan density creates enough cross-partite richness to contain a complete blow-up $K_{r+1}(t)$,
and therefore any fixed $H$ with $\chi(H)=r+1$.

## Proof artifact

The final artifact should state the ESS theorem in chromatic-number form, then explain the two
matching directions:

1. Lower bound: $T_{\chi(H)-1}(n)$ has the Turan edge density and is $H$-free.
2. Upper bound: the Erdos-Stone blow-up theorem says that exceeding that density by any fixed
   $\epsilon n^2$ forces a $K_{\chi(H)}(t)$ for every fixed $t$ once $n$ is large enough.
3. Embedding: every fixed $H$ with $\chi(H)=r+1$ is a subgraph of $K_{r+1}(t)$ for some fixed $t$.

This is the distinctive insight: the extremal problem for a particular forbidden graph collapses,
at first order, to the color-count obstruction. The specific graph affects which blow-up size is
needed and may affect lower-order terms, but the $n^2$ coefficient is the Turan coefficient
determined only by $\chi(H)$.
