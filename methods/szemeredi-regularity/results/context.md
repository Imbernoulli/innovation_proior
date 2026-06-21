# Context: approximating arbitrary dense graphs by random-looking ones

## Research question

When does a large, arbitrary, dense graph contain a prescribed substructure — a triangle, a
$K_4$, a fixed subgraph $H$, an arithmetic-progression pattern encoded as a graph? For *random*
graphs the answer is easy: in $G(n,p)$ every fixed $H$ appears the expected number of times,
because between any two reasonably large vertex sets $X, Y$ the number of edges is, with
overwhelming probability, close to $p\,|X|\,|Y|$. That single uniformity property is what makes
counting in random graphs routine.

Arbitrary dense graphs offer no such guarantee. A graph can have edge density $1/2$ overall yet
hide all its edges inside one half, leaving the rest empty; local densities can swing wildly. The
broad question is how to leverage the tools developed for random graphs — subgraph counting,
extremal thresholds, removal arguments — when the input graph has no randomness built in.

## Background

**Uniformity as local discrepancy.** For disjoint $X, Y \subseteq V(G)$ write the density
$d(X,Y) = e(X,Y)/(|X|\,|Y|)$, the fraction of possible $X$–$Y$ edges present. A random graph of
density $p$ has $d(X,Y) \approx p$ simultaneously for *all* large $X, Y$. The natural way to ask a
deterministic graph to imitate this is to demand that the density of every pair of largish
subsets stay close to the overall density of the block — a one-block, finitary form of
*discrepancy control*. This is exactly the language in which quasirandomness was being formalized.

**Quasirandomness (Thomason 1987; Chung–Graham–Wilson 1989).** A sequence of graphs $(G_n)$ of
density $\sim 1/2$ is "quasirandom" if it shares the salient first- and second-order statistics of
a true random graph, and the striking discovery was that a long list of seemingly different
conditions are all *equivalent*: that every fixed $H$ on $\nu$ vertices occurs
$(1+o(1))\,n^\nu 2^{-\binom{\nu}{2}}$ times; that $e(G_n) \ge \tfrac14 n^2 + o(n^2)$ while the
$4$-cycle count is at most $(n/2)^4 + o(n^4)$ (the single pattern $C_4$ already forces everything);
that every vertex subset $X$ spans $\tfrac14 |X|^2 + o(n^2)$ edges (edge-discrepancy); that
$\sum_{x,y}\big||N(x)\cap N(y)| - n/4\big| = o(n^3)$ (codegrees concentrate). The lesson: *uniformity of
edge distribution across all large sets is the same thing as control of subgraph counts.*

**The counting/embedding heuristic.** If between two blocks $A, B$ the edges are distributed
uniformly with density $d$, then for almost every vertex of $A$ its number of neighbours inside
any large subset $Y \subseteq B$ is about $d\,|Y|$. That single fact is enough to embed a bounded-
degree graph greedily, vertex by vertex, choosing each next vertex from the still-large set of
candidates that respect all previously-committed adjacencies. So a graph that is a union of a
*few* uniform blocks should contain a fixed pattern $H$ whenever the obvious density obstruction is
absent — the pattern survives from a coarse weighted blueprint down to the actual graph.

**The motivating prize: arithmetic progressions.** The pressure for such a tool came from number
theory. Van der Waerden (1927) showed any finite colouring of the integers yields monochromatic
arithmetic progressions of any length; Erdős and Turán (1936) conjectured this is really a
*density* phenomenon — any set of positive upper density contains $k$-term progressions. Roth
(1953) proved the $k=3$ case analytically (giving $r_3(n) = O(n/\log\log n)$); the $k=4$ case fell
in 1969; the general case was settled in 1975 by a long elementary-combinatorial argument whose
engine was a graph-decomposition statement — a "lemma on bipartite graphs" asserting that any
large bipartite graph can be broken into nearly regular bipartite subgraphs. (Behrend's 1946
construction, giving sets of density $\exp(-c\sqrt{\log n})$ with no $3$-term progression, set the
lower-bound backdrop and showed that any density theorem must survive surprisingly dense
progression-free sets.)

## Baselines

**The 1975 bipartite lemma (the direct ancestor).** Inside the proof of the density theorem for
progressions sits a self-standing lemma: given a bipartite graph on $A \cup B$ with a marked edge
set, one can find subsets on which the relative density is *nearly constant under further
restriction* — formally, sets $C, C'$ with $\beta(S,T) \ge \beta(C,C') - \delta$ for all largish
$S \subseteq C$, $T \subseteq C'$ — and one extracts these nearly-regular pieces by iterating a
density-defect argument until the residual is small. The bookkeeping ($\varepsilon(t)$ schedules,
nested $Z_j, \bar C_j$ sequences) is tailored to the bipartite setting within the progression proof.

**Turán-type and Erdős–Stone counting (Turán 1941; Erdős–Stone 1946).** Turán: a graph with more
than $\big(1 - \tfrac{1}{p-1}\big)\tfrac{n^2}{2}$ edges contains $K_p$; Erdős–Stone:
$\mathrm{ex}(n, K_p(t,\dots,t)) = \big(1 - \tfrac{1}{p-1}\big)\binom{n}{2} + o(n^2)$, which pins
down $\mathrm{ex}(n, L)$ asymptotically via $\chi(L)$. These give *thresholds* for the existence of
a subgraph.

**Sieve / inclusion–exclusion subgraph counts.** Counts of fixed subgraphs can be produced
directly by sieve-type formulas; effective for specific patterns when the graph structure is
otherwise known.

**Random-graph counting (the gold standard one wants to imitate).** In $G(n,p)$ subgraph counts are
forced by the uniform edge distribution. The whole point of any structural approximation is to let
an arbitrary graph inherit this.

## Evaluation settings

The natural testbeds — all pre-existing problems whose statements predate any decomposition tool —
are extremal and Ramsey-type:
- **Ramsey–Turán problems**, e.g. the maximal edge count of a $K_4$-free graph with only $o(n)$
  independent vertices (a setting where the extremal density is $1/2$, bounded away from $0$ and
  $1$, so the uniform-block view is essential rather than cosmetic).
- **The $(6,3)$ / Brown–Erdős–Sós hypergraph problem**: the maximum number of triples on $n$
  vertices with no $6$ points spanning $3$ triples.
- **Erdős–Stone / Turán-type extremal counts** ($\mathrm{ex}(n,L)$ and $H$-count thresholds).
- **The arithmetic-progression density problem**: bounding $r_k(n)$, the largest progression-free
  subset of $[n]$, and proving positive density forces $k$-term progressions.
- Metrics are the usual ones: number of copies $H \to G$, edges removable to destroy all copies of
  $H$, density thresholds. The yardstick parameters are the uniformity tolerance and the density
  floor below which a block is treated as empty.
