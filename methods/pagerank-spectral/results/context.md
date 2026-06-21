# Context: ranking the pages of the Web by importance

## Research question

The Web is enormous, heterogeneous, and uncurated — current estimates put it at over 150 million pages with a doubling time under a year — and a keyword query typically matches far more pages than any human could read. The pressing problem is not *finding* pages that contain the query words (a text index does that) but *ordering* them: presenting the few genuinely important pages first. We need a single, query-independent score — one number per page, computed once over the whole link graph — that measures a page's overall *importance*, and that ordering must be:

- **objective and mechanical** — computed from the link structure of the Web, not from human judgement or editorial lists, so it scales to the whole Web;
- **resistant to manipulation** — the Web contains parties actively trying to inflate their ranking, so the score should not be easily inflated by cheaply replicable features;
- **faithful to a common-sense notion of importance** — a single link from a hub like the Yahoo home page should count for more than thousands of links from obscure pages.

The score must be definable over the entire crawlable graph (≈150M nodes, ≈1.7B edges at the time), computable at that scale, and well-defined even though the graph is directed, far from strongly connected, full of cycles, and riddled with pages that have no outgoing links at all.

## Background

**Importance as a recursive, link-derived quantity.** The starting intuition comes from academic citation analysis (Garfield's citation indexing; Goffman's epidemic model of information flow): a paper cited by many papers is important. Transplanted to the Web, a link is a citation, and a page's importance grows with the number of pages that link to it (its *backlinks*). A natural refinement is that a backlink from an important page should count more than a backlink from an unimportant one. That refinement makes importance **recursive** — a page is important if important pages link to it — and the central technical question becomes whether a self-referential definition like that even has a consistent solution, and if so whether it is unique and computable.

**The recursive-importance / status lineage.** Sociology and bibliometrics had already wrestled with recursive prestige:

- *Seeley (1949)* proposed that a person's status is proportional to the summed status of those who choose them, each chooser splitting a unit of endorsement among the people they choose — a self-referential balance that is, in matrix terms, a stochastic-matrix eigenvector condition.
- *Katz (1953)* defined a status index by attenuated path-counting: status accumulates over paths of every length k, each discounted by a factor αᵏ, giving x = (Σ_{k≥1} αᵏ Aᵏ)·1 = ((I − αA)⁻¹ − I)·1 for adjacency matrix A and small α < 1/ρ(A). The attenuation α keeps the sum finite and damps long indirect paths.
- *Bonacich (1972)* made the recursion exact: status is the leading eigenvector of the adjacency matrix, x = (1/λ)A x — *eigenvector centrality*. A node is central if central nodes point to it; the fixed point is the dominant eigenvector.

These are the load-bearing ancestors: each encodes "importance begets importance" as a linear-algebra fixed point.

**Linear algebra and Markov-chain facts available at the time.** The tools needed to turn recursive importance into a solvable object were standard:

- A **column-stochastic** matrix (nonnegative, every column sums to 1) always has 1 as an eigenvalue, because its transpose fixes the all-ones vector e (Aᵀe = e). Equivalently, a row-stochastic matrix is the transition matrix of a finite Markov chain.
- A **substochastic** matrix (some column sums < 1) need not have 1 as an eigenvalue; mass leaks out under iteration.
- The **Perron–Frobenius theorem**: a strictly positive matrix has a simple dominant real eigenvalue r, strictly larger in modulus than every other eigenvalue, with a unique (up to scale) strictly positive eigenvector; the spectral projection Aᵏ/rᵏ converges. For a nonnegative *irreducible* matrix r is still a simple positive eigenvalue with a positive eigenvector, but h eigenvalues share the maximal modulus when the period is h; only a *primitive* (irreducible **and** aperiodic) matrix has the single strictly dominant eigenvalue. The stochastic corollary: an irreducible aperiodic Markov chain has a unique strictly positive stationary distribution π, and any starting distribution converges to it.
- The **power method**: for a matrix whose dominant eigenvalue is strictly larger in modulus than the rest, repeatedly applying the matrix to a generic start vector converges to the dominant eigenvector after normalization, with error contracting like |λ₂/λ₁|ᵏ — geometric in the ratio of the second-largest to the largest eigenvalue modulus. This needs only matrix–vector products, never a full eigendecomposition.

**Diagnostic facts about the Web graph itself.** The crawlable Web is *not* strongly connected: it has many disjoint and one-way-connected components; it contains large numbers of *dangling* pages (no out-links — un-crawled URLs, PDFs, images, leaf pages; in a 24M-page crawl, ≈51M referenced URLs were not yet downloaded and hence dangling); and it is sparse (≈11 links per page on average) and empirically *expander-like* / rapidly mixing — a random walk on it spreads quickly.

## Baselines

- **Raw backlink count.** Score a page by the number of links pointing to it. Links are citation votes; more backlinks means higher score.

- **Eigenvector centrality (Bonacich 1972).** x = (1/λ)A x: a node's score is proportional to the summed scores of its neighbours; the score vector is the dominant eigenvector of the adjacency matrix A. Central nodes confer centrality — the exact recursive fixed point.

- **Katz status index (1953).** x = ((I − αA)⁻¹ − I)·1: count all paths, attenuating a length-k path by αᵏ with α small enough (α < 1/ρ(A)) for the Neumann series to converge. Indirect endorsements count, but decreasingly with distance, and the attenuation factor α tames the recursion.

- **Seeley's row-normalized status (1949).** Split each node's unit of endorsement evenly among the nodes it chooses, then solve the resulting stochastic fixed point. Normalizes the *outgoing* endorsement so each node distributes exactly one unit of status.

## Evaluation settings

- **The link graph as data.** The natural input is a crawl of the Web converted to a sparse directed graph: each URL a node, each hyperlink a directed edge; out-degrees and in-degrees per node; dangling nodes flagged. Scales of interest at the time: tens to hundreds of millions of nodes, hundreds of millions to billions of edges (e.g. maps of 161M and 322M links; a 24M-page / 75M-URL crawl).
- **What a score is judged against.** Correspondence with a common-sense notion of importance (does the Stanford home page outrank a random page mentioning "Stanford"?); correspondence with observed usage / traffic (e.g. proxy-cache hit counts as an external signal); predictiveness of future backlinks (whether the score, computed on a partial crawl, orders pages the way full-information citation counts eventually would). As a search aid, the score is used to *re-order* the pages a title/text match returns.
- **Computational protocol.** Build the sparse link matrix once; compute the score by iterating to a convergence tolerance ε on the L₁ change between iterates; report iterations-to-converge versus graph size (the relevant question being whether the iteration count stays roughly constant — i.e. logarithmic in n — as the graph grows). Memory budget matters: at 75M URLs × 4 bytes the score vector alone is ≈300 MB, so the iteration must use sparse, streamable matrix access.

## Code framework

The scaffold ingests a crawl into a sparse directed graph, keeps the out-degree information needed for link-based flow, and iterates a ranking update until the L₁ change between successive score vectors falls below a tolerance. The score definition and the update rule are the empty slot.

```python
import numpy as np
import scipy.sparse as sp

def load_link_graph(edges, n):
    """Sparse directed link graph from a crawl: edges = list of (src, dst).
    Returns an n x n adjacency in sparse form; column j = out-links of page j."""
    rows = [d for (_, d) in edges]
    cols = [s for (s, _) in edges]
    A = sp.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    return A

def out_degrees(A):
    """Number of out-links per page (column sums of A)."""
    return np.asarray(A.sum(axis=0)).ravel()

def ranking_update(A):
    """Return an update function r -> next_r for a link-based importance score."""
    # TODO: define a well-posed link-based update
    pass

def rank_scores(A, tol=1e-7, max_iter=200):
    """Compute one importance score per page by iterating the operator
    to convergence (an L1 tolerance on the change between iterates)."""
    n = A.shape[1]
    r = np.full(n, 1.0 / n)          # generic positive start, sums to 1
    apply_update = ranking_update(A)
    for _ in range(max_iter):
        r_next = apply_update(r)
        if np.abs(r_next - r).sum() < tol:
            return r_next
        r = r_next
    return r
```
