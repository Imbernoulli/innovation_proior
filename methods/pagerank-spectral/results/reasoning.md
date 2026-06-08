I have a crawl of the Web in front of me — call it tens of millions of pages, a directed graph of hyperlinks — and a query for "university" returns more pages than anyone could ever read. The text index already tells me which pages contain the word. What I cannot do yet is *order* them: put the genuinely important pages first. So the real problem isn't matching, it's importance. I want one number per page, computed once over the whole graph, query-independent, that says how important this page is.

Let me start from what "important" should mean and refuse to let it be anything gameable. The cheapest idea is: a page is important if many pages link to it — count the backlinks, treat each link as a citation, like Garfield's citation counts for papers. And that's not nothing; the Netscape home page has tens of thousands of backlinks and obscure pages have a handful, and the ranking that produces is roughly sane. But I'm building this in an environment where people are actively trying to fool me. Anyone can write a program that spins up ten thousand pages all linking to their own page, and raw backlink count rewards exactly that. The feature is replicable, therefore worthless against an adversary. And it's wrong even without adversaries: a single link from the Yahoo home page ought to mean more than a thousand links from pages nobody visits. Backlink counting throws away the obvious thing — *who* is doing the linking.

So the count of backlinks isn't it; the *quality* of the backlinks is. A page is important if *important* pages link to it. Say that out loud and it's circular — importance defined in terms of importance. The instinct is to recoil from a circular definition, but I shouldn't; this is exactly the recursion I want. The question is whether the circle closes — whether there's a consistent assignment of scores that satisfies it, and whether it's unique, and whether I can actually compute it on a hundred million pages.

People have been here before, in a different language. Sociologists measuring status had the same circularity: you have high status if high-status people choose you. Seeley in '49 wrote it as a balance — each person hands out one unit of endorsement, split among the people they choose, and your status is the sum of what flows into you. Katz in '53 counted endorsements over paths of every length, discounting a length-k path by a factor αᵏ so the indirect, far-away endorsements fade and the infinite sum converges. Bonacich in '72 made the recursion exact and named the fixed point: status is the dominant eigenvector of the adjacency matrix, x = (1/λ)Ax — you're central if central nodes point to you. All three are saying "importance begets importance," and all three resolve it as a linear-algebra fixed point. That's the shape of the answer. But every one of them is clean only on a connected, essentially undirected graph, and my graph is a sprawling directed mess that is nowhere near strongly connected. I'll have to find out exactly where they break.

Let me just write my recursion down and see what object it is. Let u be a page, let B_u be the set of pages that link to u, and for a page v let N_v be its number of out-links. I want a page that endorses to *divide* its endorsement among the pages it links to — not shower full weight on each — because that's the anti-manipulation property: if your endorsement is a fixed quantity split among your out-links, then adding out-links doesn't manufacture influence, it only dilutes what you already had. So:

  r(u) = Σ_{v ∈ B_u} r(v) / N_v.

Each page v passes r(v)/N_v along each of its out-links; r(u) is the total that lands on u. Now make it a matrix. Define M by M_{uv} = 1/N_v if v links to u, and 0 otherwise. Then r(u) = Σ_v M_{uv} r(v), i.e.

  r = M r.

Stare at that. r is an eigenvector of M with eigenvalue 1. And M has a structure I should name: every column v sums to Σ_u M_{uv} = N_v · (1/N_v) = 1 (the page v spreads exactly one unit across its N_v out-links). So M is **column-stochastic** — nonnegative, columns summing to one. That's not a coincidence I imposed; it fell out of "divide your endorsement evenly," which I chose for manipulation-resistance. The recursion *is* a stochastic-matrix eigenvector equation. Good — that means I get something for free: a column-stochastic matrix always has 1 as an eigenvalue. Take the all-ones vector e; M column-stochastic means Mᵀe = e (the rows of Mᵀ sum to one), so 1 is an eigenvalue of Mᵀ, hence of M, since a matrix and its transpose share eigenvalues. So a solution r to r = Mr *exists*. The recursion closes.

And there's a second reading of M r that I like even better than "endorsement flow," because it tells me the score is a probability. Mᵀ is row-stochastic, so it's the transition matrix of a Markov chain: from page v, jump to a uniformly random one of its out-links. Picture a surfer who just keeps clicking links at random, forever. The vector r I'm solving for, the eigenvalue-1 eigenvector of M, is the stationary distribution of that walk — the long-run fraction of time the surfer spends on each page. So importance, the recursive thing, is literally *how often a random clicker lands here*. That's a satisfying definition: a page is important to exactly the degree that aimless link-following keeps returning to it.

Now I want to actually compute r on a graph too big to ever form M explicitly, let alone diagonalize it. Diagonalizing an n×n matrix is O(n³) and n is a hundred million — out of the question. But I don't need the whole spectrum; I need one eigenvector, the eigenvalue-1 one, and M is sparse (eleven nonzeros per column on average). That's the power method's home turf: start with any reasonable vector, apply M over and over, and converge to the dominant eigenvector — using nothing but sparse matrix–vector products. Each step r ← Mr is just "every page pushes its current score, split, along its out-links," an O(number of links) sweep. So *if* eigenvalue 1 is the strictly dominant eigenvalue and *if* the eigenvector is unique, I have a scalable algorithm. Two big ifs. Let me poke at them, because this is where I expect the ancestors to break.

First if — uniqueness. The web is not one connected blob; it's full of disjoint and one-way-connected pieces. Suppose two clumps of pages that point richly within themselves but never to each other. Then M is block-diagonal-ish, each block has its own eigenvalue-1 eigenvector, and the eigenspace for eigenvalue 1 is multi-dimensional. Which means r = Mr has a whole family of solutions and "the ranking" is ambiguous — I could weight one component's pages up and the other's down arbitrarily and still satisfy the equation. That's the eigenvector-centrality failure on a disconnected directed graph, made concrete. The ranking isn't determined.

Worse, watch a small trap. Two pages that link only to each other, with some outside page feeding one of them. The surfer wanders in, and now there are no out-links escaping the pair, so every step the two of them keep handing their score back and forth and accumulating whatever drips in from outside. Under iteration their scores run away — they hoard rank and never give it back to the rest of the graph. A rank sink. The cyclic pocket eats all the importance. So even when a solution exists, the iteration can pile everything into a cul-de-sac.

Second if — and it's nastier. **Dangling pages**: pages with no out-links at all. There are vast numbers of them — every URL I've seen referenced but not yet crawled, every PDF and image and dead-end leaf. In my crawl, tens of millions of URLs are referenced but un-downloaded, hence dangling. A dangling page v has N_v = 0, so its column in M is all zeros — the surfer arrives and has nowhere to go. That column doesn't sum to 1; it sums to 0. So M isn't column-stochastic anymore, it's column-*sub*stochastic. And the argument that 1 is an eigenvalue used Mᵀe = e, which now fails on those columns. Every step, total rank Σ_u r(u) drops by exactly the amount sitting on dangling pages, because they absorb their incoming rank and emit nothing. Iterate and the whole score vector decays toward zero. The fixed point I wanted has leaked out the bottom of the graph.

So the bare recursion — beautiful as r = Mr is — is broken three ways on the real Web: non-unique when the graph is disconnected, divergent into rank sinks at cyclic traps, and degenerate-to-zero at dangling nodes. The mathematicians' eigenvector centrality assumed those problems away by assuming connectivity; I can't.

Let me go back to the random-surfer picture, because the *physical* fix is obvious there and I can read the math off it afterward. A real person browsing doesn't follow links forever, and certainly doesn't get permanently trapped in a two-page loop or stranded on a dead-end page. They get bored. Periodically they stop following links and just jump — type in some other URL, hit a bookmark — to a page chosen out of the blue. So: with probability α the surfer clicks a random out-link as before, and with probability 1−α they teleport to a page chosen uniformly at random over the whole Web. That single behavioral change repairs all three failures at once, and I want to see *why* in the matrix.

A teleport to a uniform-random page is the operator (1/n)·1·1ᵀ — every entry 1/n, every column the uniform distribution. Before I mix it with the link-following walk, I have to make the symbol M mean a true column-stochastic link operator. For a page with out-links, the column is still the split endorsement M_{uv} = 1/N_v when v links to u. For a dangling page, the column cannot stay zero; the surfer at a dead end has to jump uniformly, so I replace that zero column by (1/n)1. Now every column of M sums to 1. With that patched M, mix in boredom:

  G = α M + (1 − α) (1/n) 1·1ᵀ.

That's the operator the surfer actually obeys. Let me check it's column-stochastic, because the whole probability interpretation rides on that. Since the patched M has column sums 1, the columns of αM each sum to α. The columns of (1−α)(1/n)1·1ᵀ each sum to (1−α). Together every column sums to α + (1−α) = 1. So G is column-stochastic — no more leak, total rank is conserved, and the dangling hole is plugged because dead-end rank now returns uniformly into circulation instead of vanishing.

And rank sinks? The two-page loop can no longer hoard, because (1−α) of whatever rank it holds teleports out to the rest of the graph every step. There's no trap anymore: from *any* page you can reach *any* page in one step with probability at least (1−α)/n > 0. Which is the property I actually need, stated cleanly: every entry of G is at least (1−α)/n, strictly positive. **G is a strictly positive matrix.** That single fact is the key, and now I can finally invoke the theorem that the ancestors couldn't.

Perron–Frobenius: for a strictly positive matrix, the dominant eigenvalue is real, simple, strictly larger in modulus than every other eigenvalue, and it has a unique eigenvector all of whose entries are strictly positive — and nothing else nonnegative is an eigenvector except scalar multiples of it. G is column-stochastic, so eᵀG = eᵀ and ‖G‖₁ = 1; that gives ρ(G) ≤ 1, while stochasticity also gives eigenvalue 1, so the Perron root is exactly 1. Because G is positive, that eigenvalue is the strict top. Therefore the equation r = G r has a *unique* solution up to scale, that solution is strictly *positive* (every page gets a positive score — no page is exactly zero, which also kills the "non-unique on a disconnected graph" problem, because positivity forces a single ranking across what used to be separate components), and I can normalize it to sum to 1 and read it as a probability distribution. The teleport made the chain irreducible — you can get from anywhere to anywhere — and aperiodic — the self-reachability and the spread of teleport destroy any cyclic period — so it's a primitive Markov chain with a unique stationary distribution π, and that π is my importance score. The recursive definition that looked circular and ill-posed is, after the teleport, the well-defined stationary distribution of an irreducible aperiodic random walk. Everything the ancestors couldn't guarantee — existence, uniqueness, positivity — Perron–Frobenius now hands me, *because* the teleport made the matrix strictly positive.

I should sanity-check that this G is the same object the surfer story describes and not some renormalized cousin. If d = Σ_{v:N_v=0} r(v) is the rank currently sitting on dangling pages, then the arriving rank on u is α times the ordinary link-flow from non-dangling in-neighbours, plus αd/n from the uniform dangling redistribution, plus (1−α)/n from boredom teleport. So

  r(u) = α Σ_{v ∈ B_u, N_v>0} r(v)/N_v + αd/n + (1 − α)/n.

That's exactly the scalar PageRank equation with dangling pages made explicit; if I fold the dangling columns into M, it is just r = Gr. In the no-dangling shorthand, and with the normalization folded so the average rank is 1, the same equation is often written PR(u) = (1−α) + α Σ PR(v)/N_v. The teleport term (1−α)/n is the "rank source" — a constant trickle of importance handed to every page just for existing — and the αd/n term is the dead-end mass being put back into circulation instead of disappearing. α around 0.85 means the surfer follows about six links before teleporting, but it cannot be only a modeling knob; it also controls computation.

Now, does the power method actually converge on G, and how fast? Convergence first. The power method's error after k steps decays like the k-th power of the modulus of the *second* eigenvalue (relative to the first, which is 1 here). Perron–Frobenius already told me |λ₂| < 1 strictly, because the dominant eigenvalue 1 is strictly larger in modulus than all others for a positive matrix. So r_k → r geometrically; the iteration converges from any positive start, and conserves the L₁ norm at 1 since G is stochastic. Fine — but "geometric at some rate < 1" is not good enough to know it's *practical* on a hundred-million-node graph. I need the actual rate, and I have a feeling it's controlled by α. Let me pin |λ₂| down.

Take any eigenvector x₂ of G with eigenvalue λ₂ ≠ 1. The eigenvalue-1 *left* eigenvector of G is e (the all-ones vector), since G column-stochastic means eᵀG = eᵀ. Eigenvectors of a matrix for distinct eigenvalues — well, x₂ is a right eigenvector for λ₂ and e is the left eigenvector for λ₁ = 1, and right and left eigenvectors for different eigenvalues are orthogonal. So eᵀx₂ = 0: the second eigenvector sums to zero. Now apply G to x₂ and look at what the teleport part does to a zero-sum vector. The teleport operator is (1−α)(1/n)1·1ᵀ, and (1·1ᵀ)x₂ = 1·(1ᵀx₂) = 1·(eᵀx₂) = 1·0 = 0. The teleport annihilates x₂ completely. So

  λ₂ x₂ = G x₂ = α M x₂ + (1−α)(1/n)1·1ᵀ x₂ = α M x₂ + 0 = α M x₂.

Divide by α: M x₂ = (λ₂/α) x₂. So x₂ is *also* an eigenvector of the patched link matrix M, with eigenvalue λ₂/α. But M is column-stochastic, so ‖M‖₁ = 1 and every eigenvalue of M has modulus at most 1. Therefore |λ₂/α| ≤ 1, i.e.

  |λ₂| ≤ α.

There it is. The second eigenvalue of the Google matrix is bounded by the damping factor. The teleport didn't just make the matrix positive — it gives a worst-case convergence factor no larger than α. If the patched link matrix has another unit-modulus eigenvalue on a zero-sum direction, the bound can be tight; I do not need equality for the algorithm. The power method converges geometrically at rate at most α.

Now the choice of α stops being arbitrary. The error after k iterations is on the order of αᵏ, so to reach an L₁ tolerance ε I need roughly k ≈ log ε / log α iterations. With α = 0.85 and ε = 10⁻⁶, that's (−6·ln 10)/ln 0.85 ≈ 13.82/0.1625 ≈ 85 iterations in the worst-case bound; with ε = 10⁻⁷ it is about 99. The important part is that this count depends on α and the tolerance, *not* on the size of the graph, so the same convergence argument applies whether the graph has a hundred million links or a billion. And it exposes the tension that fixes α. Push α toward 1 and the ranking follows the link structure more faithfully (less teleport noise), but |λ₂| can approach 1 and convergence slows sharply — infinitely many iterations in the limit, and the graph's disconnected pockets and sinks start to reassert themselves. Push α down and convergence is lightning-fast but the score washes out toward uniform (1/n everywhere as α → 0), ignoring the links I care about. α = 0.85 sits where the ranking still strongly reflects link structure yet the chain mixes fast enough to converge in a practical number of sparse sweeps. The "damping factor" is simultaneously the surfer's persistence, the guarantor of a unique positive ranking, and the convergence-rate control — one number doing three jobs, and that's why it's the parameter.

Let me also be honest about the dangling patch in code, because I cannot literally store a uniform column for every dead-end page without making the sparse matrix dense. Algebraically, the patched M replaces each zero column by (1/n)1. Computationally, I keep the sparse matrix with zero dangling columns, measure the current dangling mass d = Σ_{v:N_v=0} r(v), and broadcast αd/n to every page in the same line that broadcasts the boredom teleport. Those two views are identical: the ordinary sparse link-flow sums to 1−d, the dangling broadcast sums to αd, and the teleport broadcast sums to 1−α, so the next vector sums to α(1−d)+αd+(1−α)=1. No separate renormalization is part of the operator; at most I can guard against floating drift, but the fixed-point step itself preserves mass.

One more efficiency point, since n is huge: I must never materialize G — it's dense (every entry ≥ (1−α)/n). But I never need to. The teleport part is rank-one: α M r is a sparse mat-vec over the non-dangling link columns, and both the redistributed dangling mass and the teleport contribution are scalar broadcasts to the uniform target. So one power-iteration step is a sparse sweep over the links plus an O(n) vector update. That's why this is feasible at Web scale and a full eigendecomposition never would be.

The pieces now line up. Importance is recursive — important pages make you important — which forces r = Mr with M the column-stochastic "split your endorsement among your out-links" matrix, so r is its principal eigenvector and equivalently the stationary distribution of a random link-follower. That bare object is broken on the real Web (non-unique on disconnected pieces, divergent at rank sinks, degenerate at dangling pages), and the single repair — a bored surfer who teleports uniformly with probability 1−α — turns M into the strictly positive, column-stochastic Google matrix G = αM + (1−α)(1/n)11ᵀ. Positivity invokes Perron–Frobenius for a unique, strictly positive principal eigenvector (existence, uniqueness, irreducibility, aperiodicity all delivered by the teleport); the same teleport bounds the second eigenvalue by α, so the power method — sparse mat-vec plus dangling and teleport broadcasts, no matrix ever formed — converges geometrically at rate at most α, in iterations independent of graph size. Let me write it.

```python
import numpy as np

def pagerank(A, alpha=0.85, tol=1.0e-6, max_iter=100):
    """Importance score = principal eigenvector / stationary distribution of
    the Google matrix G = alpha*M + (1-alpha)*(1/n)*1*1^T, by power iteration.
    A is the sparse adjacency: A[i, j] = 1 if page j links to page i."""
    n = A.shape[0]
    if n == 0:
        return np.array([])

    # Sparse normalized link matrix for non-dangling columns. Dangling columns
    # stay zero here and are patched by the dangle_mass broadcast below.
    out_deg = np.asarray(A.sum(axis=0)).ravel()          # out-links per page
    dangling = (out_deg == 0)                             # dead-end pages
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out_deg[~dangling]
    M = A.multiply(inv)                                   # M[i,j] = 1/N_j if j->i

    p = np.full(n, 1.0 / n)        # uniform teleport target (the rank source)
    r = np.full(n, 1.0 / n)        # generic positive start, sums to 1

    for _ in range(max_iter):
        r_last = r
        # dangling mass would leak out of M; send it to the teleport target,
        # which is exactly the zero-column -> uniform-column patch.
        dangle_mass = alpha * r_last[dangling].sum()
        # one power-iteration step on G, never forming G:
        #   alpha * (sparse link flow)  +  redistributed dangling mass
        #                               +  (1-alpha) uniform teleport
        r = alpha * (M @ r_last) + (dangle_mass + (1.0 - alpha)) * p
        # converged when the L1 change between iterates is below the
        # NetworkX-style average-per-node tolerance; for an L1 tolerance eps,
        # the worst-case bound is ~ log(eps)/log(alpha) steps, independent of n.
        if np.abs(r - r_last).sum() < n * tol:
            return r
    raise RuntimeError("power iteration did not converge in max_iter")
```
