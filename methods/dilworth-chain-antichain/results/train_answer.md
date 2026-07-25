I want to know how few chains it takes to cover a finite partially ordered set. The order $\le$ is reflexive, antisymmetric and transitive but not total, so the set splits naturally into two opposite shapes: a chain, a subset all of whose elements are pairwise comparable — a thread running with the order — and an antichain, a subset no two of whose elements are comparable, a flat layer cutting across it. Partition $P$ into disjoint chains; I want the minimum number of them, and, more than the number, I want to know what feature of the order itself pins that minimum down. One bound comes for free and costs no machinery. A chain and an antichain meet in at most one element, since two shared elements would have to be comparable (both on one chain) and incomparable (both in one antichain) at once. So a single chain can absorb at most one element of any given antichain, and covering an antichain of size $w$ needs at least $w$ distinct chains, giving

$$\#\{\text{chains in any cover}\} \;\ge\; \text{largest antichain size}$$

for every cover and every antichain. The two extreme orders show this is not slack in the trivial cases: a single long chain $a_1 < a_2 < \dots < a_n$ has largest antichain $1$ and is covered by one chain, and $n$ pairwise-incomparable elements have largest antichain $n$ and genuinely need $n$ singleton chains. The whole question is whether the inequality is tight in general.

What that question looks like is worth reading carefully, because its shape names the tool. "Minimum size of a cover equals maximum size of an obstruction" is the signature of a duality theorem, a covering minimum squeezed down until it meets a packing maximum, and by 1950 several such theorems were on the shelf — yet none of them applies to a bare poset. Menger's theorem (Menger 1927) equates the maximum number of vertex-disjoint paths between two vertices with the minimum separating vertex set, but a poset comes with no pair of distinguished terminals to separate, and its chains are threads of an order rather than $s$–$t$ paths in a graph, so there is nothing to instantiate. Hall's theorem (Hall 1935) gives the exact condition $|N(S)| \ge |S|$ for saturating one side of a bipartite graph by a matching; it is a feasibility criterion about representing a family of sets by distinct elements, and it certifies saturation rather than exhibiting a min–max value one could read a width off. Dushnik–Miller (1941) does address posets, but it decomposes an order into linear extensions whose intersection recovers it and measures the least number of those — total orders laid over the whole set, and a controlling number (the dimension) that is a different invariant entirely, not the chain count. That leaves König's theorem (König 1931; Egerváry 1931): in a bipartite graph the maximum matching equals the minimum vertex cover, and it is proved constructively, through augmenting paths, so it hands over both optimal objects rather than merely asserting a number. It is also the one whose primitive object, a set of pairwise-disjoint edges, is exactly the atom out of which order-threads get spliced. The obstruction is that König lives on an undirected bipartite graph — two disjoint sides, every edge crossing between them, no edge inside a side and none from a vertex to itself — and a poset is one set carrying a reflexive, transitive relation. There is no bipartite graph inside the statement of the problem. So the order has to be rebuilt into one.

I propose the decomposition theorem: in any finite poset $P$ the minimum number of chains in a chain decomposition equals the maximum size of an antichain, this common value being the width of $P$. What proves it, and what I regard as the actual method, is a reframing I will call the split-graph successor-link reduction — split every element into two role-copies, read a bipartite matching as a set of successor-links, and let König's dual pair of optimal objects hand back the chains and the antichain together. Concretely, take two disjoint copies of the ground set, the lower copies $U = \{u_a : a \in P\}$ and the upper copies $V = \{v_b : b \in P\}$, and draw

$$u_a \,-\, v_b \quad\Longleftrightarrow\quad a < b \quad (\text{strict: } a \le b,\ a \ne b).$$

This is bipartite by construction — every edge runs from a $U$-vertex to a $V$-vertex — so König applies with no bipartiteness left to verify and no odd cycles to rule out. The strictness is a deliberate choice, not a formality: allowing $a \le b$ would, by reflexivity, create the edge $u_a - v_a$ for every element, and a matched edge is supposed to assert "$b$ is the successor of $a$ on its chain." An element being its own successor is meaningless, and worse, every element could then be matched to itself, inflating the matching to size $n$ and destroying the count I am about to build. The relation a matching should draw from is exactly "$b$ can directly follow $a$," which is strict comparability; reflexivity is discarded on purpose.

Now read a matching $M$ as a set of links, each edge $u_a - v_b \in M$ meaning "$a$'s chosen successor is $b$." The matching condition says each $u_a$ is touched at most once, so each element has at most one chosen successor, and each $v_b$ is touched at most once, so each element has at most one chosen predecessor. A relation with out-degree $\le 1$ and in-degree $\le 1$ at every element is a disjoint union of simple paths and cycles, and cycles are impossible here: a cycle of links reads $a_1 < a_2 < \dots < a_t < a_1$, which transitivity collapses to $a_1 < a_1$, forbidden by antisymmetry. So the chosen links thread the elements into vertex-disjoint paths, and each path is a genuine chain rather than merely a sequence of consecutive relations, because transitivity makes all of its elements pairwise comparable, not just neighbouring ones. The count is what turns minimization into maximization. With $M$ empty there are $n$ singleton chains. Switching on a link $u_a - v_b$ glues the fragment that ends at $a$ to the fragment that starts at $b$, and it always joins two distinct fragments: if $a$ and $b$ already lay on a common fragment with $a$ before $b$, then $a$ already has a successor there, contradicting out-degree $\le 1$ at $a$; and if $b$ came before $a$, the fragment asserts $b < a$ while the new link asserts $a < b$, giving $a < b < a$, impossible. Each link therefore drops the fragment count by exactly one, and a matching of size $m$ yields

$$n - m \ \text{chains}.$$

Fewer chains means more links means a larger matching, so the minimum chain cover is $n - m^{*}$ with $m^{*}$ the maximum matching size — the minimization I could not attack directly has become a maximization König's machinery solves.

That builds the chains. The witnessing antichain has to come from König's dual object, the minimum vertex cover, and the translation back is where one has to be careful, because every element wears two hats in the split graph. Let $C$ be a minimum vertex cover, so $|C| = m^{*}$. Call an element $a \in P$ free if neither $u_a$ nor $v_a$ lies in $C$, and let $A$ be the set of free elements. Then $A$ is an antichain: if $a, b \in A$ with $a < b$, the edge $u_a - v_b$ exists, and $a$ free puts $u_a \notin C$ while $b$ free puts $v_b \notin C$, so that edge has both endpoints outside $C$ and is uncovered — contradicting that $C$ is a vertex cover. This is exactly why the *complement* of the cover, and not the cover itself, is the object to translate: a vertex cover is a hitting set for the strict relations, so what it misses entirely is a set carrying no relation internally, which is what an antichain is. Counting is then immediate. Each element owns exactly two vertices, $u_a$ and $v_a$, and these belong to it alone; an element fails to be free precisely when at least one of its two vertices is in $C$, so distinct non-free elements account for distinct vertices of $C$ and there are at most $|C| = m^{*}$ of them. Hence

$$|A| \;=\; n - \#\{\text{non-free elements}\} \;\ge\; n - m^{*}.$$

Now the two halves meet. The maximum matching gives a chain cover with exactly $n - m^{*}$ chains, the minimum vertex cover gives an antichain with $|A| \ge n - m^{*}$, and the cheap bound from the start says any antichain is at most the number of chains in any cover, so $|A| \le n - m^{*}$. The two squeeze to $|A| = n - m^{*}$, equal to the chain count, and therefore the minimum chain cover equals the largest antichain. Both certificates — the chains from $M$, the antichain from the complement of $C$ — are produced by one run, so the optimum is not merely asserted but exhibited.

Because that argument leans on König for the single fact that a matching and a cover of equal size exist simultaneously, and because it is the constructive form that makes the width computable rather than just bounded, I want that step airtight rather than cited as a black box. Weak duality is the easy half: a cover must place a vertex on each edge of a matching, matching edges are vertex-disjoint, so it spends a distinct vertex per matching edge and $|C| \ge |M|$ for every cover and matching. For the tight half I first need a certificate that a given matching is maximum, and that is Berge's criterion: $M$ is maximum if and only if it admits no augmenting path, an alternating path whose two endpoints are unmatched. One direction is a direct construction — such a path begins and ends with non-matching edges (its endpoints carry no matching edge), so it reads non, match, non, $\dots$, non with $k$ matching and $k+1$ non-matching edges; flipping every edge along it admits $k+1$ and removes $k$, a net gain of one, and the result is still a matching because each internal vertex lay on exactly one matched and one unmatched path edge and ends up with exactly one matched, each endpoint gains its first, and off-path edges are untouched. The converse is the symmetric difference argument. Suppose some $M^{*}$ has $|M^{*}| > |M|$ and set $Q = M \triangle M^{*}$. Every vertex carries at most one $M$-edge and at most one $M^{*}$-edge, so $Q$ has maximum degree $2$ and decomposes into disjoint simple paths and cycles, along which edges alternate between $M$ and $M^{*}$ (two consecutive edges of the same type would meet at a vertex twice). Cycles are even and contribute equally to both matchings, so the surplus must live on a path, and a path with more $M^{*}$-edges than $M$-edges begins and ends with $M^{*}$-edges. Its endpoints are unmatched in $M$ — an $M$-edge at an endpoint would itself lie in $Q$ and the path would continue through it — so that path is an augmenting path for $M$, contradiction.

The failed search for an augmenting path is then exactly what hands over the small cover. Take a maximum $M$ and let $X$ be all vertices reachable from unmatched $U$-vertices by alternating paths: leave a $U$-vertex along a non-matching edge into $V$, return from a $V$-vertex along its matching edge into $U$, repeat. Define

$$C := (U \setminus X) \,\cup\, (V \cap X).$$

Every vertex of $U \setminus X$ is matched, since unmatched $U$-vertices are search roots and hence in $X$; every vertex of $V \cap X$ is matched too, since an unmatched $V$-vertex reached by an alternating path from an unmatched $U$-vertex would be an augmenting path, contradicting maximality. Their match edges exhaust $M$ without double counting: a match edge $u - v$ with $u \in U \setminus X$ has $v \notin X$, because $v \in X$ would let the search cross that matching edge and put $u \in X$; symmetrically a match edge with $v \in V \cap X$ has $u \in X$. So $|C| = |M|$. And $C$ covers every edge $u - v$: if $u \notin X$ then $u \in U \setminus X \subseteq C$; and if $u \in X$ then $v \in X$, because either $u - v$ is a non-matching edge, along which the search steps from $u$ to $v$, or $u - v$ is $u$'s matching edge, in which case $u$ was reached into along a non-matching edge and the search continues out of $u$ along precisely this matching edge (an unmatched root $u$ has no matching edge at all). Either way $v \in V \cap X \subseteq C$. So min cover $\le |M| \le$ max matching, and with weak duality König's equality holds, both objects falling out of a single alternating search.

Everything is therefore effective: building the split graph is $O(n^2)$ edges, the augmenting-path matching is polynomial, the alternating search for the cover is linear in the graph, so the width of a finite poset is computed — with certificates on both sides — rather than merely characterized. Two further consequences come cheaply. The mirror statement, that the size of the largest chain equals the fewest antichains needed to cover $P$, needs no matching at all: the easy direction is the same one-element intersection bound read the other way, and for the reverse, set $\ell(x)$ to be the length of the longest chain ending at $x$; if $a < b$ then any chain ending at $a$ extends by $b$, so $\ell(a) < \ell(b)$, which makes each level set $\{x : \ell(x) = k\}$ an antichain and produces exactly (longest chain length) antichains covering $P$, which the easy bound then forces to be optimal. It is a pleasant asymmetry that the antichain-cover side is the trivial one and only the chain-cover side needed König. The other consequence concerns infinite orders. The argument above is finite throughout — $n$ elements, a finite matching, a finite search — but the conclusion survives whenever the width is finite: a cover of $P$ by $w$ chains is the same thing as a proper $w$-coloring of the incomparability graph, since each color class is then pairwise comparable; every finite subset of $P$ has width $\le w$ and so is $w$-colorable by the finite theorem; and De Bruijn–Erdős compactness lifts a coloring from all finite subgraphs to the whole graph. So an infinite poset of finite width $w$ still decomposes into $w$ chains, while the equality can fail once the width itself is infinite — the finite-width hypothesis is exactly the right one.

On the divisibility order of $\{1,2,3,4,6,12\}$ the construction returns the chains $\{1,2,4,12\}$ and $\{3,6\}$ together with the antichain $\{4,6\}$: two chains, an antichain of two, and since $4$ and $6$ are incomparable no single chain could cover the set, so the width is $2$ and the pair of certificates proves it. Here is the construction that certifies the width on a concrete finite poset.

```python
from typing import List, Set

def chain_cover_and_antichain(n: int, leq: List[List[bool]]):
    """Finite poset on {0..n-1} given by its <= matrix `leq`.
    Returns (chains, antichain): a minimum chain cover and a witnessing antichain
    of equal size, certifying width(P) = max-antichain = min-chain-cover."""

    # reframe comparability as a bipartite graph: lower copies (left) vs upper
    # copies (right); edge u_a - v_b iff a < b (strict).
    adj: List[List[int]] = [[] for _ in range(n)]      # adj[a] = uppers b with a<b
    for a in range(n):
        for b in range(n):
            if a != b and leq[a][b]:
                adj[a].append(b)                        # successor-link candidate

    # ---- maximum bipartite matching by augmenting paths (Berge/Kuhn) ----
    match_v = [-1] * n          # match_v[b] = lower a matched to upper b, or -1
    succ    = [-1] * n          # succ[a] = chosen successor (upper) of a, or -1

    def try_augment(a: int, seen: List[bool]) -> bool:
        for b in adj[a]:                      # walk non-matching edge u_a - v_b
            if not seen[b]:
                seen[b] = True
                if match_v[b] == -1 or try_augment(match_v[b], seen):
                    match_v[b] = a
                    succ[a]   = b
                    return True
        return False

    for a in range(n):
        try_augment(a, [False] * n)
    m = sum(1 for b in range(n) if match_v[b] != -1)    # maximum matching size

    # ---- chains = connected components of the chosen successor-links ----
    pred = [-1] * n
    for b in range(n):
        if match_v[b] != -1:
            pred[b] = match_v[b]
    chains: List[List[int]] = []
    for a in range(n):
        if pred[a] == -1:                      # chain head (no predecessor)
            chain, x = [], a
            while x != -1:
                chain.append(x)
                x = succ[x]
            chains.append(chain)               # transitivity => genuine chain
    # exactly n - m chains

    # ---- minimum vertex cover via alternating search from exposed U-vertices ----
    inX_u = [False] * n
    inX_v = [False] * n
    matched_u = [succ[a] != -1 for a in range(n)]

    def visit(a: int):
        inX_u[a] = True
        for b in adj[a]:                       # non-matching step u_a -> v_b
            if not inX_v[b]:
                inX_v[b] = True
                if match_v[b] != -1 and not inX_u[match_v[b]]:
                    visit(match_v[b])          # matching step v_b -> u_(its match)
    for a in range(n):
        if not matched_u[a]:                   # unmatched lower vertex = root
            visit(a)
    cover_u = {a for a in range(n) if not inX_u[a]}   # U \ X
    cover_v = {b for b in range(n) if inX_v[b]}       # V ∩ X   (|C| = m)

    # ---- antichain = elements with NEITHER copy in the cover (the "free" ones) ----
    antichain: Set[int] = {a for a in range(n)
                           if a not in cover_u and a not in cover_v}
    # |antichain| = n - |C| = n - m = number of chains  => optimum certified
    return chains, antichain
```
