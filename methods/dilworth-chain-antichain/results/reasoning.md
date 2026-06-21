OK, let me think this through from scratch. I have a finite partial order $P$ — a set with a relation $\le$ that is reflexive, antisymmetric, transitive, but where plenty of pairs are simply incomparable. I want to chop $P$ into chains: disjoint totally-ordered threads whose union is everything. And I want the *fewest* of them. So the object I'm after is the minimum number of chains in a chain cover, and I want to know what in the order itself pins that number down.

Let me first feel out a lower bound before reaching for any machinery, because the cheap half usually tells me the shape of the answer. What forces me to use *many* chains? An antichain — a set of pairwise-incomparable elements. Take any antichain $A$, say of size $w$, and any chain cover at all. A single chain is totally ordered, so any two of its elements are comparable; but any two elements of $A$ are incomparable. So a chain can contain at most *one* element of $A$ — if it held two, those two would be both comparable (same chain) and incomparable (same antichain), which can't be. Hence to cover the $w$ elements of $A$ I need at least $w$ different chains. That holds for every antichain, so

$$\#\text{chains in any cover} \;\ge\; \text{size of any antichain} \;\ge\; \text{(largest antichain size)}.$$

So the minimum chain cover is at least the width — the size of the largest antichain. Clean, forced, costs nothing. The whole question is whether this bound is *tight*: can I always find a chain cover with exactly (largest antichain) chains?

Let me sanity-check on the two extremes. If $P$ is one long chain $a_1 < a_2 < \dots < a_n$, the largest antichain is a single element and indeed one chain covers everything — $1 = 1$. If $P$ is one big antichain, $n$ pairwise-incomparable elements, the largest antichain is $n$ and I genuinely need $n$ singleton chains — $n = n$. Both tight. So the conjecture is: *minimum chain cover = largest antichain, always.* Now I have to either prove the reverse inequality or break it.

Stare at the *shape* of what I'm claiming. "Minimum size of a cover equals maximum size of an obstruction." A covering quantity squeezed down until it meets a packing quantity. That is exactly the smell of a duality theorem — min cover = max packing. I've seen this shape before, and not vaguely: Menger (1927) proved that the minimum number of vertices separating two nodes equals the maximum number of vertex-disjoint paths between them — min cut = max packing of paths. And König (1931), in the bipartite world, proved that the minimum vertex cover equals the maximum matching. Those are *constructive* min–max theorems sitting on the shelf. If I can make my poset *look like* one of those structures, I might get the reverse inequality for free instead of inventing a whole new induction.

Which one? Menger is about disjoint paths between two fixed terminals — there's no natural pair of terminals in a bare poset, and its "chains" are paths in a graph, not threads of an order. König is about a *bipartite* graph: a matching versus a vertex cover. And matchings have a property I should look at hard, because a matching is exactly "a set of pairwise-disjoint edges," and disjoint edges are the atoms out of which I splice threads. Let me chase König.

König's theorem lives on an undirected bipartite graph: two *disjoint* sides $L$ and $R$, edges only crossing between them, no edge inside a side, and certainly no edge from a vertex to itself. My poset is none of that. It's *one* set, with a relation that goes both ways (if $a \le b$ then I might also reason from $b$ down to $a$), reflexive (every element relates to itself), and transitive (relations chain through). There is no poset inside König's statement. So I can't apply it as-is; I have to *build* a bipartite graph out of the order and hope the matching means something about chains.

So how do I turn an order into a bipartite graph? I want a matching to encode chains. A chain is $c_1 < c_2 < \dots < c_t$, a sequence of "next" steps: $c_2$ follows $c_1$, $c_3$ follows $c_2$, and so on. So the *atom* of a chain is a "successor link" — a pair $(a, b)$ with $a < b$ meaning "on its chain, $b$ comes right after $a$." If I had a set of such links and I could guarantee each element is the *predecessor* of at most one other and the *successor* of at most one other, then following the links would thread the elements into disjoint chains. "Predecessor of at most one" and "successor of at most one" — those are *two independent degree-one constraints*, one on the "I'm the lower end of a link" role and one on the "I'm the upper end of a link" role. Two roles per element, each used at most once. That is *crying out* to be a matching: split every element into two copies, a "lower" copy and an "upper" copy, put the lower copies on one side and the upper copies on the other, and let a successor link $(a,b)$ be the edge from the lower copy of $a$ to the upper copy of $b$. Then "each element is the lower end of at most one link" is "each lower copy is touched at most once," and "each element is the upper end of at most one link" is "each upper copy is touched at most once" — and a set of edges touching each vertex at most once is *exactly a matching*.

Let me write it down. Take two copies of the ground set $P$. Call the lower copies $U = \{u_a : a \in P\}$ and the upper copies $V = \{v_b : b \in P\}$. Put an edge

$$u_a \;-\; v_b \quad\text{whenever}\quad a < b \;\;(\text{i.e. } a \le b \text{ and } a \ne b).$$

This is bipartite by construction — every edge runs from a $U$-vertex to a $V$-vertex, the two sides are disjoint, done. König applies directly, no bipartiteness to check.

Why *strict* $a < b$ and not $a \le b$? Because if I allowed $a \le b$ I'd get the edge $u_a - v_a$ for every $a$ (reflexivity). A matched edge is supposed to mean "$b$ is the successor of $a$ on a chain"; an element being its own successor is meaningless, and worse, it would let me "match" every element to itself trivially and wreck the count I'm about to set up. The relation I want a matching to draw from is precisely "$b$ can directly follow $a$," which is strict comparability $a < b$, $a \ne b$. Reflexivity gets thrown out; that's correct.

Now the crucial bookkeeping: if $M$ is a matching in this graph, what does it do to chains? Read each edge $u_a - v_b \in M$ as the link "$a$'s successor is $b$." Because $M$ is a matching, each $u_a$ is touched at most once — so each element has at most one chosen successor; and each $v_b$ is touched at most once — so each element has at most one chosen predecessor. A set of links with in-degree $\le 1$ and out-degree $\le 1$ at every element is a disjoint union of simple paths (and, in principle, cycles). Are cycles possible? A cycle of links would be $a_1 < a_2 < \dots < a_t < a_1$, which by transitivity forces $a_1 < a_1$ — impossible in a strict order (antisymmetry kills $a_1 < a_1$). So no cycles: the chosen links partition the elements into disjoint *paths*, and each path $a_1 < a_2 < \dots$ is a genuine chain because transitivity makes all its elements pairwise comparable, not just consecutive ones. Good — a matching gives me an honest set of disjoint chains.

How many chains? Start from $M$ empty: that's $n$ singleton chains, one per element. Now switch the links on one at a time. Each link $u_a - v_b$ glues the chain-fragment that *ends* at $a$ to the chain-fragment that *starts* at $b$. I have to check it always joins two *distinct* fragments — otherwise it might close a loop or do nothing. Since $M$ is a matching, $a$ was the upper end of no chosen link before (it's about to be a lower end now, $u_a$ used once) — wait, let me argue it cleanly. Suppose adding link $(a,b)$ joined a fragment to *itself*. Then $a$ and $b$ were already on the same fragment, meaning there was already a chain $\dots a \dots b \dots$ or $\dots b \dots a \dots$. If $a$ comes before $b$ on it, then we'd be adding $a$'s successor to be $b$ while $a$ already has a successor on that fragment (the next element) — contradicts out-degree $\le 1$ at $a$. If $b$ comes before $a$, then $b < a$ along the fragment yet the new link says $a < b$, and with $b < a$ that's $a < b < a$, impossible. So every added link joins two *distinct* fragments into one, dropping the fragment count by exactly $1$. Therefore a matching of $m$ links yields

$$n - m \quad\text{chains.}$$

Fewer chains $\iff$ more links $\iff$ larger matching. So the *minimum* chain cover is $n - (\text{maximum matching size})$. And maximizing a matching is precisely what König's machinery does. The minimization has become a maximization that I can actually solve.

So I have one half of the bridge: a chain cover of size $n - m^\*$ where $m^\*$ is the maximum matching. I need to show $n - m^\*$ also equals the largest antichain — that this chain cover is *optimal* and the antichain *witnesses* it. This is where König's *dual* object — the minimum vertex cover — has to do the work, and I have to figure out how a vertex cover of the split graph becomes an antichain of the poset.

A vertex cover $C$ is a set of vertices (here, lower copies $u_a$ and upper copies $v_b$) meeting every edge. König says I can find one with $|C| = m^\*$ (= the maximum matching). What is the *complement* of a vertex cover? An *independent set* — a set of vertices with no edge among them. Translate back: a set of elements such that no edge $u_a - v_b$ has *both* its... hmm, I have to be careful because each element wears two hats, $u_a$ and $v_b$. Let me define the candidate antichain as the set of elements that are "completely missed" by the cover: call $a \in P$ *free* if **neither** $u_a$ **nor** $v_a$ is in $C$. Let $A$ = the free elements. Claim: $A$ is an antichain.

Suppose not — suppose $a, b \in A$ with $a < b$ (relabel so the smaller is $a$). Then the edge $u_a - v_b$ exists in the graph. Is it covered by $C$? Its endpoints are $u_a$ and $v_b$. But $a$ free means $u_a \notin C$; and $b$ free means $v_b \notin C$. So *neither* endpoint of this edge is in $C$ — the edge is uncovered, contradicting that $C$ is a vertex cover. Hence no two free elements are comparable: $A$ is an antichain. The complement-of-a-cover *is* an antichain — that's the whole reason a vertex cover is the right dual object to translate.

Now count $|A|$. Each element $a$ owns exactly two vertices, its copies $u_a$ and $v_a$. An element is *not* free exactly when at least one of its two copies is in $C$, so each non-free element accounts for $\ge 1$ vertex of $C$; and distinct elements use distinct vertices ($u_a, v_a$ belong to $a$ alone), so those counted vertices are distinct. The number of non-free elements is therefore at most $|C|$. So

$$|A| = n - \#\{\text{non-free elements}\} \;\ge\; n - |C| \;=\; n - m^\*.$$

Put the two pieces together. I have a chain cover with exactly $n - m^\*$ chains (from the maximum matching), and an antichain $A$ with $|A| \ge n - m^\*$ (from the minimum vertex cover, which König makes the same size $m^\*$). But the cheap lower bound from the start says every antichain is *at most* the number of chains in any cover, so $|A| \le n - m^\*$. The two squeeze: $|A| = n - m^\*$ exactly, and it equals the number of chains. So

$$\text{minimum chain cover} \;=\; n - m^\* \;=\; |A| \;=\; \text{largest antichain.}$$

The bound is tight. And it's *constructive*: the maximum matching builds the chains, the minimum vertex cover (its complement) builds the witnessing antichain, both at once. That is exactly the reverse inequality I needed, and it dropped out of König instead of out of a bespoke induction.

But I've been leaning on König as a black box — "there exists a matching $M$ and a cover $C$ with $|M| = |C|$." That's the load-bearing fact and I should not take it on faith; let me actually prove it, because the *constructive* version is what gives me both objects simultaneously and makes the width computable. König's theorem: in a bipartite graph the maximum matching equals the minimum vertex cover.

One direction is the weak duality I already used implicitly: any vertex cover must put at least one vertex on each edge of any matching, and a matching's edges are vertex-disjoint, so a cover spends at least one *distinct* vertex per matching edge — hence $|C| \ge |M|$ for *every* cover and matching, so (min cover) $\ge$ (max matching). The content is the reverse: I must exhibit a cover as small as the largest matching.

How do I even know a given matching is the *largest*? I need a certificate. Here's the idea, due to Berge: a matching $M$ is maximum iff there is no *augmenting path*. An augmenting path is a path that starts and ends at *unmatched* vertices and alternates non-matching / matching / non-matching / … edges along the way. If I have one, I can enlarge $M$: flip every edge on the path (matched $\to$ unmatched and vice versa). Let me check that's legal and gains one. The path begins and ends with non-matching edges (its endpoints are unmatched, so they carry no matching edge), so the pattern is non, match, non, match, …, non — $k$ matching edges and $k+1$ non-matching ones. After flipping, the $k+1$ formerly-non-matching edges enter $M$ and the $k$ formerly-matching edges leave: net $+1$. Still a matching? Each internal vertex lay on exactly two path edges, one matched and one not; after the flip exactly one of the two is matched, and it had no other matching edge (the one it had was on the path). Each endpoint was unmatched and now carries one new matching edge. Off-path edges of $M$ are untouched. So the flip yields a valid matching one larger. Hence if an augmenting path exists, $M$ is not maximum.

Conversely — the part I need — *no augmenting path* forces $M$ maximum. Suppose, for contradiction, some matching $M^\*$ is strictly larger, $|M^\*| > |M|$, while $M$ has no augmenting path. Look at the symmetric difference $Q = M \triangle M^\*$, the edges in exactly one of the two. At any vertex, $Q$ has at most one $M$-edge and at most one $M^\*$-edge, so every vertex has degree $\le 2$ in $Q$; thus $Q$ is a disjoint union of simple paths and cycles. Along any of these, consecutive edges alternate between $M$ and $M^\*$ (two consecutive same-type edges at a vertex would give it two $M$-edges or two $M^\*$-edges). Cycles must then be even, contributing equally to $M$ and $M^\*$. So the surplus $|M^\*| - |M| > 0$ must come from the *paths*; at least one path has more $M^\*$-edges than $M$-edges, so it begins and ends with $M^\*$-edges. Its two endpoints are matched in $M^\*$ but unmatched in $M$ (if an endpoint had an $M$-edge, that edge would be in $Q$ — the path would continue through it — contradicting that the path ends there). An alternating path whose endpoints are both $M$-unmatched is an augmenting path for $M$. Contradiction. So no larger $M^\*$ exists: **no augmenting path $\Rightarrow$ $M$ maximum.** That's Berge's criterion, and it's the hinge.

Now turn the "no augmenting path" certificate into the small vertex cover. Take a maximum matching $M$ (no augmenting path). Run an alternating search from the unmatched vertices on the left side $U$. Formally: let $X$ be the set of all vertices reachable from an unmatched $U$-vertex by alternating paths — leave a $U$-vertex along a *non*-matching edge into $V$, return from a $V$-vertex along its *matching* edge to $U$, repeat. Let me define the cover as

$$C := (U \setminus X) \;\cup\; (V \cap X).$$

First, $|C| = |M|$. Every vertex of $U \setminus X$ is matched: an *unmatched* $U$-vertex is a search root, hence in $X$, so it's excluded from $U \setminus X$. Every vertex of $V \cap X$ is matched too: if some $v \in V \cap X$ were unmatched, the alternating path that reached it would start at an unmatched $U$-vertex and end at an unmatched $V$-vertex — an augmenting path, contradicting maximality. So both $U \setminus X$ and $V \cap X$ consist of matched vertices. Moreover their matching edges are distinct and account for all of $M$: each matched $U$-vertex in $U \setminus X$ contributes its match edge; each matched $V$-vertex in $V \cap X$ contributes its match edge; and these don't double-count because a match edge $u - v$ with $u \in U \setminus X$ has $v \notin X$ (if $v \in X$ then $u$ would be reached from $v$ along this matching edge, putting $u \in X$ — contradiction), so $v \notin V \cap X$; symmetrically a match edge with $v \in V \cap X$ has $u \in X$ so $u \notin U \setminus X$. Every match edge is thus counted exactly once, giving $|C| = |M|$.

Second, $C$ is a vertex cover. Take any edge $u - v$, $u \in U$, $v \in V$; I must show it meets $C$. If $u \notin X$ then $u \in U \setminus X \subseteq C$, done. So suppose $u \in X$. Then I claim $v \in X$, which puts $v \in V \cap X \subseteq C$. Why is $v \in X$? The edge $u - v$ is either matching or not. If it's non-matching, then from the reached vertex $u \in X$ I can step along this non-matching edge to $v$, so $v \in X$. If it's matching, then $u$ is matched via this very edge; $u$ being in $X$ and matched means $u$ was reached *into* along a non-matching edge from some $v' \in X$, and the search continues from $u$ along its matching edge $u - v$, landing on $v$ — so $v \in X$. (Or $u$ is an unmatched root, but then $u - v$ can't be a matching edge.) Either way $v \in X$, so the edge is covered. Hence $C$ covers every edge. A cover of size $|M|$ — so (min cover) $\le |M| \le$ (max matching), and with weak duality, **König's theorem holds: max matching $=$ min vertex cover**, both produced constructively from one alternating search.

That closes the loop. The maximum matching of the split graph builds the minimum chain cover ($n - m^\*$ chains); the complementary minimum vertex cover builds the witnessing antichain (the free elements, size $n - m^\*$); the trivial bound seals equality. Everything is constructive and runs in polynomial time — building the graph is $O(n^2)$ edges, the augmenting-path matching is polynomial, the alternating search for the cover is linear in the graph — so the width of any finite poset is computable, not just bounded.

Let me make it concrete on a small poset so I trust the bookkeeping. Take the divisibility order on $\{1,2,3,4,6,12\}$: $a \le b$ iff $a \mid b$. Strict relations $a < b$: $1<2,1<3,1<4,1<6,1<12,\;2<4,2<6,2<12,\;3<6,3<12,\;4<12,\;6<12$. Build the split graph: $U=\{u_1,u_2,u_3,u_4,u_6,u_{12}\}$, $V=\{v_1,\dots,v_{12}\}$, edge $u_a - v_b$ for each strict relation. Find a maximum matching. Try $u_1-v_2,\;u_2-v_4,\;u_4-v_{12},\;u_3-v_6$. Check it's a matching: lower ends $1,2,4,3$ distinct, upper ends $2,4,12,6$ distinct — yes, size $4$. Read the links: $1$'s successor $2$, $2$'s successor $4$, $4$'s successor $12$, $3$'s successor $6$. Thread them: $1<2<4<12$ is one chain; $3<6$ is another; $6$ is already used as a successor of $3$, and nothing's successor is... wait, $6$ has no chosen successor, so $3<6$ closes. Elements $1,2,4,12,3,6$ — all six covered. Chains: $\{1,2,4,12\}$ and $\{3,6\}$. That's $2$ chains $= n - m = 6 - 4$. 

Now is $4$ the maximum matching? Is there an antichain of size $6-4 = 2$? The complement of a minimum vertex cover should give it. By inspection $\{4,6\}$: is $4 \le 6$? $4 \nmid 6$. Is $6 \le 4$? No. Incomparable — an antichain of size $2$. And I cannot cover with $1$ chain (it's not a total order: $4,6$ incomparable), so $2$ chains is optimal, matching the antichain $\{4,6\}$ of size $2$. The minimum chain cover ($2$) equals the largest antichain ($2$). The construction works exactly as derived.

One more thing nags at me — the *infinite* case. My whole argument is finite: $n$ elements, a finite matching, a finite search. Does "min chain cover = largest antichain" survive when $P$ is infinite? Suppose the width is *finite*, $w$ — i.e. every antichain has at most $w$ elements — even though $P$ is infinite. Then I'd want to cover $P$ with $w$ chains. A chain cover into $w$ chains is the same as a proper coloring of the *incomparability graph* (vertices = elements, edges between incomparable pairs) with $w$ colors, since each color class is then pairwise-comparable, a chain. Every *finite* subset $S$ of $P$ has width $\le w$, so by the finite theorem its incomparability graph is $w$-colorable. A graph all of whose finite subgraphs are $w$-colorable is itself $w$-colorable — that's the De Bruijn–Erdős compactness theorem. So the whole infinite $P$ has a $w$-coloring of its incomparability graph, i.e. a cover by $w$ chains. So the theorem extends *as long as the width is finite*. If the width itself is infinite the clean equality breaks — the largest antichain and the fewest chains can diverge — so the finite-width hypothesis is exactly the right one, and I'll state the finite theorem as the core.

Let me also notice the *dual* statement, because the easy direction I started from is completely symmetric in "chain" and "antichain." A chain and an antichain meet in at most one element — that's symmetric. So the mirror claim should be: the largest *chain* (the height) equals the fewest *antichains* needed to cover $P$. The easy half is identical: a chain of length $h$ needs $h$ different antichains to cover it (each antichain holds at most one of its elements), so (min antichain cover) $\ge$ (longest chain). And the reverse is even easier here — I don't need König at all. For each element $x$, let $\ell(x)$ = the length of the longest chain *ending* at $x$. If $x < y$ then any chain ending at $x$ extends by $y$, so $\ell(y) > \ell(x)$; hence $\ell$ is strictly increasing along comparable pairs, so each level set $\{x : \ell(x) = k\}$ is an antichain (two elements with the same $\ell$ can't be comparable). The number of distinct values of $\ell$ is exactly the longest chain length $h$, and the level sets are $h$ antichains covering $P$. So (min antichain cover) $\le h \le$ (min antichain cover): equality. The dual falls straight out of leveling by longest-chain-length, no matching needed — a nice asymmetry, the antichain-cover side is the *easy* one and the chain-cover side is the one that needed König.

So, to recap the causal chain. I wanted the fewest chains covering a poset. The trivial bound — a chain meets an antichain in $\le 1$ element — gave (chains) $\ge$ (largest antichain) and the *shape* min-cover $=$ max-packing, which is the fingerprint of a duality theorem. König's bipartite matching/cover duality was the constructive one with usable machinery, but it lives on bipartite graphs, so I reframed comparability as a bipartite graph by splitting every element into a lower copy and an upper copy and drawing $u_a - v_b$ for each strict relation $a < b$. A matching then encodes successor-links, threading the elements into $n - m$ disjoint chains (transitivity makes each thread a genuine chain; strictness kills self-links; no cycles because the order is strict). Maximizing the matching minimizes the chains. König's dual minimum vertex cover, complemented to the "free" elements, yields an antichain of size $\ge n - m^\*$, which the trivial bound forces to equal $n - m^\*$ exactly. To make König airtight I proved Berge's criterion (a matching is maximum iff no augmenting path) and turned the failed alternating search into a vertex cover $C = (U \setminus X) \cup (V \cap X)$ of size $|M|$. The result: minimum chain cover $=$ largest antichain, for every finite poset, constructively and in polynomial time — and it persists to infinite posets of finite width via De Bruijn–Erdős, with the dual height/antichain-cover statement falling out by leveling.

The theorem, stated cleanly:

> **Decomposition theorem.** In a finite partially ordered set $P$, the minimum number of chains into which $P$ can be partitioned equals the maximum size of an antichain in $P$. (This common number is the *width* of $P$.)

and the construction that proves it, on a concrete poset:

```python
from typing import List, Set

def chain_cover_and_antichain(n: int, leq: List[List[bool]]):
    """Finite poset on {0..n-1} given by its <= matrix `leq`.
    Returns (chains, antichain): a minimum chain cover and a witnessing antichain
    of equal size, proving width(P) = max-antichain = min-chain-cover."""

    # reframe comparability as a bipartite graph: lower copy U=0..n-1 (left),
    # upper copy V=0..n-1 (right); edge u_a - v_b iff a < b (strict, a != b).
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
                # v_b free, or its current owner can be rerouted (alternating path)
                if match_v[b] == -1 or try_augment(match_v[b], seen):
                    match_v[b] = a
                    succ[a]   = b
                    return True
        return False

    for a in range(n):
        try_augment(a, [False] * n)
    m = sum(1 for b in range(n) if match_v[b] != -1)    # size of maximum matching

    # ---- chains = connected components of the chosen successor-links ----
    pred = [-1] * n
    for b in range(n):
        if match_v[b] != -1:
            pred[b] = match_v[b]
    chains: List[List[int]] = []
    for a in range(n):
        if pred[a] == -1:                      # a is a chain head (no predecessor)
            chain, x = [], a
            while x != -1:
                chain.append(x)
                x = succ[x]
            chains.append(chain)               # transitivity => genuine chain
    # there are exactly n - m chains

    # ---- minimum vertex cover via alternating search from exposed U-vertices ----
    # X = vertices reachable by alternating paths from unmatched lower vertices.
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
    # cover C = (U \ X) ∪ (V ∩ X); |C| = m
    cover_u = {a for a in range(n) if not inX_u[a]}
    cover_v = {b for b in range(n) if inX_v[b]}

    # ---- antichain = elements with NEITHER copy in the cover (the "free" ones) ----
    antichain: Set[int] = {a for a in range(n)
                           if a not in cover_u and a not in cover_v}
    # |antichain| = n - |C| = n - m = number of chains  => optimum certified
    return chains, antichain
```
