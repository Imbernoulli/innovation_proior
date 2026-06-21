# Dilworth's Decomposition Theorem (chain cover = largest antichain)

## Problem

In a finite partially ordered set $P$ (relation $\le$ reflexive, antisymmetric,
transitive; not every pair comparable), a **chain** is a pairwise-comparable subset
and an **antichain** is a pairwise-incomparable subset. Partition $P$ into disjoint
chains using as few as possible. What is the minimum number of chains, and what in
the order controls it?

## Theorem

> **Dilworth's theorem.** In any finite poset $P$, the minimum number of chains in a
> chain decomposition (partition of $P$ into disjoint chains) equals the maximum size
> of an antichain in $P$. This common value is the **width** of $P$.

**Easy direction (weak duality).** A chain and an antichain share at most one element
(two common elements would be both comparable and incomparable). So to cover an
antichain of size $w$ one needs $\ge w$ distinct chains: every chain cover has at
least (largest antichain) chains. The theorem's content is that this is tight.

## Key idea — reframe comparability as a bipartite graph

The claim has the form *min cover $=$ max packing*, the signature of a duality
theorem; the constructive finite duality with the right machinery is **König's
theorem** (König 1931): in a bipartite graph, max matching $=$ min vertex cover. To
invoke it, build a bipartite graph from the order: split each element $a$ into a
**lower copy** $u_a$ (left side $U$) and an **upper copy** $v_a$ (right side $V$), and
draw

$$u_a \,-\, v_b \quad\Longleftrightarrow\quad a < b \ \ (\text{strict: } a \le b,\ a \ne b).$$

A matching $M$ is then a set of **successor-links**: each $u_a$ used once $=$ each
element has $\le 1$ chosen successor; each $v_b$ used once $=$ each element has $\le 1$
chosen predecessor. Following the links threads the elements into vertex-disjoint
paths; transitivity makes each path a genuine chain, strictness forbids self-links,
and a strict order admits no cycle $a_1 < \dots < a_t < a_1$. Each link merges two
distinct chain-fragments into one, so a matching of size $m$ turns the $n$ singleton
chains into

$$n - m \ \text{chains}.$$

Minimizing chains $=$ maximizing the matching.

## Proof

Let $m^\*$ be the maximum matching size and, by König, $C$ a minimum vertex cover with
$|C| = m^\*$. The chosen successor-links of a maximum matching give a chain cover of
size $n - m^\*$.

Call $a \in P$ **free** if neither $u_a$ nor $v_a$ lies in $C$; let $A$ be the free
elements. $A$ is an antichain: if $a, b \in A$ with $a < b$, the edge $u_a - v_b$
exists and has *both* endpoints outside $C$, contradicting that $C$ covers every edge.
Each non-free element accounts for $\ge 1$ distinct vertex of $C$, so the number of
non-free elements is $\le |C| = m^\*$, giving

$$|A| \ \ge\ n - m^\*.$$

But the easy direction says every antichain is $\le$ the number of chains in any
cover, so $|A| \le n - m^\*$. Hence $|A| = n - m^\*$ exactly, equal to the chain
count. Therefore

$$\text{minimum chain cover} \ =\ n - m^\* \ =\ |A| \ =\ \text{largest antichain.} \qquad\blacksquare$$

Both objects — the chains (from $M$) and the witnessing antichain (the free elements,
from $C$) — are produced simultaneously, in polynomial time.

**König's theorem (used above), constructively.** *Weak duality:* a cover spends a
distinct vertex on each edge of a matching, so $|C| \ge |M|$ always. *Tight
direction:* by Berge's criterion a matching $M$ is maximum iff it has no augmenting
path (an alternating path between two unmatched vertices) — flipping such a path
enlarges $M$ by one, and via the symmetric difference $M \triangle M^\*$ a larger
matching always yields one. Given a maximum $M$, let $X$ be all vertices reachable
from unmatched $U$-vertices by alternating paths (non-matching edges $U\to V$,
matching edges $V\to U$). Then $C := (U \setminus X) \cup (V \cap X)$ is a vertex cover
with $|C| = |M|$: every vertex in it is matched and accounts for a distinct match
edge, and any edge $u-v$ with $u \in X$ forces $v \in X$, so it is covered. Hence
max matching $=$ min vertex cover.

## Dual (Mirsky's theorem)

By the symmetric easy direction, the mirror holds and needs no matching. Define
$\ell(x)$ = length of the longest chain ending at $x$; then $a < b \Rightarrow
\ell(a) < \ell(b)$, so each level set $\{x : \ell(x) = k\}$ is an antichain, giving
(longest chain length) antichains covering $P$, which the easy bound forces optimal:

> **Mirsky's theorem.** The maximum size of a chain (the *height*) equals the minimum
> number of antichains needed to cover $P$.

## Infinite extension

If $P$ is infinite but has **finite width** $w$ (every antichain has $\le w$
elements), it still decomposes into $w$ chains: a $w$-chain cover is a $w$-coloring of
the incomparability graph, every finite subset is $w$-colorable by the finite
theorem, and De Bruijn–Erdős compactness lifts this to all of $P$. The equality can
fail when the width itself is infinite.

## Construction (certifies the width on a concrete poset)

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

On the divisibility order of $\{1,2,3,4,6,12\}$ this returns chains
$\{1,2,4,12\}$ and $\{3,6\}$ and the antichain $\{4,6\}$ — two chains, an antichain of
two, so the width is $2$.
