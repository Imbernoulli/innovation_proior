I was handed a finite family of finite sets $T_1, \dots, T_n$ drawn from a common ground set $S$, and asked the blunt question: when can I pick one element from each set so that all the chosen elements are distinct — a tuple $(a_1, \dots, a_n)$ with $a_i \in T_i$ and $a_i \ne a_j$ for $i \ne j$, a system of distinct representatives (SDR)? The same object wears many costumes — people choosing acceptable partners, tasks choosing machines, a finite-index subgroup choosing a common transversal of its left and right cosets — so a single sharp criterion settles all of them. What I want is not merely a sufficient condition nor merely a necessary one but the *exact* dividing line, and ideally one that, when no SDR exists, points a finger at *why*.

The right picture is the bipartite graph: indices $\{1,\dots,n\}$ on one side (call it $A$), ground-set elements on the other ($B$), and an edge $i$–$x$ exactly when $x \in T_i$. An SDR is then a set of pairwise-disjoint edges, one per index — a matching that *saturates* $A$. The prevailing tools in this arena are all minimax theorems: König (1931) says the maximum matching equals the minimum vertex cover in a bipartite graph; Egerváry (1931) says the same in 0/1-matrix language; Menger (1927) gives the more general max-disjoint-paths = min-cut; Frobenius (1917) gives the algebraic determinant precursor. Each is powerful, but each falls short of what I am after in the same way: they certify optimality with a dual object of equal *size*, yet none is phrased directly on the family of sets, and none, on failure, hands me the specific offending subfamily. König tells me an SDR exists iff the minimum cover has size $n$, but "the minimum cover has size $n$" is not yet a transparent, checkable condition on the neighborhoods $N(S)$. The naive alternative — greedy or exhaustive assignment — is correct but opaque and exponential, and when it fails it says only "no assignment found," with no reason.

So I propose the **marriage condition**, also called **Hall's theorem**. Writing $N(S) = \{\,x \in B : x \in T_i \text{ for some } i \in S\,\}$ for the union of a subfamily, the theorem is: a finite family $(T_1,\dots,T_n)$ has an SDR **if and only if** $|N(S)| \ge |S|$ for every $S \subseteq \{1,\dots,n\}$ — that is, any $k$ of the sets contain between them at least $k$ elements, for every $k$. The condition is phrased entirely on the sets and their unions, exactly what the minimax theorems left implicit, and the obstruction it names is a *deficient subfamily*: a set of indices whose sets jointly span too few elements.

Necessity is the cheap half and costs nothing. If a saturating matching $M$ exists, then for any $S \subseteq A$ each index $a \in S$ is matched to a distinct element $M(a) \in T_a \subseteq N(S)$, so $a \mapsto M(a)$ injects $S$ into $N(S)$ and forces $|S| \le |N(S)|$. The deficiency, when it appears, is never about a single set $T_i$ being small — it is the *union* of a subfamily being too small.

The real content is that this condition is sufficient, and I want to prove it *constructively*, so that when no SDR exists the deficient subfamily falls out of the argument rather than being conjured. One could outsource sufficiency to König (assume the condition, show every cover has size $\ge n$, invoke the minimax to get a matching of size $n$), but that launders away the witness — it proves existence without producing the bottleneck. Instead I prove the contrapositive with a witness, and the engine is augmenting paths. With respect to a matching $M$, an alternating path alternates non-$M$ and $M$ edges; an augmenting path is one whose two endpoints are both exposed. The first lemma is the improvement move: if $P$ is augmenting then $M \triangle P$ is a matching of size $|M|+1$, because $P$ starts and ends on non-$M$ edges so it carries one more non-$M$ edge than $M$-edge, and flipping membership along it leaves every internal vertex with exactly one $M$-edge and gives each exposed endpoint exactly one. The second lemma (Berge) is that $M$ is maximum iff it has *no* augmenting path: if a larger $M^\*$ existed, the symmetric difference $M \triangle M^\*$ has max degree two, hence splits into paths and even cycles alternating between the two matchings, and the surplus $|M^\*| - |M| > 0$ must sit on some path with more $M^\*$-edges than $M$-edges — a path augmenting for $M$, a contradiction. This makes "getting stuck" meaningful: a stuck matching is genuinely maximum.

Now the load-bearing construction. Suppose the marriage condition holds but, for contradiction, no matching saturates $A$. Take a maximum matching $M$; some index $r \in A$ is exposed. Grow the **alternating tree** from $r$: from an $A$-vertex step out along non-$M$ edges to $B$, and from a $B$-vertex step back along its $M$-edge to $A$. Let $Z_A, Z_B$ be the reachable indices and elements. Three facts pin down the structure when this search *dies*. First, no element of $Z_B$ is exposed — if some $b \in Z_B$ were exposed, the walk $r \rightsquigarrow b$ would be augmenting, contradicting maximality. Second, $N(Z_A) = Z_B$ exactly: every reachable element is a neighbor of a reachable index, and conversely every neighbor $b$ of some $a \in Z_A$ lands in $Z_B$ — if the edge $a$–$b$ is non-matching, $a$ steps to $b$ directly; if it is matching, then $a \ne r$ (the exposed root has no $M$-edge), so $a$ was reached *into* along that very edge and $b$ preceded it. Nothing leaks out of the neighborhood. Third, the matching is a bijection $Z_A \setminus \{r\} \to Z_B$: every $b \in Z_B$ is matched and its partner lies in $Z_A$ (the search continues from $b$ along its $M$-edge), the partner is not the exposed $r$, and conversely every non-root $a \in Z_A$ was reached along its $M$-edge from some $b \in Z_B$; injectivity is just $M$ being a matching. Hence $|Z_B| = |Z_A| - 1$. Combining the last two,
$$|N(Z_A)| = |Z_B| = |Z_A| - 1 < |Z_A|,$$
so $Z_A$ is a deficient subfamily — a contradiction. Therefore a saturating matching exists, and when one does not, this very tree names the offending set with deficiency exactly one.

I checked the bookkeeping on the smallest obstructed instance to be sure the sign is right: $T_1 = \{a\}$, $T_2 = \{a\}$. Match $1$–$a$, leaving $2$ exposed; the tree from $r=2$ reaches $a$ (matched to $1$), follows back to $1$, finds nothing new, and dies with $Z_A = \{1,2\}$, $Z_B = \{a\}$, so $|N(Z_A)| = 1 = 2 - 1 < 2$ — it fingers exactly the two indices fighting over the single element. The same structure also reproves the minimax for free: running the tree from all exposed indices at once, $C = (A \setminus Z_A) \cup Z_B$ is a vertex cover (no edge can have its $A$-end in $Z_A$ and its $B$-end outside $Z_B$, since $N(Z_A) = Z_B$) of size $|A| - |Z_A| + |Z_B| = |M|$, which is therefore minimum — König's $\max|M| = \min|C|$, with matching and cover emerging from one search. The criterion and the minimax are two readings of the same alternating-tree structure. Finiteness is genuinely essential: for an infinite family one can satisfy the condition on every finite subfamily yet have no global SDR — take the positive integers as ground set, $T_0 = S$, and $T_i = \{i\}$ for $i \ge 1$; each $T_i$ forces $a_i = i$, exhausting the integers and leaving nothing for $a_0$.

The whole thing assembles into a single decision procedure that either returns a saturating matching (the SDR) or returns the deficient subfamily as a certificate of impossibility. Starting from the empty matching, it repeatedly grows the alternating tree from an exposed index; finding an exposed element flips the augmenting path and enlarges the matching, and a dead search hands back $Z_A$. Each augmentation raises $|M|$ by one, so it halts in at most $n$ steps.

```python
from collections import deque

def admits_SDR(A, B, T):
    """
    A : indices of the family.   B : the finite ground set.
    T : dict, T[a] = set of elements joined to index a (= T_a).

    Returns (True, M)  with M : index -> distinct representative, if an SDR exists;
    else    (False, S) with S a deficient subfamily, |N(S)| < |S| (the witness).

    Constructive proof of Hall's theorem: enlarge the matching by augmenting
    paths; when an alternating search from an exposed index dies, its reachable
    A-side Z_A satisfies N(Z_A) = Z_B and |Z_B| = |Z_A| - 1, i.e. is deficient.
    """
    match_A, match_B = {}, {}          # index->element, element->index

    def neighborhood(S):
        out = set()
        for a in S:
            out |= T[a]
        return out

    for root in A:
        if root in match_A:
            continue
        parent_B = {}                  # element -> index it was reached from
        Z_A, Z_B = {root}, set()
        frontier = deque([root])
        augmented = False

        while frontier and not augmented:
            a = frontier.popleft()
            for b in T[a]:             # non-matching step A -> B
                if b in Z_B:
                    continue
                Z_B.add(b); parent_B[b] = a
                if b not in match_B:   # exposed element: augmenting path found
                    while True:        # flip back along the path
                        a2 = parent_B[b]
                        prev = match_A.get(a2)
                        match_A[a2] = b; match_B[b] = a2
                        if prev is None:
                            break
                        b = prev
                    augmented = True
                    break
                else:                  # matched: follow matching edge back to A
                    a2 = match_B[b]
                    if a2 not in Z_A:
                        Z_A.add(a2); frontier.append(a2)

        if not augmented:              # search died: Z_A is the deficient set
            assert neighborhood(Z_A) == Z_B and len(Z_B) == len(Z_A) - 1
            return (False, frozenset(Z_A))

    return (True, dict(match_A))
```
