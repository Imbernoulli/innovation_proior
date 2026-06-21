# Hall's marriage theorem (the marriage condition)

## Problem

Given a finite family of finite sets $T_1, \dots, T_n$ over a ground set $S$, decide when a **system of distinct representatives** (SDR) exists: distinct elements $a_1, \dots, a_n$ with $a_i \in T_i$ for all $i$. Equivalently, in the bipartite graph $G$ with index side $A = \{1, \dots, n\}$, element side $B = S$, and an edge $i$–$x$ iff $x \in T_i$: decide when $G$ has a matching that **saturates** $A$ (covers every index).

## Key idea

The exact obstruction is a *deficient subfamily*: a set of indices whose sets, taken together, span too few elements. The marriage condition rules these out, and a constructive proof shows it is the only obstruction — when a maximum matching fails to saturate $A$, the alternating tree grown from any exposed index produces a deficient subfamily on the spot.

Write $N(S) = \{\, x \in B : x \in T_i \text{ for some } i \in S \,\}$ for the neighborhood (union) of a subfamily $S \subseteq A$.

## Theorem

**(Marriage / SDR condition.)** A finite family $(T_1, \dots, T_n)$ has a system of distinct representatives **if and only if**
$$|N(S)| \ge |S| \quad\text{for every } S \subseteq \{1, \dots, n\},$$
i.e. any $k$ of the sets contain between them at least $k$ elements, for every $k$. Equivalently: a finite bipartite graph $G = (A \cup B, E)$ has a matching saturating $A$ iff $|N(S)| \ge |S|$ for every $S \subseteq A$.

## Proof

**Necessity.** Suppose a saturating matching $M$ exists. Fix $S \subseteq A$. Each $a \in S$ is matched to a distinct element $M(a)$, and $M(a) \in T_a \subseteq N(S)$. Thus $a \mapsto M(a)$ is an injection $S \hookrightarrow N(S)$, so $|S| \le |N(S)|$.

**Sufficiency (constructive, via augmenting paths).** We use two lemmas.

*Definitions.* With respect to a matching $M$, an **alternating path** alternates between non-$M$ edges and $M$-edges; an **augmenting path** is an alternating path whose two endpoints are both exposed (unmatched). For edge sets $X, Y$, write $X \triangle Y = (X \setminus Y) \cup (Y \setminus X)$.

**Lemma 1 (augmenting move).** If $P$ is an augmenting path for $M$, then $M' = M \triangle P$ is a matching with $|M'| = |M| + 1$.

*Proof.* $P$ begins and ends with non-$M$ edges (its exposed endpoints have no $M$-edge), so $P$ has one more non-$M$ edge than $M$-edge; flipping membership along $P$ adds one to $|M|$. It remains a matching: each internal vertex of $P$ lies on exactly two path edges, one in $M$ and one not, so after the flip exactly one is in $M'$; each endpoint gains exactly one $M'$-edge and had none; vertices off $P$ are untouched. $\square$

**Lemma 2 (Berge: maximality criterion).** $M$ is a maximum matching iff there is no augmenting path for $M$.

*Proof.* ($\Rightarrow$) An augmenting path would, by Lemma 1, give a larger matching, contradicting maximality. ($\Leftarrow$) Suppose a larger matching $M^\*$ exists, $|M^\*| > |M|$. In $Q = M \triangle M^\*$ every vertex has at most one $M$-edge and one $M^\*$-edge, so degree $\le 2$; hence $Q$ is a disjoint union of simple paths and cycles, and along each, edges alternate between $M$ and $M^\*$. Cycles have even length with equal numbers of $M$- and $M^\*$-edges, so the surplus $|M^\* \setminus M| - |M \setminus M^\*| > 0$ lies on the paths: some path has more $M^\*$-edges than $M$-edges. Such a path starts and ends with $M^\*$-edges; each endpoint is therefore $M^\*$-matched but $M$-exposed (a degree-1 endpoint of $Q$ has no $M$-edge in $Q$, and any $M$-edge at it lying in $M^\*$ would give that endpoint two $M^\*$-edges). This path is augmenting for $M$ — a contradiction. $\square$

**Main argument.** Suppose the marriage condition holds but, for contradiction, no matching saturates $A$. Let $M$ be a maximum matching; by assumption some index $r \in A$ is exposed. Grow the **alternating tree** from $r$: let $Z$ be all vertices reachable from $r$ by alternating walks (from an $A$-vertex step along non-$M$ edges to $B$; from a $B$-vertex step along its $M$-edge to $A$). Put $Z_A = Z \cap A$, $Z_B = Z \cap B$.

1. *No element of $Z_B$ is exposed.* If some $b \in Z_B$ were exposed, the alternating walk $r \rightsquigarrow b$ would be an augmenting path, contradicting maximality (Lemma 2). So every $b \in Z_B$ is matched.

2. *$N(Z_A) = Z_B$.* ($\subseteq$) Take $a \in Z_A$ and $b \in T_a$. If $a$–$b \notin M$, then $b$ is reachable from $a$ by a non-$M$ step, so $b \in Z_B$. If $a$–$b \in M$, then $a \ne r$ (the exposed root has no $M$-edge), so $a$ was reached *into* along its $M$-edge, which is $a$–$b$; hence $b$ preceded $a$ on the walk and $b \in Z_B$. ($\supseteq$) Every $b \in Z_B$ is reached from some index in $Z_A$, so $b \in N(Z_A)$.

3. *The matching is a bijection $Z_A \setminus \{r\} \to Z_B$.* Every $b \in Z_B$ is matched (step 1) and the search continues from $b$ along its $M$-edge, so its partner lies in $Z_A$; the partner is not $r$ ($r$ is exposed). Conversely every $a \in Z_A$ with $a \ne r$ was reached along its $M$-edge from some $b \in Z_B$, so $a$ is matched into $Z_B$; and $r$ is exposed. Injectivity is from $M$ being a matching. Hence $|Z_B| = |Z_A| - 1$.

Combining (2) and (3):
$$|N(Z_A)| = |Z_B| = |Z_A| - 1 < |Z_A|,$$
so $Z_A$ violates the marriage condition — contradiction. Therefore a saturating matching exists. $\blacksquare$

The proof is **constructive**: starting from any matching, repeatedly find an augmenting path (alternating search) and apply Lemma 1; each augmentation raises $|M|$ by one, so after at most $n$ steps either $A$ is saturated (an SDR) or some alternating search dies, returning the deficient set $Z_A$ as a certificate of impossibility.

## Corollaries

**König–Egerváry minimax (1931).** At a maximum matching $M$, run the alternating tree from all exposed indices simultaneously; with $Z_A, Z_B$ as above, $C = (A \setminus Z_A) \cup Z_B$ is a vertex cover (no edge has $A$-endpoint in $Z_A$ and $B$-endpoint outside $Z_B$, since $N(Z_A) = Z_B$), and $|C| = |A| - |Z_A| + |Z_B| = |M|$. Since every matching needs a cover at least its size, this cover is minimum: **max matching $=$ min vertex cover** in bipartite graphs.

**Defect form.** If for every $S$ we have $|N(S)| \ge |S| - d$, then there is a matching of size $\ge n - d$. (Add $d$ dummy elements common to all sets, apply the theorem, discard representatives that are dummies.) Equivalently, the maximum matching size is $|A| - \max_{S \subseteq A}\bigl(|S| - |N(S)|\bigr)$, the deficiency version.

**Regular-degree special case.** If every $T_i$ has size $r$ and every element lies in exactly $r$ of the sets, an SDR exists: counting incidences, any $k$ sets carry $rk$ incidences spread over points each used $\le r$ times, so they touch $\ge k$ points; the condition holds. (This is what extends a partial Latin rectangle to a full Latin square: each unfilled column's set of available letters is such a regular family.)

## Algorithm

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

**Worked example (a deficient family).** $T_1 = \{a\}$, $T_2 = \{a\}$, $T_3 = \{a, b, c\}$. The subfamily $S = \{1, 2\}$ has $N(S) = \{a\}$, so $|N(S)| = 1 < 2 = |S|$: the condition fails and no SDR exists ($T_1, T_2$ both need $a$). Trace a run that processes index $3$ first: it matches $3$–$a$. Now index $1$ is exposed; grow the tree from $r = 1$: it reaches $a$ (matched to $3$), follows back to $3$, finds $3$'s other neighbors $b, c$ exposed — so the augmenting path $1$–$a$–$3$–$b$ flips $3$–$a$ to $1$–$a$, $3$–$b$. Now index $2$ is exposed, its only neighbor $a$ is matched to $1$, so the tree from $r = 2$ is $Z_A = \{2, 1\}$, $Z_B = \{a\}$, the search dies, and it returns the deficient set $\{1, 2\}$ with $|N(\{1,2\})| = 1 < 2$. (Processing the indices in a different order matches different edges first but lands on the same deficient set $\{1,2\}$.)

**Note on the finite hypothesis.** Finiteness is essential. For an infinite family the condition can hold while no SDR exists — e.g. take $S = \{1, 2, 3, \dots\}$ the positive integers, $T_0 = S$, and $T_i = \{i\}$ for $i \ge 1$: any $k$ of the sets contain between them either $k$ or infinitely many elements (depending on whether $T_0$ is among them), so every finite subfamily satisfies $|N(\cdot)| \ge |\cdot|$. Yet each $T_i$ ($i\ge1$) forces $a_i = i$, so $\{a_i : i \ge 1\}$ exhausts all the positive integers, leaving no legal value in $T_0 = S$ for $a_0$.
