OK, let me think this through from scratch. I have a finite family of sets $T_1, \dots, T_n$ sitting inside a finite ground set $S$, and I want to choose one element from each, all distinct — a system of distinct representatives. Sometimes I can, sometimes I can't, and I want to know *exactly when*. Not a sufficient condition, not a necessary one — the precise dividing line, and ideally one that tells me what's wrong when it can't be done.

First, get the picture right. Indices on one side, ground-set elements on the other, an edge from index $i$ to element $x$ whenever $x \in T_i$. A choice of distinct representatives is a set of edges, one per index, no two touching the same element — and since they're one per index they also don't share an index. So it's a set of pairwise-disjoint edges that covers every index. A set of pairwise-disjoint edges is a matching; "covers every index" means it saturates the index side $A$. The whole question is: *when does this bipartite graph have a matching that saturates $A$?* Good, now I'm reasoning about edges, not tuples.

Let me feel out the obstruction before I reach for any machinery. When is it *obviously* impossible? If two indices, say $T_1 = T_2 = \{a\}$, both insist on the single element $a$, I'm dead — two indices, one element to go around. Generalize: if some collection of $k$ indices has, between all of them, fewer than $k$ distinct elements available, then those $k$ indices are fighting over fewer than $k$ chairs and somebody is left standing. In neighborhood language: if there is a set $S \subseteq A$ of indices with $|N(S)| < |S|$, no SDR exists. That direction is forced and immediate — and notice it's not just one set $T_i$ being too small, it's the *union* of a subfamily being too small.

So the necessary condition writes itself: for every $S \subseteq A$,
$$|N(S)| \ge |S|.$$
Equivalently, in the original language: any $k$ of the sets $T_i$ contain between them at least $k$ elements of the ground set, for every $k$. Let me nail the necessity cleanly, because it's the cheap half and I want it airtight. Suppose an SDR exists — call the matching $M$. Restrict $M$ to the indices in $S$: each index $a \in S$ is matched to a distinct element $M(a)$, and $M(a) \in T_a \subseteq N(S)$. Distinct indices get distinct elements, so $a \mapsto M(a)$ is an injection from $S$ into $N(S)$. An injection forces $|S| \le |N(S)|$. Done. Necessity costs nothing.

Now the real content: is this condition *sufficient*? Is "$|N(S)| \ge |S|$ for all $S$" enough to guarantee an SDR? My gut says it should be — it says there's no local bottleneck anywhere, and intuitively if there's no bottleneck then there's room to assign everyone. But "intuitively there's room" is not a proof, and conditions like this are exactly where intuition lies to you, because the condition ranges over *all* $2^n$ subsets $S$ at once and the interactions between subsets are not obvious. I need to actually prove it, or find a counterexample.

Let me first see what the tools I already have can do, because maybe I don't have to build anything. König's theorem says the maximum matching equals the minimum vertex cover. An $A$-saturating matching exists iff the maximum matching has size $n$. By König that's iff the minimum vertex cover has size $n$. So I could try: assume the neighborhood condition, and show every vertex cover has size at least $n$. A vertex cover $C$ — say it uses $t$ vertices from $A$ and the rest from $B$. The $n - t$ indices of $A$ that $C$ *doesn't* cover: every edge out of them must be covered on the $B$ side, so all of their neighbors lie in $C \cap B$. Those uncovered indices form a set $S$ of size $n - t$, and $N(S) \subseteq C \cap B$, so $|C \cap B| \ge |N(S)| \ge |S| = n - t$ by the condition. Then $|C| = t + |C \cap B| \ge t + (n - t) = n$. Every cover has size $\ge n$; König upgrades that to a matching of size $n$, which saturates $A$. That works — sufficiency follows from König.

But I'm not satisfied. I've just outsourced the whole thing to König's minimax, which is itself a theorem I'd have to prove, and which is stated about *sizes* of covers — it certifies optimality with a dual object but it doesn't, on its own, hand me the bottleneck subfamily. The criterion I want is supposed to be *about the sets*, and when it fails I want it to point at the *specific* offending subfamily $S$ with $|N(S)| < |S|$. The König route proves existence but it launders away the witness. I want a proof that, when an SDR doesn't exist, actually *produces* the deficient set. Let me try to prove sufficiency directly and constructively, so the obstruction falls out of the argument itself rather than being invoked by fiat.

Here's the constructive reframing. Instead of "assume the condition, build the SDR," prove the contrapositive *with a witness*: if I cannot saturate $A$, then I will exhibit a set $S$ with $|N(S)| < |S|$. That's the same theorem — condition $\Rightarrow$ SDR is logically equivalent to no-SDR $\Rightarrow$ condition-fails — but stated this way it demands a construction, and a construction is what gives me the bottleneck for free. So: suppose I've got a matching that does *not* saturate $A$, and I want to manufacture a violating set out of it.

How do I even get my hands on a matching that's stuck? Start from any matching — empty, or whatever I can build greedily — and try to improve it. If I can always either grow it or extract a deficient set, I'm done. So the engine I need is: given a non-saturating matching, *either* enlarge it *or* prove no enlargement is possible and read off the obstruction.

Let me think about how to enlarge a matching. Greedy is no good on its own — I can paint myself into a corner where no edge can be *added* even though a *bigger* matching exists, by having matched the wrong pairs early. The classic trap: matching $1$–$a$ and $2$–$b$ looks maximal if those are the only edges I added, but maybe the truly good matching is $1$–$b$ and $2$–$a$ plus a third edge I now can't reach. So I can't just add edges; I have to be willing to *rearrange* the ones I already placed. What's the minimal rearrangement that grows a matching by one?

Suppose I have a matching $M$ and I want it bigger. Picture a path that starts at an exposed vertex, goes along a non-matching edge to a matched vertex, then back along its matching edge, then out along another non-matching edge, and so on, *alternating* matching and non-matching edges, and finally ends at another exposed vertex. An **alternating path** between two exposed endpoints. If I have one of these, look at what happens when I swap every edge's status along it — non-matching edges become matching, matching edges become non-matching. The endpoints were exposed and now get matched; every internal vertex was matched before (to one edge of the path) and is matched after (to the other path edge), so nobody loses their match. The path had one more non-matching edge than matching edge — because it starts and ends with non-matching edges (the exposed endpoints had no matching edge), so the pattern is non, match, non, match, …, non, with $k$ matching edges and $k+1$ non-matching ones. After the swap I've turned $k+1$ edges into matching edges and $k$ out, net $+1$. The matching grew by exactly one, and it's still a matching — I should double check no vertex got two matching edges. Internal vertices: each lies on exactly two path edges, one previously matching and one previously non; after the flip exactly one of those two is in the matching, and the vertex has no other matching edge (it had only its single $M$-edge, which was on the path). Endpoints: were exposed, now have one new matching edge each, and had no others. Off-path edges of $M$ untouched. So yes — flipping an alternating path between two exposed vertices yields a valid matching, one larger. Call such a path **augmenting**.

That gives me the improvement move. But I need the converse to make it a *decision* procedure: if there is *no* augmenting path, is the matching already maximum? Because if "no augmenting path" could happen while a bigger matching still exists, then getting stuck would tell me nothing. Let me prove that getting stuck means truly maximum.

Take my matching $M$ and suppose, for contradiction, that some strictly larger matching $M^\*$ exists, $|M^\*| > |M|$, yet $M$ has no augmenting path. Look at the symmetric difference $Q = M \mathbin\triangle M^\* = (M \setminus M^\*) \cup (M^\* \setminus M)$ — the edges in exactly one of the two matchings. Consider the degree of any vertex $v$ in the graph $(V, Q)$. It can have at most one edge from $M$ (matchings have max degree one) and at most one from $M^\*$, so its degree in $Q$ is at most $2$. A graph with all degrees $\le 2$ is a disjoint union of simple paths and cycles. Now, along any of these paths or cycles, consecutive edges must alternate between $M$ and $M^\*$ — two consecutive edges of $Q$ at a shared vertex can't both be from $M$ (that vertex would have two $M$-edges) nor both from $M^\*$, so they alternate. The cycles therefore have even length, with equally many $M$- and $M^\*$-edges. So the cycles contribute equal counts to $M$ and $M^\*$; all the *imbalance* $|M^\*| - |M| > 0$ must come from the paths. Hence at least one path has strictly more $M^\*$-edges than $M$-edges. An alternating path with more $M^\*$-edges than $M$-edges must begin and end with $M^\*$-edges, which means both endpoints have no $M$-edge in $Q$ — and they have no $M$-edge outside $Q$ either, because an $M$-edge at an endpoint that's also an $M^\*$-edge would be in neither $M\setminus M^\*$ nor $M^\*\setminus M$, fine, but if it were an $M$-edge not in $M^\*$ it would be in $Q$, contradicting "the path ends here." Let me state it carefully: an endpoint of such a path is matched in $M^\*$ (the path's last edge is an $M^\*$-edge incident to it) but *unmatched in $M$* — if it were $M$-matched, that $M$-edge is either in $M^\*$ too (then both edges at the endpoint... no, the endpoint has degree one in $Q$, so its $M$-edge, if not in $M^\*$, would add a second $Q$-edge, contradiction; and if its $M$-edge *is* in $M^\*$, then $M^\*$ would have two edges at this endpoint, the path edge and this one, contradiction). Either way the endpoint is exposed in $M$. So this path is an alternating path (alternating $M^\*$/$M$, equivalently $M$/non-$M$) with both endpoints exposed in $M$ — an augmenting path for $M$. That contradicts "no augmenting path." Therefore no larger $M^\*$ exists: **a matching is maximum iff it has no augmenting path.** That's the hinge I needed. Getting stuck is meaningful.

Now I can run the program. Build a matching, look for an augmenting path; if found, flip and repeat; the size strictly increases each time and is bounded by $n$, so this halts after at most $n$ augmentations at a maximum matching. And now the question that the whole approach was built to answer: when the search for an augmenting path fails at a non-saturating matching, why should a deficient set appear?

So let me actually run the search and watch what it builds. I have a matching $M$ that doesn't saturate $A$; pick an exposed index $r \in A$. I want an augmenting path from $r$. An augmenting path from $r$ leaves $r$ on a non-matching edge (r is exposed, it has no matching edge), reaches some element $y \in B$; if $y$ is exposed I'm done, that's a length-one augmenting path; otherwise $y$ is matched, so follow its matching edge back to an index $x \in A$; from $x$ leave again on a non-matching edge to a new element, and so on. So I should explore from $r$ by alternating: from an $A$-vertex, take *non-matching* edges to $B$; from a $B$-vertex, take its *matching* edge back to $A$. Let me grow the set of everything reachable from $r$ by such alternating walks. Call it the alternating tree, or just the reachable set $Z$. Split it: $Z_A = Z \cap A$, $Z_B = Z \cap B$. The root $r \in Z_A$.

Let me trace the structure of $Z$ when the search *dies* — i.e. when I never hit an exposed element in $B$, so there's no augmenting path. Watch the two sides.

Every element of $Z_B$ got there from some index in $Z_A$ via a *non-matching* edge — that's how I step from $A$ into $B$. So $Z_B \subseteq N(Z_A)$: every reachable element is a neighbor of a reachable index. Could there be a neighbor of $Z_A$ that's *not* in $Z_B$? Take any index $a \in Z_A$ and any neighbor $b \in N(\{a\}) = T_a$. The edge $a$–$b$ is either a matching edge or not. If it's a non-matching edge, then from $a$ (which is reachable) I can step to $b$ along it, so $b$ is reachable: $b \in Z_B$. If it's a matching edge, then $a$ was reached, and $a \neq r$ (the root is exposed, so it has no matching edge), so $a$ was reached *into* — the last step landing on $a$ came from $B$ along $a$'s matching edge, which is exactly the edge $a$–$b$; so $b$ was on the path just before $a$, hence $b \in Z_B$. Either way $b \in Z_B$. So **every neighbor of $Z_A$ lies in $Z_B$**: $N(Z_A) \subseteq Z_B$. Combined with $Z_B \subseteq N(Z_A)$, I get the clean equality
$$N(Z_A) = Z_B.$$
That's a strong fact — the reachable indices' neighborhood is *exactly* the reachable elements, nothing leaks out. Now I just need to count $Z_A$ against $Z_B$.

Here's where "the search died" pays off. The search dying means *no exposed element in $B$ was reached* — every element of $Z_B$ is matched (if some $b \in Z_B$ were exposed, the alternating walk from $r$ to $b$ would be an augmenting path, contradicting that I'm stuck). So each $b \in Z_B$ has a matching partner. Where does that partner live? $b$ was reached from some index along a non-matching edge, and then the search continues from $b$ by following $b$'s *matching* edge — so $b$'s matched partner is itself reachable, in $Z_A$. Hence the matching maps $Z_B$ injectively into $Z_A$ (it's a matching, so injective), and every $b \in Z_B$ has its partner in $Z_A$. So $|Z_B| \le |\{\text{matched indices in } Z_A\}| \le |Z_A|$. I can do better and get the exact gap. Which indices of $Z_A$ are matched, and which are exposed? The root $r$ is exposed. Any *other* index $a \in Z_A$, $a \ne r$ — it was reached by stepping from some $b' \in Z_B$ along $a$'s matching edge, so $a$ is matched (to $b'$). So inside $Z_A$, exactly one vertex is exposed, the root $r$, and all the others are matched, each to a *distinct* element of $Z_B$ (the matching is injective and, as just argued, those partners lie in $Z_B$). So the matching restricted to $Z_A \setminus \{r\}$ is a bijection onto $Z_B$: every reachable element is matched, its partner is a non-root reachable index, and distinct elements have distinct partners. Both inclusions are equalities of a bijection. Therefore
$$|Z_B| = |Z_A| - 1.$$

Put the two facts together. $Z_A$ is a subset of $A$ with
$$|N(Z_A)| = |Z_B| = |Z_A| - 1 < |Z_A|.$$
There it is — the deficient set, with deficiency exactly one, handed to me by the dead search. If $M$ is a maximum matching that fails to saturate $A$, then growing the alternating tree from any exposed index $r$ produces a concrete subfamily $Z_A$ whose neighborhood is strictly smaller than itself. So *no* matching can saturate $A$, and the neighborhood condition is violated on $Z_A$. Contrapositive complete: if the condition holds, a saturating matching exists; and when it fails, this very construction names the offending subfamily.

Let me make sure I didn't fool myself on the counting, because the whole theorem balances on $|N(Z_A)| = |Z_A| - 1$ and a sign slip there would wreck it. $Z_A$ contains the root $r$ (exposed) and some matched indices. The bijection is between $Z_A \setminus \{r\}$ and $Z_B$: forward map is "take your matching partner," and I argued (a) every non-root reachable index is matched into $Z_B$, (b) every reachable element is matched into $Z_A$ and being matched its partner is a non-root index (the root has no matching edge), (c) the matching is injective both ways. So it's a genuine bijection, $|Z_A| - 1 = |Z_B|$, and $N(Z_A) = Z_B$, so $|N(Z_A)| = |Z_A| - 1$. The strict inequality is real. Good.

And let me sanity-check on the dead-simple instance so I trust the machine: $T_1 = \{a\}$, $T_2 = \{a\}$. Start matching empty, augment once, say $1$–$a$. Now $2$ is exposed; grow the tree from $r = 2$. From $2$ take a non-matching edge — its only neighbor is $a$, so $Z_B = \{a\}$. Is $a$ exposed? No, it's matched to $1$. Follow $a$'s matching edge back: lands on $1$, so $1 \in Z_A$. From $1$, neighbors are just $a$, already in $Z_B$, nothing new. Search dies. $Z_A = \{1, 2\}$, $Z_B = \{a\}$, $N(Z_A) = \{a\}$, and indeed $|N(Z_A)| = 1 = 2 - 1 = |Z_A| - 1 < |Z_A|$. The construction fingers exactly the obstruction $\{1, 2\}$, the two indices fighting over the single element $a$. Exactly right.

I should pause on whether there was an easier proof and why I'm choosing this one. There is a slicker-looking induction on $n$: throw away the last set $T_n$, get an SDR for the first $n-1$ by induction, and try to extend it — if $T_n$ has an element not already used, great; if not, every element of $T_n$ is spoken for, and you argue some "critical" subfamily (one that's already tight, $|N(S)| = |S|$) forces a recursion onto $S$ and onto its complement separately, both of which inherit the condition. It works, but it's a case analysis that *invokes* a critical tight set rather than *constructing* it, and the recursion onto two pieces is fiddly to get right. More to the point, it proves existence without putting the augmenting machinery in my hands, and it's the augmenting machinery — the ability to actually search for an improvement and, on failure, read off the bottleneck — that I wanted from the start. The alternating-tree proof gives the theorem *and* the algorithm *and* the witness in one motion. So I'll land on it.

One more thing to record, because it's the same coin's other face and it's worth seeing fall out. When the search dies on a maximum matching $M$, look at the set
$$C = (A \setminus Z_A) \cup Z_B.$$
Is this a vertex cover? An edge is uncovered only if both endpoints are outside $C$, i.e. its $A$-endpoint is in $Z_A$ and its $B$-endpoint is outside $Z_B$. But I proved $N(Z_A) = Z_B$ — every neighbor of a $Z_A$-vertex is in $Z_B$ — so no such edge exists. $C$ is a cover. And its size: $|C| = (|A| - |Z_A|) + |Z_B| = |A| - |Z_A| + (|Z_A| - 1) = |A| - 1$. More generally, run the tree from *all* exposed indices at once and the same bookkeeping gives a cover of size exactly $|M|$, while every matching needs a cover at least its own size — so this cover is minimum and $\max|M| = \min|C|$. That's König's minimax, reproved constructively, with the matching and the cover coming out of the *same* search. The neighborhood criterion and the minimax are two readings of one structure: the alternating tree at a stuck matching. I came in wanting the criterion phrased on the sets, and not only did I get it, the dual cover and König's theorem are byproducts.

So let me write the result down cleanly. The condition is: *for every subfamily of indices $S$, the union of their sets contains at least $|S|$ elements*, $|N(S)| \ge |S|$ for all $S \subseteq A$. This is necessary by the trivial injection. It is sufficient because, were it impossible to saturate $A$, a maximum matching would leave some index exposed, and the alternating tree from that index would exhibit a subfamily $Z_A$ with $|N(Z_A)| = |Z_A| - 1$, violating the condition. The proof is constructive: it either augments toward a full assignment or returns the deficient subfamily. Here is the whole thing assembled.

```python
from collections import deque

def admits_SDR(A, B, T):
    """
    A : list of indices (the family is indexed by A).
    B : the finite ground set.
    T : dict, T[a] = the set of ground-set elements joined to index a (= the set T_a).

    Returns (True, M) where M maps each index to its distinct representative,
    if a system of distinct representatives exists; otherwise (False, S) where
    S is a deficient subfamily with |N(S)| < |S| -- the witness of impossibility.

    Necessity: a saturating matching injects each index a into T_a, so any S
    satisfies |N(S)| >= |S|. Sufficiency is proved constructively below: we
    repeatedly enlarge the matching via augmenting paths; if at some exposed
    index the alternating search dies, its reachable A-side is exactly the
    deficient set.
    """
    match_A = {}          # index -> element  (current matching, A side)
    match_B = {}          # element -> index  (current matching, B side)

    def neighborhood(S):
        out = set()
        for a in S:
            out |= T[a]
        return out

    for root in A:
        if root in match_A:
            continue
        # Grow the alternating tree from the exposed index `root`.
        # From an A-vertex we step out along NON-matching edges to B;
        # from a B-vertex we step back along its MATCHING edge to A.
        parent_B = {}                  # element -> index it was reached from
        Z_A = {root}                   # reachable indices
        Z_B = set()                    # reachable elements
        frontier = deque([root])
        augmented = False

        while frontier and not augmented:
            a = frontier.popleft()
            for b in T[a]:             # non-matching step A -> B
                if b in Z_B:
                    continue
                Z_B.add(b)
                parent_B[b] = a
                if b not in match_B:
                    # b is an exposed element: we found an augmenting path.
                    # Flip it: walk back parent_B / match_A, swapping edges.
                    while True:
                        a2 = parent_B[b]
                        prev = match_A.get(a2)     # a2's old partner (if any)
                        match_A[a2] = b            # match a2 -- b
                        match_B[b] = a2
                        if prev is None:           # reached the exposed root
                            break
                        b = prev                   # continue back along the path
                    augmented = True
                    break
                else:
                    # b is matched: follow its matching edge back into A.
                    a2 = match_B[b]
                    if a2 not in Z_A:
                        Z_A.add(a2)
                        frontier.append(a2)

        if not augmented:
            # The search died with no augmenting path. Then N(Z_A) = Z_B and
            # |Z_B| = |Z_A| - 1, so Z_A is a deficient set: |N(Z_A)| < |Z_A|.
            assert neighborhood(Z_A) == Z_B
            assert len(Z_B) == len(Z_A) - 1
            return (False, frozenset(Z_A))

    # Every index got matched: the matching is an SDR.
    return (True, dict(match_A))
```
