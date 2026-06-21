Fix a ground set $[n] = \{1,\dots,n\}$ and call a family $F$ of $k$-element subsets *intersecting* when every two of its members share a point, $A \cap B \neq \emptyset$ for all $A, B \in F$. I want the largest such $F$. There is one obvious large family: fix a point $x$ and take every $k$-set through it, $S_x = \{A : x \in A\}$; any two of these meet at $x$, and there are $\binom{n-1}{k-1}$ of them. Every small case I check by hand stops at exactly this number, so the live conjecture is that for $n \ge 2k$ no intersecting $k$-family beats the star, while below $2k$ the problem collapses — two $k$-sets in $[n]$ with $n < 2k$ are forced to meet, so *all* $\binom{n}{k}$ sets are intersecting and there is nothing to prove. The regime of interest is therefore $n \ge 2k$.

The proof I had on the table for the bound is an induction, and I find it unsatisfying. One fixes the parameters, picks the intersecting family minimizing a potential like the sum of all elements over all sets, argues it sits in a compressed canonical position, invokes a Sperner-type covering lemma — count, two ways, the incidences between small sets and the slightly larger sets covering them, get $n_0(m-\ell_0) \le n_1(\ell_0+1)$ — and trades the family down to a smaller ground set or subset size, splitting into sub-cases at every step until base cases finish it. It is correct, but at the end I have *verified* $\binom{n-1}{k-1}$ without *understanding* it. Two things bother me. Why is the answer exactly $\binom{n-1}{k-1}$ and not some other binomial? And why is the hypothesis exactly $n \ge 2k$ — where does the $2$ come from? The induction produces both as outputs of bookkeeping rather than explaining their meaning. I want a proof where the constant and the threshold are forced and visible.

The number itself is the clue. We have $\binom{n-1}{k-1} = \tfrac{k}{n}\binom{n}{k}$, so the bound is exactly a $k/n$ fraction of *all* the $k$-sets. A fraction smells like averaging: if I could show that on average an intersecting family takes up at most a $k/n$ share of some structured slice of the $k$-sets, and the slices tile the whole space symmetrically, I would be done — and the $k/n$ would *be* the answer instead of being reverse-engineered. Sperner's world tells me how such an averaging works: the LYM proof weights each set by $1/\binom{m}{\ell}$ so each level contributes total weight $1$, and the chain-decomposition proof cuts the cube into symmetric saturated chains an antichain hits at most once. Both convert a hard global extremal count into an easy local per-level or per-chain statement by exploiting symmetry. The obstruction is that "antichain" and "intersecting" are different constraints, so Sperner's slices, fitted to containment, don't transfer. I need slices fitted to *intersection*.

So what structured slice of $k$-sets interacts cleanly with intersection? The induction pushed sets *down* toward a fixed point; let me instead think about *position*. An abstract $k$-set like $\{2,5,9\}$ has no shape, but if I lay $[n]$ out in an arrangement, the nicest $k$-sets relative to it are the *consecutive blocks*. Trying a line first — let the slice be the $n-k+1$ windows of $k$ consecutive points — gives the right local number (an intersecting set of windows is one whose starts lie in an interval of length $< k$, so at most $k$ of them), but the line has *ends*: the boundary windows behave differently, the slice has size $n-k+1$ rather than $n$, and a cyclic shift would push a window off the end, so the symmetry group of $[n]$ does not act transitively on linear windows. The boundary wrecks the symmetry. The fix is to bend the line into a circle.

I propose the **cycle method** (the circular-arrangement averaging argument). Put $1,\dots,n$ around a circle and call a block of $k$ consecutive points an **arc**. Now there are exactly $n$ arcs, none special; a rotation permutes them cyclically and is an automorphism of $[n]$, so this slice is genuinely symmetric, and $|Y| = n$ is exactly what I need. The whole theorem rests on two facts: a *local* bound that on any single circle an intersecting family contains at most $k$ of the $n$ arcs, and a *global* averaging step that, since each $k$-set is an arc in a $k/n$ fraction of the circles, turns the local $\le k$ into $|F| \le \tfrac{k}{n}\binom{n}{k} = \binom{n-1}{k-1}$.

For the local bound, fix an arc $A = \{a_1,\dots,a_k\}$ (consecutive) in $F$; if no arc lies in $F$ the count is $0 \le k$ trivially. Any other arc $B \in F$ satisfies $B \cap A \neq \emptyset$ and $B \neq A$, and since an arc is a contiguous block, $B$ slides off $A$ at exactly one end — for some internal gap $i \in \{1,\dots,k-1\}$ between adjacent points $a_i, a_{i+1}$, the arc $B$ contains exactly one of that pair. Exactly two arcs separate a given gap: the arc $L_i$ ending at $a_i$ (extending $k-1$ points to the left of $A$) and the arc $R_i$ starting at $a_{i+1}$ (extending $k-1$ points to the right). Here is the crux: $L_i$ and $R_i$ each have length $k$, so together they occupy $2k$ distinct points, which they can do without colliding around the back of the circle precisely when $2k \le n$. **That is where the hypothesis lives** — $n \ge 2k$ is exactly the condition making the two opposite arcs around a gap disjoint, so an intersecting $F$ holds at most one of $\{L_i, R_i\}$. Summing over the $k-1$ gaps gives at most $k-1$ arcs besides $A$, hence at most $k$ in total. If $n < 2k$ the two blocks wrap around and overlap and the argument breaks — rightly, since for $n < 2k$ the theorem is false anyway. So the threshold is not a technical convenience; it is the geometric disjointness of opposite arcs. The local bound is tight: $k$ consecutive arcs through one common point pairwise intersect.

For the global step I average directly so the $k/n$ appears in the open. Choose a circular arrangement uniformly at random. For a fixed $k$-set $A$, it becomes an arc when its $k$ points form one consecutive block: there are $n$ rotational positions for the block, $k!$ internal orders of $A$, and $(n-k)!$ orders of the rest, against $n!$ total, giving

$$P(A \text{ is an arc}) = \frac{n \cdot k! \cdot (n-k)!}{n!} = \frac{n}{\binom{n}{k}}.$$

By linearity of expectation the expected number of members of $F$ that are arcs of the random circle is $|F| \cdot n/\binom{n}{k}$. Since the local lemma bounds that count by $k$ for *every* arrangement, the expectation cannot exceed $k$:

$$|F|\cdot\frac{n}{\binom{n}{k}} \le k \quad\Longrightarrow\quad |F| \le \frac{k}{n}\binom{n}{k} = \binom{n-1}{k-1}.$$

The last identity, $\tfrac{k}{n}\binom{n}{k} = \tfrac{(n-1)!}{(k-1)!(n-k)!} = \binom{n-1}{k-1}$, is the whole point made flesh: a circle has $n$ arcs, an intersecting family eats at most $k$ of them, so it occupies at most a $k/n$ share of the symmetric slice, and averaging over all circles spreads that share evenly over all $\binom{n}{k}$ sets. The constant is forced and the threshold is forced, with not a single induction or sub-case in sight. The same fact can be stated as a pure double count over the $(n-1)!$ circular arrangements, in each of which a fixed $k$-set is an arc in $k!(n-k)!$ of them: $|F|\cdot k!(n-k)! \le k\cdot(n-1)!$, the same $\binom{n-1}{k-1}$.

The bound is tight because the star attains it, so the maximum is *exactly* $\binom{n-1}{k-1}$ for $n \ge 2k$. The extremal structure then splits at the threshold. For $n > 2k$ the star is the *unique* optimum. Suppose $F$ is intersecting with $|F| = \binom{n-1}{k-1}$ but is not a star. First, there is no saturated direction: if for some $x, y$ every $k$-set containing $x$ but not $y$ were in $F$, then $F = S_x$, because for any $K \not\ni x$, since $n > 2k$ there are $n-1-k \ge k$ points besides $y$ outside $K$, so a $k$-set $L \ni x$, $y \notin L$, $L \cap K = \emptyset$ exists; $L \in F$ forces $K \notin F$, whence $F \subseteq S_x$ and by cardinality $F = S_x$, a contradiction. So for every $x, y$ some $k$-set with $x$ and without $y$ is missing from $F$. Second, there is a boundary swap: two $k$-sets $K, K'$ with $|K \cap K'| = k-1$, $K \in F$, $K' \notin F$ — otherwise single-element swaps, which connect all of $\binom{[n]}{k}$ through the Johnson graph, would force $F$ to be everything, which is not intersecting. To finish, label $K \setminus K' = \{0\}$ and $K' \setminus K = \{k\}$, and pick $K'' \notin F$ with $0 \in K''$, $k \notin K''$. Choose a circular arrangement making $K = [0,k-1]$ an arc, with element $k$ at position $k$ so the arc $[1,k]$ equals $K' \notin F$, and $K''$ realized as the arc on the other side of $K$. The equality case of the local lemma — the only way to hit $k$ pairwise-intersecting arcs is $k$ *consecutive* arcs sharing a common point — forces the arcs of $F$ to be one of the $k$ consecutive runs through $K$; but each such run contains either $[1,k] = K'$ or $K''$, both outside $F$, so no run survives. The contradiction shows $F$ must be a star. The argument genuinely used $n > 2k$ (in the slack $n-1-k \ge k$), and at $n = 2k$ exactly that slack vanishes — which is precisely why the boundary behaves differently. There, each $k$-set has a unique disjoint complement, the $\binom{2k}{k}$ sets pair off into $\tfrac12\binom{2k}{k} = \binom{2k-1}{k-1}$ complementary pairs, and any *transversal* of this matching (one set from each pair $\{A, [n]\setminus A\}$) is intersecting and of extremal size. Stars are special transversals, but most transversals are not stars. So the picture is fully consistent: a tight bound $\binom{n-1}{k-1}$ for all $n \ge 2k$, the unique star strictly above $2k$, and many complementary-pair transversals exactly at $2k$.

The result is pure combinatorics and needs no implementation; but the two load-bearing identities and the bound itself can be confirmed by direct enumeration on small parameters, including the $n = 2k$ boundary case $(6,3)$:

```python
from itertools import combinations
from math import comb, factorial

def k_subsets(n, k):
    return [frozenset(s) for s in combinations(range(n), k)]

def is_intersecting(family):
    fam = list(family)
    return all(fam[i] & fam[j]
               for i in range(len(fam)) for j in range(i + 1, len(fam)))

def star(n, k, x=0):
    return [s for s in k_subsets(n, k) if x in s]

# Identity 1: the bound is the k/n share of all k-sets.
def check_bound_identity(n, k):
    return comb(n - 1, k - 1) * n == k * comb(n, k)

# Identity 2: a fixed k-set is an arc with probability n / C(n,k).
def check_arc_probability(n, k):
    favorable = n * factorial(k) * factorial(n - k)   # n*k!*(n-k)!
    from fractions import Fraction
    return Fraction(favorable, factorial(n)) == Fraction(n, comb(n, k))

# Brute-force largest intersecting family for tiny (n, k), compared to bound.
def brute_force_max(n, k):
    sets = k_subsets(n, k)
    best = 0
    # exact search via a simple branch over sets, feasible only for small n,k
    def extend(idx, chosen):
        nonlocal best
        best = max(best, len(chosen))
        if len(chosen) + (len(sets) - idx) <= best:
            return
        for j in range(idx, len(sets)):
            if all(sets[j] & c for c in chosen):
                chosen.append(sets[j]); extend(j + 1, chosen); chosen.pop()
    extend(0, [])
    return best

if __name__ == "__main__":
    for (n, k) in [(4, 2), (5, 2), (6, 3)]:   # last is the n = 2k boundary
        assert check_bound_identity(n, k)
        assert check_arc_probability(n, k)
        assert len(star(n, k)) == comb(n - 1, k - 1)
        assert is_intersecting(star(n, k))
        assert brute_force_max(n, k) == comb(n - 1, k - 1)
    print("all checks pass")
```
