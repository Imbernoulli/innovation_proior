Fix a ground set $[n]=\{1,\dots,n\}$ and order its power set $2^{[n]}$ by inclusion. A family $\mathcal{F}\subseteq 2^{[n]}$ is an antichain when no member of it is contained in another, and the question is how large such a family can be; write $w(n)$ for the largest possible $|\mathcal{F}|$. One family comes for free. Containment always raises cardinality, so an entire layer $\binom{[n]}{k}$ — all $k$-element subsets — is automatically an antichain, of size $\binom{n}{k}$. Which layer is fattest is settled by comparing neighbours,
$$
\frac{\binom{n}{k}}{\binom{n}{k-1}}=\frac{n!/(k!\,(n-k)!)}{n!/((k-1)!\,(n-k+1)!)}=\frac{n-k+1}{k},
$$
a ratio that exceeds $1$ exactly when $n-k+1>k$, i.e. when $k<\tfrac{n+1}{2}$. So $\binom{n}{k}$ strictly increases up to the middle and strictly decreases after it: a single hump, peaking at $k=\lfloor n/2\rfloor$, and tied with $\lceil n/2\rceil$ when $n$ is odd because $\binom{n}{k}=\binom{n}{n-k}$. That gives $w(n)\ge\binom{n}{\lfloor n/2\rfloor}$ at no cost, and the whole difficulty sits in the matching upper bound: can a family drawing on several layers at once ever beat the best single layer?

Mixing is certainly legal — $\{\{1\},\{2,3\}\}$ is a perfectly good antichain in $2^{[3]}$ — it is only small, and the reason it is small is that admitting a set spends its entire cone. The moment $\{2,3\}$ goes in, $\{2\}$, $\{3\}$ and $\{1,2,3\}$ are all barred: every subset and every superset is forfeited. Accounting for that forfeiture simultaneously across many cardinalities is the actual problem, and the naive bookkeeping does not do it. Writing $a_k$ for the number of members of size $k$, so $|\mathcal{F}|=\sum_k a_k$, the layerwise bound $a_k\le\binom{n}{k}$ yields only $|\mathcal{F}|\le\sum_k\binom{n}{k}=2^n$. That is worthless, and it is worthless for a diagnosable reason: it treats the layers as independent and never once uses incomparability to couple them.

The concrete way to force a coupling is to move the family rather than count it in place — to compress any antichain into the middle layer without losing members, and then read the bound off there, since a single layer has at most $\binom{n}{\lfloor n/2\rfloor}$ sets. Take the members of size $k>n/2$ and replace that subfamily $\mathcal{A}$, of size $m$, by its lower shadow $\partial\mathcal{A}$, all $(k-1)$-subsets of members of $\mathcal{A}$. The count is controlled by an incidence double count between the two layers: each $k$-set has exactly $k$ subsets of size $k-1$, giving $m\,k$ incidences, while each $(k-1)$-set lies in at most $n-(k-1)=n-k+1$ of the $k$-sets, so the number $r=|\partial\mathcal{A}|$ of distinct shadow sets obeys
$$
m\,k\ \le\ r\,(n-k+1)\qquad\Longrightarrow\qquad r\ \ge\ \frac{m\,k}{\,n-k+1\,}.
$$
When $k>\tfrac{n+1}{2}$ we have $k>n-k+1$, the fraction $k/(n-k+1)$ exceeds $1$, and therefore $r\ge m+1$: the shadow is strictly larger, so the replacement never shrinks the family. Complementation gives the dual statement, that $m$ sets of size $k<n/2$ have at least $m+1$ supersets of size $k+1$, so the bottom can be pushed up by the shade in the same way. Iterating drives everything into the middle in finitely many steps and the bound follows. This route works, and I do not doubt it, but it is heavy out of all proportion to its conclusion. The shadow inequality degenerates exactly at the boundary $k=\tfrac{n+1}{2}$ for odd $n$, where it only gives $r\ge m$ with equality possible, so a separate and genuinely fiddly argument is needed there to produce an extra $(k-1)$-set whenever $m<\binom{n}{\lfloor n/2\rfloor}$; the whole thing is then an induction over the replacement steps, each of which must be re-checked for preservation of incomparability against the layers already present; and at the end it hands over the bare inequality $|\mathcal{F}|\le\binom{n}{\lfloor n/2\rfloor}$ with no structure attached, so the extremal families still have to be dug out by hand. The alternative of covering $2^{[n]}$ by disjoint chains is cleaner to state — an antichain meets each chain at most once, so a partition into $t$ chains bounds $|\mathcal{F}|$ by $t$ — but it only relocates the work into constructing an economical chain partition, which is its own combinatorial problem.

So I stop moving mass and instead reweight it. The defect in $|\mathcal{F}|=\sum_k a_k$ is that a set in the fat middle layer is charged the same as a set in a thin outer layer, even though the two layers offer wildly different numbers of candidates. Normalize so that a full layer always costs exactly $1$: charge a $k$-set the price $1/\binom{n}{k}$. The target then becomes the weighted statement $\sum_k a_k/\binom{n}{k}\le 1$, which is strictly stronger than the size bound and, as usual when a statement is strengthened correctly, easier — there is more structure to hold on to. It also implies the size bound in a single line, because unimodality gives $\binom{n}{k}\le\binom{n}{\lfloor n/2\rfloor}$ and hence $1/\binom{n}{k}\ge 1/\binom{n}{\lfloor n/2\rfloor}$ for every $k$.

The method is the LYM inequality, and its mechanism is that the weight $1/\binom{n}{k}$ is not an accounting convenience at all but a genuine count of objects. Write it as $1/\binom{n}{k}=k!\,(n-k)!/n!$ and the shape is a probability: favourable arrangements over $n!$ total arrangements. The $n!$ is the number of permutations of $[n]$, and $k!\,(n-k)!$ is the number of orderings in which a distinguished $k$-block comes first in some internal order and the rest follows in some internal order. Read a permutation as an instruction for assembling the ground set one element at a time, starting from $\emptyset$ and adding elements in the given order; it traces out a maximal chain
$$
\emptyset=C_0\subset C_1\subset\cdots\subset C_n=[n],\qquad |C_i|=i,
$$
each step adding a single element. Permutations and maximal chains are the same data — the chain records the order of insertion — so there are exactly $n!$ maximal chains. A fixed $A$ with $|A|=k$ lies on such a chain precisely when the chain builds $A$ first, in one of $k!$ orders, and then completes $[n]\setminus A$, in one of $(n-k)!$ orders; the two halves are independent, so exactly $k!\,(n-k)!$ maximal chains pass through $A$. The weight $1/\binom{n}{k}$ is therefore literally the fraction of maximal chains through a $k$-set. It matters here that the chains are maximal rather than arbitrary: a saturated chain meets every layer exactly once and is in bijection with a permutation, which is what produces the clean $n!$ and $k!\,(n-k)!$; arbitrary chains carry no such correspondence and no such counts.

With that reading, the antichain hypothesis enters exactly once, and it is the entire argument. A maximal chain passes through at most one member of $\mathcal{F}$, because any two sets lying on one chain are some $C_i$ and $C_j$ with $i<j$, hence nested, hence forbidden in an antichain. The families "maximal chains through $A$", as $A$ ranges over $\mathcal{F}$, are therefore pairwise disjoint subcollections of the $n!$ maximal chains, and disjoint pieces cannot overshoot the whole:
$$
\sum_{A\in\mathcal{F}}|A|!\,(n-|A|)!\ =\ \sum_{k=0}^{n}a_k\,k!\,(n-k)!\ \le\ n!.
$$
Dividing by $n!$ and using $k!\,(n-k)!/n!=1/\binom{n}{k}$ gives
$$
\sum_{k=0}^{n}\frac{a_k}{\binom{n}{k}}\ \le\ 1,
$$
which is the weighted inequality I wanted, obtained from one double count of maximal chains — no shadow lemma, no degenerate boundary case, no induction over replacement steps. The whole weight of the compression argument has collapsed into the single sentence "two sets on one chain are comparable". The same content in probabilistic dress confirms that no factor has been dropped: choose a maximal chain uniformly at random, equivalently a uniformly random permutation, and it passes through a fixed $k$-set with probability $k!\,(n-k)!/n!=1/\binom{n}{k}$; if $X$ counts the members of $\mathcal{F}$ that the random chain meets, then $\mathbb{E}[X]=\sum_k a_k/\binom{n}{k}$ by linearity, while $X\le 1$ pointwise because a chain meets at most one member, and an expectation of a quantity that never exceeds $1$ is at most $1$.

Sperner's bound is now immediate. Each denominator satisfies $\binom{n}{k}\le\binom{n}{\lfloor n/2\rfloor}$, so
$$
\frac{|\mathcal{F}|}{\binom{n}{\lfloor n/2\rfloor}}=\sum_k\frac{a_k}{\binom{n}{\lfloor n/2\rfloor}}\ \le\ \sum_k\frac{a_k}{\binom{n}{k}}\ \le\ 1,
$$
giving $|\mathcal{F}|\le\binom{n}{\lfloor n/2\rfloor}$, which the middle layer attains. Hence $w(n)=\binom{n}{\lfloor n/2\rfloor}$ exactly.

The weighted form also hands over the equality cases, which the compression route made me hunt for. Equality $|\mathcal{F}|=\binom{n}{\lfloor n/2\rfloor}$ forces both inequalities in the display above to be tight. The second one, where the denominators were replaced by the largest binomial coefficient, is tight only if $a_k=0$ for every $k$ with $\binom{n}{k}<\binom{n}{\lfloor n/2\rfloor}$: the family uses only layers that achieve the maximum, which for even $n$ is the single layer $k=n/2$ and for odd $n$ is the pair $k=\tfrac{n-1}{2}$, $k=\tfrac{n+1}{2}$. The first one, $\sum_A |A|!\,(n-|A|)!\le n!$, is tight only if the disjoint chain-families exhaust all $n!$ maximal chains, i.e. every maximal chain meets a member of $\mathcal{F}$. For even $n$ these two conditions already close the case: a family inside layer $n/2$ that misses even one $(n/2)$-set also misses the chains through that set, so the unique extremal antichain is the full middle layer $\binom{[n]}{n/2}$. For odd $n$ the family lives in the two middle layers and must meet every maximal chain exactly once; since a maximal chain passes through exactly one set of each size, covering every chain from two adjacent layers with never two members on a chain leaves no room to split between the layers without creating either a gap or a collision, and the extremal antichains are exactly the two full middle layers $\binom{[n]}{(n-1)/2}$ and $\binom{[n]}{(n+1)/2}$. The optimum is therefore the construction I began with, now known to be the only one.

One more thing falls out of the same count, and it is worth recording because it shows what the argument really used. Suppose $A_1,\dots,A_m$ and $B_1,\dots,B_m$ are subsets of $[n]$ with $A_i\cap B_j=\emptyset$ if and only if $i=j$, and put $a_i=|A_i|$, $b_i=|B_i|$. Then
$$
\sum_{i=1}^{m}\frac{1}{\binom{a_i+b_i}{a_i}}\ \le\ 1 .
$$
Count the permutations of $[n]$ that place all of $A_i$ before all of $B_i$: among the $(a_i+b_i)!$ internal orders of those distinguished elements, $a_i!\,b_i!$ put $A_i$ first, a fraction $1/\binom{a_i+b_i}{a_i}$, so there are $n!/\binom{a_i+b_i}{a_i}$ such permutations. No permutation qualifies for two indices $i\ne j$: if it did, say with the last element of $A_i$ coming no later than the last element of $A_j$, then all of $A_i$ precedes all of $B_j$, forcing $A_i\cap B_j=\emptyset$ and contradicting $i\ne j$. The qualifying sets are thus disjoint and their sizes sum to at most $n!$. Taking $B_i=[n]\setminus A_i$ turns the hypothesis $A_i\cap B_j=\emptyset$ into $A_i\subseteq A_j$, which on an antichain happens only for $i=j$, and the inequality becomes the LYM inequality again, and with it Sperner's bound.

The formula and the equality statement are exhaustively checkable at small $n$, where the answer can be found by brute force over all families: $w(0)=w(1)=1$, $w(2)=2$, $w(3)=3$, $w(4)=6=\binom{4}{2}$, and the full middle layer is in each case an antichain whose LYM weight is exactly $1$.

```python
from itertools import combinations
from math import comb, factorial

def chains_through_set_fraction(n, k):
    # k!*(n-k)! maximal chains through a fixed k-set, out of n! total == 1 / C(n,k)
    return factorial(k) * factorial(n - k) / factorial(n)

def lym_lhs(n, antichain):
    # the LYM left-hand side: sum of 1 / C(n, |A|) over members A
    return sum(1 / comb(n, len(A)) for A in antichain)

def is_antichain(family):
    fam = list(map(set, family))
    return all(not (fam[i] <= fam[j]) for i in range(len(fam))
               for j in range(len(fam)) if i != j)

def brute_force_max_antichain(n):
    subs = [frozenset(s) for k in range(n + 1) for s in combinations(range(n), k)]
    from itertools import combinations as C
    best = 0
    for r in range(len(subs) + 1):
        for fam in C(subs, r):
            if is_antichain(fam):
                best = max(best, r)
    return best

for n in range(5):
    assert brute_force_max_antichain(n) == comb(n, n // 2)         # Sperner's bound is tight
    middle = [frozenset(s) for s in combinations(range(n), n // 2)]
    assert abs(lym_lhs(n, middle) - 1.0) < 1e-9                    # full middle layer: LYM sum = 1
    for k in range(n + 1):
        assert abs(chains_through_set_fraction(n, k) - 1 / comb(n, k)) < 1e-12
print("verified: max antichain = C(n, floor(n/2)); middle layer attains LYM equality")
```
