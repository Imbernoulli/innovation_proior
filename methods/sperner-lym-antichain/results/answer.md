# Sperner's theorem via the LYM inequality

## Problem

In the subset lattice $2^{[n]}$ of $[n]=\{1,\dots,n\}$ ordered by inclusion, an **antichain** (Sperner family) is a family $\mathcal{F}$ of subsets no one of which contains another. How large can $\mathcal{F}$ be?

## Key idea

Weight each subset by the fraction of *maximal chains* of the lattice that pass through it. A maximal chain $\emptyset=C_0\subset C_1\subset\cdots\subset C_n=[n]$ ($|C_i|=i$) is the same data as a permutation of $[n]$, so there are $n!$ of them, and exactly $k!\,(n-k)!$ of them pass through a fixed $k$-set. An antichain meets each maximal chain at most once (two sets on one chain are nested). Counting maximal chains two ways gives the **LYM inequality**, and bounding each weight by the middle (largest) binomial coefficient gives the size bound.

## The LYM inequality

**Theorem (LYM).** Let $\mathcal{F}$ be an antichain in $2^{[n]}$ and let $a_k$ be the number of its members of size $k$. Then
$$
\sum_{k=0}^{n} \frac{a_k}{\binom{n}{k}} \ \le\ 1 .
$$

**Proof.** Call $\emptyset=C_0\subset C_1\subset\cdots\subset C_n=[n]$ with $|C_i|=i$ a *maximal chain*. Building $[n]$ one element at a time identifies maximal chains with permutations of $[n]$, so there are exactly $n!$ of them. A fixed $A$ with $|A|=k$ lies on a maximal chain iff the chain builds $A$ first ($k!$ orders) and then $[n]\setminus A$ ($(n-k)!$ orders); hence exactly $k!\,(n-k)!$ maximal chains pass through $A$. No maximal chain passes through two distinct members of $\mathcal{F}$, since any two sets on one chain are comparable and $\mathcal{F}$ is an antichain. Therefore the sets "chains through $A$", over $A\in\mathcal{F}$, are pairwise disjoint subfamilies of all $n!$ maximal chains, so
$$
\sum_{A\in\mathcal{F}} |A|!\,(n-|A|)! \;=\; \sum_{k=0}^{n} a_k\,k!\,(n-k)! \ \le\ n!.
$$
Dividing by $n!$ and using $\dfrac{k!\,(n-k)!}{n!}=\dfrac{1}{\binom{n}{k}}$ gives $\displaystyle\sum_{k} \frac{a_k}{\binom{n}{k}}\le 1$. $\qquad\blacksquare$

*(Equivalent probabilistic phrasing: a uniformly random maximal chain hits a fixed $k$-set with probability $1/\binom{n}{k}$, so the number $X$ of antichain members it hits has $\mathbb{E}[X]=\sum_k a_k/\binom{n}{k}$; since $X\le1$ always, $\mathbb{E}[X]\le1$.)*

## Sperner's theorem (the size bound)

**Theorem (Sperner).** The largest antichain in $2^{[n]}$ has size $\binom{n}{\lfloor n/2\rfloor}$.

**Proof.** *Lower bound.* The middle layer $\binom{[n]}{\lfloor n/2\rfloor}$ is an antichain of size $\binom{n}{\lfloor n/2\rfloor}$.

*Upper bound.* The binomial coefficients are unimodal with maximum at the middle: from
$$
\frac{\binom{n}{k}}{\binom{n}{k-1}}=\frac{n-k+1}{k}\ \begin{cases}>1 & k<\tfrac{n+1}{2}\\[2pt] <1 & k>\tfrac{n+1}{2}\end{cases}
$$
we get $\binom{n}{k}\le\binom{n}{\lfloor n/2\rfloor}$ for all $k$. Hence $1/\binom{n}{k}\ge 1/\binom{n}{\lfloor n/2\rfloor}$, and by the LYM inequality
$$
\frac{|\mathcal{F}|}{\binom{n}{\lfloor n/2\rfloor}}=\sum_k\frac{a_k}{\binom{n}{\lfloor n/2\rfloor}}\le\sum_k\frac{a_k}{\binom{n}{k}}\le 1,
$$
so $|\mathcal{F}|\le\binom{n}{\lfloor n/2\rfloor}$. $\qquad\blacksquare$

## Equality characterization

Equality $|\mathcal{F}|=\binom{n}{\lfloor n/2\rfloor}$ forces both LYM inequalities to be tight:
- $\sum_k a_k/\binom{n}{\lfloor n/2\rfloor}=\sum_k a_k/\binom{n}{k}$ requires $a_k=0$ whenever $\binom{n}{k}<\binom{n}{\lfloor n/2\rfloor}$, i.e. $\mathcal{F}$ uses only the maximizing layer(s);
- $\sum_A |A|!(n-|A|)!=n!$ requires every maximal chain to pass through a member of $\mathcal{F}$.

Together: for **even $n$** the unique extremal antichain is the full middle layer $\binom{[n]}{n/2}$; for **odd $n$** the extremal antichains are exactly the two full middle layers $\binom{[n]}{(n-1)/2}$ and $\binom{[n]}{(n+1)/2}$.

## A stronger ambient statement (Bollobás)

The same double count over the $n!$ permutations gives a generalization. If $A_1,\dots,A_m$ and $B_1,\dots,B_m$ are subsets of $[n]$ with $A_i\cap B_j=\emptyset$ **iff** $i=j$, then with $a_i=|A_i|,\,b_i=|B_i|$,
$$
\sum_{i=1}^{m}\frac{1}{\binom{a_i+b_i}{a_i}}\ \le\ 1 .
$$
(Count permutations placing all of $A_i$ before all of $B_i$: among the $(a_i+b_i)!$ orders of those distinguished elements, $a_i!\,b_i!$ put $A_i$ first, a fraction $1/\binom{a_i+b_i}{a_i}$, so there are $n!/\binom{a_i+b_i}{a_i}$ such permutations. No permutation qualifies for two indices $i\ne j$: qualifying for both puts $A_i$ before $B_i$ and $A_j$ before $B_j$, but say the last element of $A_i$ comes no later than the last of $A_j$; then all of $A_i$ precedes all of $B_j$, so $A_i$ and $B_j$ are disjoint — contradicting $A_i\cap B_j\ne\emptyset$. Hence the qualifying sets are disjoint, summing to $\le n!$.) Taking $B_i=[n]\setminus A_i$ gives $A_i\cap B_j=\emptyset\iff A_i\subseteq A_j\iff i=j$ on an antichain, recovering the LYM inequality, and then Sperner's bound, as above.

## Sanity check (small $n$)

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
