# The Kruskal–Katona theorem

## Problem

For a $k$-uniform family $\mathcal{F}$ (a set of $k$-element sets), the **lower shadow** is
$$ \partial\mathcal{F} = \{\,E : E = F\setminus\{x\},\ F\in\mathcal{F},\ x\in F\,\}, $$
the $(k-1)$-uniform family of all $(k-1)$-subsets contained in some member of $\mathcal{F}$. Fixing
$|\mathcal{F}| = m$, what is the minimum possible $|\partial\mathcal{F}|$, and which families attain it?

## Key idea

To minimise the boundary, maximise subset-sharing: pack the sets onto as few elements as possible.
The extremal family is the **colexicographic initial segment** — the first $m$ sets in colex order,
$A < B \iff \max(A\triangle B)\in B$, which avoids large elements as long as possible. Equivalently it
is the greedy stack of cliques: the largest clique $[a_k]^{(k)}$ that fits, then a $(k-1)$-clique on
top using the next element, and so on. Writing $m$ in this greedy form is the **$k$-cascade
representation**, and the minimum shadow is obtained by pushing every cascade index down by one.

## The cascade representation

Every $m\ge1$ has a unique
$$ m = \binom{a_k}{k} + \binom{a_{k-1}}{k-1} + \cdots + \binom{a_s}{s}, \qquad a_k > a_{k-1} > \cdots > a_s \ge s \ge 1, $$
obtained greedily: $a_k$ is the largest integer with $\binom{a_k}{k}\le m$; subtract and recurse at
index $k-1$. Define the shadow function
$$ \partial^{(k)}(m) := \binom{a_k}{k-1} + \binom{a_{k-1}}{k-2} + \cdots + \binom{a_s}{s-1}. $$

## Theorem

For every $k$-uniform family $\mathcal{F}$,
$$ \boxed{\,|\partial\mathcal{F}| \ \ge\ \partial^{(k)}(|\mathcal{F}|)\,}, $$
with equality for the first $|\mathcal{F}|$ sets in colexicographic order. The shadow of that colex
initial segment is itself the colex initial segment of the same cascade parameters one layer down, so
the bound is exactly attained. When $|\mathcal{F}| = \binom{a}{k}$ for an integer $a$, equality holds
**iff** $\mathcal{F}$ is a clique on $a$ vertices, i.e. $[a]^{(k)}$ after relabelling.

**Lovász form.** Extend $\binom{x}{k} = x(x-1)\cdots(x-k+1)/k!$ to real $x\ge k$. If
$|\mathcal{F}| = \binom{x}{k}$ then
$$ |\partial\mathcal{F}| \ge \binom{x}{k-1}. $$
This is weaker than the cascade bound strictly between integer points but agrees at $m=\binom{n}{k}$
and is far easier to compute with.

## Proof (Frankl's 1984 shifting argument)

**Shift operator.** For $i\ge2$, $S_i(\mathcal{F}) = \{S_i(F):F\in\mathcal{F}\}$ where $S_i(F) =
F\setminus\{i\}\cup\{1\}$ if $i\in F$, $1\notin F$, and $F\setminus\{i\}\cup\{1\}\notin\mathcal{F}$
(else $F$, *blocked*).

1. **Size is preserved**, $|S_i(\mathcal{F})| = |\mathcal{F}|$: $S_i$ is a bijection (a moved set's
   target is unoccupied, so no collisions).
2. **Shadow does not grow**, $\partial S_i(\mathcal{F}) \subseteq S_i(\partial\mathcal{F})$ (four-case
   analysis on how $\{1,i\}$ meets $S_i(F)$), hence $|\partial S_i(\mathcal{F})| \le |\partial\mathcal{F}|$.
3. **Reduce to stable families.** Iterating $S_i$ strictly increases the number of sets containing
   $1$, so it terminates at a *stable* $\mathcal{G}$ ($S_i(\mathcal{G})=\mathcal{G}\ \forall i$) with
   $|\mathcal{G}|=m$, $|\partial\mathcal{G}|\le|\partial\mathcal{F}|$.

**Structure of a stable family.** Split $\mathcal{F}=\mathcal{F}_0\sqcup\mathcal{F}_1$ by membership of
$1$ and set $\mathcal{F}_1' = \{F\setminus\{1\}:F\in\mathcal{F}_1\}$, a $(k-1)$-family with
$|\mathcal{F}_1'|=|\mathcal{F}_1|$.

- $\partial\mathcal{F}_0\subseteq\mathcal{F}_1'$, so $|\mathcal{F}_1'|\ge|\partial\mathcal{F}_0|$.
  (Any $E=F\setminus\{x\}\in\partial\mathcal{F}_0$ has $x\ge2$; stability means $F$ was blocked, so
  $F\setminus\{x\}\cup\{1\}\in\mathcal{F}_1$ and $E\in\mathcal{F}_1'$.)
- $\partial\mathcal{F}_1 = \mathcal{F}_1'\ \sqcup\ \{1\}\!\cdot\!\partial\mathcal{F}_1'$ (disjoint), so
  $$ |\partial\mathcal{F}| = |\mathcal{F}_1'| + |\partial\mathcal{F}_1'|. $$

**Double induction** (outer on $k$, inner on $m$), with $m$ of cascade $(a_k,\dots,a_s)$:

- *Lower bound on $|\mathcal{F}_1'|$.* If $|\mathcal{F}_1'| < \sum_j\binom{a_j-1}{j-1}$, then by
  $\binom{a}{i}-\binom{a-1}{i-1}=\binom{a-1}{i}$ (Pascal), $|\mathcal{F}_0| > \sum_j\binom{a_j-1}{j}$;
  after dropping any zero term $\binom{s-1}{s}$, monotonicity of the cascade shadow function and the
  inner induction on $\mathcal{F}_0$ ($|\mathcal{F}_0|<m$) give $|\partial\mathcal{F}_0| \ge
  \sum_j\binom{a_j-1}{j-1} \le |\mathcal{F}_1'|$ — contradiction. So
  $|\mathcal{F}_1'|\ge\sum_j\binom{a_j-1}{j-1}$.
- *Shadow of $\mathcal{F}_1'$.* Outer induction (on $k$): $|\partial\mathcal{F}_1'| \ge
  \sum_j\binom{a_j-1}{j-2}$.
- *Combine.* By $\binom{a_j-1}{j-1}+\binom{a_j-1}{j-2}=\binom{a_j}{j-1}$ (Pascal),
  $$ |\partial\mathcal{F}| = |\mathcal{F}_1'|+|\partial\mathcal{F}_1'| \ge \sum_j\binom{a_j}{j-1} = \partial^{(k)}(m). $$

Base cases: $k=1$ gives $|\partial\mathcal{F}|=|\{\emptyset\}|=1=\binom{m}{0}$; $m=1$ gives
$|\partial\mathcal{F}|=k=\binom{k}{k-1}$. The Lovász form follows by carrying the real-$x$
$\binom{x}{k}$ calculation through the same induction (with
$\binom{x}{k}=\binom{x-1}{k}+\binom{x-1}{k-1}$). If $x\in[k,k+1)$, then $x-1<k$ and the outer
induction cannot be invoked for the $x-1$ parameter; the fallback is that
$|\mathcal{F}_0|>\binom{x-1}{k}\ge0$ makes $\mathcal{F}_0$ nonempty, so
$|\partial\mathcal{F}_0|\ge k=\binom{k}{k-1}\ge\binom{x-1}{k-1}$.

**Compression variant.** For $i<j$, the $(i,j)$-compression $C_{ij}$ (replace $j$ by $i$ where
possible, with blocked sets retained) preserves size, and nontrivial compressions lower
$w(A)=\sum_{x\in A}2^x$. The compression proof must choose compressions in an order that keeps the
shadow from increasing; the naive analogue
$\partial C_{ij}(\mathcal{F})\subseteq C_{ij}(\partial\mathcal{F})$ is not valid for arbitrary
compression steps, and fully compressed families of a fixed size are not uniquely colex. With the
correct order, the family can still be compressed to the colex initial segment; once the ordered
compression reaches the needed compressed form, the same split by element $1$ gives
$\partial\mathcal{F}_0\subseteq\mathcal{F}_1'$ and the same double induction.

## Consequences

- **Erdős–Ko–Rado:** an intersecting $\mathcal{A}\subseteq X^{(r)}$, $2\le r<n/2$, has
  $|\mathcal{A}|\le\binom{n-1}{r-1}$ — via complements, the iterated shadow stays disjoint from
  $\mathcal{A}$ and the theorem bounds it; equality is the star.
- **$f$-vectors of simplicial complexes:** $(f_0,\dots,f_{d-1})$ is realisable iff
  $f_{i-1}\ge\partial^{(i+1)}(f_i)$ for all $i\ge1$, where $f_i$ counts $i$-dimensional faces.

## Code

```python
from math import comb
from itertools import combinations

def k_cascade(m, k):
    """Unique greedy k-cascade: m = sum_i C(a_i, i), a_k > a_{k-1} > ... ."""
    rep, i = [], k
    while i >= 1 and m > 0:
        a = i
        while comb(a + 1, i) <= m:        # largest a with C(a, i) <= m
            a += 1
        rep.append((a, i))
        m -= comb(a, i)
        i -= 1
    return rep

def min_shadow_size(m, k):
    """Kruskal-Katona minimum: push every cascade index down by one."""
    return sum(comb(a, i - 1) for (a, i) in k_cascade(m, k))

def colex_key(S):
    """Colex order: A < B iff max(A symdiff B) in B; equiv. sum of 2^element."""
    return sum(1 << x for x in S)

def extremal_family(m, k, ground=None):
    """First m k-sets in colex order — the minimiser of the shadow."""
    if ground is None:
        ground = m + k
    sets = sorted((frozenset(c) for c in combinations(range(1, ground + 1), k)),
                  key=colex_key)
    return sets[:m]

def shadow(family):
    out = set()
    for S in family:
        for x in S:
            out.add(frozenset(S - {x}))
    return out

# Sanity check on small n: compare len(shadow(extremal_family(m, k)))
# with brute_force_min_shadow(n, m, k).
```
