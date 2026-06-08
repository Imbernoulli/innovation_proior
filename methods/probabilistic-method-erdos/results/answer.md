# The Probabilistic Method (first moment + deletion), and the Ramsey lower bound

## Problem

Prove that a combinatorial object with a desired property *exists*, in a setting where no explicit construction is known. The motivating instance: lower-bound the diagonal Ramsey number $R(k,k)=f(k,k)$ — the least $n$ such that every 2-coloring of the edges of $K_n$ has a monochromatic $K_k$ — by certifying that order-free colorings exist for $n$ as large as possible. The standing upper bound is $R(k,k)\le\binom{2k-2}{k-1}<4^{k-1}$ (Erdős–Szekeres 1935); the lower side had no good explicit family.

## Key idea

To show an object with a property $P$ exists, **do not construct it**. Put a probability distribution on a space of candidate objects and show
$$\Pr[\text{a random candidate has }P]>0,$$
which forces at least one candidate to have $P$. Two refinements:

- **First moment / union bound.** For "no bad event occurs," bound $\Pr[\bigcup_i B_i]\le\sum_i\Pr[B_i]$; if this is $<1$, a candidate avoiding all $B_i$ exists. Equivalently, if a counting variable $X\ge0$ has $\mathbb E[X]<1$ then $\Pr[X=0]>0$.
- **Deletion (alteration).** Don't demand a flawless random object. Take one whose number of blemishes $X$ is at most $\mathbb E[X]$ (some object is below its mean), then *remove* one element per blemish; what remains has the property. This beats the union bound when $\mathbb E[X]$ is small but not below $1$.
- **Linearity of expectation** drives both: $\mathbb E[\sum_i X_i]=\sum_i\mathbb E[X_i]$ with **no independence** required, and some point of the space achieves at least (or at most) the mean.

## Theorems landed on

**Theorem 1 (Ramsey lower bound, first moment).** If $\binom nk\,2^{1-\binom k2}<1$ then $R(k,k)>n$. In particular $R(k,k)>\lfloor 2^{k/2}\rfloor$ for all $k\ge3$; since $R(k,k)$ is an integer, this is the usual $f(k,k)>2^{k/2}$ lower-bound form.

*Proof.* Color each edge of $K_n$ red/blue by an independent fair coin. For a fixed $k$-set $R$, $\Pr[R\text{ monochromatic}]=2\cdot2^{-\binom k2}=2^{1-\binom k2}$. Union bound over the $\binom nk$ sets: $\Pr[\exists\text{ mono }K_k]\le\binom nk2^{1-\binom k2}$. If this is $<1$, some coloring has no monochromatic $K_k$, witnessing $R(k,k)>n$. For $n=\lfloor 2^{k/2}\rfloor$, $k\ge3$,
$$\binom nk2^{1-\binom k2}<\frac{n^k}{k!}\,2^{1-k(k-1)/2}\le\frac{2^{k^2/2}}{k!}\,2^{1-k(k-1)/2}=\frac{2^{1+k/2}}{k!}<1.$$
(Erdős's original count: among the $2^{\binom n2}$ labeled $n$-vertex graphs, the number containing a *fixed* $K_k$ is exactly $2^{\binom n2-\binom k2}$, because the $\binom k2$ clique edges are forced and all other edges are free. Hence the number containing *some* $K_k$ is at most
$$\binom nk2^{\binom n2-\binom k2}<\frac{n^k}{k!}\,2^{\binom n2-\binom k2},$$
which is $<\tfrac12\,2^{\binom n2}$ whenever $2n^k<k!\,2^{k(k-1)/2}$. This condition holds for $n\le2^{k/2}$, $k\ge3$; by the complement bijection the graphs with an independent $k$-set are also $<\tfrac12\,2^{\binom n2}$, so a graph avoiding both exists.) $\square$

A careful asymptotic optimization of the same union bound gives $R(k,k)>\tfrac{1}{e\sqrt2}(1+o(1))\,k\,2^{k/2}$.

**Theorem 2 (Ramsey lower bound, deletion).** For every $n$, $R(k,k)>n-\binom nk\,2^{1-\binom k2}$.

*Proof.* Let $X=\sum_R \mathbf 1[R\text{ mono}]$. By linearity $\mathbb E[X]=\binom nk2^{1-\binom k2}=:m$, so some coloring has $X\le m$. Delete one vertex from each monochromatic $k$-set: at most $m$ vertices removed, leaving $s\ge n-m$ vertices on which no monochromatic $k$-set survives. Hence $R(k,k)>n-m$. $\square$

Using $\binom nk\sim n^k/k!$, the derivative condition for maximizing $n-\frac{n^k}{k!}2^{1-\binom k2}$ is $n^{k-1}=(k-1)!2^{\binom k2-1}$, so the optimal $n\sim\tfrac ke\,2^{k/2}$ and
$$R(k,k)>\frac1e(1+o(1))\,k\,2^{k/2},$$
a factor $\sqrt2$ improvement over the careful union bound.

**Theorem 3 (Linearity-of-expectation existence).**
(a) Some tournament on $n$ players has at least $n!\,2^{-(n-1)}$ Hamiltonian paths. *Proof.* Random tournament (each arc by a fair coin); for a fixed ordering $\sigma$, $\Pr[\sigma\text{ is a directed Ham path}]=2^{-(n-1)}$; summing over $n!$ orderings, $\mathbb E[\#\text{Ham paths}]=n!\,2^{-(n-1)}$; some tournament meets the mean. $\square$
(b) Every graph with $e$ edges has a bipartite subgraph (cut) with at least $e/2$ edges. *Proof.* Assign each vertex to $T$ or $B$ by a fair coin; an edge crosses with probability $1/2$; $\mathbb E[\#\text{crossing}]=e/2$; some split meets it. $\square$ Equivalently (Max-SAT form): any $m$ clauses, each falsified with probability $\le\tfrac12$ under a uniform random assignment, admit an assignment satisfying $\ge m/2$ of them.

In all three, no independence among the summed indicators is used — only linearity of expectation and the existence of a point at least/at most the mean.
