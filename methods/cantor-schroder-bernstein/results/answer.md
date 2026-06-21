# The Cantor–Schröder–Bernstein theorem

## Problem

Compare set sizes by injections: write $A \preceq B$ when there is an injection $A \to B$. Reflexivity and
transitivity are easy; the decisive property is **antisymmetry**. The theorem supplies it: from two one-sided
embeddings, build a two-sided correspondence.

> **Theorem (Cantor–Schröder–Bernstein).** Let $A, B$ be sets. If there exist injections $f : A \to B$ and
> $g : B \to A$, then there exists a bijection $h : A \to B$. Consequently $A \preceq B$ and $B \preceq A$ imply
> $A \approx B$, so $\preceq$ is a (partial) order on cardinalities.

## Key idea

Any bijection assembled from the data must be $f$ on some part $C \subseteq A$ and $g^{-1}$ on the rest
$A \setminus C$ (each map repairs exactly the other's defect: $f$ is total but not onto, $g^{-1}$ is onto but only
partial, defined on $g[B]$). Requiring the two image-pieces to tile $B$ with no overlap and no gap forces the cut
to satisfy a self-referential equation, $C = A \setminus g[\,B \setminus f[C]\,]$. The operator on the right is
**monotone** (its two set-complements cancel two order-reversals), so a canonical fixed point exists and is
pinned down by $f, g$ alone — no Axiom of Choice, only excluded middle. That fixed point is exactly **Dedekind's
chain** of the $g$-preimage-less seed $A \setminus g[B]$ under $g \circ f$: the elements of $A$ whose backward
$f,g$-trace originates in $A$, which is precisely where $f$ must be used.

## Construction and proof

Define $m : \wp(A) \to \wp(A)$ by $m(X) = A \setminus g[\,B \setminus f[X]\,]$.

**$m$ is monotone.** If $X \subseteq Y$: $f[X] \subseteq f[Y] \Rightarrow B\setminus f[X] \supseteq B\setminus f[Y]
\Rightarrow g[B\setminus f[X]] \supseteq g[B\setminus f[Y]] \Rightarrow m(X) \subseteq m(Y)$.

**Least fixed point.** Let $C = \bigcap\{X \subseteq A : m(X) \subseteq X\}$. For every such $X$, $C \subseteq X$,
so $m(C) \subseteq m(X) \subseteq X$; intersecting, $m(C) \subseteq C$. Then $m(C)\subseteq C$ gives
$m(m(C))\subseteq m(C)$, so $C \subseteq m(C)$. Hence $m(C) = C$. Equivalently $C$ is the chain of $A\setminus g[B]$
under $\psi = g\circ f$ and satisfies the **structural identity**
$$ C = (A\setminus g[B]) \cup g[f[C]], \qquad\text{i.e.}\qquad A\setminus C = g[\,B\setminus f[C]\,]. \tag{$\dagger$}$$

**The bijection.**
$$ h(x) = \begin{cases} f(x), & x \in C,\\ g^{-1}(x), & x \in A\setminus C. \end{cases} $$

- *Total / well defined.* On $C$, $f$ applies. On $A\setminus C$, by $(\dagger)$ each $x = g(b)$ with
  $b\in B\setminus f[C]$, so $x\in g[B]$ and $g^{-1}(x)$ is the unique preimage ($g$ injective).
- *Injective.* $f(x)=f(x')$ on $C\Rightarrow x=x'$; $g^{-1}(x)=g^{-1}(x')$ off $C\Rightarrow x=x'$ (apply $g$).
  Cross case $f(x)=g^{-1}(x')$ with $x\in C,\,x'\notin C$: then $x'=g(f(x))\in g[f[C]]\subseteq C$ by $(\dagger)$,
  contradicting $x'\notin C$.
- *Surjective.* For $b\in B$: if $g(b)\in A\setminus C$ then $h(g(b))=g^{-1}(g(b))=b$. If $g(b)\in C$, then
  $g(b)\in g[B]$ cannot lie in $A\setminus C = g[B\setminus f[C]]$, so injectivity of $g$ forces $b\in f[C]$, say
  $b=f(x)$ with $x\in C$, and $h(x)=b$.

Thus $h$ is a bijection $A \to B$. $\blacksquare$

## The orbit picture (why the cut is right)

Take $A, B$ disjoint and draw an arrow $x \to f(x)$ for $x\in A$ and $y \to g(y)$ for $y\in B$. Every vertex has
**out-degree $1$** ($f,g$ total) and **in-degree $\le 1$** ($f,g$ injective). Such a digraph splits into disjoint
components, each a **cycle**, a **doubly-infinite path**, an **$A$-started path** (backward trace stops at
$a_0\in A\setminus g[B]$), or a **$B$-started path** (stops at $b_0\in B\setminus f[A]$); vertices alternate
$A,B,A,B,\dots$.

| Component type | Forced map | Reason |
|---|---|---|
| Cycle / doubly-infinite path | either $f$ or $g^{-1}$ | both pair the $A$- and $B$-vertices perfectly |
| $A$-started path | $f$ | origin $a_0$ has no $g$-preimage, so $g^{-1}(a_0)$ is undefined |
| $B$-started path | $g^{-1}$ | $f$ misses the origin $b_0$ (no $f$-preimage), so $f$ isn't onto here |

The set $C$ above is exactly the union of the $A$-vertices on $A$-started paths (least fixed point puts cycles and
doubly-infinite paths on the $g^{-1}$ side; the greatest fixed point puts them on the $f$ side — the two canonical
bijections, differing only on the "either-works" components). The classification is computed from $f, g$ alone, so
the construction is **uniquely determined**: no Axiom of Choice. The only nonconstructive step is deciding which
type each component is (a possibly non-terminating backward search), which uses **excluded middle**.

## Worked witnesses

- $A=B=\mathbb N$, $f(n)=n+1$, $g=\mathrm{id}$: seed $A\setminus g[B]=\emptyset$, so $C=\emptyset$, $h=g^{-1}=\mathrm{id}$.
- $A=B=\mathbb N$, $f=\mathrm{id}$, $g(n)=n+1$: seed $\{0\}$, chain under $\psi(n)=n+1$ is all of $\mathbb N$, so
  $C=\mathbb N$, $h=f=\mathrm{id}$. The origin $0$ (no $g$-preimage) drags $\mathbb N$ onto the $f$-side.
- $[0,1]\preceq[0,1)$ via $x\mapsto x/2$ and $[0,1)\preceq[0,1]$ via inclusion: the construction shifts the dyadic
  orbit of the unmatched endpoint and fixes everything else, yielding an explicit bijection.

## Equivalent forms (same $h$)

- **Dedekind's fundamental lemma.** If $h_0:a\to e$ is a bijection and $e\subseteq d\subseteq a$, then $a\approx d$
  (take $c=$ chain of $a\setminus d$ under $h_0$, set $h^*=h_0$ on $c$ and $\mathrm{id}$ off $c$). CSB follows via
  $g\circ f: A\approx g[f[A]]\subseteq g[B]\subseteq A$, then compose with $g^{-1}$.
- **Bernstein's approximation from below.** $A_0=A,\ B_0=B,\ A_n=g[B_{n-1}],\ B_n=f[A_{n-1}]$; on the disjoint
  shells $A_n^*=A_n\setminus A_{n+1}$ alternate $f$ and $g^{-1}$, and on the cores $\bar A=\bigcap A_n$ use $f$
  (since $f[\bar A]=\bar B$). The disjoint-gluing lemma assembles the same bijection.
- **Knaster–Tarski.** $C$ is the least fixed point of the monotone $m$; the greatest fixed point gives the dual
  canonical bijection. (Dedekind = least, Zermelo = greatest.)
