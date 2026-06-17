# Counting $p$-subsets of $\{1,\dots,2p\}$ with sum divisible by $p$ — the roots-of-unity filter

## Problem

Let $p$ be an odd prime. Determine the number of $p$-element subsets $A\subseteq\{1,2,\dots,2p\}$ whose element sum is divisible by $p$.

## Answer

$$\boxed{\,N=\frac{\dbinom{2p}{p}-2}{p}+2\,}\qquad\Bigl(=\tfrac1p\bigl[\tbinom{2p}{p}+2(p-1)\bigr]\Bigr).$$

For example $p=3\Rightarrow N=8$, $p=5\Rightarrow N=52$, $p=7\Rightarrow N=492$.

## Key idea

Track subset *size* and subset *sum* simultaneously with a two-variable generating function, then use a roots-of-unity filter to keep only the size $p$, sum $\equiv 0 \pmod p$ part. Each element $k$ contributes a factor $1+x\,y^{k}$ — "$1$" omits it, "$x\,y^{k}$" includes it (one unit of $x$ for size, $y^{k}$ for its contribution to the sum) — so

$$F(x,y)=\prod_{k=1}^{2p}\bigl(1+x\,y^{k}\bigr),\qquad [x^{m}y^{s}]\,F=\#\{m\text{-subsets of sum }s\}.$$

To select $s\equiv 0\pmod p$, average $y$ over the $p$-th roots of unity $\omega=e^{2\pi i/p}$, using orthogonality $\sum_{j=0}^{p-1}\omega^{js}=p\,[\,p\mid s\,]$:

$$N=\frac1p\sum_{j=0}^{p-1}[x^{p}]\,F(x,\omega^{j}).$$

## Derivation

**Trivial root $j=0$.** Every $y^{k}\mapsto 1$, so $F(x,1)=(1+x)^{2p}$ and
$$[x^{p}]\,F(x,1)=\binom{2p}{p}.$$

**Nontrivial roots $j=1,\dots,p-1$.** Here $\omega^{j}$ is a primitive $p$-th root. As $k$ runs over $1,\dots,2p$, the residue $k\bmod p$ takes each value in $\{0,\dots,p-1\}$ **exactly twice** (the ground set has length $2p=2\cdot p$), so

$$F(x,\omega^{j})=\Biggl[\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr)\Biggr]^{2}.$$

The bracket is evaluated from $z^{p}-1=\prod_{r=0}^{p-1}(z-\omega^{r})$. Writing $1+x\omega^{r}=(-x)\bigl((-1/x)-\omega^{r}\bigr)$ and using $p$ odd so $(-x)^{p}=-x^{p}$:

$$\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr)=(-x)^{p}\Bigl[(-\tfrac1x)^{p}-1\Bigr]=1-(-x)^{p}=1+x^{p}.$$

Hence $F(x,\omega^{j})=(1+x^{p})^{2}=1+2x^{p}+x^{2p}$ and $[x^{p}]\,F(x,\omega^{j})=2$ for every nontrivial $j$ (the dependence on $j$ disappears because multiplying exponents by $j$ merely permutes the $p$-th roots).

**Assembly.**
$$N=\frac1p\Bigl[\binom{2p}{p}+(p-1)\cdot 2\Bigr]=\frac{\binom{2p}{p}+2p-2}{p}=\frac{\binom{2p}{p}-2}{p}+2.$$

## Integrality

$N$ is a count, so $p$ must divide $\binom{2p}{p}-2$. Directly,
$$
\binom{2p}{p}
=\frac{(p+1)(p+2)\cdots(2p)}{1\cdot2\cdots p}
=2\prod_{i=1}^{p-1}\frac{p+i}{i}.
$$
For $1\le i\le p-1$, $i$ is invertible mod $p$ and $p+i\equiv i\pmod p$, so each factor $(p+i)/i$ is congruent to $1$ mod $p$. Thus
$$\binom{2p}{p}\equiv2\pmod p,$$
hence $p\mid\bigl(\binom{2p}{p}-2\bigr)$ and $\dfrac{\binom{2p}{p}-2}{p}$ is an integer.

## Where the coefficient $2$ comes from

The two boundary subsets $\{1,\dots,p\}$ and $\{p+1,\dots,2p\}$ both have sums divisible by $p$: the first has sum $\tfrac{p(p+1)}{2}=p\cdot\tfrac{p+1}{2}$, and the second has sum $\tfrac{p(p+1)}{2}+p^{2}$. Algebraically, after the ground set is split into two full residue cycles, the coefficient of $x^p$ in $(1+x^p)^2$ comes from $x^p\cdot1$ or $1\cdot x^p$, matching those two full-block choices. The final count is still produced by the root average, not by a separate partition into a uniform part plus those two subsets.
