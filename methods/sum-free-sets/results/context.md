# Context: large sum-free sets, and counting sum-free sets

## Research question

A set $A$ of integers is *sum-free* if there is no solution to $x+y=z$ with $x,y,z\in A$; the variables need not be distinct, so $2x=z$ is forbidden. Two basic questions sit next to one another.

1. **Extremal / Ramsey-type.** Given a set $A$ of $n$ nonzero integers, how large a sum-free subset must it contain? Write $s(A)$ for the largest size and $f(n)=\min_{|A|=n}s(A)$. The leading constant is the central issue, and the lower-order surplus over that constant is the hard part.

2. **Enumerative.** How many sum-free subsets does $[N]=\{1,\dots,N\}$ have? There are obvious families of size about $2^{N/2}$: all subsets of the odd numbers, and all subsets of the strict upper half $\{\lfloor N/2\rfloor+1,\dots,N\}$. The question is whether these essentially account for the count, and how the parity-dependent constant appears.

## Background

**Sum-free regions on the circle.** Work on $\mathbb{T}=\mathbb{R}/\mathbb{Z}$. A measurable region $B\subseteq\mathbb{T}$ is itself sum-free when $(B+B)\cap B=\emptyset$, where the sumset is taken modulo $1$; symmetric arcs $(\alpha,1-\alpha)$ around $1/2$ are the simplest candidates, and the half-open and open versions of an arc carry the same measure.

**Dilations equidistribute.** For a fixed nonzero integer $a$, the map $\theta\mapsto \{\theta a\}$ pushes Haar measure on $\mathbb{T}$ to Haar measure. Hence $\Pr_\theta[\{\theta a\}\in B]=1/3$ for every nonzero $a$, with no independence assumption across different elements of $A$.

**Fourier on $\mathbb{T}$ and the character mod 3.** Let
$$f(x)=1_{[1/3,2/3)}(x)-\frac13.$$
Then $f$ has mean zero and cosine expansion
$$f(x)=-\frac{\sqrt3}{\pi}\sum_{m\ge1}\frac{\chi(m)}{m}\cos(2\pi m x),$$
where $\chi$ is the nonprincipal Dirichlet character modulo $3$:
$$\chi(1)=1,\qquad \chi(2)=-1,\qquad \chi(3k)=0.$$
The negative sign is fixed by the first coefficient,
$$2\int_{1/3}^{2/3}\cos(2\pi x)\,dx=-\frac{\sqrt3}{\pi}.$$
For $A$, write $f_A(x)=\sum_{a\in A}f(ax)$ and $m_A=\max_x f_A(x)$. Then a middle-third dilation has size $|A_x|=|A|/3+f_A(x)$.

**Residues modulo 3.** The function $f$ is built from thirds, so $f(ax)$ responds to $x\mapsto x+1/3$ differently according to whether $a$ is a multiple of $3$ or coprime to $3$: a multiple of $3$ is unchanged, while an element coprime to $3$ has its argument cycle through the three third-translates.

**Additive triples and almost-sum-free sets.** A natural quantitative relaxation of sum-freeness is to count additive triples: an *almost sum-free* set is one with $o(N^2)$ triples $(x,y,z)$ satisfying $x+y=z$. Exact sum-free sets have no triples at all and sit at one end of this scale.

**Fourier analysis on $\mathbb{Z}/p\mathbb{Z}$.** Embed $[N]$ into $\mathbb{Z}/p\mathbb{Z}$ for a prime $p\in[2N,4N]$. Since sums of two elements of $[N]$ are $<2N\le p$, there is no wraparound. On this group,
$$\widehat A(r)=\sum_x A(x)e(rx/p),\qquad \sum_r|\widehat A(r)|^2=p|A|,$$
and large Fourier coefficients are sparse by Parseval.

**Prior structural facts.** Freiman and Deshouillers-Freiman-Sos-Temkin showed that very dense sum-free subsets of $[N]$ are essentially interval-like or residue-class-like. Freiman also proved that the number of sum-free subsets of $[N]$ of size at least $5N/12+2$ is $O(2^{N/2})$. These results are for dense, exactly sum-free sets; the counting problem needs comparable control for almost sum-free containers.

**Known counting scale.** Alon, Calkin, and Erdos-Granville independently proved $|\mathrm{SF}(N)|=2^{N/2+o(N)}$. This fixes the exponential scale but leaves a factor $2^{o(N)}$, much too large for the Cameron-Erdos constant.

**Container line.** For related enumeration problems, Green's Fourier granularization and Sapozhenko's combinatorial container method are the standard tools for reducing a count over many constrained objects to a count over a smaller family of coarser supersets.

## Baselines

**Erdos's averaging bound and the strict Alon-Kleitman gain.** Choose $\theta\in\mathbb{T}$ uniformly and set
$$A_\theta=\{a\in A:\{\theta a\}\in(1/3,2/3)\}.$$
For every $\theta$, this set is sum-free because it is the preimage of a sum-free arc under an additive map. Equidistribution gives
$$\mathbb{E}_\theta |A_\theta|=\sum_{a\in A}\Pr[\{\theta a\}\in(1/3,2/3)]=\frac n3,$$
so some $\theta$ has $|A_\theta|\ge n/3$. Using the half-open arc gives $|A_\theta|=n/3+f_A(\theta)$ exactly. The average is still $n/3$, while $\theta=0$ keeps no elements, so the integer-valued function $|A_\theta|$ is not constant and its maximum is strictly larger than $n/3$. Therefore
$$s(A)\ge \lfloor n/3\rfloor+1=\left\lceil\frac{n+1}{3}\right\rceil\ge\frac{n+1}{3}.$$

**Counting baselines.** Every subset of the odd numbers is sum-free, giving $2^{\lceil N/2\rceil}$ examples. Every subset of the strict upper half $\{\lfloor N/2\rfloor+1,\dots,N\}$ is also sum-free. Cameron-Erdos counted the sum-free subsets of $\{\lceil(N+1)/3\rceil,\dots,N\}$ as $\sim c(N)2^{N/2}$, with $c(N)$ depending on the parity of $N$.

## Evaluation settings

For the extremal problem, the natural tests are worst-case sets $A$ of size $n$: intervals, residue classes, and structured sets such as $\{u,2u,v,2v,\dots\}$. The key metric is the integer value of $s(A)$, or equivalently the discrete surplus $m_A$ with $|A_x|=n/3+f_A(x)$. The residue of $n$ modulo $3$ matters, since the integer rounding in the averaging bound behaves differently across the three classes.

For the counting problem, the yardstick is $|\mathrm{SF}(N)|$ compared with $2^{N/2}$, together with the parity of $N$: the target is to pin down the constant in front of $2^{N/2}$, not merely the exponent. Small-$N$ exhaustive enumeration is a useful code check.
