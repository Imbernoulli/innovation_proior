# Context: large sum-free sets, and counting sum-free sets

## Research question

A set $A$ of integers is *sum-free* if there is no solution to $x+y=z$ with $x,y,z\in A$; the variables need not be distinct, so $2x=z$ is forbidden. Two basic questions sit next to one another.

1. **Extremal / Ramsey-type.** Given a set $A$ of $n$ nonzero integers, how large a sum-free subset must it contain? Write $s(A)$ for the largest size and $f(n)=\min_{|A|=n}s(A)$. The leading constant is the central issue, and the lower-order surplus over that constant is the hard part.

2. **Enumerative.** How many sum-free subsets does $[N]=\{1,\dots,N\}$ have? There are obvious families of size about $2^{N/2}$: all subsets of the odd numbers, and all subsets of the strict upper half $\{\lfloor N/2\rfloor+1,\dots,N\}$. The question is whether these essentially account for the count, and how the parity-dependent constant appears.

A solution to the extremal question has to keep a positive fraction of an arbitrary additive set without inspecting all triples one by one. A solution to the counting question has to turn the local prohibition $x+y=z$ into global structural control; otherwise the number of possible constrained subsets is too large to count directly.

## Background

**The middle-third arc is sum-free.** Work on $\mathbb{T}=\mathbb{R}/\mathbb{Z}$. The open arc $B=(1/3,2/3)$ satisfies $(B+B)\cap B=\emptyset$: if $u,v\in(1/3,2/3)$, then $u+v\in(2/3,4/3)$, which modulo $1$ lies in $(2/3,1)\cup(0,1/3)$. The half-open arc $[1/3,2/3)$ has the same measure and is also sum-free, since the endpoint $2/3$ is excluded. Among symmetric arcs $(\alpha,1-\alpha)$, self-sum-avoidance is exactly the condition $\alpha\ge 1/3$, so the densest symmetric arc of this kind has measure $1/3$.

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

**Residues modulo 3.** The function $f$ is built from thirds. Multiples of $3$ are unchanged by $x\mapsto x+1/3$, while elements coprime to $3$ cycle through the three third-translates. This makes it natural to split $A=A_0\sqcup A_1$, where $A_0$ consists of elements coprime to $3$ and $A_1$ consists of multiples of $3$.

**Additive triples and almost-sum-free sets.** For counting, exact sum-freeness is too rigid for a small covering family. The useful relaxation is an *almost sum-free* set, one with $o(N^2)$ triples $(x,y,z)$ satisfying $x+y=z$. Exact sum-free sets have no triples, and almost sum-free supersets can still be forced into the two large shapes that drive the enumeration.

**Fourier analysis on $\mathbb{Z}/p\mathbb{Z}$.** Embed $[N]$ into $\mathbb{Z}/p\mathbb{Z}$ for a prime $p\in[2N,4N]$. Since sums of two elements of $[N]$ are $<2N\le p$, there is no wraparound. On this group,
$$\widehat A(r)=\sum_x A(x)e(rx/p),\qquad \sum_r|\widehat A(r)|^2=p|A|,$$
and large Fourier coefficients are sparse by Parseval.

**Prior structural facts.** Freiman and Deshouillers-Freiman-Sos-Temkin showed that very dense sum-free subsets of $[N]$ are essentially interval-like or residue-class-like. Freiman also proved that the number of sum-free subsets of $[N]$ of size at least $5N/12+2$ is $O(2^{N/2})$. These results are for dense, exactly sum-free sets; the counting problem needs comparable control for almost sum-free containers.

**Known counting scale.** Alon, Calkin, and Erdos-Granville independently proved $|\mathrm{SF}(N)|=2^{N/2+o(N)}$. This fixes the exponential scale but leaves a factor $2^{o(N)}$, much too large for the Cameron-Erdos constant.

**Container line.** Green's Fourier granularization and Sapozhenko's independent combinatorial container approach put the counting problem into the same broad shape: cover every exact sum-free set by one of only $2^{o(N)}$ almost-sum-free supersets, then prove that any large superset with few triples is forced into the interval/parity alternatives.

## Baselines

**Erdos's averaging bound and the strict Alon-Kleitman gain.** Choose $\theta\in\mathbb{T}$ uniformly and set
$$A_\theta=\{a\in A:\{\theta a\}\in(1/3,2/3)\}.$$
For every $\theta$, this set is sum-free because it is the preimage of a sum-free arc under an additive map. Equidistribution gives
$$\mathbb{E}_\theta |A_\theta|=\sum_{a\in A}\Pr[\{\theta a\}\in(1/3,2/3)]=\frac n3,$$
so some $\theta$ has $|A_\theta|\ge n/3$. Using the half-open arc gives $|A_\theta|=n/3+f_A(\theta)$ exactly. The average is still $n/3$, while $\theta=0$ keeps no elements, so the integer-valued function $|A_\theta|$ is not constant and its maximum is strictly larger than $n/3$. Therefore
$$s(A)\ge \lfloor n/3\rfloor+1=\left\lceil\frac{n+1}{3}\right\rceil\ge\frac{n+1}{3}.$$
The gap is that this is a one-step integrality gain.

**Bourgain, $s(A)\ge(n+2)/3$ and the $L^1$ surplus.** For $A$ with coprime positive elements, the Bourgain theorem says that either $A=\{1,2\}$ or
$$s(A)\ge\frac{n+2}{3},$$
and the reusable Fourier function is
$$F_A(x)=\sum_{a\in A}\sum_{m\ge1}\frac{\chi(m)}m\cos(2\pi ma x),\qquad f_A=-(\sqrt3/\pi)F_A.$$
Since $F_A$ has mean zero, $\max(-F_A)\ge\|F_A\|_1/2$, so $L^1$ mass gives surplus over $n/3$. A Mobius-sifting step isolates the first harmonic $c_A(x)=\sum_{a\in A}\cos(2\pi ax)$ at a logarithmic cost, giving
$$s(A)\ge\frac n3+c\frac{\|c_A\|_{L^1(\mathbb{T})}}{\log n}.$$
This gives a surplus for sets with large Littlewood norm; the structured case with small norm needs separate arithmetic input.

**The test-function viewpoint.** To certify $\max_x f_A(x)\ge t$, it suffices to find $\varphi\ge0$ with $\int\varphi=1$ and $\int\varphi f_A\ge t$. Nonnegative trigonometric polynomials built from factors $1-\cos(2\pi ux)$ give such weights, and the value of $\int\varphi f_A$ is computed from the coefficients $-\frac{\sqrt3}{\pi}\chi(m)/m$.

**Finite certificates after mod-3 descent.** The descent for a fixed surplus $S/3$ reduces the proof to sizes $N_S<n\le 3N_S+2S$. For $S=2$, the binding residue is $n\equiv2\pmod3$, and the finite cases are $n=5$ and $n=8$. At $n=5$, the hard shape has $A_1=\{v,2v\}$ and then $1,2\in A$; the polynomial $1-\frac43\cos(2\pi x)+\frac23\cos(4\pi x)$ and its dilation by the remaining coprime element give an integral $>1/3$. At $n=8$, the hard shape again has $A_1=\{v,2v\}$ with known elements $1,2,u,2u$; after the balance check at $x=1/(6v)$, the product $(1-\cos 2\pi x)(1-\cos 2\pi vx)$ gives the final certificate.

**Counting baselines.** Every subset of the odd numbers is sum-free, giving $2^{\lceil N/2\rceil}$ examples. Every subset of the strict upper half $\{\lfloor N/2\rfloor+1,\dots,N\}$ is also sum-free. Cameron-Erdos counted the sum-free subsets of $\{\lceil(N+1)/3\rceil,\dots,N\}$ as $\sim c(N)2^{N/2}$, with $c(N)$ depending on the parity of $N$. The upper-bound baseline $2^{N/2+o(N)}$ leaves the main enumerative gap.

## Evaluation settings

For the extremal problem, the natural tests are worst-case sets $A$ of size $n$: intervals, residue classes, and structured sets such as $\{u,2u,v,2v,\dots\}$. The key metric is the integer value of $s(A)$, or equivalently the discrete surplus $m_A$ with $|A_x|=n/3+f_A(x)$. Once the Alon-Kleitman bound is known, the residue class $n\equiv2\pmod3$ is the binding case for the $(n+2)/3$ theorem; the other two residue classes round up from the previous bound.

For the counting problem, the yardstick is $|\mathrm{SF}(N)|$ compared with $2^{N/2}$, together with the parity of $N$. The proof has to separate small containers of size at most $(1/2-1/120)N$ from large almost-sum-free containers, then verify that large containers are interval-like or almost all odd. Small-$N$ exhaustive enumeration is a useful code check.

## Code framework

The objects fixed before any selector is chosen are few and concrete. A *candidate set* is a finite collection $A$ of nonzero integers, and the *sum-free predicate* on a subset $S\subseteq A$ is the assertion that no triple $x,y,z\in S$ (repetitions allowed, so including $2x=z$) satisfies $x+y=z$; checking it is a direct scan over pairs. From these two primitives the extremal question asks only for the largest $S$ the predicate admits, and the enumerative question asks only for the number of $S$ that satisfy it. Brute-force enumeration over all subsets of $[N]$ for small $N$ gives exact counts to test any later formula against, and the three explicit baseline families — all subsets of the odd numbers, all subsets of the strict upper half, and the Cameron-Erdos interval $\{\lceil (N+1)/3\rceil,\dots,N\}$ — are constructible outright and supply the $2^{N/2}$ scale to beat.

The available machinery is the averaging idea together with harmonic analysis. The averaging idea is to replace the search for one good subset by a *membership criterion that depends on a random parameter*: each element of $A$ is admitted or rejected according to where some parameter-dependent quantity lands, and one averages the resulting size (or the resulting count of triples) over the parameter, so that a single good value is guaranteed without examining the parameter space exhaustively. The harmonic tools stand ready to evaluate such averages: Fourier expansion on $\mathbb{T}$ with the nonprincipal character modulo $3$ for the extremal side, Fourier analysis on $\mathbb{Z}/p\mathbb{Z}$ for a prime $p\in[2N,4N]$ on the counting side, and the test-function / nonnegative-trigonometric-polynomial device for turning a mean-zero surplus into a pointwise lower bound. These are the load-bearing instruments, not yet aimed at a particular criterion.

What is left open is exactly the choice of criterion and the certificate it must produce. The first empty slot is the *random selector itself*: which parameter is averaged over, what region the parameter-dependent quantity is tested against, and how an element's eligibility is read off — none of this is fixed in advance, and in particular whether the right device is a dilation of $A$ landing in some self-sum-avoiding region of $\mathbb{T}$ is left undetermined here. The second empty slot is the *averaging certificate* on the extremal side: a nonnegative weight, of mean one, whose pairing against the relevant mean-zero function provably exceeds the integrality-improved threshold, witnessing a surplus over $n/3$. The third empty slot is the *counting certificate*: the structural guarantee that turns the local prohibition $x+y=z$ into a small family of almost-sum-free containers, each forced toward the interval or all-odd shape, tight enough to recover the parity-dependent constant rather than a $2^{o(N)}$ slack. Filling these slots — naming the selector, the weight, and the container certificate — is the content of a solution, and is deliberately left unspecified above.
