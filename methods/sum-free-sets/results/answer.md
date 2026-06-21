# Large sum-free sets in sets of integers, and counting sum-free sets

## Problem

A set is *sum-free* if it has no solution to $x+y=z$. The two targets are:

1. For every set $A$ of $n$ nonzero integers, lower-bound the largest sum-free subset size $s(A)$.
2. Count the sum-free subsets of $[N]=\{1,\dots,N\}$.

## Method

**Random dilation.** The arc $B=(1/3,2/3)\subset\mathbb{T}$ is sum-free, and for a random $\theta$ each nonzero integer $a$ has $\{\theta a\}$ uniformly distributed. Thus
$$A_\theta=\{a\in A:\{\theta a\}\in B\}$$
is sum-free for every $\theta$ and $\mathbb{E}|A_\theta|=n/3$, giving $s(A)\ge n/3$. With the half-open arc $[1/3,2/3)$, the same averaging is nonconstant because $\theta=0$ selects nothing, so some dilation selects strictly more than $n/3$ elements.

**Integrality.** With $f=1_{[1/3,2/3)}-\tfrac13$ and $f_A(x)=\sum_{a\in A}f(ax)$, one has $|A_x|=n/3+f_A(x)$. Since $|A_x|$ is integer-valued, has average $n/3$, and is not constant, its maximum is at least $\lfloor n/3\rfloor+1=\lceil(n+1)/3\rceil$. Hence $s(A)\ge(n+1)/3$.

**Fourier surplus.** The exact expansion is
$$f(x)=-\frac{\sqrt3}{\pi}\sum_{m\ge1}\frac{\chi(m)}m\cos(2\pi m x),$$
where $\chi$ is the nonprincipal character mod $3$. If
$$F_A(x)=\sum_{a\in A}\sum_{m\ge1}\frac{\chi(m)}m\cos(2\pi ma x),$$
then $f_A=-(\sqrt3/\pi)F_A$, so the one-sided maximum of $f_A$ is controlled by the $L^1$ mass of $F_A$. Mobius sifting of the higher harmonics gives the plain-cosine corollary
$$s(A)\ge\frac n3+c\frac{\left\|\sum_{a\in A}\cos(2\pi a\,\cdot)\right\|_{L^1(\mathbb{T})}}{\log n}.$$
For structured sets, nonnegative test functions certify $m_A=\max f_A$: if $\varphi\ge0$ and $\int\varphi=1$, then $m_A\ge\int\varphi f_A$. In the case $1\notin A$, with $u=\min A$ and $v$ the smallest element not divisible by $u$,
$$\varphi=(1-\cos 2\pi ux)(1-\cos 2\pi vx)$$
gives
$$\int\varphi f_A=\frac{\sqrt3}{2\pi}\left(2-\frac12 1_A(u+v)\right)>1/3.$$

**Mod-3 descent.** Split $A=A_0\sqcup A_1$ into elements coprime to $3$ and divisible by $3$. Then
$$f_A(x)+f_A(x+1/3)+f_A(x+2/3)=3f_{A_1}(x),$$
so $m_A\ge m_{A_1}$. Also
$$m_A\ge\frac{|A_0|}{6}-\frac{|A_1|}{3}.$$
This reduces the fixed-surplus conjecture $m_A\ge S/3$ to finitely many sizes. For $S=2$, the remaining $n=5,8$ cases are handled by explicit nonnegative trigonometric certificates: for $n=5$, first
$$1-\frac43\cos(2\pi x)+\frac23\cos(4\pi x),$$
then the same polynomial at the remaining coprime frequency; for $n=8$, after reducing to $A_1=\{v,2v\}$ and known elements $1,2,u,2u$, the product $(1-\cos 2\pi x)(1-\cos 2\pi vx)$ gives a value $>1/3$. Therefore, for coprime positive $A$, either $A=\{1,2\}$ or
$$s(A)\ge\frac{n+2}{3}.$$

**Counting.** The container strategy embeds $[N]$ into $\mathbb{Z}/p\mathbb{Z}$, partitions into arithmetic progressions, and chooses a good common difference $d$ satisfying
$$\left\|\frac{dr}{p}\right\|\le\frac1{4L}\left(\frac{\delta p}{|\widehat A(r)|}\right)^{1/2}$$
on every large Fourier coefficient. The granularization covers all but $\epsilon p$ points of a sum-free set and has at most $\epsilon p^2$ triples; all such containers form a family of size $2^{o(N)}$. A popular-difference/Kneser structure theorem says every large almost-sum-free container is essentially contained in a short interval or is essentially all odd. Pairing arguments discard the exceptional choices, leaving only entirely odd sets and sets inside $\{\lceil(N+1)/3\rceil,\dots,N\}$. Cameron-Erdos count the latter, giving
$$|\mathrm{SF}(N)|\sim c(N)2^{N/2},$$
with $c(N)$ depending only on the parity of $N$.

