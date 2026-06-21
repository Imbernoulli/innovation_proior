# Concentration of measure: Hoeffding, Azuma–Hoeffding, McDiarmid

## Problem

For independent $X_1,\dots,X_n$, give a **finite-sample, exponentially small** upper bound on the probability that a statistic deviates from its mean — first for the sum $S=\sum_i X_i$, then for any function $f(X_1,\dots,X_n)$ that no single coordinate can swing by much. The variance-based Chebyshev bound $\sigma^2/(nt^2)$ decays only polynomially in $t$; the goal is to match the Gaussian-type tail $e^{-t^2/(2v)}$ that the CLT predicts in the limit, but with explicit constants and at fixed $n$.

## Key idea

Bound the **moment generating function**, not the probability. Markov applied to $e^{hZ}$ gives the Chernoff bound $\Pr\{Z\ge t\}\le e^{-ht}E e^{hZ}$; optimizing the free parameter $h$ produces a Gaussian-type exponent. For a bounded centered variable, convexity of the exponential pins the MGF below the Gaussian one (Hoeffding's lemma). Independence is used only to factor the MGF — and that factorization survives, one increment at a time, for any **martingale** with conditionally bounded increments (Azuma–Hoeffding). Finally, any function of independent variables becomes a martingale by revealing the coordinates one at a time (the **Doob martingale**); the bounded-differences condition forces each conditional increment into a bounded window, so the same tail applies (McDiarmid). One one-dimensional MGF bound, used three times.

## The Cramér–Chernoff method

For $\lambda>0$, Markov on $e^{\lambda Z}$ gives $\Pr\{Z\ge t\}\le e^{-\lambda t}E e^{\lambda Z}=\exp(-(\lambda t-\psi_Z(\lambda)))$, $\psi_Z(\lambda)=\log E e^{\lambda Z}$. Optimizing,
$$\Pr\{Z\ge t\}\le \exp(-\psi_Z^\*(t)),\qquad \psi_Z^\*(t)=\sup_{\lambda\ge0}\big(\lambda t-\psi_Z(\lambda)\big),$$
the Cramér transform (Fenchel–Legendre dual of $\psi_Z$). For a centered Gaussian of variance $v$, $\psi_Z(\lambda)=\lambda^2 v/2$ and $\psi_Z^\*(t)=t^2/(2v)$, so $\Pr\{Z\ge t\}\le e^{-t^2/(2v)}$. For an independent sum, $\psi_S(\lambda)=\sum_i\psi_{X_i}(\lambda)$ — the log-MGF is additive, which is why the method is tailored to sums.

## Hoeffding's lemma

If $EX=0$ and $a\le X\le b$ a.s., then for every real $h$,
$$E\,e^{hX}\le \exp\!\Big(\frac{h^2(b-a)^2}{8}\Big).$$

**Proof.** Convexity of $x\mapsto e^{hx}$ on $[a,b]$: $e^{hx}\le \frac{b-x}{b-a}e^{ha}+\frac{x-a}{b-a}e^{hb}$. Taking $E$ (with $EX=0$) gives $E e^{hX}\le e^{L(u)}$, where $p=\frac{-a}{b-a}\in[0,1]$, $u=h(b-a)$, and $L(u)=-pu+\log(1-p+pe^{u})$. Then $L(0)=0$, $L'(0)=0$, and $L''(u)=q(1-q)$ with $q=\frac{pe^{u}}{1-p+pe^{u}}\in[0,1]$, so $L''(u)\le\frac14$. Taylor: $L(u)=\frac12 L''(\xi)u^2\le u^2/8$. The boundary cases $p=0$ or $p=1$ are degenerate and satisfy the same bound. $\square$

## Hoeffding's inequality

$X_1,\dots,X_n$ independent, $X_i\in[a_i,b_i]$, $S=\sum_i(X_i-EX_i)$. For $t>0$,
$$\Pr\{S\ge t\}\le \exp\!\Big(-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\Big).$$

**Proof.** Chernoff + independence + the lemma: $\Pr\{S\ge t\}\le e^{-ht}\prod_i E e^{h(X_i-EX_i)}\le \exp\big(-ht+\frac{h^2}{8}\sum_i(b_i-a_i)^2\big)$. Minimize over $h>0$ at $h=4t/\sum_i(b_i-a_i)^2$. $\square$ For $X_i\in[0,1]$: $\Pr\{\bar X-\mu\ge t\}\le e^{-2nt^2}$ (constant matches the fair-coin Gaussian tail).

## Azuma–Hoeffding inequality

Let $Z_0,\dots,Z_n$ be a martingale with differences $d_k=Z_k-Z_{k-1}$, $E[d_k\mid\mathcal F_{k-1}]=0$, $|d_k|\le c_k$ a.s. For $t>0$,
$$\Pr\{Z_n-Z_0\ge t\}\le \exp\!\Big(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\Big),\qquad \Pr\{|Z_n-Z_0|\ge t\}\le 2\exp\!\Big(-\frac{t^2}{2\sum_k c_k^2}\Big).$$

**Proof.** Conditionally, $d_k$ is centered in an interval of length $2c_k$, so the lemma gives $E[e^{hd_k}\mid\mathcal F_{k-1}]\le e^{h^2(2c_k)^2/8}=e^{h^2c_k^2/2}$. By the tower property, peeling one increment at a time, $E e^{h(Z_n-Z_0)}\le \exp\big(\frac{h^2}{2}\sum_k c_k^2\big)$. Chernoff, minimize at $h=t/\sum_k c_k^2$. $\square$ (Equivalently, via $\cosh(hc)\le e^{h^2c^2/2}$ from $\cosh z=\sum z^{2m}/(2m)!\le\sum z^{2m}/(2^m m!)=e^{z^2/2}$.)

## McDiarmid's (bounded-differences) inequality

$X_1,\dots,X_n$ independent. $f$ satisfies the bounded-differences condition: changing only coordinate $i$ changes $f$ by at most $c_i$. With $Z=f(X_1,\dots,X_n)$, for $t>0$,
$$\Pr\{Z-EZ\ge t\}\le \exp\!\Big(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\Big),$$
and the two-sided bound carries a factor $2$.

**Proof.** Form the Doob martingale $Z_i=E[Z\mid X_1,\dots,X_i]$ ($Z_0=EZ$, $Z_n=Z$); then $\Delta_i=Z_i-Z_{i-1}$ has conditional mean zero. Bounded differences imply that, conditionally on $X_1,\dots,X_{i-1}$, $\Delta_i$ lies in an interval of width $\le c_i$ (the sup minus the inf over $X_i$ of the conditional expectation, with the remaining coordinates averaged out). The lemma applied conditionally gives $E[e^{h\Delta_i}\mid X_1,\dots,X_{i-1}]\le e^{h^2c_i^2/8}$; peeling the Doob martingale, $E e^{h(Z-EZ)}\le \exp\big(\frac{h^2}{8}\sum_i c_i^2\big)$; Chernoff and minimize at $h=4t/\sum_i c_i^2$. $\square$ Taking $f=\sum_i X_i$ with $X_i\in[a_i,b_i]$ gives $c_i=b_i-a_i$ and recovers Hoeffding's inequality.

## Causal chain

Markov $\to$ (apply to $e^{hZ}$) Chernoff, whose optimum is the Legendre dual, with the Gaussian as the target shape $\to$ (convexity of $e^{hx}$) Hoeffding's lemma: a bounded centered variable has a sub-Gaussian MGF with proxy variance $(b-a)^2/4$ $\to$ (Chernoff over an independent sum, optimize $h$) Hoeffding's inequality $\to$ (only independence-via-factorization was used; the tower property preserves it for martingale differences) Azuma–Hoeffding $\to$ (the Doob martingale turns any function of independent inputs into martingale increments, bounded differences cap each one) McDiarmid. A not-too-sensitive function of many independent variables is sharply concentrated around its mean.

