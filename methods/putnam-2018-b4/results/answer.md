# A Fibonacci–cosine substitution for a quadratic recurrence

## Statement

Given a real number $a$, define $x_0=1$, $x_1=x_2=a$, and
$$x_{n+1}=2x_nx_{n-1}-x_{n-2}\qquad(n\ge2).$$
If $x_n=0$ for some $n$, then the sequence $\{x_n\}$ is periodic.

## Key idea

Once a zero is possible, $|a|\le1$, so write $a=\cos b$. Then $x_n=\cos(F_n b)$, where $F_n$ is the Fibonacci sequence ($F_0=0$, $F_1=F_2=1$, $F_{n+1}=F_n+F_{n-1}$). A vanishing term forces $b$ to be a rational multiple of $2\pi$, after which periodicity follows because the Fibonacci numbers are periodic modulo a fixed integer.

## Proof

**Step 1: the case $|a|>1$ admits no zero.** We show by induction that $|x_{n+1}|\ge|x_n|$ and $|x_n|\ge1$ for all $n$. For $n=0,1$ this holds since $|x_0|=1$ and $|x_1|=|x_2|=|a|>1$, giving $|x_2|\ge|x_1|\ge|x_0|=1$. Assume $|x_k|\ge|x_{k-1}|\ge\cdots\ge|x_0|=1$ for $k\le n$. Then, using $|x_{n-2}|\le|x_n|$ and $|x_{n-1}|\ge1$,
$$|x_{n+1}|=|2x_nx_{n-1}-x_{n-2}|\ge 2|x_n||x_{n-1}|-|x_{n-2}|\ge|x_n|\bigl(2|x_{n-1}|-1\bigr)\ge|x_n|.$$
Hence $|x_n|\ge1$ for all $n$ and no term is $0$. Therefore a zero can occur only if $|a|\le1$.

**Step 2: the cosine formula.** Suppose $|a|\le1$ and choose $b\in[0,\pi]$ with $a=\cos b$. We prove $x_n=\cos(F_n b)$ for all $n\ge0$ by induction. It holds for $n=0,1,2$:
$$\cos(F_0 b)=\cos0=1=x_0,\quad \cos(F_1 b)=\cos(F_2 b)=\cos b=a.$$
Assume $x_k=\cos(F_k b)$ for $k\le n$. By the product-to-sum identity $2\cos\alpha\cos\beta=\cos(\alpha+\beta)+\cos(\alpha-\beta)$, together with $F_n+F_{n-1}=F_{n+1}$ and $F_n-F_{n-1}=F_{n-2}$,
$$2x_nx_{n-1}=2\cos(F_n b)\cos(F_{n-1}b)=\cos(F_{n+1}b)+\cos(F_{n-2}b),$$
so
$$x_{n+1}=2x_nx_{n-1}-x_{n-2}=\cos(F_{n+1}b)+\cos(F_{n-2}b)-\cos(F_{n-2}b)=\cos(F_{n+1}b).$$

**Step 3: a zero makes the angle rational.** Suppose $x_n=0$ for some $n$. Since $x_0=1\ne0$ we have $n\ge1$, so $F_n\ge1$. From $\cos(F_n b)=0$,
$$F_n b=\frac{k\pi}{2}\quad\text{for some odd integer }k.$$
Put $c=k$ and $d=4F_n$ (integers, $d\ge4$). Then
$$b=\frac{k\pi}{2F_n}=\frac{c}{d}\,2\pi,\qquad\text{so}\qquad x_m=\cos(F_m b)=\cos\!\Big(\frac{F_m c}{d}\,2\pi\Big).$$
Thus $x_m$ depends only on the residue $F_m\bmod d$.

**Step 4: Fibonacci numbers are periodic modulo $d$.** Consider the pairs $(F_m,F_{m+1})\in(\mathbb{Z}/d\mathbb{Z})^2$. This set is finite, so there exist $n_1<n_2$ with $(F_{n_1},F_{n_1+1})\equiv(F_{n_2},F_{n_2+1})\pmod d$. The forward step $(u,v)\mapsto(v,u+v)$ is a bijection of $(\mathbb{Z}/d\mathbb{Z})^2$ (its inverse is $(v,w)\mapsto(w-v,v)$). Applying the inverse $n_1$ times gives, with $\ell=n_2-n_1$,
$$(F_0,F_1)\equiv(F_\ell,F_{\ell+1})\pmod d,$$
and advancing forward,
$$(F_m,F_{m+1})\equiv(F_{m+\ell},F_{m+\ell+1})\pmod d\quad\text{for all }m\ge0.$$
In particular $F_{m+\ell}\equiv F_m\pmod d$ for all $m\ge0$.

**Step 5: conclusion.** For each $m$, $d\mid(F_{m+\ell}-F_m)$, so $\dfrac{F_{m+\ell}c}{d}-\dfrac{F_m c}{d}\in\mathbb{Z}$. Since $\cos(\theta+2\pi\mathbb{Z})=\cos\theta$,
$$x_{m+\ell}=\cos\!\Big(\frac{F_{m+\ell}c}{d}2\pi\Big)=\cos\!\Big(\frac{F_m c}{d}2\pi\Big)=x_m\qquad(m\ge0).$$
Therefore $\{x_n\}$ is periodic with period $\ell$. $\blacksquare$

## Remark

The case split is only apparent: writing $a=\cos b$ with $b$ allowed complex unifies both regimes. If $a>1$, take $b=it$ with $a=\cosh t$; then $\cos(F_n b)=\cosh(F_n t)$. If $a<-1$, take $b=\pi+it$ with $a=-\cosh t$; then $\cos(F_n b)=(-1)^{F_n}\cosh(F_n t)$. In either subcase every term has absolute value at least $1$, so no term vanishes, consistent with Step 1. A zero forces $b$ real, which is exactly the case where the cosine values can repeat.
