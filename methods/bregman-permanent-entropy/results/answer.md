# The Bregman–Minc inequality, via Radhakrishnan's entropy proof

## Statement

Let $A=(a_{ij})$ be an $n\times n$ matrix with entries in $\{0,1\}$ and row sums $r_i=\sum_{j=1}^n a_{ij}$. If some $r_i=0$, then $\operatorname{per}(A)=0$ and the claim is trivial. Otherwise, with all $r_i>0$, its permanent
$$\operatorname{per}(A)=\sum_{\sigma\in S_n}\prod_{i=1}^n a_{i\sigma(i)}$$
satisfies
$$\boxed{\;\operatorname{per}(A)\;\le\;\prod_{i=1}^{n}(r_i!)^{1/r_i}.\;}$$

Equivalently, if $G$ is the bipartite graph with rows $\{1,\dots,n\}$ on one side, columns $\{1,\dots,n\}$ on the other, and an edge $i\!-\!j$ iff $a_{ij}=1$ (so row $i$ has degree $r_i$), then the number of perfect matchings of $G$ is at most $\prod_i (r_i!)^{1/r_i}$. The bound is tight: when $d$ divides $n$, a disjoint union of $n/d$ complete bipartite graphs $K_{d,d}$ attains it, with $(d!)^{n/d}$ perfect matchings.

## Key idea

For a $0/1$ matrix, $\operatorname{per}(A)$ counts the permutations $\sigma$ with $a_{i\sigma(i)}=1$ for all $i$ — i.e. the perfect matchings of $G$. In the nonempty case, put the uniform distribution on the set $\mathcal{M}$ of perfect matchings, so $|\mathcal M|=2^{H}$ where $H$ is the Shannon entropy of a uniformly random matching. Decompose $H$ by the chain rule along a *random* reveal order: averaging over orders, the number of still-available choices for each row becomes uniform on $\{1,\dots,r_i\}$, and the average of $\log$ over that uniform law is $\frac1{r_i}\log(r_i!)$. Summing and exponentiating gives the bound.

## Entropy toolbox

All logarithms are base $2$; $0\log0:=0$. For a finite-valued random variable $X$, $H(X)=-\sum_x p(x)\log p(x)$.

1. **Maximality of the uniform.** If $X$ takes $m$ values with positive probability, then $H(X)\le\log m$, with equality iff $X$ is uniform. *Proof.* $H(X)=\sum_x p(x)\log\frac1{p(x)}\le\log\big(\sum_x p(x)\cdot\frac1{p(x)}\big)=\log m$ by Jensen, as $\log$ is concave. In particular, for $X$ uniform on a finite set $C$, $\;|C|=2^{H(X)}$. The same bound holds conditionally: $H(X\mid E)\le\log|\mathrm{range}(X\mid E)|$.

2. **Chain rule (exact).** $H(X,Y)=H(X)+H(Y\mid X)$, where $H(Y\mid X)=\sum_x p(x)\,H(Y\mid X=x)$. *Proof.* $H(X,Y)-H(X)=-\sum_{x,y}p(x)p(y\mid x)[\log p(x)+\log p(y\mid x)]+\sum_x p(x)\log p(x)$; the $\log p(x)$ terms cancel (using $\sum_y p(y\mid x)=1$), leaving $-\sum_x p(x)\sum_y p(y\mid x)\log p(y\mid x)=H(Y\mid X)$. Iterating,
$$H(X_1,\dots,X_n)=\sum_{k=1}^n H\big(X_k\mid X_1,\dots,X_{k-1}\big).$$

3. **Dropping conditioning.** $H(X\mid Y)\le H(X)$. *Proof.* $H(X\mid Y)=\sum_x p(x)\sum_y p(y\mid x)\log\frac1{p(x\mid y)}\le\sum_x p(x)\log\big(\sum_y p(y\mid x)\frac{1}{p(x\mid y)}\big)=\sum_x p(x)\log\frac1{p(x)}=H(X)$, using $p(y)p(x\mid y)=p(x)p(y\mid x)$ and Jensen. (Subadditivity $H(X_1,\dots,X_n)\le\sum_k H(X_k)$ then follows from the chain rule.)

## Proof

If $\mathcal M$ is empty, then $\operatorname{per}(A)=0$ and there is nothing to prove. Otherwise view a perfect matching as a bijection $f:[n]\to[n]$ with $a_{i,f(i)}=1$, and let $f$ be **uniform** on $\mathcal M$. By maximality of the uniform, $H(f)=\log|\mathcal M|=\log\operatorname{per}(A)$, so it suffices to show
$$H(f)\;\le\;\sum_{k=1}^n \frac{1}{r_k}\log(r_k!).$$

Subadditivity alone gives only $H(f)\le\sum_k H(f(k))\le\sum_k\log r_k$ (each $f(k)$ has $\le r_k$ values), i.e. the trivial bound $\operatorname{per}(A)\le\prod_k r_k$; it discards the dependence between rows. The chain rule keeps it, but for a fixed reveal order the number of already-used neighbors of row $k$ is uncontrolled. **Fix this by revealing the rows in a uniformly random order $\tau\in S_n$.** For every fixed $\tau$, the chain rule gives
$$H(f)=\sum_{k=1}^n H\big(f(k)\mid \{f(\tau_\ell):\ell<\tau^{-1}_k\}\big),\qquad \tau^{-1}_k=\text{position of }k\text{ in }\tau.$$
Averaging over the uniform $\tau$ (the left side is unchanged):
$$H(f)=\sum_{k=1}^n \frac{1}{n!}\sum_{\tau\in S_n} H\big(f(k)\mid k\text{'s predecessors under }\tau\big).$$

Fix $k$. For fixed $\tau,f$, let $N_k(\tau,f)$ be the number of neighbors of row $k$ not yet used when $k$ is revealed. Once the previous revealed values are fixed, every possible value of $f(k)$ lies among those live neighbors, so the conditional range size is at most $N_k$. Then $1\le N_k\le r_k$, and by conditional maximality of the uniform,
$$H\big(f(k)\mid k\text{'s predecessors under }\tau\big)\le \mathbb{E}_f\big[\log N_k(\tau,f)\big].$$
Hence $H(f)\le \sum_{k=1}^n \mathbb{E}_{\tau,f}\big[\log N_k(\tau,f)\big]$, with $\tau,f$ independent and uniform.

**Distribution of $N_k$.** Fix any matching $f$. A neighbor $i$ of $k$ is already used at $k$'s turn iff its owner $f^{-1}(i)$ precedes $k$ in $\tau$. So $N_k$ is determined by the position of $k$ among the $r_k$ rows $\{k\}\cup\{f^{-1}(i): i\sim k,\,i\ne f(k)\}$ — the owners of $k$'s $r_k$ neighbors (one of which is $k$ itself). Under uniform $\tau$, $k$ is equally likely (probability $1/r_k$) to be in each relative position $m=1,\dots,r_k$ among these rows; in position $m$, exactly $m-1$ neighbors are used, so $N_k=r_k-m+1$. Thus, for *every* $f$, $N_k(\tau,f)$ is **uniform on $\{1,\dots,r_k\}$**, and
$$\mathbb{E}_\tau[\log N_k]=\frac{1}{r_k}\sum_{i=1}^{r_k}\log i=\frac{1}{r_k}\log(r_k!).$$

Summing over $k$,
$$H(f)\le\sum_{k=1}^n \frac{1}{r_k}\log(r_k!),$$
and exponentiating base $2$,
$$\operatorname{per}(A)=|\mathcal M|=2^{H(f)}\le \prod_{k=1}^n (r_k!)^{1/r_k}. \qquad\blacksquare$$

## A small worked check

For the disjoint union of $n/d$ copies of $K_{d,d}$, with $d$ dividing $n$: each $K_{d,d}$ has $d!$ perfect matchings, so $\operatorname{per}(A)=(d!)^{n/d}$, while $\prod_{k=1}^n (d!)^{1/d}=(d!)^{n/d}$. The inequality is an equality on these block-diagonal all-ones matrices.
