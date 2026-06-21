# Solving Kadison-Singer via Interlacing Families of Mixed Characteristic Polynomials

## Problem

The Kadison-Singer problem (1959): does every pure state on the abelian von
Neumann algebra of bounded diagonal operators on $\ell_2$ extend uniquely to a
pure state on $B(\ell_2)$? It is equivalent to Anderson's paving conjecture and to
Weaver's discrepancy conjecture $KS_2$, all of which reduce to a single existence
statement about sums of independent random rank-one positive semidefinite
matrices, required at a **dimension-free** scale and with an **existence** (not
high-probability) guarantee — precisely what matrix concentration, which only
gives $\|\sum_i v_iv_i^*\|\le C(\epsilon)\log n$ with high probability, cannot
provide.

## Key idea

Bound the operator norm $\|\sum_i v_iv_i^*\|$ — the largest root of the
characteristic polynomial — through the *expected* characteristic polynomial,
using three ingredients:

1. **Interlacing families.** Organize the per-outcome characteristic polynomials
   over the product of the supports into an interlacing family. Such a family
   always contains a leaf whose largest root is at most the largest root of the sum
   of the family (the expected polynomial). This converts the average into an
   existence statement.
2. **Mixed characteristic polynomials.** The expected characteristic polynomial of
   $\sum_i v_iv_i^*$ is
   $\mu[A_1,\dots,A_m](x)=\big(\prod_i(1-\partial_{z_i})\big)\det(xI+\sum_i z_iA_i)|_{z=0}$,
   $A_i=\mathbb{E}\,v_iv_i^*$. The seed $\det(xI+\sum_i z_iA_i)$ is real stable, the
   operators $1-\partial_{z_i}$ preserve real stability, and setting variables real
   preserves it — so every mixed characteristic polynomial is real-rooted, which is
   what makes the outcome family an interlacing family.
3. **A multivariate barrier argument.** Bound the largest root of
   $\mu[A_1,\dots,A_m]$ by climbing a single point up the diagonal one coordinate at
   a time, applying one $1-\partial_{z_i}$ per step, while a vector-valued barrier
   function (kept bounded below 1 via convexity) controls how far the point must
   climb.

The output is a universal constant $(1+\sqrt\epsilon)^2$, yielding Weaver's $KS_2$
with $\eta=18,\theta=2$ and Anderson paving with $r=(6/\epsilon)^4$.

## Main theorem and proof

**Theorem (existence at constant scale).** Let $v_1,\dots,v_m$ be independent
random vectors in $\mathbb{C}^d$ with finite support such that
$\sum_{i=1}^m\mathbb{E}\,v_iv_i^*=I_d$ and $\mathbb{E}\,\|v_i\|^2\le\epsilon$ for all
$i$. Then
$$\Pr\Big[\Big\|\sum_{i=1}^m v_iv_i^*\Big\|\le(1+\sqrt\epsilon)^2\Big]>0.$$

The proof is the composition of the four results below.

### 1. Interlacing families: the existence principle

A real-rooted $g$ of degree $n-1$ *interlaces* a real-rooted $f$ of degree $n$,
$f(x)=\prod(x-\beta_i)$, $g(x)=\prod(x-\alpha_i)$, if
$\beta_1\le\alpha_1\le\dots\le\alpha_{n-1}\le\beta_n$. Polynomials
$f_1,\dots,f_k$ have a *common interlacing* if one $g$ interlaces all of them.

**Lemma 1.** If $f_1,\dots,f_k$ are real-rooted of equal degree with positive
leading coefficients and have a common interlacing, then some $f_i$ has largest
root $\le$ the largest root of $f_\emptyset=\sum_i f_i$.

*Proof.* Let $\alpha_{n-1}$ be the largest root of the interlacer $g$. Each $f_i$
has positive leading coefficient (positive at $+\infty$) and exactly one root
$\ge\alpha_{n-1}$, so $f_i(\alpha_{n-1})\le0$; hence $f_\emptyset(\alpha_{n-1})\le0$
and $f_\emptyset$ has a largest root $\beta_n\ge\alpha_{n-1}$. Since
$\sum_i f_i(\beta_n)=0$, some $f_i$ has $f_i(\beta_n)\ge0$; as that $f_i$ is $\le0$
at $\alpha_{n-1}$ with only one root past $\alpha_{n-1}$, its largest root lies in
$[\alpha_{n-1},\beta_n]$. $\square$

An *interlacing family* indexed by $S_1\times\cdots\times S_m$ is a set of
real-rooted leaf polynomials with positive leading coefficients such that for every
partial assignment $s_1,\dots,s_k$ ($k<m$) the children
$\{f_{s_1,\dots,s_k,t}\}_{t\in S_{k+1}}$ have a common interlacing (here
$f_{s_1,\dots,s_k}:=\sum_{\text{rest}}f_{s_1,\dots,s_m}$).

**Theorem 1 (interlacing-family principle).** An interlacing family contains a leaf
$f_{s_1,\dots,s_m}$ whose largest root is at most the largest root of $f_\emptyset$.

*Proof.* By Lemma 1 the children over $S_1$ have a common interlacing summing to
$f_\emptyset$, so some $f_{s_1}$ has largest root $\le$ that of $f_\emptyset$.
Inductively, at any $s_1,\dots,s_k$ the children over $S_{k+1}$ have a common
interlacing summing to $f_{s_1,\dots,s_k}$, so some $f_{s_1,\dots,s_{k+1}}$ has
largest root $\le$ that of $f_{s_1,\dots,s_k}$. Descend to a leaf. $\square$

Common interlacing is verified through real-rootedness: $f_1,\dots,f_k$ of equal
degree with positive leading coefficients have a common interlacing **iff** every
convex combination $\sum_i\lambda_i f_i$ is real-rooted (Dedieu; Fell;
Chudnovsky-Seymour).

### 2. The mixed characteristic polynomial and its real-rootedness

**Lemma 2 (one rank-one update is a differential operator).** For a square matrix
$A$ and a random vector $v$,
$$\mathbb{E}\,\det(A-vv^*)=(1-\partial_t)\det(A+t\,\mathbb{E}\,vv^*)\big|_{t=0}.$$
*Proof.* For invertible $A$, the matrix determinant lemma gives
$\mathbb{E}\,\det(A-vv^*)=\det(A)-\det(A)\,\mathrm{Tr}(A^{-1}\mathbb{E}\,vv^*)$;
Jacobi's formula gives $(1-\partial_t)\det(A+t\mathbb{E}\,vv^*)|_{t=0}=\det(A)-\det(A)\,\mathrm{Tr}(A^{-1}\mathbb{E}\,vv^*)$.
Both sides are polynomial in the entries of $A$; extend from invertible $A$ by
continuity. $\square$

**Theorem 2 (expected characteristic polynomial).** For independent finite-support
$v_1,\dots,v_m$ with $A_i=\mathbb{E}\,v_iv_i^*$,
$$\mathbb{E}\,\chi\Big[\sum_{i=1}^m v_iv_i^*\Big](x)=\Big(\prod_{i=1}^m(1-\partial_{z_i})\Big)\det\Big(xI+\sum_{i=1}^m z_iA_i\Big)\Big|_{z=0}=:\mu[A_1,\dots,A_m](x).$$
*Proof.* Induct on $k$: $\mathbb{E}\,\det(M-\sum_{i\le k}v_iv_i^*)=\big(\prod_{i\le k}(1-\partial_{z_i})\big)\det(M+\sum_{i\le k}z_iA_i)|_{z=0}$.
Peel off $v_k$ with Lemma 2 (with $M-\sum_{i<k}v_iv_i^*$ as the matrix), use
linearity to move $1-\partial_{z_k}$ and the evaluation outside the remaining
expectation, then apply the hypothesis with $M+z_kA_k$. Set $M=xI$, $k=m$. $\square$

A polynomial $p\in\mathbb{C}[z_1,\dots,z_m]$ is *real stable* if it has real
coefficients and no zero with all $\mathrm{Im}(z_i)>0$; univariate real stable =
real-rooted.

**Theorem 3 (real-rootedness).** If $A_1,\dots,A_m\succeq0$ then
$\mu[A_1,\dots,A_m](x)$ is real-rooted.

*Proof.* (i) $\det(xI+\sum_i z_iA_i)$ is real stable: if all variables have
positive imaginary part then $\mathrm{Im}(xI+\sum_i z_iA_i)\succ0$, so the matrix is
nonsingular. (ii) Each $1-\partial_{z_i}$ preserves real stability: fix the other
variables to numbers with positive imaginary part; the univariate restriction
$q(z)=c\prod_j(z-w_j)$ has all $\mathrm{Im}\,w_j\le0$, and
$q-q'=q\big(1-\sum_j\frac1{z-w_j}\big)$ is nonzero for $\mathrm{Im}\,z>0$ because
each $\frac1{z-w_j}$ has negative imaginary part. (iii) Setting all $z_i=0$
preserves real stability. The result is univariate, hence real-rooted. $\square$

**Theorem 4 (the outcomes are an interlacing family).** Let $v_i$ take values
$w_{i,1},\dots,w_{i,l_i}$ with probabilities $p_{i,1},\dots,p_{i,l_i}$ and set
$q_{j_1,\dots,j_m}=\big(\prod_i p_{i,j_i}\big)\chi[\sum_i w_{i,j_i}w_{i,j_i}^*](x)$.
These form an interlacing family.

*Proof.* At a node $j_1,\dots,j_k$, a convex combination $\sum_t\lambda_t
q_{j_1,\dots,j_k,t}$ equals (up to the fixed constant $\prod_{i\le k}p_{i,j_i}$) the
mixed characteristic polynomial obtained by letting coordinate $k+1$ be a random
vector $u_{k+1}$ taking $w_{k+1,t}$ with probability $\lambda_t$ — real-rooted by
Theorem 3. So every convex combination is real-rooted, giving a common interlacing
at every node. $\square$

### 3. The multivariate barrier bound

**Theorem 5 (root bound).** If $A_1,\dots,A_m\succeq0$ with $\sum_i A_i=I$ and
$\mathrm{Tr}(A_i)\le\epsilon$, then the largest root of $\mu[A_1,\dots,A_m]$ is at
most $(1+\sqrt\epsilon)^2$.

Write $\mu[A_1,\dots,A_m](x)=Q(x,\dots,x)$ where
$Q(y)=\big(\prod_i(1-\partial_{y_i})\big)P(y)$ and $P(y)=\det(\sum_i y_iA_i)$ (the
shift $y_i=z_i+x$ uses $\sum_i A_i=I$). Say $z\in\mathbb{R}^m$ is *above the roots*
of $p$ if $p(z+t)>0$ for all $t\ge0$ coordinatewise. For real stable $p$ and $z$
above its roots, define the barrier
$$\Phi^i_p(z)=\frac{\partial_{z_i}p(z)}{p(z)}=\partial_{z_i}\log p(z)=\sum_j\frac1{z_i-\lambda_j},$$
$\lambda_j$ the roots of the univariate restriction in coordinate $i$.

**Lemma 5a (monotone, convex).** Above the roots, $\partial_{z_j}\Phi^i_p\le0$ and
$\partial^2_{z_j}\Phi^i_p\ge0$ for all $i,j$.
*Proof.* For $i=j$, $\partial^2_{z_i}\frac1{z_i-\lambda_j}=\frac2{(z_i-\lambda_j)^3}>0$.
For $i\ne j$, the bivariate restriction is $\det(z_iB_i+z_jB_j+C)$ with
$B_i,B_j\succeq0$, $C$ Hermitian (Helton-Vinnikov), $M=z_iB_i+z_jB_j+C\succ0$ above
the roots; by Jacobi $\Phi^i_p=\mathrm{Tr}(M^{-1}B_i)$, so
$\partial_{z_j}\Phi^i_p=-\mathrm{Tr}(B_jM^{-1}B_iM^{-1})\le0$ and
$\partial^2_{z_j}\Phi^i_p=\mathrm{Tr}\big((B_jM^{-1}B_j)(M^{-1}B_iM^{-1})\big)\ge0$,
both traces of products of PSD matrices. $\square$

**Lemma 5b (above roots).** If $z$ above roots of $p$ and $\Phi^i_p(z)<1$, then $z$
is above roots of $p-\partial_{z_i}p$. *Proof.* For $t\ge0$, monotonicity gives
$\Phi^i_p(z+t)<1$, i.e. $\partial_{z_i}p(z+t)<p(z+t)$. $\square$

**Lemma 5c (barrier stays bounded).** If $z$ above roots of $p$ and
$\Phi^j_p(z)\le1-\frac1\delta$ ($\delta>0$), then
$\Phi^i_{p-\partial_{z_j}p}(z+\delta e_j)\le\Phi^i_p(z)$ for all $i$.
*Proof.* Using $\partial_i\Phi^j_p=\partial_j\Phi^i_p$,
$\Phi^i_{p-\partial_j p}=\Phi^i_p-\frac{\partial_j\Phi^i_p}{1-\Phi^j_p}$.
The claim is
$-\frac{\partial_j\Phi^i_p(z+\delta e_j)}{1-\Phi^j_p(z+\delta e_j)}\le\Phi^i_p(z)-\Phi^i_p(z+\delta e_j)$.
Convexity gives the right side $\ge\delta(-\partial_j\Phi^i_p(z+\delta e_j))$, so it
suffices that $\frac1{1-\Phi^j_p(z+\delta e_j)}\le\delta$ (dividing by
$-\partial_j\Phi^i_p\ge0$); monotonicity gives
$\Phi^j_p(z+\delta e_j)\le\Phi^j_p(z)$, so the hypothesis $\Phi^j_p(z)\le1-\frac1\delta$
suffices. $\square$

*Proof of Theorem 5.* Set $t=\sqrt\epsilon+\epsilon$. Then
$\det(\sum_i tA_i)=\det(tI)=t^d>0$, so $t\mathbf1$ is above the roots of $P$. By
Jacobi, $\Phi^i_P(t\mathbf1)=\mathrm{Tr}((tI)^{-1}A_i)=\mathrm{Tr}(A_i)/t\le\epsilon/t=:\phi$,
and $\delta:=\frac1{1-\phi}=1+\sqrt\epsilon$. Let
$P_k=\big(\prod_{i\le k}(1-\partial_{y_i})\big)P$ ($P_m=Q$) and let $x^k$ be
$t+\delta$ in the first $k$ coordinates, $t$ elsewhere. By Lemmas 5b and 5c,
inductively $x^k$ is above the roots of $P_k$ with all $\Phi^i_{P_k}(x^k)\le\phi$.
Thus $(t+\delta)\mathbf1$ is above the roots of $Q$, so the largest root of
$\mu(x)=Q(x,\dots,x)$ is at most $t+\delta=(\sqrt\epsilon+\epsilon)+(1+\sqrt\epsilon)=(1+\sqrt\epsilon)^2$.
$\square$ ($t=\epsilon+\sqrt\epsilon$ minimizes $t+\frac t{t-\epsilon}$.)

### 4. Assembling the main theorem

$A_i=\mathbb{E}\,v_iv_i^*$ satisfy $\mathrm{Tr}(A_i)=\mathbb{E}\,\|v_i\|^2\le\epsilon$
and $\sum_i A_i=I$. By Theorem 2 the expected characteristic polynomial is
$\mu[A_1,\dots,A_m]$, whose largest root is $\le(1+\sqrt\epsilon)^2$ (Theorem 5).
The outcome polynomials form an interlacing family (Theorem 4), so by Theorem 1
some outcome $\sum_i w_{i,j_i}w_{i,j_i}^*$ has characteristic-polynomial largest
root — i.e. operator norm — $\le(1+\sqrt\epsilon)^2$. That outcome has positive
probability. $\blacksquare$

## Consequences (explicit universal constants)

**Corollary (partition).** If $u_1,\dots,u_m\in\mathbb{C}^d$ satisfy
$\sum_i u_iu_i^*=I$ and $\|u_i\|^2\le\delta$, then for every integer $r$ there is a
partition $S_1,\dots,S_r$ of $[m]$ with
$$\Big\|\sum_{i\in S_k}u_iu_i^*\Big\|\le\Big(\frac1{\sqrt r}+\sqrt\delta\Big)^2\quad\text{for all }k.$$
*Proof.* For $i\in[m]$, $k\in[r]$ let $w_{i,k}\in\mathbb{C}^{rd}$ be $u_i$ in the
$k$-th block, $0$ elsewhere; let $v_i=\sqrt r\,w_{i,k}$ with probability $1/r$. Then
$\sum_i\mathbb{E}\,v_iv_i^*=I_{rd}$, $\|v_i\|^2=r\|u_i\|^2\le r\delta$, and the main
theorem with $\epsilon=r\delta$ gives an assignment (= partition
$S_k=\{i:v_i=\sqrt r\,w_{i,k}\}$) with the block-diagonal norm $\le(1+\sqrt{r\delta})^2$;
the $k$-th block contributes $r\big\|\sum_{i\in S_k}u_iu_i^*\big\|$, so dividing by
$r$ gives the bound. $\square$

**Weaver's $KS_2$.** With $r=2,\delta=1/18$: each part is
$\le(1/\sqrt2+1/(3\sqrt2))^2=16/18=8/9<1$. In Weaver's normalization
($u_i=w_i/\sqrt\eta$, $\eta=18$, $\delta=1/\eta$), multiplying back by $\eta=18$
gives $\eta-\theta=16$, i.e. **$\eta=18,\theta=2$**. Since $KS_2$ implies
Akemann-Anderson projection paving, which implies Kadison-Singer, **the
Kadison-Singer problem has a positive answer.**

**Anderson paving (explicit).** By the Casazza-Edidin-Kalra-Paulsen reduction it
suffices to $(r,\frac{1+\epsilon}2)$-pave every $2n\times2n$ projection $Q$ with
diagonal $1/2$. Such $Q$ is the Gram matrix of vectors $u_i$ with $\|u_i\|^2=1/2$;
the partition corollary with $\delta=1/2$ gives
$\|P_kQP_k\|\le(1/\sqrt r+1/\sqrt2)^2<1/2+3/\sqrt r\le\frac{1+\epsilon}2$ for
$r=36/\epsilon^2$. Propagating through the reduction yields: **for every
$\epsilon>0$, every zero-diagonal complex self-adjoint matrix can be
$(r,\epsilon)$-paved with $r=(6/\epsilon)^4$** — a constant independent of the
matrix size.

## Why each design choice

- **Characteristic polynomials, not matrices:** they satisfy multilinear
  identities that make averages computable and admit real-stability/interlacing
  tools; concentration on matrices only reaches the wrong $\log n$ scale.
- **Expected characteristic polynomial (first moment), not moments/Stieltjes:** we
  need existence (nonzero probability), and a root of the average can dominate a
  root of some summand — concentration cannot certify a rare good outcome.
- **Interlacing family:** roots of a generic average are unrelated to summands;
  interlacing is exactly the structure under which "some summand's largest root
  $\le$ the average's" holds, lifted up the product tree.
- **Mixed characteristic polynomial via $\prod(1-\partial_{z_i})$:** a random
  rank-one isotropic update *is* the operator $1-\partial$ on the characteristic
  polynomial (Lemma 2); keeping a separate variable per non-commuting covariance
  lets the multivariate real-stability theory apply where the univariate
  $(1-cD)$-on-one-polynomial picture (valid only for identically-distributed
  isotropic vectors) fails.
- **Real stability + $\det(\sum z_iA_i)$:** the natural multilinear real stable
  seed for PSD $A_i$; its closure under $1-\partial_{z_i}$ and under setting
  variables real delivers real-rootedness of every mixed characteristic polynomial.
- **Vector barrier, climb with shift $\delta$:** $\Phi^i<1$ alone keeps a point
  above the roots after one operator but the barriers creep up; bounding $\Phi^j$
  below $1$ by $1/\delta$ and stepping coordinate $j$ up by $\delta$ uses convexity
  to keep every barrier non-increasing, so the $m$-step induction closes.
- **$t=\sqrt\epsilon+\epsilon$, $\delta=1+\sqrt\epsilon$:** minimizes the final
  climb $t+\delta$, giving the tight $(1+\sqrt\epsilon)^2$ that matches the
  associated-Laguerre / matching-polynomial sharpness benchmarks.
