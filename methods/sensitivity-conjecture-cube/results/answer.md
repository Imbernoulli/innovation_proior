# Signed adjacency matrices of the hypercube and the Sensitivity Conjecture

## Problem

Among the complexity measures of a Boolean function $f:\{0,1\}^n\to\{0,1\}$, almost all — block sensitivity $bs(f)$, certificate complexity, decision-tree depth, polynomial degree $\deg(f)$, query complexity, approximate degree — are mutually polynomially related. Sensitivity $s(f)$ (the maximum, over inputs $x$, of the number of single-bit flips that change $f$) was the lone holdout: only $s(f)\le bs(f)$ was trivial, while the best upper bound on $bs(f)$ in terms of $s(f)$ was exponential. The **Sensitivity Conjecture** asks for an absolute constant $C$ with $bs(f)\le s(f)^C$.

## Key idea

The Gotsman–Linial equivalence reduces the conjecture to a purely combinatorial statement about the hypercube graph $Q^n$ (vertices $\{0,1\}^n$, edges between vectors at Hamming distance $1$): it suffices to show that every induced subgraph on more than half the vertices has a high-degree vertex. The new ingredient is **spectral**: instead of the $0/1$ adjacency matrix of $Q^n$ — whose middle eigenvalues collapse to $0$, making eigenvalue interlacing useless — assign signs $\pm1$ to the edges so that the resulting symmetric matrix squares to $nI$. Such a matrix has only the eigenvalues $\pm\sqrt n$, each with multiplicity $2^{n-1}$, so its $2^{n-1}$-th eigenvalue is exactly $\sqrt n$. Cauchy interlacing then forces every $(2^{n-1}+1)$-vertex principal submatrix to have largest eigenvalue $\ge\sqrt n$, and the largest eigenvalue lower-bounds the maximum degree. The correct signing is the one in which **every $4$-cycle of the cube carries an odd number of $-1$ edges**, produced cleanly by a block recursion.

## The theorem

**Main Theorem.** For every integer $n\ge 1$, every $(2^{n-1}+1)$-vertex induced subgraph $H$ of $Q^n$ has
$$\Delta(H)\ \ge\ \sqrt n .$$
This is tight when $n$ is a perfect square (the construction of Chung, Füredi, Graham, and Seymour gives a $(2^{n-1}+1)$-vertex induced subgraph of maximum degree $\lceil\sqrt n\rceil$).

**Corollaries.** Via the Gotsman–Linial equivalence and the known $bs(f)\le\deg(f)^2$:
$$s(f)\ \ge\ \sqrt{\deg(f)}\qquad\text{(equivalently }\deg(f)\le s(f)^2\text{), and}\qquad bs(f)\ \le\ s(f)^4 .$$
The first is tight for the AND-of-ORs function $\bigwedge_{i=1}^m\bigvee_{j=1}^m x_{ij}$ on $m^2$ variables ($\deg=m^2$, $s=m$). The Sensitivity Conjecture holds with $C=4$.

## Proof

Throughout, $\Delta$ is maximum degree, $\lambda_1$ the largest eigenvalue of a symmetric matrix, and a *principal submatrix* deletes the same set of rows and columns.

**Lemma 1 (eigenvalue lower bound on the maximum degree).** Let $H$ be an $m$-vertex graph and $A$ a symmetric matrix with entries in $\{-1,0,1\}$, rows and columns indexed by $V(H)$, with $A_{uv}=0$ whenever $u,v$ are non-adjacent in $H$. Then $\Delta(H)\ge\lambda_1:=\lambda_1(A)$.

*Proof.* Let $\vec v$ be an eigenvector for $\lambda_1$, so $A\vec v=\lambda_1\vec v$, and let $i$ be a coordinate with $|v_i|$ maximal. Then
$$|\lambda_1||v_i|=\Big|\sum_{j}A_{ij}v_j\Big|=\Big|\sum_{j\sim i}A_{ij}v_j\Big|\le\sum_{j\sim i}|A_{ij}|\,|v_i|\le\Delta(H)\,|v_i|,$$
using $A_{ij}=0$ for $j\not\sim i$, $|v_j|\le|v_i|$, and $|A_{ij}|\le1$. Dividing by $|v_i|>0$ gives $|\lambda_1|\le\Delta(H)$. $\qquad\square$

**Lemma 2 (a signed matrix of the cube with $A_n^2=nI$).** Define symmetric matrices recursively by
$$A_1=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad A_n=\begin{bmatrix}A_{n-1}&I\\ I&-A_{n-1}\end{bmatrix}.$$
Then $A_n$ is $2^n\times2^n$, $A_n^2=nI$, and its eigenvalues are $\sqrt n$ and $-\sqrt n$, each with multiplicity $2^{n-1}$.

*Proof.* Induct on $n$. For $n=1$, $A_1^2=I$. If $A_{n-1}^2=(n-1)I$, then
$$A_n^2=\begin{bmatrix}A_{n-1}^2+I & A_{n-1}-A_{n-1}\\ A_{n-1}-A_{n-1} & I+A_{n-1}^2\end{bmatrix}=\begin{bmatrix}A_{n-1}^2+I & 0\\ 0 & A_{n-1}^2+I\end{bmatrix}=nI.$$
Hence every eigenvalue satisfies $\lambda^2=n$, so $\lambda=\pm\sqrt n$. Since $\operatorname{Tr}(A_n)=0$, the eigenvalues sum to zero, so $+\sqrt n$ and $-\sqrt n$ occur with equal multiplicity $2^{n-1}$. $\qquad\square$

*Why this is a signing of $Q^n$.* The nonzero entries of $A_n$ are exactly the edges of $Q^n$: the two $A_{n-1}$ blocks sit on the two $(n-1)$-subcubes ($x_n=0$ and $x_n=1$), and the two $I$ blocks are the perfect matching joining them. Replacing every $-1$ by $+1$ recovers the ordinary $0/1$ adjacency matrix of $Q^n$. (Geometrically, $(A_n^2)_{ii}=n$ counts the $n$ closed length-$2$ walks at $i$, each contributing $(\pm1)^2=1$; for $i\ne j$ at distance $2$, $(A_n^2)_{ij}$ is the sum of the two signed length-$2$ paths around their common $4$-cycle, which cancel because the recursion puts an odd number of $-1$ edges on every $4$-cycle.)

**Lemma 3 (Cauchy's interlace theorem).** Let $A$ be symmetric $n\times n$ with eigenvalues $\lambda_1\ge\dots\ge\lambda_n$, and let $B$ be an $m\times m$ principal submatrix with eigenvalues $\mu_1\ge\dots\ge\mu_m$. Then for all $1\le i\le m$, $\ \lambda_i\ge\mu_i\ge\lambda_{i+n-m}$.

*Proof.* It suffices to remove one row/column at a time and iterate. Permuting rows and columns, write $A=\bigl[\begin{smallmatrix}B&c\\ c^\top&d\end{smallmatrix}\bigr]$ with $B$ obtained by deleting the last row and column. By linearity of the determinant in the last column,
$$\det\begin{bmatrix}B-xI&c\\ c^\top&d-x+\alpha\end{bmatrix}=\det\begin{bmatrix}B-xI&c\\ c^\top&d-x\end{bmatrix}+\det\begin{bmatrix}B-xI&c\\ 0&\alpha\end{bmatrix}=\det(A-xI)+\alpha\det(B-xI).$$
The left-hand side is (up to sign) the characteristic polynomial of a symmetric matrix for every real $\alpha$, hence has all real roots; so $\det(A-xI)$ and $\det(B-xI)$ interlace, giving the inequalities for the single-deletion step. Stacking $n-m$ such steps yields the general statement. $\qquad\square$

**Proof of the Main Theorem.** Let $H$ be a $(2^{n-1}+1)$-vertex induced subgraph of $Q^n$ and let $A_H$ be the principal submatrix of $A_n$ on $V(H)$. By Lemma 2, $A_H$ is symmetric, $\{-1,0,1\}$-valued, and vanishes on non-edges of $H$, so Lemma 1 gives
$$\Delta(H)\ge\lambda_1(A_H).$$
By Lemma 2 the eigenvalues of $A_n$ are $\underbrace{\sqrt n,\dots,\sqrt n}_{2^{n-1}},\underbrace{-\sqrt n,\dots,-\sqrt n}_{2^{n-1}}$. Applying Lemma 3 with $n\!\to\!2^n$, $m=2^{n-1}+1$, $i=1$:
$$\lambda_1(A_H)\ \ge\ \lambda_{1+2^n-(2^{n-1}+1)}(A_n)=\lambda_{2^{n-1}}(A_n)=\sqrt n.$$
Combining, $\Delta(H)\ge\sqrt n$. Tightness when $n$ is a perfect square follows from the CFGS construction. $\qquad\square$

**Strengthening (eigenvalue form).** The proof in fact yields $\lambda_1(H)\ge\lambda_1(A_H)\ge\sqrt n$, where $\lambda_1(H)$ is the largest eigenvalue of the *ordinary* $0/1$ adjacency matrix of $H$ (since the all-positive adjacency entrywise dominates $|A_H|$, its Perron eigenvalue is at least $\lambda_1(A_H)$). As $\Delta(H)\ge\lambda_1(H)$ always, this is stronger, and $\lambda_1(H)\ge\sqrt n$ is best possible for *every* $n$: taking all $2^{n-1}$ even vertices plus one odd vertex induces a star $K_{1,n}$ together with isolated vertices, with $\lambda_1(K_{1,n})=\sqrt n$.

**Deriving the corollaries.** One of $H$, $Q^n-H$ has $\ge2^{n-1}+1$ vertices and $\Delta$ is monotone, so $\Gamma(H)=\max\{\Delta(H),\Delta(Q^n-H)\}\ge\sqrt n$ for every induced subgraph with $|V(H)|\ne2^{n-1}$. The Gotsman–Linial equivalence with $h(n)=\sqrt n$ converts this into $s(f)\ge\sqrt{\deg(f)}$, i.e. $\deg(f)\le s(f)^2$. With the known $bs(f)\le\deg(f)^2$,
$$bs(f)\le\deg(f)^2\le\bigl(s(f)^2\bigr)^2=s(f)^4,$$
proving the Sensitivity Conjecture with $C=4$. $\qquad\square$
