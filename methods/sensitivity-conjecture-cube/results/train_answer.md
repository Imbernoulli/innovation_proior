For a Boolean function $f:\{0,1\}^n\to\{0,1\}$, almost every complexity measure we care about — block sensitivity $bs(f)$, certificate complexity, decision-tree depth, the degree $\deg(f)$ of the unique multilinear polynomial representing $f$, randomized and quantum query complexity, approximate degree — sits inside a single equivalence class: each is bounded by a fixed power of any other. Sensitivity $s(f)$, the largest number of single-coordinate flips at any one input that change the value, is the one measure that refused to join. The trivial direction $s(f)\le bs(f)$ holds because a single flipped bit is a block of size one; the reverse was the whole problem, and after three decades the best known bound on block sensitivity in terms of sensitivity was exponential, $bs(f)=O\!\big(e^{s(f)}\sqrt{s(f)}\big)$. The difficulty is an asymmetry: $bs$ is a global quantity that can gather evidence from $k$ disjoint blocks scattered across the input, while $s$ is local, looking only at one point and its $n$ immediate neighbors, so we are trying to bound something global by something local that can be much smaller. We want an absolute constant $C$ with $bs(f)\le s(f)^C$.

Attacking $s$ versus $bs$ head-on is where everyone had been stuck, so the move is to use what has already been converted into a more tractable form. Gotsman and Linial turned "$s$ versus $\deg$" into a pure question about the hypercube graph $Q^n$. Identify $f$ with a $\pm1$ coloring of the vertices of $Q^n$ (edges between vectors at Hamming distance one), and twist it by the parity function $\chi_{[n]}(x)=(-1)^{\sum_i x_i}$, which flips sign across every edge because $Q^n$ is bipartite. The product $g=f\cdot\chi_{[n]}$ then satisfies $s(g,x)=n-s(f,x)$, and in Fourier terms $\widehat g(S)=\widehat f([n]\setminus S)$, so $\mathbb E[g]=\widehat g(\varnothing)=\widehat f([n])$, which is nonzero exactly when $\deg(f)=n$. Low sensitivity of $f$ means $s(g,x)=n-s(f,x)$ is large everywhere, so every vertex agrees with almost all its neighbors under $g$ — that is, in its own monochromatic induced subgraph it has high degree. Writing $\Gamma(H)=\max\{\Delta(H),\Delta(Q^n-H)\}$, the exact equivalence is: for every induced $H$ with $|V(H)|\ne 2^{n-1}$ we have $\Gamma(H)\ge h(n)$ if and only if $s(f)\ge h(\deg(f))$ for every $f$. So the entire question becomes one about the cube: how large must the maximum degree of an induced subgraph of $Q^n$ on more than half the vertices be?

The target is fixed. Chung, Füredi, Graham, and Seymour built a $(2^{n-1}+1)$-vertex induced subgraph of maximum degree exactly $\lceil\sqrt n\rceil$, so no proof can exceed $\sqrt n$. Their own lower bound, by counting, only reached $\big(\tfrac12-o(1)\big)\log_2 n$. The reason counting tops out at a logarithm is structural: their engine says a subgraph of average degree $\bar d$ has at least $2^{\bar d}$ vertices, so inverting it extracts only a logarithm of the size as a degree. Closing the gap from $\log n$ to $\sqrt n$ needs an object that sees the whole subgraph at once rather than walking edge by edge — and eigenvalues are exactly that handle.

I propose to attack the cube statement spectrally, through a signed adjacency matrix of the hypercube whose square is $nI$. The starting point is the elementary inequality $\Delta(H)\ge\lambda_1(H)$: if $A\vec v=\lambda_1\vec v$ and $|v_i|$ is the largest coordinate, then $|\lambda_1||v_i|=\big|\sum_{j\sim i}A_{ij}v_j\big|\le\Delta(H)|v_i|$. Crucially this derivation uses only that $A$ is symmetric, supported on the edges, and has entries bounded by $1$ in absolute value — it never demands $0/1$ entries. That freedom to choose signs is the lever. To transfer a large eigenvalue from the whole cube down to every large induced subgraph, the tool is Cauchy interlacing: an $m\times m$ principal submatrix $B$ of an $N\times N$ symmetric $A$ has $\lambda_i(A)\ge\mu_i(B)\ge\lambda_{i+N-m}(A)$. An induced subgraph $H$ on $m=2^{n-1}+1$ vertices is precisely a principal submatrix, so $\lambda_1(A_H)\ge\lambda_{2^{n-1}}(A)$, and everything reduces to the $2^{n-1}$-th largest eigenvalue of the cube's matrix.

With the plain $0/1$ adjacency matrix this fails: its eigenvalues are the integers $n-2k$ with multiplicity $\binom{n}{k}$, and the multiplicities pile up near the middle where the eigenvalue is $\approx 0$, so the halfway-index eigenvalue $\lambda_{2^{n-1}}$ is $0$ for even $n$ and $o(n)$ in general — interlacing yields nothing near $\sqrt n$. The fix is to reshape the spectrum by signing the edges. The ideal is a matrix with only two distinct eigenvalues, $+\theta$ and $-\theta$, each of multiplicity $2^{n-1}$, for then $\lambda_{2^{n-1}}=+\theta$ is as large as it could possibly be. A matrix with eigenvalues only $\pm\theta$ satisfies $A^2=\theta^2 I$, so I seek a symmetric $\{-1,0,1\}$ matrix supported on the edges of $Q^n$ with $A^2=\theta^2 I$; the trace then does the bookkeeping, since $\operatorname{Tr}(A)=0$ (zero diagonal) forces the two multiplicities equal at $2^{n-1}$. The value of $\theta$ is forced by the diagonal of $A^2$: $(A^2)_{ii}=\sum_j A_{ij}^2=n$ because each vertex has $n$ neighbors and $(\pm1)^2=1$, so necessarily $A^2=nI$ and $\theta=\sqrt n$ — exactly the number the CFGS construction sits at.

The only remaining demand is that the off-diagonal of $A^2$ vanish. For $i\ne j$ at Hamming distance $2$ there are exactly two common neighbors, giving two length-two paths that together with $i,j$ form a $4$-cycle of the cube; $(A^2)_{ij}$ is the sum of the two signed paths, two terms each $\pm1$, which cancel precisely when the product of the four edge-signs around that square is $-1$. So the condition is purely local: sign the edges so that every $4$-cycle of $Q^n$ carries an odd number of $-1$ edges. This is realized by a block recursion. Splitting $Q^n$ into the two subcubes $x_n=0$ and $x_n=1$ joined by the direction-$n$ matching, take the matching block to be $I$ and demand the cross term vanish: with diagonal blocks $A_{n-1}$ and $B$, the square has off-diagonal block $A_{n-1}+B$, which dies exactly when $B=-A_{n-1}$. The recursion is therefore
$$A_1=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad A_n=\begin{bmatrix}A_{n-1}&I\\ I&-A_{n-1}\end{bmatrix},$$
and one computes $A_n^2=\begin{bmatrix}A_{n-1}^2+I&0\\0&A_{n-1}^2+I\end{bmatrix}$, so $A_{n-1}^2=(n-1)I$ gives $A_n^2=nI$, with base case $A_1^2=I$. Replacing every $-1$ by $+1$ recovers the ordinary adjacency matrix of $Q^n$, confirming $A_n$ is genuinely a signing of the cube; the $-A_{n-1}$ block is what flips an odd number of signs on every square straddling the two subcubes. The blocks behave like anticommuting square-roots of the identity, which is why squaring the sum of directions gives $\sum(\pm1)^2 I=nI$ with no cross terms.

What makes the argument close is the off-by-one at the interlacing index, and it is exactly tight. The eigenvalues of $A_n$ are $\sqrt n$ ($2^{n-1}$ times) then $-\sqrt n$ ($2^{n-1}$ times); with $m=2^{n-1}+1$ the lower interlacing index is $1+2^n-(2^{n-1}+1)=2^{n-1}$, landing on the last $+\sqrt n$, so $\lambda_1(A_H)\ge\sqrt n$. One fewer vertex would push the index to $2^{n-1}+1$, the first $-\sqrt n$, collapsing the bound — which is precisely why the threshold is "more than half," and why half itself (the even vertices, an independent set with $\Delta=0$) forces nothing. The bound jumps from $0$ to $\sqrt n$ the instant we cross the halfway line. Tightness against the CFGS construction makes $\sqrt n$ best possible for perfect squares; the eigenvalue form $\lambda_1(H)\ge\sqrt n$, obtained because the all-positive adjacency entrywise dominates $|A_H|$, is sharp for every $n$ via the star $K_{1,n}$ (all even vertices plus one odd vertex), which has $\lambda_1(K_{1,n})=\sqrt n$.

Backing out through the bridge with $h(n)=\sqrt n$ turns $\Gamma(H)\ge\sqrt n$ into $s(f)\ge\sqrt{\deg(f)}$, i.e. $\deg(f)\le s(f)^2$, tight for the AND-of-ORs function on $m^2$ variables ($\deg=m^2$, $s=m$). Composing with the known quadratic tie $bs(f)\le\deg(f)^2$ gives $bs(f)\le\deg(f)^2\le(s(f)^2)^2=s(f)^4$, so the Sensitivity Conjecture holds with $C=4$ and sensitivity finally joins the polynomial cluster.

The complete result, exactly as the chain produces it:

```
Main Theorem. For every integer n >= 1, every (2^{n-1}+1)-vertex induced subgraph H
of Q^n has
        Delta(H) >= sqrt(n).
This is tight when n is a perfect square (the Chung–Füredi–Graham–Seymour construction
gives a (2^{n-1}+1)-vertex induced subgraph of maximum degree ceil(sqrt(n))).

Corollaries. Via the Gotsman–Linial equivalence and the known bs(f) <= deg(f)^2:
        s(f) >= sqrt(deg(f))   (equivalently deg(f) <= s(f)^2),   and   bs(f) <= s(f)^4.
The first is tight for the AND-of-ORs function AND_{i=1}^m OR_{j=1}^m x_{ij} on m^2
variables (deg = m^2, s = m). The Sensitivity Conjecture holds with C = 4.


Lemma 1 (eigenvalue lower bound on the maximum degree).
Let H be an m-vertex graph and A a symmetric matrix with entries in {-1,0,1}, rows and
columns indexed by V(H), with A_{uv} = 0 whenever u,v are non-adjacent in H. Then
Delta(H) >= lambda_1 := lambda_1(A).

Proof. Let v be an eigenvector for lambda_1, so A v = lambda_1 v, and let i be a
coordinate with |v_i| maximal. Then
        |lambda_1| |v_i| = | sum_j A_{ij} v_j | = | sum_{j ~ i} A_{ij} v_j |
                         <= sum_{j ~ i} |A_{ij}| |v_i| <= Delta(H) |v_i|,
using A_{ij} = 0 for j not ~ i, |v_j| <= |v_i|, and |A_{ij}| <= 1. Dividing by
|v_i| > 0 gives |lambda_1| <= Delta(H).  QED


Lemma 2 (a signed matrix of the cube with A_n^2 = n I).
Define symmetric matrices recursively by
        A_1 = [ 0 1 ; 1 0 ],     A_n = [ A_{n-1}  I ; I  -A_{n-1} ].
Then A_n is 2^n x 2^n, A_n^2 = n I, and its eigenvalues are sqrt(n) and -sqrt(n), each
with multiplicity 2^{n-1}.

Proof. Induct on n. For n = 1, A_1^2 = I. If A_{n-1}^2 = (n-1) I, then
        A_n^2 = [ A_{n-1}^2 + I,  A_{n-1} - A_{n-1} ; A_{n-1} - A_{n-1},  I + A_{n-1}^2 ]
              = [ A_{n-1}^2 + I,  0 ; 0,  A_{n-1}^2 + I ] = n I.
Hence every eigenvalue satisfies lambda^2 = n, so lambda = +- sqrt(n). Since Tr(A_n) = 0,
the eigenvalues sum to zero, so +sqrt(n) and -sqrt(n) occur with equal multiplicity
2^{n-1}.  QED

Why this is a signing of Q^n. The nonzero entries of A_n are exactly the edges of Q^n:
the two A_{n-1} blocks sit on the two (n-1)-subcubes (x_n = 0 and x_n = 1), and the two I
blocks are the perfect matching joining them. Replacing every -1 by +1 recovers the
ordinary 0/1 adjacency matrix of Q^n. (Geometrically, (A_n^2)_{ii} = n counts the n closed
length-2 walks at i, each contributing (+-1)^2 = 1; for i != j at distance 2, (A_n^2)_{ij}
is the sum of the two signed length-2 paths around their common 4-cycle, which cancel
because the recursion puts an odd number of -1 edges on every 4-cycle.)


Lemma 3 (Cauchy's interlace theorem).
Let A be symmetric n x n with eigenvalues lambda_1 >= ... >= lambda_n, and let B be an
m x m principal submatrix with eigenvalues mu_1 >= ... >= mu_m. Then for all 1 <= i <= m,
        lambda_i >= mu_i >= lambda_{i+n-m}.

Proof. It suffices to remove one row/column at a time and iterate. Permuting rows and
columns, write A = [ B  c ; c^T  d ] with B obtained by deleting the last row and column.
By linearity of the determinant in the last column,
        det[ B - xI,  c ; c^T,  d - x + alpha ]
          = det[ B - xI,  c ; c^T,  d - x ] + det[ B - xI,  c ; 0,  alpha ]
          = det(A - xI) + alpha det(B - xI).
The left-hand side is (up to sign) the characteristic polynomial of a symmetric matrix
for every real alpha, hence has all real roots; so det(A - xI) and det(B - xI) interlace,
giving the inequalities for the single-deletion step. Stacking n - m such steps yields the
general statement.  QED


Proof of the Main Theorem. Let H be a (2^{n-1}+1)-vertex induced subgraph of Q^n and let
A_H be the principal submatrix of A_n on V(H). By Lemma 2, A_H is symmetric, {-1,0,1}-valued,
and vanishes on non-edges of H, so Lemma 1 gives
        Delta(H) >= lambda_1(A_H).
By Lemma 2 the eigenvalues of A_n are sqrt(n) (2^{n-1} times) then -sqrt(n) (2^{n-1} times).
Applying Lemma 3 with n -> 2^n, m = 2^{n-1}+1, i = 1:
        lambda_1(A_H) >= lambda_{1 + 2^n - (2^{n-1}+1)}(A_n) = lambda_{2^{n-1}}(A_n) = sqrt(n).
Combining, Delta(H) >= sqrt(n). Tightness when n is a perfect square follows from the CFGS
construction.  QED

Strengthening (eigenvalue form). The proof in fact yields lambda_1(H) >= lambda_1(A_H)
>= sqrt(n), where lambda_1(H) is the largest eigenvalue of the ordinary 0/1 adjacency
matrix of H (since the all-positive adjacency entrywise dominates |A_H|, its Perron
eigenvalue is at least lambda_1(A_H)). As Delta(H) >= lambda_1(H) always, this is stronger,
and lambda_1(H) >= sqrt(n) is best possible for every n: taking all 2^{n-1} even vertices
plus one odd vertex induces a star K_{1,n} together with isolated vertices, with
lambda_1(K_{1,n}) = sqrt(n).

Deriving the corollaries. One of H, Q^n - H has >= 2^{n-1}+1 vertices and Delta is
monotone, so Gamma(H) = max{Delta(H), Delta(Q^n - H)} >= sqrt(n) for every induced
subgraph with |V(H)| != 2^{n-1}. The Gotsman–Linial equivalence with h(n) = sqrt(n)
converts this into s(f) >= sqrt(deg(f)), i.e. deg(f) <= s(f)^2. With the known
bs(f) <= deg(f)^2,
        bs(f) <= deg(f)^2 <= (s(f)^2)^2 = s(f)^4,
proving the Sensitivity Conjecture with C = 4.  QED
```
