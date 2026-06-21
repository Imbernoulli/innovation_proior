Let me start from where the time actually goes. I have a tall matrix $A \in \mathbb{R}^{n\times d}$ with $n \gg d$ and rank $r$, and a family of tasks — overconstrained least squares, low-rank approximation, leverage-score estimation, $\ell_p$ regression — that all reduce to the same move: pick a random linear map $S:\mathbb{R}^n \to \mathbb{R}^t$ with $t \ll n$, form the sketch $SA$, solve the tiny problem on the sketch, and lift the answer back. This works the instant $S$ is a *subspace embedding* for the column space $C(A)$, meaning $\|SAx\|_2 = (1\pm\varepsilon)\|Ax\|_2$ for every $x$ simultaneously — equivalently $\|SUz\|_2 = (1\pm\varepsilon)\|z\|_2$ for all $z$, where $U$ is an orthonormal basis of the relevant column space. Once that holds, the running time is dominated entirely by the cost of forming $SA$. So the whole game is to build a good subspace embedding fast.

And here is the waste that drives everything. With a dense Gaussian $S$, computing $SA$ costs $\Theta(ndt)$; with the clever structured sketches — Fast-JL, the subsampled randomized Hadamard transform — it drops to $\Theta(nd\log n)$, which is essentially optimal for a *dense* $A$. But the matrices people actually feed in — document-term, adjacency, recommender, web — are wildly sparse: writing $\mathrm{nnz}(A)$ for the number of nonzeros, $\mathrm{nnz}(A)$ is often $O(n)$, not $nd$. Every one of these sketches reads all $nd$ entries even though almost all are zero, paying $nd$ to process $O(n)$ numbers. The dream is a leading term of $O(\mathrm{nnz}(A))$: touch each nonzero a constant number of times, plus a lower-order additive overhead that is polynomial in $d,k,1/\varepsilon$ and *not* polynomial in $n$.

What does it take for $S\cdot A$ to cost $O(\mathrm{nnz}(A))$? Each nonzero $A_{ij}$ contributes to $SA$ only through column $j$ of $S$; if column $j$ has $c_j$ nonzeros, $A_{ij}$ fans out to $c_j$ sketch entries, and the total work is $\sum_j c_j\cdot(\text{nonzeros of }A\text{ in column }j)$. The cheapest possible sketch makes $S$ have *exactly one nonzero per column*. Then applying $S$ is nothing but "for each nonzero $A_{ij}$, add it with a sign into a single bucket" — one add per nonzero, $O(\mathrm{nnz}(A))$ flat.

I propose the **sparse embedding matrix** $S = \Phi D$ — the CountSketch matrix used as a subspace embedding. Concretely, draw a hash $h:[n]\to[t]$ uniformly, let $\Phi \in \{0,1\}^{t\times n}$ have $\Phi_{h(i),i}=1$ and zeros elsewhere (one nonzero per column), and let $D$ be diagonal with $D_{ii}\in\{\pm1\}$ i.i.d. uniform. Then $S = \Phi D$, and $(SA)_{h(i),\cdot}$ accumulates $\pm A_{i,\cdot}$ over all rows $i$ hashing to the same bucket. The random signs are essential: without them $\|Sy\|^2 = \sum_j(\sum_{i:h(i)=j} y_i)^2$ carries off-diagonal cross terms $2\sum_{i<i',\,h(i)=h(i')} y_i y_{i'}$ that do not vanish in expectation, whereas with the signs $\|\Phi D y\|^2 = \sum_j(\sum_{i:h(i)=j} D_{ii} y_i)^2$ and the cross terms $y_i y_{i'} D_{ii} D_{i'i'}$ have mean zero since $\mathbb{E}[D_{ii}D_{i'i'}]=0$ for $i\neq i'$. Hence $\mathbb{E}[\|\Phi D y\|^2] = \sum_i y_i^2 = \|y\|^2$ — unbiased — exactly the random-sign-hash trick of Alon–Matias–Szegedy and the structure of the CountSketch data structure of Charikar–Chen–Farach-Colton.

The hard part is that this map, taken naively, *fails* the standard embedding proof, and seeing why is what reveals how to fix it. The usual recipe is: show $\|Sy\| = (1\pm\varepsilon)\|y\|$ for a fixed $y$ with failure $e^{-\Omega(r)}$, place an $\varepsilon$-net of $e^{O(r)}$ points on the unit sphere of $C(A)$, union-bound over the net, and extend by linearity. But CountSketch does not give $e^{-\Omega(r)}$ failure for a worst-case fixed vector: take $y = (e_1+e_2)/\sqrt2$; with probability $1/t$ the two heavy coordinates collide in one bucket and $\|\Phi D y\|^2$ is either $2$ or $0$ — full distortion — a failure probability of only $\sim 1/t$, polynomial, not $e^{-\Omega(r)}$. The net union bound would then demand $t \ge e^{\Omega(r)}$, which is absurd. The standard proof is dead.

The resolution is that the net points are *not* arbitrary — they live in one fixed rank-$r$ subspace, and that subspace pins down where heavy coordinates can sit. Write a unit $y \in C(A)$ as $y = Ux$ for unit $x$; then $y_i = U_{i,\cdot}\cdot x$, so by Cauchy–Schwarz $y_i^2 \le \|U_{i,\cdot}\|^2 = u_i$, the $i$-th leverage score, *independent of which $y$ I picked*. Since $\sum_i u_i = r$, calling a coordinate "heavy" when $u_i$ exceeds a threshold $T$ means there are at most $r/T$ heavy coordinates, and they form a *single fixed set* $H$ — the large leverage scores cannot roam. This is the heavy/light split from streaming norm estimation: track the few heavy coordinates exactly, handle the many light ones by random aggregation. So I order rows by non-increasing $u_i$, set $s = \min\{i : u_i \le T\}$, and split a unit $y$ into the heavy part $y^H = y_{1:(s-1)}$ (with $u_i > T$) and the light part $y^L = y_{s:n}$ (with $u_i \le T$).

The heavy part costs nothing once perfectly hashed. Let $E_B$ be the event that $h$ is injective on $\{1,\dots,s-1\}$; each pair collides with probability $1/t$, so by the birthday union bound $\Pr[\neg E_B] \le s^2/t$. On $E_B$ the distinct heavy coordinates hit distinct buckets, and $\Phi$ restricted to $H$ is a signed permutation — an exact isometry — so $\|\Phi D y^H\|^2 = \sum_{i<s} y_i^2 = \|y^H\|^2$ exactly. This is what forces $t = \Omega(s^2)$.

The light part has small $\infty$-norm, the regime where sparse hashing concentrates. First I control the bucket loads: for bucket $j$ let $X_i = u_i\,\mathbb{1}[h(i)=j,\,i\ge s] \in [0,T]$, so $\mathbb{E}[\sum X_i] \le r/t$ and $V = \|u_{s:n}\|^2/t$; Bernstein in the form "$X_i\in[0,T]$, $V \le LT^2/6 \Rightarrow \Pr[\sum X_i \ge \sum\mathbb{E}[X_i] + LT] \le e^{-L}$" with $L = \log(t/\delta_h)$, when $t \ge 6\|u_{s:n}\|^2/(LT^2)$, gives — after a union bound over the $t$ buckets — that with probability $\ge 1-\delta_h$ (call this event $E_h$) every bucket holds light squared-mass at most
$$W \equiv T\log(t/\delta_h) + r/t,$$
using $y_i^2 \le u_i$. Now $\|\Phi D y^L\|^2 = z^\top B z$ with $z = \mathrm{diag}(D)\in\{\pm1\}^n$ and $B_{ii'} = y_i y_{i'}\,\mathbb{1}[h(i)=h(i')]$ for $i,i'\ge s$, so $\mathrm{tr}(B) = \|y^L\|^2$. On $E_h$ both relevant norms of $B$ are controlled by $W$: $\|B\|_F^2 = \sum_{i\ge s} y_i^2 \sum_{i':h(i')=h(i)} y_{i'}^2 \le W$, and since $B$ is block-diagonal across buckets its eigenvectors are the per-bucket vectors with eigenvalues equal to the bucket masses, so $\|B\|_2 = \max_j(\text{bucket mass}) \le W$. With $\ell \le 1/W$, the Hanson–Wright bound $\mathbb{E}|z^\top B z - \mathrm{tr}\,B|^\ell \le (CQ)^\ell$ has $Q = \max\{\sqrt\ell\,\|B\|_F,\ \ell\|B\|_2\} \le \sqrt{\ell W}$, and Markov with $\ell = \log(1/\delta_L)$ yields
$$\bigl|\,\|\Phi D y^L\|^2 - \|y^L\|^2\,\bigr| \le K_L\sqrt{W\log(1/\delta_L)}$$
with failure $\delta_L$. (Going through the moments via Hanson–Wright in the spirit of Kane–Nelson, rather than the direct small-$\infty$-norm route of Dasgupta–Kumar–Sarlós, is what lets me reuse the single *per-bucket* mass bound $W$ from the subspace instead of re-deriving concentration for each net point — and it is what shrinks $t$ later.)

The cross term closes the same way. Expanding $\|\Phi D y\|^2 = \|\Phi D y^H\|^2 + \|\Phi D y^L\|^2 + 2\langle \Phi D y^H, \Phi D y^L\rangle$, the first term is exact on $E_B$ and the second is handled above. On $E_B$ each heavy bucket holds at most one heavy coordinate, so the cross term is a signed sum $\sum_{i\ge s} y_i D_{ii} z_i$ where $z_i$ is the colliding heavy value (or $0$). Khintchine gives $\mathbb{E}[(\cdot)^{2p}]^{1/p} \le C_p \sum_{i\ge s} y_i^2 z_i^2 = C_p\sum_{i'<s} y_{i'}^2 \sum_{i:h(i)=i'} y_i^2 \le C_p W$ on $E_h$, so $|\text{cross}| \le K_C\sqrt{W\log(1/\delta_C)}$ with failure $\delta_C$. Putting the three pieces together, on $E_h\cap E_B$,
$$\bigl|\,\|\Phi D y\|^2 - \|y\|^2\,\bigr| \le K_L\sqrt{W\log(1/\delta_L)} + 2K_C\sqrt{W\log(1/\delta_C)};$$
setting $\delta_L=\delta_C=\delta_y/2$ and forcing $W \le K_y\varepsilon^2/\log(1/\delta_y)$ with $K_y \le 1/(9(K_L+K_C)^2)$ makes the right side $\le \varepsilon$. The payoff: $\delta_y$ can be pushed to $e^{-\Omega(r)}$ by taking $\ell = \Theta(r)$ moments — and $W$ does not depend on $\delta_y$ — so the fixed-vector guarantee finally has the $e^{-\Omega(r)}$ failure the naive analysis could never reach, *recovered by using the subspace structure*.

Now the net is affordable. Take the grid net $E = \{w \in (\gamma/\sqrt r)\mathbb{Z}^r : \|w\|\le 1\}$ with $\gamma = 1-1/\sqrt2$; by Arora–Hazan–Kale, $|E| \le e^{cr}$ with $c = 1/\gamma + 2$, and $|u^\top J v| \le \varepsilon\ \forall u,v\in E$ implies $|w^\top J w| \le \varepsilon/(1-\gamma)^2$ for every unit $w$. With $J = U^\top S^\top S U - I_r$, applying the fixed-$y$ bound to $Ux$, $Uy$, $U(x+y)$ for $x,y\in E$ and expanding gives $|x^\top J y| \le \varepsilon/2$; a union bound over the pairs (failure $\delta_y K_{\mathrm{sub}}^r \le 1/10$) then gives $\|Sw\|^2 = (1\pm\varepsilon)\|w\|^2$ for all $w \in C(A)$. The net works *only* because the dimension is $r$: $e^{O(r)}$ points meet $e^{-\Omega(r)}$ per-point failure, whereas in ambient $\mathbb{R}^n$ a net would have $e^{O(n)}$ points and the argument would die.

Counting up the constraints — $t \ge s^2$ from perfect hashing, $W = O(\varepsilon^2/r)$ from the fixed-vector error (with $\log(1/\delta_y) = \Theta(r)$), the mild Bernstein condition $t = O((r/\varepsilon)^2\log t)$ since $\|u_{s:n}\|^2 \le Tr$, and $s \le r/T$ with $T = \Theta(\varepsilon^2/(r\log)) \Rightarrow s = \Theta((r/\varepsilon)^2)$ — gives
$$t = O\!\left((r/\varepsilon)^4 \log^2(r/\varepsilon)\right),$$
an $\varepsilon$-subspace embedding applied in $O(\mathrm{nnz}(A))$ time, with no polynomial dependence on $n$. The $(r/\varepsilon)^4$ comes from $t = \Omega(s^2)$, the price of perfectly hashing all $s = (r/\varepsilon)^2$ heavy coordinates. But that is wasteful: only the $\approx r$ genuinely unit leverage scores (e.g. when $A$ contains $I_d$) truly must avoid all collisions; scores of value $1/2, 1/4,\dots$ can collide a little. Partitioning the heavy scores into geometric groups $G_j = \{i : 2^{-j} < u_i \le 2^{-j+1}\}$ and bounding the damage of within-group collisions by the spectral norm $\|\hat U_j\|_2^2$ of the collided submatrix of $U$ — controlled via matrix-Bernstein, since the collision set is a random subset and the squared spectral norm of the sampled submatrix is exactly $\|\sum_m \hat H_m\|_2$ — lets smaller groups collide while paying only their (small) spectral norm, dropping the dimension to
$$t = r^2\varepsilon^{-2}\,\mathrm{polylog}(r/\varepsilon).$$
A third variant is needed for the applications that union-bound over many blocks: replacing the single $\pm1$-per-column by a small $O(\log n/\varepsilon^2)$-dimensional JL transform *inside* each bucket (hashing into $\approx d^2/\log n$ buckets so each holds $O(\log n)$ heavy coordinates) upgrades the success probability to $1 - 1/\mathrm{poly}(n)$ at time $O(\mathrm{nnz}(A)\log n/\varepsilon)$.

With the embedding in hand, every task is "embed the column space, then solve small." Least squares: embed $C([A\mid b])$, solve $\arg\min_x \|\Phi D A x - \Phi D b\|$ on the $t$-row sketch — the sketch minimizer is a $(1+\varepsilon)$-approximation — in $O(\mathrm{nnz}(A)) + \tilde O(d^3\varepsilon^{-2})$. The $\varepsilon^{-2}$ can be turned into $\log(1/\varepsilon)$ by using the embedding only as a *preconditioner*: read a change-of-basis $R$ off the sketched QR so that $AR$ has all singular values $1\pm\varepsilon_0$, then iterate the fixed point $x^{(m+1)} = x^{(m)} + R^\top A^\top(b - ARx^{(m)})$, which contracts because $AR(x^{(m+1)}-x^*) = U(\Sigma-\Sigma^3)V^\top(x^{(m)}-x^*)$ and the diagonal of $\Sigma-\Sigma^3$ is at most $\sigma_i((1+\varepsilon_0)^2-1) \le 3\varepsilon_0\sigma_i$, so with a small constant $\varepsilon_0$ each step shrinks the residual by $3\varepsilon_0 < 1$ and $O(\log(1/\varepsilon))$ steps suffice — condition-independent, since $R$ came from the embedding and not from $A$'s spectrum. Leverage scores: take $\Pi_1 = (\text{Fast-JL})\circ(\text{sparse embedding})$, QR it for $R$ so that $AR$ is orthonormal up to the embedding error, and read row norms $\|(AR)_{i,\cdot}\|^2 = (1\pm\varepsilon)u_i$ — multiplying by a tiny $O(\log n)$-width JL $\Pi_2$ reads all $n$ of them at once — in $O(\mathrm{nnz}(A)\log n + r^3\log^2 r + r^2\log n)$. Low rank uses affine embeddings (a single $S$ that is both a subspace embedding and satisfies approximate matrix multiplication, which the sparse embedding does by the Thorup–Zhang second-moment bound): right-sketch $AR^\top$ for a rank-$k$ space within $(1+\varepsilon)\Delta_k$, left affine-embed, and solve the rank-$k$ restriction by SVD inside the sketch, giving $\|A - LDW^\top\|_F \le (1+\varepsilon)\Delta_k$ in $O(\mathrm{nnz}(A)) + \tilde O(nk^2\varepsilon^{-4} + k^3\varepsilon^{-5})$. And $\ell_p$ regression rides on the high-probability generalized embedding: block the rows, embed each block to $\ell_2$, build a well-conditioned $\ell_p$ basis, sample rows by their $\ell_p$ norms — $O(\mathrm{nnz}(A)\log n) + \mathrm{poly}(r/\varepsilon)$ for any $p\in[1,\infty)$.

The concrete code is the basic embedding — one $\pm1$ per column, built as a sparse CSC matrix and applied as $S @ A$ — together with the sketch-and-solve regression on $[A\mid b]$, the preconditioned-iteration variant, and the leverage-score reader. The embedding dimension is chosen from the analysis above, $t = O((r/\varepsilon)^4\log^2(r/\varepsilon))$ for the basic construction or $t = r^2\varepsilon^{-2}\,\mathrm{polylog}(r/\varepsilon)$ for the partition-refined one; in practice far smaller $t$ already yields error ratios near $1$.

```python
import numpy as np
from scipy import sparse
from scipy import linalg


def cwt_matrix(t, n, rng):
    """Sparse embedding S = Phi D in R^{t x n}: exactly one +-1 per column.

    Column i has its single nonzero in row h(i) (the hash) with value the random sign
    D_ii. As a CSC matrix the column pointers are 0,1,...,n, so each column holds one
    entry. Applying S to A then hashes every nonzero of A into a bucket with a sign and
    adds -- O(nnz(A)) time.
    """
    rows = rng.integers(0, t, size=n)          # h : [n] -> [t]
    cols = np.arange(n + 1)                     # one nonzero per column
    signs = rng.choice([1, -1], size=n)        # diag(D)
    return sparse.csc_matrix((signs, rows, cols), shape=(t, n))


def sparse_embedding(A, t, rng):
    """Form S A in time proportional to nnz(A)."""
    S = cwt_matrix(t, A.shape[0], rng)
    return S @ A


def least_squares(A, b, t, rng):
    """(1+eps)-approximate argmin_x ||A x - b||_2 via sketch-and-solve on [A | b]."""
    b = np.asarray(b).reshape(-1, 1)
    if sparse.issparse(A):
        Ab = sparse.hstack([A, sparse.csc_matrix(b)]).tocsc()
    else:
        Ab = np.hstack([np.asarray(A), b])
    SAb = sparse_embedding(Ab, t, rng)                       # O(nnz(A)) time
    SAb = SAb.toarray() if sparse.issparse(SAb) else SAb
    SA, Sb = SAb[:, :-1], SAb[:, -1]
    x, *_ = linalg.lstsq(SA, Sb)                             # small dense least squares
    return x


def least_squares_iterative(A, b, t, rng, n_iter=40):
    """log(1/eps) variant: sketch -> well-conditioned R -> iterate the fixed point
    x <- x + R^T A^T (b - A R x), which contracts the residual by ~3*eps0 < 1 / step."""
    A = (A.toarray() if sparse.issparse(A) else np.asarray(A)).astype(float)
    b = np.asarray(b).ravel()
    SA = sparse_embedding(A, t, rng)
    SA = SA.toarray() if sparse.issparse(SA) else SA
    _, Rmat = np.linalg.qr(SA)
    Rinv = np.linalg.inv(Rmat)                               # A @ Rinv is well-conditioned
    AR = A @ Rinv
    y = np.zeros(AR.shape[1])
    for _ in range(n_iter):
        y = y + AR.T @ (b - AR @ y)
    return Rinv @ y


def leverage_scores(A, t, rng):
    """Constant-factor approx to all u_i = ||U_{i,*}||^2: embed, basis-change R so that
    (A R) is ~orthonormal, then ||(A R)_{i,*}||^2 = (1 +- eps) u_i; read row norms with a
    tiny JL Pi2 (width O(log n))."""
    A = (A.toarray() if sparse.issparse(A) else np.asarray(A)).astype(float)
    n = A.shape[0]
    SA = sparse_embedding(A, t, rng)
    SA = SA.toarray() if sparse.issparse(SA) else SA
    _, Rmat = np.linalg.qr(SA)
    R = np.linalg.inv(Rmat)
    p = max(1, int(np.ceil(8 * np.log(max(n, 2)))))
    Pi2 = rng.standard_normal((R.shape[1], p)) / np.sqrt(p)
    Y = A @ (R @ Pi2)
    return np.sum(Y ** 2, axis=1)
```
