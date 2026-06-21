I want the top-$k$ principal components of a data matrix $X \in \mathbb{R}^{n\times d}$ — the eigenvectors of $M = X^\top X$, ordered by eigenvalue, each one a genuine component and not merely some basis of the top-$k$ subspace. On a small problem this is settled: call SVD and walk away. The problems I actually care about have $n$ near a million and $d$ in the tens of millions — flattened activations of a deep network over a whole image dataset. A full SVD costs $O(\min\{nd^2, n^2d\})$ time and $O(nd)$ space, and I cannot even form $M$, which has $d^2$ entries. So the question is not *what* the answer is — linear algebra answered that a century ago — but *what iterative procedure recovers it* when the data is enormous and the only compute I have is a farm of accelerators connected by fast interconnects. That hardware clause is load-bearing: the natural move is to give each accelerator one eigenvector to learn, feed it a shard of data, and let it talk to the others as little as possible. I am really after a procedure that *decentralizes* — per-vector updates, minimal communication, and no step that forces all $k$ vectors to synchronize at once.

The standard tools do not give me this. The top-1 eigenvector falls out of Hebb's rule $v \leftarrow v + \eta M v$, with Oja's rule $v \leftarrow v + \eta(I - vv^\top)Mv$ or an explicit renormalization to keep the norm bounded; Krasulina is Oja-with-renormalization. To get $k$ of them, everyone re-orthonormalizes the whole bundle: a $QR$ of the $d\times k$ iterate every step (top-$k$ Oja's algorithm), a Stiefel projection plus $QR$ (Matrix Krasulina), or deflation. That per-iteration $QR$ is a global, synchronizing operation across all $k$ vectors — exactly the serial bottleneck I need to avoid, and it grows with $d$. Sanger's Generalized Hebbian Algorithm folds the deflation into a single update, $\Delta\hat v_i = 2[M\hat v_i - (\hat v_i^\top M\hat v_i)\hat v_i - \sum_{j<i}(\hat v_i^\top M\hat v_j)\hat v_j]$, recovers the actual components, and can be distributed — so it nearly does what I want. But stare at it and ask what each vector is *trying* to do. Is there a payoff $u_i$ whose gradient is this update? For that, the Jacobian of the update must be symmetric (a Hessian). The penalty term contributes $-\sum_{j<i}\hat v_j\hat v_j^\top M$ to the Jacobian, and $\hat v_j\hat v_j^\top M \neq M\hat v_j\hat v_j^\top$ in general, so the Jacobian is not symmetric: the GHA update is the gradient of nothing. It works blindly. I cannot ask what player $i$ wants, cannot import optimization or equilibrium theory cleanly. That is the gap I set out to close: I want the component recovery to fall out of an *objective* each vector is maximizing.

I propose EigenGame, which casts PCA as a many-player game and recovers the ordered components as its unique equilibrium. The construction begins from $R(\hat V) := \hat V^\top M\hat V$, whose diagonal $R_{ii} = \langle\hat v_i, M\hat v_i\rangle$ is the variance captured along $\hat v_i$ (a Rayleigh quotient, since $\|\hat v_i\| = 1$) and whose off-diagonals $R_{ij} = \langle\hat v_i, M\hat v_j\rangle$ measure alignment under the $M$-inner product. The obvious objective, maximize the trace, is a trap: at $k=d$, $\hat V$ is square orthonormal so $\hat V\hat V^\top = I$ and $\operatorname{Tr}(\hat V^\top M\hat V) = \operatorname{Tr}(M)$, a constant independent of $\hat V$. Trace-maximization recovers only the *subspace*; it expresses no preference for which directions inside it the columns point. What distinguishes components from subspace is the eigenvalue equation: orthonormal $V$ with $MV = V\Lambda$ satisfies $V^\top MV = \Lambda$, a *diagonal* matrix. So $\hat V$ is the true eigenvector set exactly when $R(\hat V)$ has no off-diagonals. The complete objective is therefore reward minus penalty — capture variance (the trace), and drive the off-diagonals to zero.

The decisive design choice is to make that penalty *asymmetric*. A symmetric penalty, where $\hat v_1$ is punished for aligning with $\hat v_k$ as much as the reverse, fights PCA's natural ordering: $\hat v_1$ estimates the *largest* eigenvector and should chase maximum variance freely, answerable to no one. So I let each vector penalize only its **parents** $j < i$ — the penalty points up the hierarchy, never back down. The instant the objectives differ across vectors, this is no longer one optimization but a game: $k$ players, player $i$ controls $\hat v_i$ and maximizes its own payoff given the others. Player $i$ maximizes, subject to $\|\hat v_i\| = 1$,

$$u_i(\hat v_i \mid \hat v_{j<i}) = \hat v_i^\top M\hat v_i - \sum_{j<i}\frac{(\hat v_i^\top M\hat v_j)^2}{\hat v_j^\top M\hat v_j} = \|X\hat v_i\|^2 - \sum_{j<i}\frac{\langle X\hat v_i, X\hat v_j\rangle^2}{\langle X\hat v_j, X\hat v_j\rangle}.$$

Two refinements make the penalty right. First, the inner product in the penalty is the *generalized* one $\langle\cdot,\cdot\rangle_M$, not a plain Euclidean overlap $\sum\langle\hat v_i,\hat v_j\rangle^2$: a bare overlap is drowned out when a parent points along a huge-eigenvalue direction, whereas including $M$ boosts the penalty to balance the reward. Second, I divide each penalty term by the parent's own Rayleigh quotient $\langle X\hat v_j, X\hat v_j\rangle$. This puts the two terms on a common scale — and, as becomes clear in the gradient, it is what makes the algebra collapse cleanly. I also drop any explicit orthogonality constraint $\hat v_i^\top\hat v_j = 0$: at the solution the penalty is $\Lambda_{jj}^2\langle\hat v_i, v_j\rangle^2$, which blows up the instant $\hat v_i$ drifts toward a parent, so the penalty enforces the orthogonality I would otherwise impose by hand. Only $\|\hat v_i\| = 1$ remains.

The right correctness notion is a strict Nash equilibrium — a choice from which no player can unilaterally improve — and the top-$k$ eigenvectors are the *unique* one. To see it, diagonalize $M = U\Lambda U^\top$; since $U$ preserves inner products I can analyze the action on the diagonal $\Lambda$ with $V$ the identity. Write a candidate as $\hat v_i = \sum_p w_p v_p$ with $\|w\| = 1$. With exact parents the reward becomes $\sum_q w_q^2\Lambda_{qq}$, and the penalty numerator for parent $j$ is $\hat v_i^\top Mv_j = \Lambda_{jj}w_j$, so the normalized penalty term is exactly $\Lambda_{jj}w_j^2$ — the parent indices cancel cleanly (this is what the Rayleigh-quotient normalization bought). Setting $z_p = w_p^2 \in \Delta^{d-1}$,

$$u_i = \sum_{q}w_q^2\Lambda_{qq} - \sum_{j<i}\Lambda_{jj}w_j^2 = \sum_{p\ge i}\Lambda_{pp}\,z_p,$$

a linear function over the simplex restricted to indices $p \ge i$, maximized at a vertex; with distinct positive top-$k$ eigenvalues the unique maximizer is $z^* = e_i$, i.e. $\hat v_i = \pm v_i$. Induction over $i$ finishes it. The hierarchy is not cosmetic: the fully symmetric variant has *no* symmetric Nash (if all $\hat v_i$ are equal, $u_i = (2-k)(\hat v^\top M\hat v) \le 0$ for $k\ge 2$, and any player can deviate to an orthogonal direction for positive payoff), and certifying uniqueness there is NP-hard in general. The asymmetry buys both uniqueness and a DAG that can be solved by sweeping in order.

The update is gradient ascent on $u_i$, and the payoff of the normalization shows up here. Differentiating, the reward gives $2M\hat v_i$ and each penalty term gives $2(\hat v_i^\top M\hat v_j)/(\hat v_j^\top M\hat v_j)\,M\hat v_j$, so

$$\nabla_{\hat v_i}u_i = 2M\Big[\hat v_i - \sum_{j<i}\frac{\hat v_i^\top M\hat v_j}{\hat v_j^\top M\hat v_j}\,\hat v_j\Big] = 2X^\top\Big[X\hat v_i - \sum_{j<i}\frac{\langle X\hat v_i, X\hat v_j\rangle}{\langle X\hat v_j, X\hat v_j\rangle}\,X\hat v_j\Big].$$

The $M$ factors all the way out to the front. The bracket is $\hat v_i$ minus its projection onto each parent under $\langle\cdot,\cdot\rangle_M$ — a single step of *generalized Gram–Schmidt* — and the outer $2M$ is precisely the matrix product at the heart of Oja's rule and power iteration. Neither ingredient was put in by hand; both fell out of differentiating one clean objective. Setting $M=I$ recovers ordinary Gram–Schmidt as the special case. This also places GHA exactly: GHA equals this gradient with the reward projected onto the sphere's tangent space but the penalty left *unprojected*, which is why it is the gradient of nothing — it does not project a single coherent vector field. In data form $M = X^\top X$ never appears explicitly: two passes through the data per step ($X\cdot$ then $X^\top\cdot$), no $d\times d$ matrix, and the only thing a child needs from a parent is its broadcast vector. That broadcast is the entire interconnect cost.

Because each $\hat v_i$ lives on the unit sphere $S^{d-1}$, I do constrained ascent the Riemannian way: project the ambient gradient onto the tangent space, $\nabla^R_{\hat v_i} = \nabla_{\hat v_i} - \langle\nabla_{\hat v_i}, \hat v_i\rangle\hat v_i$, take a step, then retract by renormalizing. There is a useful second variant: near the optimum the ambient gradient is nearly radial, and stepping with it and then renormalizing lets the renormalization eat the radial part, which effectively shrinks the step as the iterate approaches equilibrium — a built-in learning-rate decay. So I keep both: the projected ($R$) variant, for which the theory is cleanest, and the unprojected one, which is often more stable and faster in practice — one line of difference.

The landscape analysis explains why this converges from any start. Parameterizing $\hat v_i = \cos(\theta_i)v_i + \sin(\theta_i)\Delta_i$ with exact parents, the utility collapses to $u_i = \Lambda_{ii} - \sin^2(\theta_i)(\Lambda_{ii} - \sum_{l>i}z_l\Lambda_{ll})$ — a sinusoid of period $\pi$ (since $\pm v_i$ are the same component): non-concave, yet every local maximum is global, so there are no spurious traps. With approximate parents the utility takes the form $A\sin^2(\theta_i) - B\sin(2\theta_i)/2 + C$ with $A,B,C$ depending on the parents' deviations but not on $\theta_i$, and the maximizer's angular error follows an $\arctan$ soft-step in the parents' error. This gives a sharp accuracy threshold: parents learned to within $c_i\,g_i/((i-1)\Lambda_{11})$ (with $c_i \le 1/16$ and gap $g_i = \Lambda_{ii} - \Lambda_{i+1,i+1}$) yield child error $\le 8c_i$. Combined with a nonconvex Riemannian rate ($1/\rho^2$ iterations to reach $\|\nabla^R\| \le \rho$), the sequential algorithm converges to within $\theta_{\text{tol}}$ *independent of initialization*, with total iteration count $T_k = \lceil O(k\cdot[(16\Lambda_{11})^k(k-1)!/(\prod_{j=1}^k g_j)\cdot 1/\theta_{\text{tol}})^2]\rceil$ — the $(k-1)!$ being the up-the-chain accuracy amplification that forces the first eigenvector to be learned most precisely. The sequential version is what I can prove, but in practice I run all players in parallel and broadcast each step: as a parent nears its optimum it becomes quasi-stationary, so a child maximizing against a slowly drifting parent is close to maximizing against a fixed one. (One honest caveat: stochastic minibatch gradients of this utility are biased, since it contains products and ratios of inner products of the same batch; larger batches reduce the bias.)

In full-batch form everything vectorizes across the $k$ players. Let $R = (X\hat V)^\top(X\hat V)$ be the $k\times k$ Gram of the projected vectors, normalize each column by its diagonal so $R_{\text{norm}}$ holds the ratios with ones on the diagonal, and apply the lower-triangular mask $\operatorname{LT}(2I_k - \mathbf 1_k)$ — $+1$ on the diagonal (the reward), $-1$ on the strict lower triangle (the parents), zero above. Then $G_s = \hat V(R_{\text{norm}}\odot\text{mask})^\top$ assembles the generalized-Gram–Schmidt bracket for every player at once, the ambient gradient is $X^\top(XG_s)$, the Riemannian projection subtracts each column's radial part, and a final column-normalize retracts to the sphere. No $QR$, no SVD, no $d\times d$ matrix; the only $k\times k$ object is $R$, and the only coupling between vectors is the masked $R_{\text{norm}}$ — exactly the parent broadcasts.

```python
import numpy as np

def reference_components(X, k):
    """Exact eigenvectors of M = X^T X, descending — validation only."""
    w, U = np.linalg.eigh(X.T @ X)
    order = np.argsort(w)[::-1]
    return U[:, order[:k]], w[order[:k]]

def normalize_columns(V):
    return V / np.linalg.norm(V, axis=0, keepdims=True)

def utility(X, V):
    """u_i = ||X v_i||^2 - sum_{j<i} <Xv_i,Xv_j>^2 / <Xv_j,Xv_j>  (per column)."""
    XV = X @ V
    R = XV.T @ XV                                # R_ij = <Xv_i, Xv_j>
    rewards = np.diag(R)
    k = V.shape[1]
    pen = np.array([sum(R[i, j] ** 2 / R[j, j] for j in range(i)) for i in range(k)])
    return rewards - pen

def eigengame_step(X, V, lr, riemannian=True):
    """Each column v_i ascends u_i(v_i | v_{j<i}).
    grad_i = 2 X^T [ X v_i - sum_{j<i} (<Xv_i,Xv_j>/<Xv_j,Xv_j>) X v_j ]
           = 2 X^T X [ v_i - (generalized Gram-Schmidt against parents) ].
    """
    k = V.shape[1]
    XV = X @ V                                   # (n, k): the "rewards" signal
    R = XV.T @ XV                                # (k, k): <Xv_i, Xv_j>
    R_norm = R / np.diag(R)                      # divide each col by parent Rayleigh
    mask = np.tril(2 * np.eye(k) - np.ones((k, k)))   # diag +1 (reward), strict-lower -1 (parents)
    G_s = V @ (R_norm * mask).T                  # column i: v_i - sum_{j<i} (...) v_j
    grad = 2.0 * (X.T @ (X @ G_s))               # Oja matrix product (never forms M)
    if riemannian:                               # project onto sphere tangent space
        grad = grad - V * np.sum(grad * V, axis=0, keepdims=True)
    V = V + lr * grad                            # ascent
    return normalize_columns(V)                  # retract to the sphere

def eigengame(X, k, lr=1e-4, iters=5000, riemannian=True, V0=None):
    d = X.shape[1]
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for _ in range(iters):                       # in parallel: one device per column
        V = eigengame_step(X, V, lr, riemannian)
    return V

def eigengame_streaming(stream, d, k, lr=1e-4, riemannian=False, V0=None):
    """Same step on a stream of minibatches; broadcast(v_i) is the only comm."""
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for Xt in stream:                            # Xt : (m, d)
        V = eigengame_step(Xt, V, lr, riemannian)
    return V

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 30))
    k = 6
    V = eigengame(X, k, lr=3e-4, iters=8000, riemannian=False)
    V_true, w_true = reference_components(X, k)
    print("alignment per component:", np.round(np.abs(np.sum(V * V_true, axis=0)), 4))
    print("recovered Rayleigh:", np.round(np.diag((X @ V).T @ (X @ V)), 2))
    print("true eigenvalues:  ", np.round(w_true, 2))
```

To recover the *smallest* eigenvectors, run the same game on $M' = \Lambda_{11}I - M$, whose top eigenvectors are $M$'s bottom ones. And because every term is a generalized inner product $\langle X\hat v_i, X\hat v_j\rangle$, replacing $X\hat v_i$ with a learned feature map $f_i(X)$ carries the same utility and gradient machinery beyond the linear case.
