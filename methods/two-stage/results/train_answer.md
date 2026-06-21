The setting is a two-layer network learning a multi-index target on isotropic Gaussian data: inputs are $z \sim \mathcal{N}(0, I_d)$, but the label depends only on a fixed low-dimensional projection, $y = f^*(z) = g^*(\langle w_1^*, z\rangle, \ldots, \langle w_r^*, z\rangle)$, with orthonormal teacher directions spanning a subspace $V^*$ of dimension $r \ll d$. The distribution of $z$ has no preferred axes; all the structure lives in the unknown subspace the target reads off. A student $\hat f(z; W, a) = \tfrac{1}{\sqrt{p}}\sum_i a_i \sigma(\langle w_i, z\rangle)$ has exactly the right form — its first-layer rows could rotate into $V^*$, after which the readout only has to fit a function of $r$ coordinates. The real question is whether training actually makes that rotation happen cheaply. The obstruction is geometric: a row $w_i$ drawn uniformly on the sphere has only $\|\Pi^* w_i\| = O(\sqrt{r/d})$ of its mass inside $V^*$, where $\Pi^*$ projects onto the target subspace. Every existing option founders on this tiny overlap. Frozen random features (conjugate-kernel ridge) never move the rows at all, so they hit a degree barrier — learning a degree-$k$ part of the target costs $n, p \sim d^k$, and the teacher subspace is never discovered. A small step from initialization stays in the lazy/linearized regime, adjusting constants and a linear component but never producing order-one row alignment. Online one-sample SGD does eventually align, but starting from $O(d^{-1/2})$ correlation it spends $d\log d$ samples to escape for information exponent $2$ and roughly $d^{\ell-1}$ for larger leap index $\ell$. One-step preprocessing methods can expose several directions at once but require a large batch plus a separate low-degree Hermite estimation stage. What is missing is a first-layer move whose entire purpose is to amplify the accidental $O(\sqrt{r/d})$ overlap into order-one alignment, together with a clean way to turn the learned features into a predictor without entangling the two analyses.

I propose two-stage large-batch training: train the two layers separately. In the first stage, freeze the output weights $a$ and update only the first-layer rows with one or a few large-batch gradient-descent steps on squared loss, using a deliberately giant learning rate. In the second stage, freeze the learned $W$ and solve ridge regression on the scaled conjugate-kernel features $\sigma(ZW^\top)/\sqrt{p}$ on an independent batch. The design is dictated by inspecting one row with the readout frozen. With squared loss the negative gradient on $w_i$ is $g_i = \tfrac{a_i}{\sqrt{p}}\,\mathbb{E}[z\,\sigma'(\langle w_i, z\rangle)(f^*(z) - \hat f(z))]$. To isolate the signal at initialization I take the residual to equal $f^*$, which the exact theory enforces by pairing hidden units with opposite output weights, $w_i^0 = w_{p-i+1}^0$ and $a_i^0 = -a_{p-i+1}^0$, so $\hat f(\cdot; W^0, a^0) \equiv 0$ (the public notebooks instead use small random output weights, where this holds only approximately at large width). Then Stein's lemma rewrites the expectation as a part along $w_i$ plus a part involving $\nabla_z f^*$:
$$\mathbb{E}[z\,\sigma'(\langle w, z\rangle) f^*(z)] = w\,\mathbb{E}[\sigma''(\langle w, z\rangle)f^*(z)] + \mathbb{E}[\sigma'(\langle w, z\rangle)\,\nabla_z f^*(z)].$$
The first term is parallel to $w$ and only rescales the row; the second can point into $V^*$. Expanding in Hermite tensors, with $c_k$ the Hermite coefficients of the activation $\sigma$ and $C_k^*$ those of the target, gives
$$\mathbb{E}[\mathbf{neg\_grad}_i] = \frac{a_i}{\sqrt{p}}\Big[\sum_k c_{k+2}\langle w_i^{\otimes k}, C_k^*\rangle\, w_i + \sum_k c_{k+1}\, C_{k+1}^*\!\cdot_{1..k}\, w_i^{\otimes k}\Big],$$
where the second sum contracts the first $k$ modes of the $(k{+}1)$-tensor and leaves a vector. Because every $C_k^*$ inherits its singular directions from the low-dimensional link, all of them lie in $V^*$; contracting one against $k$ copies of a random row costs $\|\Pi^* w\|^k \approx d^{-k/2}$ up to fixed $r$ and logs. So if $\ell$ is the first nonzero Hermite degree of $f^*$ — the leap index — the leading useful target-subspace vector is $\tfrac{a_i}{\sqrt{p}}\,c_\ell\, C_\ell^* \cdot_{1..(\ell-1)} (w_i^0)^{\otimes(\ell-1)}$, of size about $d^{-(\ell-1)/2}/p$ since $a_i = O(p^{-1/2})$.

That scale is the whole reason an $O(1)$ step is useless and why the learning rate must be enormous. To make the new $V^*$-component order one, the step has to cancel both the Hermite smallness $d^{-(\ell-1)/2}$ and the two factors of $p^{-1/2}$ (one from $a_i$, one from the $1/\sqrt{p}$ in the network), which forces
$$\eta = \Theta\!\big(p\, d^{(\ell-1)/2}\big),$$
equivalently $\eta = \Theta(p\sqrt{n/d})$ when the one-step batch is $n = \Theta(d^\ell)$ — the operational form, which says that with linear-size batches $n = \Theta(d)$ the feature-learning rate is order $p$, and with higher leap order both batch and step grow. The noise floor confirms the threshold: the empirical average estimates the leap-order contraction, so if $n = O(d^{\ell-\delta})$ the $V^*$-signal is buried in sampling fluctuations and the post-step alignment stays vanishing, but once $n = \Omega(d^\ell)$ and the student activation has the matching nonzero Hermite coefficient, the giant step makes $\|\Pi^* w_i^1\|^2/\|w_i^1\|^2$ bounded away from zero for each row with high probability. The row norm explodes because the step is giant, but the quantity that matters is the ratio, and numerator and denominator are scaled by the same $\eta$.

One step has a structural ceiling: its leading vector lives in $V_\ell^*$, the span of the higher-order singular vectors of $C_\ell^*$, so it can only learn the directions visible in the first nonzero Hermite tensor (for $\ell = 1$, $C_1^*$ is a single vector and every row is pulled toward the same one-dimensional spike). If $V_\ell^* \subsetneq V^*$, no later readout can invent the missing directions. The way to recover more directions without raising the batch to $d^2$ is to repeat the large-batch step with fresh data: after step one the rows have order-one components in the already-learned subspace, and conditioning on those components changes the next leading Hermite coefficient. Writing $U_t^*$ for the subspace learned by time $t$ and the conditioned function on $U_t^{*\perp}$ as $f^*_{U_t^*, x}(x_\perp) = f^*(x + x_\perp)$ for fixed $x \in U_t^*$, the new first-order signal is $\mu_{U_t^*, x}(f^*) = \mathbb{E}_{x_\perp}[\nabla_{x_\perp} f^*_{U_t^*, x}(x_\perp)]$, and the learned subspace grows by the conditioning recursion
$$U_0^* = \{0\}, \qquad U_{t+1}^* = U_t^* + \mathrm{span}\{\mu_{U_t^*, x}(f^*) : x \in U_t^*\}.$$
A direction appears at the next step exactly when conditioning on what is already learned makes it linearly visible — this is the staircase condition, not a tuning choice. For $f^*(z) = z_1 + z_2 + z_1^2 - z_2^2$ the first Hermite coefficient points along $v = (e_1 + e_2)/\sqrt 2$; rewriting with $u = \langle v, z\rangle$, $s = \langle v_\perp, z\rangle$ gives $\sqrt 2\,u + 2us$, and conditioning on $u = \lambda$ leaves orthogonal-derivative mean $2\lambda$, so step two learns $v_\perp$. For $z_1 + z_2 + z_1^2 + z_2^2$ the rewrite is $\sqrt 2\,u + u^2 + s^2$, conditioning leaves derivative $2s$ with Gaussian mean zero, and $v_\perp$ is never learned by finitely many linear-batch steps.

Once the rows have moved I stop treating the readout as part of the nonconvex problem. With $W$ fixed the feature matrix is just $\Phi = \sigma(ZW^\top)/\sqrt{p}$ and fitting $a$ is a convex ridge problem, $\min_a \|Y - \Phi a\|^2 + \lambda\|a\|^2$, with closed form $\hat a = (\Phi^\top\Phi + \lambda I_p)^{-1}\Phi^\top Y$ when $n \ge p$ and, by the push-through identity, $\hat a = \Phi^\top(\Phi\Phi^\top + \lambda I_n)^{-1}Y$ when $n < p$. The load-bearing invariants are the $1/\sqrt p$ feature scaling and the two correct linear-system branches. I keep the data split clean — fresh batches for each feature step, an independent batch for the readout — so that after $W$ becomes data-dependent the readout is still analyzable as ridge on a fixed learned feature map. On signs: the canonical implementation forms a negative-gradient matrix with $(Y - \hat f)$ and updates $W \leftarrow W + \eta\,G^\top$, equivalent to the true-gradient convention with $(\hat f - Y)$ and $W \leftarrow W - \eta\,\mathrm{grad}$. There is no per-step projection of rows back to the sphere; the dynamics are analyzed through the projection ratio after the update, not by normalizing every step. The reachable subspace is decided entirely by the first stage — $V_\ell^*$ for one step at batch $d^\ell$, or the staircase-reachable subspace for several linear-batch steps — and the second stage attains the best predictor available from those features, bounded below by the variance of $f^*$ left after conditioning on the learned subspace $U$: $\mathbb{E}[(f^*(z) - \hat f(z))^2] \ge \mathbb{E}[\mathrm{Var}(f^*(z)\mid P_U z)] - o(1)$.

```python
import numpy as np


def sigma(x):
    return np.maximum(x, 0.0)


def sigma_prime(x):
    return (x > 0).astype(float)


def net(Z, W, a):
    p = W.shape[0]
    return sigma(Z @ W.T) @ a / np.sqrt(p)


def sample_data(n, d, teacher, link):
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)
    return Z, Y


def init_weights(p, d, symmetric=True):
    """Sphere init; symmetric=True gives exact zero initial output."""
    if symmetric:
        if p % 2:
            raise ValueError("symmetric initialization requires even p")
        half = p // 2
        W_half = np.random.randn(half, d)
        W_half /= np.linalg.norm(W_half, axis=1, keepdims=True)
        a_half = np.random.uniform(-1.0, 1.0, size=half) / np.sqrt(p)
        W = np.vstack([W_half, W_half])
        a = np.concatenate([a_half, -a_half])
        return W, a

    W = np.random.randn(p, d)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    a = np.random.uniform(-1.0, 1.0, size=p) / np.sqrt(p)
    return W, a


def negative_first_layer_gradient(Z, Y, W, a):
    """Return G with shape (d, p), so the update is W <- W + eta * G.T."""
    n, p = Z.shape[0], W.shape[0]
    residual = Y - net(Z, W, a)
    gates = sigma_prime(Z @ W.T)
    weighted = (residual[:, None] * a[None, :] / np.sqrt(p)) * gates
    return (Z.T @ weighted) / n


def ridge_estimator(Phi, y, lam):
    """Solve min_b ||y - Phi b||^2 + lam ||b||^2."""
    n, p = Phi.shape
    if n >= p:
        return np.linalg.solve(Phi.T @ Phi + lam * np.eye(p), Phi.T @ y)
    return Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)


def two_stage(d, p, teacher, link, n_feature, n_readout, steps, eta_scale=1.0, lam=1.0):
    W, a0 = init_weights(p, d, symmetric=True)
    eta = eta_scale * p * np.sqrt(n_feature / d)

    for _ in range(steps):
        Z, Y = sample_data(n_feature, d, teacher, link)
        G = negative_first_layer_gradient(Z, Y, W, a0)
        W = W + eta * G.T

    Z2, Y2 = sample_data(n_readout, d, teacher, link)
    Phi = sigma(Z2 @ W.T) / np.sqrt(p)
    a_hat = ridge_estimator(Phi, Y2, lam)

    def predict(Ztest):
        return sigma(Ztest @ W.T) @ a_hat / np.sqrt(p)

    return W, a_hat, predict
```
