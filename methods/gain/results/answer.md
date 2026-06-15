# GAIN, distilled

GAIN (Generative Adversarial Imputation Nets) adapts the GAN framework to missing-data
imputation. A generator `G` takes a partially observed vector and outputs candidate values for every
coordinate; the completed vector keeps the observed entries and uses `G` only in the holes. A
discriminator `D` looks at the completed vector and predicts, *componentwise*, which entries were
actually observed and which were imputed (i.e. it predicts the mask). They are trained
adversarially. Because that game alone has non-unique optima, `D` is given a **hint** that reveals
most of the mask, forcing the adversarial comparison onto unrevealed component(s) — which makes the
optimal generator distribution unique and equal to the true data distribution. A reconstruction loss
matches `G`'s raw output to the known values on observed entries. GAIN learns from incomplete data
(no complete training set required) and models the conditional *distribution* `P(X | X̃)`, so it
supports multiple imputation.

## Problem it solves

Impute missing entries of a mixed continuous/binary tabular dataset by sampling from
`P(X | X̃ = x̃)` (the distribution, not just the mean), exploiting cross-feature dependencies, and
learning entirely from data that is itself incomplete. Theory is under MCAR (`M ⟂ X`).

## Setup

- `X ∈ X` data vector; `M ∈ {0,1}^d` mask (`1` = observed). `X̃_i = X_i` if `M_i = 1`, else `*`.
- Generator `G : X̃ × {0,1}^d × [0,1]^d → X`, noise `Z ⟂` everything:
  - `X̄ = G(X̃, M, (1 − M) ⊙ Z)`  — imputations (noise masked to the holes, since `P(X|X̃)` is
    only `‖1 − M‖_1`-dimensional). `G` outputs a value for *every* coordinate.
  - `X̂ = M ⊙ X̃ + (1 − M) ⊙ X̄`  — completed vector (keep observed, fill holes with `G`).
- Discriminator `D : X × H → [0,1]^d`, `D(x̂, h)_i = P(M_i = 1 | x̂, h)` (predicts the mask
  componentwise; not a whole-vector real/fake verdict, because `X̂` is part real, part fake).

## Objective

```
V(D, G) = E_{X̂, M, H}[ M^T log D(X̂, H) + (1 − M)^T log(1 − D(X̂, H)) ],   min_G max_D V(D, G).
```

(Componentwise cross-entropy of `D` predicting the mask `M`; dependence on `G` is through `X̂`.)

## Hint mechanism

`H` is a random variable depending on `M`, passed to `D`, that the designer defines. Construction:
sample `k ~ Uniform{1,…,d}`, set `B_j = 1` for `j ≠ k` and `B_k = 0`, and

```
H = B ⊙ M + 0.5 (1 − B)  ∈ {0, 0.5, 1}^d.
```

So `H_i = t ∈ {0,1} ⟹ M_i = t` (revealed), and `H_i = 0.5` ⟹ nothing about `M_i` (the one hidden
coordinate). `H` reveals `d − 1` components of `M`. The canonical code uses the practical
Bernoulli-hint variant `B = binary_sampler(hint_rate)`, `H = M ⊙ B`, with `hint_rate ≈ 0.9`.

## Theory (MCAR)

**Lemma 1 (optimal discriminator).** For fixed `G`, maximizing `V` componentwise uses the pointwise
fact that `y ↦ a log y + b log(1 − y)` is maximized on `[0,1]` at `y = a/(a + b)`. Hence
```
D*(x, h)_i = p(x, h, m_i = 1) / [ p(x, h, m_i = 1) + p(x, h, m_i = 0) ] = p_m(m_i = 1 | x, h).
```

**Theorem 1 (generator criterion).** Substituting `D*` into `V` gives `C(G)`; expanding
`p_m(m_i=t|x,h) = p̂(x|h,m_i=t) p_m(m_i=t|h) / p̂(x|h)` and `log(ab) = log a + log b`, the
`x`-dependent part becomes a sum of KL divergences and the rest is `G`-independent:
```
C(G) = Σ_{i=1}^d Σ_{t∈{0,1}} ∫ p_m(m_i = t, h) · D_KL( p̂(· | h, m_i = t) ∥ p̂(· | h) ) dh  +  const.
```
Since each `D_KL ≥ 0`, `C(G)` is minimized iff for all `i, t, h` (with `p_h(h | m_i=t) > 0`) and a.e. `x`,
```
p̂(x | h, m_i = t) = p̂(x | h).                                   (*)
```
The minimum value is the constant — the best `D` can do is predict the mask from the hint alone.

**Proposition 1 (no hint ⟹ non-unique).** If `H ⟂ M` (≡ no hint), (*) reduces to
`p̂(x | m_i = 0) = p̂(x | m_i = 1)` for all `i`. Counting (e.g. `X = (X_1, X_2, X_3)` Bernoulli: 38
generator degrees of freedom vs. 24 linear equalities) shows `p̂` is not uniquely determined.
Adversarial training alone does not force `G` to the true distribution.

**Lemma 2 (`D` on revealed slots).** With the `B`-hint, `h_i = t ∈ {0,1} ⟹ p_m(m_i=t | h_i=t) = 1`,
so `D*(x, h)_i = h_i` there for all `x` — those outputs carry no information about `G`. Train `D`
only on the hidden (`b_i = 0`) slots.

**Proposition 2 (`B`-hint ⟹ unique = truth).** For masks `m_0, m_1` differing in exactly the `i`-th
component, the hint `h_j = m_j (j≠i), h_i = 0.5` is reachable from both (`p_h(h|m_i=t) > 0`,
`t = 0,1`). By Theorem 1, `p̂(x|h,m_i=0) = p̂(x|h,m_i=1)`. Conditioning on this `h` and `m_i=t`
pins down the full mask `m_t` and selector `B=b`; since `B` is sampled independently of `(X,M)`,
conditioning on `B=b` adds no information about `X` once `M=m_t` is fixed, so
`p̂(x|h,m_i=t) = p̂(x|m_t)`. Equivalently, in an unnormalized density calculation the same
`P(B=b)` factor appears on both sides and cancels. Thus `p̂(x|m_0) = p̂(x|m_1)`. Chaining single-coordinate flips,
`p̂(x|m) = p̂(x|1)` for all `m`; and `p̂(x|1)` is the density of `X` (since `X̂ = X` when fully
observed, and `M ⟂ X`). So `p̂` is the true data distribution, uniquely. ∎

## Algorithm

The proof-level pseudocode alternates stochastic-gradient steps on `D` and `G` over minibatches,
with fully connected networks. The released implementation injects `Z ~ U(0, 0.01)` into the holes.

- **`D` step** (only hidden slots): `L_D(m, m̂, b) = Σ_{i: b_i=0} [ m_i log m̂_i + (1−m_i) log(1−m̂_i) ]`;
  minimize `−Σ_j L_D`.
- **`G` step**: adversarial (non-saturating; push `D` toward "observed" on imputed slots)
  `L_G(m, m̂, b) = −Σ_{i: b_i=0} (1 − m_i) log m̂_i`, plus reconstruction comparing the raw
  generator output `x' = G(·)` to the known values on observed slots:
  `L_M(x, x') = Σ_i m_i L_M(x_i, x_i')`, with `L_M(x_i, x_i') = (x_i' − x_i)²` (continuous) or
  `−x_i log x_i'` (binary). Minimize `Σ_j L_G + α L_M` (`α` cross-validated, e.g. `{0.1,0.5,1,2,10}`).

The canonical TF1 implementation uses the Bernoulli hint `H=M⊙binary_sampler(hint_rate)`, the full
componentwise mask-prediction BCE for `D`, `G_loss_adv + alpha*MSE` for `G`, Adam, min-max
normalization, 3-layer ReLU MLPs with hidden width `d`, default `batch_size=128`, `hint_rate=0.9`,
`alpha=100`, and rounding for near-categorical columns after renormalization.

Why each piece: noise `(1−M)⊙Z` matches the target's dimension; mask-predicting `D` matches the
part-real/part-fake structure; the hint removes the non-uniqueness (Props 1–2); the non-saturating
`G` loss avoids early gradient starvation; the reconstruction term pins `G`'s observed-slot outputs and forces
the hidden layers to encode `X̃` (autoencoder effect); `α` trades distributional vs. pointwise fit.

## Working code

Faithful to the canonical implementation (TF1; `G`/`D` as 3-layer MLPs):

```python
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def _xavier(shape):
    return tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0))


def gain(data_x, batch_size=128, hint_rate=0.9, alpha=100.0, iterations=10000):
    """Impute NaNs in data_x with GAIN. Returns a finite same-shape array."""
    data_m = 1.0 - np.isnan(data_x).astype(np.float32)        # 1 = observed, 0 = missing
    no, dim = data_x.shape
    h_dim = dim

    # min-max normalize columns to [0,1] (matches sigmoid output)
    mn = np.nanmin(data_x, axis=0)
    rng = np.nanmax(data_x, axis=0) - mn + 1e-6
    norm_x = np.nan_to_num((data_x - mn) / rng, nan=0.0)

    X = tf.placeholder(tf.float32, [None, dim])   # data (holes filled with noise)
    M = tf.placeholder(tf.float32, [None, dim])   # mask
    H = tf.placeholder(tf.float32, [None, dim])   # hint

    # Discriminator vars: input = completed vector + hint (2d wide)
    D_W1 = tf.Variable(_xavier([dim * 2, h_dim])); D_b1 = tf.Variable(tf.zeros([h_dim]))
    D_W2 = tf.Variable(_xavier([h_dim, h_dim]));   D_b2 = tf.Variable(tf.zeros([h_dim]))
    D_W3 = tf.Variable(_xavier([h_dim, dim]));     D_b3 = tf.Variable(tf.zeros([dim]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator vars: input = noise-filled partial vector + mask (2d wide)
    G_W1 = tf.Variable(_xavier([dim * 2, h_dim])); G_b1 = tf.Variable(tf.zeros([h_dim]))
    G_W2 = tf.Variable(_xavier([h_dim, h_dim]));   G_b2 = tf.Variable(tf.zeros([h_dim]))
    G_W3 = tf.Variable(_xavier([h_dim, dim]));     G_b3 = tf.Variable(tf.zeros([dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(x, m):
        inp = tf.concat([x, m], axis=1)
        h1 = tf.nn.relu(tf.matmul(inp, G_W1) + G_b1)
        h2 = tf.nn.relu(tf.matmul(h1, G_W2) + G_b2)
        return tf.nn.sigmoid(tf.matmul(h2, G_W3) + G_b3)      # value per coordinate, in [0,1]

    def discriminator(x, h):
        inp = tf.concat([x, h], axis=1)
        h1 = tf.nn.relu(tf.matmul(inp, D_W1) + D_b1)
        h2 = tf.nn.relu(tf.matmul(h1, D_W2) + D_b2)
        return tf.nn.sigmoid(tf.matmul(h2, D_W3) + D_b3)      # per-coordinate P(observed)

    G_sample = generator(X, M)                  # X_bar (every coordinate)
    Hat_X = X * M + G_sample * (1 - M)          # X_hat: keep observed, impute holes
    D_prob = discriminator(Hat_X, H)

    D_loss = -tf.reduce_mean(M * tf.log(D_prob + 1e-8)
                             + (1 - M) * tf.log(1.0 - D_prob + 1e-8))   # predict the mask
    G_loss_adv = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))       # non-saturating
    MSE = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)  # reconstruct observed
    G_loss = G_loss_adv + alpha * MSE

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess = tf.Session(); sess.run(tf.global_variables_initializer())
    for _ in range(iterations):
        idx = np.random.permutation(no)[:batch_size]
        X_mb, M_mb = norm_x[idx], data_m[idx]
        Z_mb = np.random.uniform(0.0, 0.01, [batch_size, dim])          # small noise in holes
        B_mb = (np.random.uniform(0.0, 1.0, [batch_size, dim]) < hint_rate).astype(np.float32)
        H_mb = M_mb * B_mb                                              # hint: reveal w.p. hint_rate
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
        sess.run(D_solver, {X: X_mb, M: M_mb, H: H_mb})                 # (1) D step
        sess.run(G_solver, {X: X_mb, M: M_mb, H: H_mb})                 # (2) G step

    Z = np.random.uniform(0.0, 0.01, [no, dim])
    X_full = data_m * norm_x + (1 - data_m) * Z
    imputed = sess.run(G_sample, {X: X_full, M: data_m})
    imputed = data_m * norm_x + (1 - data_m) * imputed                 # only fill holes
    imputed = imputed * rng + mn                                       # denormalize
    for i in range(dim):                                               # round near-categorical cols
        col = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(col)) < 20:
            imputed[:, i] = np.round(imputed[:, i])
    return imputed
```

Wrapped as a fit/transform imputer, `fit` runs the adversarial training on the incomplete matrix
(never using test labels) and `transform` runs `G` to fill the holes, keeping observed entries and
renormalizing — a same-shape array with no `NaN`s.
