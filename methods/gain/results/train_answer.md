Imputation asks us to complete a feature vector from a partial observation, and the task is harder than it first appears. A hole in one coordinate is informed by the other coordinates of the same record, so any method that fills each column independently throws away the structure that makes imputation accurate. At the same time, many real datasets are intrinsically incomplete: there is no clean training set hiding somewhere, so the learner must train on the incomplete data itself. Existing approaches each fail one of these constraints. Column-mean and k-nearest-neighbors imputation are fast point predictors that collapse the conditional distribution to a single number and ignore feature correlations or scale poorly. MICE and MissForest chain per-column regressions, but the conditionals are fit separately and need not come from any coherent joint distribution. Low-rank matrix completion assumes a linear subspace, while EM commits to a parametric family that breaks on mixed continuous and categorical features. Even flexible denoising autoencoders usually require complete examples to manufacture corrupted-clean training pairs. What is missing is a way to learn a flexible, nonparametric model of the full conditional distribution P(X | X̃) directly from data that is itself incomplete.

The method that fills this gap is GAIN, Generative Adversarial Imputation Nets. It adapts the GAN framework to missing data. A generator network looks at the partial record and outputs candidate values for every coordinate. The completed vector keeps the observed entries unchanged and uses the generator only in the holes. A discriminator network then examines the completed vector and predicts, for each coordinate independently, whether that coordinate was originally observed or imputed. This componentwise mask prediction is the right adversarial question because every completed vector is part real and part fake. Running the usual optimal-discriminator argument shows that the discriminator learns the posterior probability of the mask given the completed vector. However, the resulting game is underdetermined unless extra structure is provided: the generator could in principle fool the discriminator without recovering the true data distribution. To fix this, the discriminator is given a hint that reveals most of the true mask and hides only a small subset. In the idealized analysis the hint reveals all but one coordinate, which forces the conditional distributions under different masks to coincide and makes the unique optimum equal to the true data distribution under the MCAR assumption.

The generator receives the partial vector with small uniform noise injected into the missing slots, together with the mask, and outputs a full vector squashed to the normalized range by a sigmoid. The completed vector is formed by masking: observed coordinates keep their true values and missing coordinates take the generator output. The discriminator sees this completed vector concatenated with the hint. In practice the hint is built by revealing each coordinate's true mask value independently with probability close to one, so most of the mask is visible and the adversarial signal concentrates on the unrevealed entries. The discriminator is trained to maximize the componentwise log-likelihood of predicting the mask. The generator is trained to make the discriminator believe its imputed coordinates were observed, using the non-saturating form to avoid the vanishing gradients that occur when the discriminator confidently rejects bad samples early in training. A reconstruction loss also encourages the generator's raw outputs on observed coordinates to match the truth, which pins those outputs and forces the hidden layers to encode the partial record like an autoencoder. These two terms pull in compatible directions: reconstruction makes the representation faithful, while the adversarial term makes the distribution of imputed values plausible.

After training, imputation is performed by running the generator on the noisy partial matrix, keeping observed values, denormalizing back to the original scales, and rounding near-categorical columns. The method learns from incomplete data alone, models the conditional distribution rather than just a conditional mean, and therefore supports multiple imputation by drawing different noise samples for the same record.

```python
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _xavier(shape):
    return tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0))


def gain(data_x, batch_size=128, hint_rate=0.9, alpha=100.0, iterations=10000):
    """Impute NaNs in data_x with GAIN. Returns a finite same-shape array."""
    data_m = 1.0 - np.isnan(data_x).astype(np.float32)  # 1 = observed, 0 = missing
    no, dim = data_x.shape
    h_dim = dim

    # min-max normalize columns to [0, 1] to match sigmoid output
    mn = np.nanmin(data_x, axis=0)
    rng = np.nanmax(data_x, axis=0) - mn + 1e-6
    norm_x = np.nan_to_num((data_x - mn) / rng, nan=0.0)

    X = tf.placeholder(tf.float32, [None, dim])  # data (holes filled with noise)
    M = tf.placeholder(tf.float32, [None, dim])  # mask
    H = tf.placeholder(tf.float32, [None, dim])  # hint

    # Discriminator: completed vector concatenated with hint
    D_W1 = tf.Variable(_xavier([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros([h_dim]))
    D_W2 = tf.Variable(_xavier([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros([h_dim]))
    D_W3 = tf.Variable(_xavier([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros([dim]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator: noise-filled partial vector concatenated with mask
    G_W1 = tf.Variable(_xavier([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros([h_dim]))
    G_W2 = tf.Variable(_xavier([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros([h_dim]))
    G_W3 = tf.Variable(_xavier([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros([dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(x, m):
        inp = tf.concat([x, m], axis=1)
        h1 = tf.nn.relu(tf.matmul(inp, G_W1) + G_b1)
        h2 = tf.nn.relu(tf.matmul(h1, G_W2) + G_b2)
        return tf.nn.sigmoid(tf.matmul(h2, G_W3) + G_b3)

    def discriminator(x, h):
        inp = tf.concat([x, h], axis=1)
        h1 = tf.nn.relu(tf.matmul(inp, D_W1) + D_b1)
        h2 = tf.nn.relu(tf.matmul(h1, D_W2) + D_b2)
        return tf.nn.sigmoid(tf.matmul(h2, D_W3) + D_b3)

    G_sample = generator(X, M)
    Hat_X = X * M + G_sample * (1 - M)  # keep observed, fill holes
    D_prob = discriminator(Hat_X, H)

    # Discriminator loss: predict the mask componentwise
    D_loss = -tf.reduce_mean(
        M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1.0 - D_prob + 1e-8)
    )
    # Generator adversarial loss: non-saturating on imputed slots
    G_loss_adv = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    # Reconstruction loss on observed slots
    MSE = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)
    G_loss = G_loss_adv + alpha * MSE

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(iterations):
        idx = np.random.permutation(no)[:batch_size]
        X_mb, M_mb = norm_x[idx], data_m[idx]
        Z_mb = np.random.uniform(0.0, 0.01, [batch_size, dim])
        B_mb = (np.random.uniform(0.0, 1.0, [batch_size, dim]) < hint_rate).astype(np.float32)
        H_mb = M_mb * B_mb
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
        sess.run(D_solver, {X: X_mb, M: M_mb, H: H_mb})
        sess.run(G_solver, {X: X_mb, M: M_mb, H: H_mb})

    # Impute the full dataset
    Z = np.random.uniform(0.0, 0.01, [no, dim])
    X_full = data_m * norm_x + (1 - data_m) * Z
    imputed = sess.run(G_sample, {X: X_full, M: data_m})
    imputed = data_m * norm_x + (1 - data_m) * imputed
    imputed = imputed * rng + mn
    for i in range(dim):
        col = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(col)) < 20:
            imputed[:, i] = np.round(imputed[:, i])
    return imputed
```
