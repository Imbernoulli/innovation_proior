I start with the actual pressure in the task. I have one sparse undirected graph, not independent labeled examples. Some real edges are hidden, and I need to rank those hidden edges above sampled non-edges. I also have node features, so if I learn embeddings only from node IDs or random walks I am throwing away a source of evidence that should matter on citation graphs. The model I need has to turn `(X, A_train)` into one vector per node and then turn a pair of node vectors into a link score, using only the observed graph as supervision.

The older unsupervised choices do part of this but not enough. Spectral embeddings give me a topology summary from graph eigenvectors, and random-walk methods like DeepWalk give me node embeddings by treating walks as sentences and running SkipGram. Both can work for link prediction after I score pairs by a similarity in the learned space, but both are structure-only. They do not let a bag-of-words vector on a document change the embedding except through a separate downstream model. On the other side, the graph convolutional encoder does combine features with the graph: after adding self-loops and symmetric normalization, one layer applies `D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H W`, so every node receives a normalized mixture of its own features and its neighbors' features. That is the right encoder shape, but its original objective is supervised node classification. I have no node labels. I need to give the encoder an unsupervised graph-reconstruction job.

The simplest decoder is the one I should try first: put linked nodes near each other in latent space and score a candidate pair by an inner product. If node `i` has embedding `z_i` and node `j` has embedding `z_j`, then `z_i^T z_j` should serve as a logit for an undirected edge, with `p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T z_j)` and reconstructed adjacency `sigmoid(Z Z^T)` in matrix form. The dot product is symmetric in its two arguments by construction, so an undirected edge gets one well-defined score instead of two that could disagree, and it carries no parameters of its own and no dependence on how the nodes happen to be indexed. A pair MLP could be more flexible, but it would add parameters and I would have to symmetrize it by hand. The dot product gives me symmetry and permutation-invariance for free, so it is the clean first choice.

With a GCN encoder and that decoder, I already have the deterministic graph auto-encoder:

```text
Z = GCN(X, A)
A_hat = sigmoid(Z Z^T).
```

Training it as a Bernoulli reconstruction loss exposes the sparsity problem again. If I average binary cross-entropy over all `N^2` pairs, zeros dominate. The dense fix is to up-weight positive entries and rescale the mean loss; the equivalent sparse fix is to sample non-edges and train on positives plus sampled negatives. These are the same case split in spirit: either reweight the rare ones or avoid showing all easy zeros at once.

Now I ask what the deterministic model is missing. It learns one point per node, and the only pressure on the space is reconstruction. A latent-variable version gives me a principled regularizer: draw node latents from an approximate posterior, reconstruct the adjacency from them, and pull the posterior toward a simple prior. The standard variational objective already has exactly that shape,

```text
ELBO = E_q[log p(A | Z)] - KL(q(Z | X,A) || p(Z)).
```

I have to adapt the VAE to a graph. A usual VAE encoder maps one independent datapoint to one posterior. Here, a node's posterior should depend on its neighborhood and features, so the recognition model has to be a GCN over the observed graph. I keep a factorized posterior over nodes for tractability, but each factor is produced by message passing:

```text
q(Z | X,A) = prod_i q(z_i | X,A)
q(z_i | X,A) = N(z_i | mu_i, diag(sigma_i^2)).
```

The encoder now needs two outputs per node. I use the same first GCN layer for both because the mean and uncertainty should be read from the same neighborhood evidence. Then two linear GCN heads produce `mu` and an unconstrained real number I read as `log sigma`. The scale is recovered as `sigma = exp(logstd)`, which is always positive whatever real value the head emits — no separate positivity constraint to enforce.

To sample while keeping gradients, I use the reparameterization trick:

```text
z_i = mu_i + sigma_i * eps_i,     eps_i ~ N(0,I).
```

The randomness is now in `eps`, not in a parameter-dependent sampling operation, so the reconstruction term can be differentiated through `mu` and `logstd`. At evaluation time I should not inject sampling noise into link scores — I want a deterministic ranking of held-out pairs, not one that shifts from run to run — so I drop the sampling step and read off the posterior mean `mu` directly.

The KL term is where a sign or factor error would silently change the method, so I derive it carefully. For one latent coordinate with `q = N(mu, sigma^2)` and `p = N(0,1)`,

```text
E_q[log p(z)] = -0.5 log(2*pi) - 0.5 (mu^2 + sigma^2)
E_q[log q(z)] = -0.5 log(2*pi) - 0.5 (1 + log sigma^2).
```

Subtracting gives the contribution that is added to the ELBO:

```text
-KL(q || p) = 0.5 * (1 + log sigma^2 - mu^2 - sigma^2).
```

which is the familiar closed form for a Gaussian against a standard normal, `-0.5 (s^2 + m^2 - 1 - log s^2)`, with the sign flipped so it reads as a bonus added to the ELBO rather than a penalty subtracted from it.

Summing over latent dimensions gives the negative KL contributed by a single node; adding those contributions across all nodes gives the total. Since my encoder emits `logstd = log sigma`, one node's contribution has the implementation form

```text
-KL = 0.5 * sum_j(1 + 2 * logstd_j - mu_j^2 - exp(2 * logstd_j)).
```

This bracket is exactly zero when the posterior already sits at the prior (`mu = 0`, `sigma = 1`) and negative whenever it drifts away — it is a bonus added to the ELBO that shrinks as the posterior deviates from the prior. When I minimize a loss rather than maximize the ELBO, I flip its sign and add the positive KL instead, which is zero at the prior and strictly positive otherwise, exactly the pull-toward-prior behavior a regularizer needs:

```text
KL = -0.5 * sum_j(1 + 2 * logstd_j - mu_j^2 - exp(2 * logstd_j)).
```

Before I fix the KL's weight I have to reckon with a tension between the prior and the decoder. The standard-normal prior pulls every `z_i` toward the origin with unit variance on each coordinate, but the inner-product decoder needs `sigmoid(z_i^T z_j)` to sit close to `1` for a true edge, which only happens when connected nodes have enough norm and enough alignment to make that dot product comfortably positive — exactly what a small, zero-centered latent space makes hard. A KL term at full node-averaged strength fights the decoder on every step: it would keep dragging the embeddings back toward a region that makes it harder for the decoder to separate true edges from non-edges. So instead of adding the node-averaged bracket at unit weight, I divide it by the number of nodes a second time before adding it to the loss:

```text
KL added to loss = (1/N) * [-0.5 * mean_i sum_j(1 + 2*logstd_ij - mu_ij^2 - exp(2*logstd_ij))].
```

The `mean_i` inside the bracket has already divided the total sum over nodes and latent dimensions by `N` once; the extra `1/N` in front divides it a second time. This keeps the KL's gradient reaching every node — the space still gets pulled toward a continuous, well-behaved prior rather than being left completely free — but at a small enough fraction of its raw strength that reconstruction wins the tug-of-war over where the embeddings actually sit. Concretely, this means the helper that returns the positive node-averaged bracket, `kl_loss()`, has to be divided by `num_nodes` once more wherever it is added to a sampled-edge loss.

There is one more notation choice to pin down: if I define the adjacency with diagonal entries already set to one, the propagation rule reads `D^{-1/2} A D^{-1/2}` directly, with self-loops built into `A`. But the GCN layer I am borrowing takes a plain adjacency without self-loops and adds `I` itself during normalization. These describe the same matrix at two different points in the pipeline — self-loops folded into the input, or added by the layer — so I only need to pick one and stay consistent: I let the GCN layer add its own self-loops during normalization and hand it a self-loop-free `A`.

Putting these pieces together gives the variational graph auto-encoder: shared GCN trunk, mean and log-standard-deviation heads, reparameterized node latents during training, posterior mean at evaluation, and the same inner-product Bernoulli decoder. The deterministic graph auto-encoder is the special case where I remove the log-standard-deviation head, the sampling step, and the KL. The featureless case is also simple: set `X = I`, so structure is the only input signal.

This analysis fixes every remaining piece: a shared GCN layer feeding two separate head convolutions for `mu` and `logstd`, and an encoder that only draws its `eps` and reparameterizes while training — at evaluation it returns `mu` untouched, for the deterministic-ranking reason above. The decoder is the bare inner product and `kl_loss` is the node-averaged bracket derived above; neither one has anything left to decide. I do not clamp `logstd`: nothing in the derivation calls for an artificial ceiling on it.

For training on sampled edges, the reconstruction term stays the plain binary cross-entropy over positive and sampled-negative pairs already given, and the one addition is the KL term at the scale worked out above, divided by the node count as derived.

If instead I train on the full dense objective, I flatten all entries of `Z Z^T` and compute `pos_weight = (N^2 - sum(adj_train)) / sum(adj_train)` and `norm = N^2 / ((N^2 - sum(adj_train)) * 2)`, minimizing `norm * weighted_BCE - negative_KL_term` against the training adjacency. To see that these two constants pull in the direction I want: take `N = 4` with `P = sum(adj_train) = 6` positive entries, so `N^2 = 16` and `neg = N^2 - P = 10`. Then `pos_weight = 10/6 = 1.667`, greater than one, so each positive entry is up-weighted relative to a zero — the correct direction for a graph that is mostly zeros. Weighting alone would inflate the mean loss by roughly `pos_weight*P + neg` against a flat `P + neg`, so `norm = 16/(10*2) = 0.8` rescales that inflated weighted mean back down to the same order of magnitude an unweighted per-entry average would already sit at, rather than one dominated by the up-weighting. That is the same imbalance fix as negative sampling, applied densely — the sparse loss above is the same model, applied to the zero-subsampling case instead.
