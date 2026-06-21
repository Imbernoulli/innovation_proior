We want an autoencoder that learns a *useful* representation — one that holds onto the high-level, semantically important structure of the data (an object spanning thousands of pixels, the words in a speech clip, the layout of a room) instead of squandering its capacity on local noise and imperceptible texture — and we want that representation to be *discrete*. The discreteness is not aesthetics: language is already a sequence of symbols, speech transcribes to phonemes, an image is well summarized by a one-sentence caption, and symbol-level codes are exactly what downstream reasoning, planning, and compression want. The obstacle is that the latent-variable machinery that trains well is built for *continuous* latents. The variational autoencoder maximizes the evidence lower bound $\log p(x) \ge \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathrm{KL}(q(z|x)\,\|\,p(z))$, and with Gaussian $q$ and $p$ it trains beautifully because the reparameterization trick $z = \mu(x) + \sigma(x)\,\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$, moves the randomness off the parameters and delivers a low-variance gradient into the encoder. The moment we make $q(z|x)$ categorical and sample a symbol, that trick is gone. The fallbacks all have a wall: score-function (REINFORCE) estimators as in NVIL and VIMCO are unbiased but their variance is brutal, so they spend their whole budget on baselines and many samples and still never close the likelihood gap with a Gaussian VAE; Gumbel-softmax / Concrete relaxations are reparameterizable but trade bias for variance along the temperature anneal — soft and biased when hot, low-bias but high-variance when nearly one-hot — and they too had not closed the gap.

There is a second, entangled wall. The strongest likelihood models around are powerful autoregressive decoders — PixelCNN over an image grid, WaveNet over audio — that factorize $p(x) = \prod_i p(x_i\,|\,x_{<i})$ and model the data extremely well on their own. Bolt one onto a VAE and the latent goes silent. The reason is precise: the KL term is minimized, driven all the way to zero, by setting $q(z|x) = p(z)$, i.e. by making the posterior ignore $x$. If the decoder can model $p(x)$ unaided, the optimizer takes the free KL reduction and loses almost nothing on reconstruction, so it switches the latent off. That is posterior collapse, and it is the objective itself instructing the model to discard exactly the thing we are trying to learn.

I propose VQ-VAE. The single decision that opens the path is to make the posterior a *deterministic one-hot* over a finite codebook and hold the prior *uniform* during autoencoder training. With a one-hot $q$ placing all its mass on one of $K$ symbols and a fixed uniform prior, the KL collapses to a constant: $\mathrm{KL}(q\,\|\,p) = \sum_z q(z)\log\frac{q(z)}{p(z)} = 1\cdot\log\frac{1}{1/K} = \log K$ per latent position, and $N\log K$ for $N$ independent positions. Being constant in $x$ and in the encoder parameters, its gradient is zero and it drops out of training entirely. This is the antidote to collapse: there is no longer any KL reward for pushing the posterior back to a data-independent prior, so even a very strong decoder gives the optimizer no free win for silencing the latent channel. The only way left to improve the objective is the data term. The uniform prior is purely a training device; to generate later we will learn a real prior over the resulting codes.

Making that one-hot concrete is vector quantization. Keep a codebook $e \in \mathbb{R}^{K\times D}$ of $K$ prototype vectors. The encoder emits a continuous vector $z_e(x)\in\mathbb{R}^D$, and we snap it to the nearest prototype, $k = \arg\min_j \|z_e(x) - e_j\|_2$, passing $z_q(x) = e_k$ to the decoder. The posterior is then $q(z=k|x)=1$, deterministic and one-hot, exactly as assumed. The forward pass is clean — encode, snap, decode — but the $\arg\min$ is piecewise constant, so $\partial z_q/\partial z_e = 0$ almost everywhere and the encoder gets no reconstruction gradient. What saves it is geometry: $z_e$ and $z_q$ live in the *same* $D$-dimensional space, and the decoder hands back $\nabla_{z_q}L$, a perfectly sensible direction in that space for where the encoder's output should move to reconstruct better. So we force it through with a straight-through estimator: forward uses the hard value, backward pretends the quantizer was the identity. In an autodiff graph this is
$$z_q = z_e + \mathrm{sg}[\,e_k - z_e\,],$$
where $\mathrm{sg}[\cdot]$ is stop-gradient. Forward, the stop-gradient evaluates to $e_k - z_e$ so $z_q = e_k$ exactly; backward, it contributes nothing, so $\nabla_{z_e}z_q = I$ and the decoder's gradient lands on $z_e$ unchanged, routed cleanly around the $\arg\min$. This is biased — the gradient was computed at $z_q$, which differs from $z_e$ by the quantization residual $e_k - z_e$ — but it is low-variance and cheap, which is precisely the property the score-function and Gumbel routes never delivered.

That same trick opens a gap: by routing the reconstruction gradient *around* the quantizer, it skips the embeddings, so the codebook receives no gradient from reconstruction and the prototypes never move. The fix is to give the codebook its own objective, and the right one is vector quantization's own — k-means: each prototype should sit at the mean of the encoder outputs assigned to it, since that is what keeps $z_q \approx z_e$ and makes the quantizer a small, faithful perturbation. A squared-error term $\|\,\mathrm{sg}[z_e(x)] - e\,\|_2^2$ drags the chosen code toward $z_e$, with the stop-gradient on $z_e$ so this term *only* moves the dictionary, not the encoder. But the latent space is dimensionless and the decoder can absorb any global rescaling, so nothing pins the scale of $z_e$: it can inflate and run away from the prototype it just selected, the prototype chases but never catches, assignments flip, and training never settles. So we add a commitment term $\beta\,\|\,z_e(x) - \mathrm{sg}[e]\,\|_2^2$ with the stop-gradient on $e$, which makes the *encoder* commit to its chosen code, killing the runaway scale and stabilizing assignments. The two stop-gradients are doing real work: there is one quantity, the gap $\|z_e - e\|^2$, and two forces that should close it — the dictionary coming to the encoder, the encoder committing to the dictionary. A single un-split term would let both $z_e$ and $e$ chase each other on the same objective; splitting it cleanly partitions responsibility, one owner per copy. The weight $\beta$ should track the scale of the reconstruction loss; in practice it is robust anywhere in $[0.1, 2.0]$, so we fix $\beta = 0.25$.

The complete training loss for one latent position, written to minimize, is the three terms
$$L = -\log p(x\,|\,z_q(x)) \;+\; \|\,\mathrm{sg}[z_e(x)] - e\,\|_2^2 \;+\; \beta\,\|\,z_e(x) - \mathrm{sg}[e]\,\|_2^2,$$
the first (negative data log-likelihood, or an MSE surrogate) training the decoder directly and the encoder through the straight-through copy, the second training only the codebook, the third only the encoder; over a whole field of latents the auxiliary terms are averaged across positions. The likelihood story holds up as a VAE: $\log p(x) = \log\sum_k p(x|z_k)p(z_k) \ge \log p(x|z_q(x))\,p(z_q(x))$ by keeping a single nonnegative summand (monotonicity of $\log$ over a partial sum), and after training the selected code is taken to dominate that sum.

There is a slicker, optimizer-independent way to handle the codebook, worth deriving because the squared-error term couples the dictionary update to whatever optimizer runs the encoder. Term two is k-means; fixing assignments, minimizing $\sum_j \|z_{i,j} - e_i\|^2$ over $e_i$ has the closed form $2\sum_j(e_i - z_{i,j}) = 0$, i.e. $e_i = \frac{1}{n_i}\sum_j z_{i,j}$, the mean of its assigned points. We cannot compute that mean exactly online, so we track it as an exponential moving average of a running count and running sum,
$$N_i \leftarrow \gamma N_i + (1-\gamma)\,n_i,\qquad m_i \leftarrow \gamma m_i + (1-\gamma)\sum_j z_{i,j},\qquad e_i = m_i / N_i,$$
with $\gamma \approx 0.99$. This is not a gradient step at all, so it is independent of the encoder/decoder optimizer and tends to train faster; a code that stops getting assignments would have $N_i \to 0$ and blow up, so the counts are Laplace-smoothed to keep things finite and give rare codes a chance. With the EMA update, only the commitment term remains in the loss.

One piece the design defers is generation. Holding the prior uniform is what made the KL constant and killed collapse, but a uniform prior cannot generate — the real codes are heavily structured (neighboring grid positions are correlated, audio codes follow temporal regularities), so sampling them uniformly and decoding yields noise. So after the autoencoder is trained and the codes are fixed, we fit a *separate*, expressive autoregressive model over the discrete codes: for images a PixelCNN over the 2D latent grid (single channel of indices, so only spatial masking is needed), for audio a WaveNet over the 1D code sequence. We then ancestral-sample a field of codes from this learned prior and decode it.

The bottleneck, computing pairwise distances to every code, snapping to the nearest, building the two latent losses with the stop-gradients in the right places, and applying the straight-through copy, is below, together with the EMA variant and the surrounding model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Codebook with the three-term loss and straight-through gradient."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.D, self.K, self.beta = embedding_dim, num_embeddings, commitment_cost
        self.embedding = nn.Embedding(self.K, self.D)              # e in R^{K x D}
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):                       # z_e: [B, D, H, W]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, self.D)

        # ||z_e - e||^2 = ||z_e||^2 - 2 z_e.e + ||e||^2
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)                                       # nearest code
        encodings = F.one_hot(idx, self.K).to(flat.dtype)
        indices = idx.view(z_e.shape[:-1])                         # [B, H, W]
        z_q = self.embedding(idx).view(z_e.shape)                 # z_q = e_k

        codebook_loss = F.mse_loss(z_q, z_e.detach())            # term 2: e -> z_e
        commit_loss   = F.mse_loss(z_q.detach(), z_e)            # term 3: z_e -> e
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q = z_e + (z_q - z_e).detach()                          # straight-through
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, vq_loss, indices, perplexity


class VectorQuantizerEMA(nn.Module):
    """Codebook updated by EMA k-means; only the commitment term is in the loss."""
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.D, self.K = embedding_dim, num_embeddings
        self.beta, self.decay, self.eps = commitment_cost, decay, eps
        embed = torch.randn(self.K, self.D)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))  # N_i
        self.register_buffer("ema_w", embed.clone())               # m_i

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, self.D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))
        idx = dist.argmin(1)
        onehot = F.one_hot(idx, self.K).to(flat.dtype)
        indices = idx.view(z_e.shape[:-1])
        z_q = self.embedding[idx].view(z_e.shape)

        if self.training:
            with torch.no_grad():
                n = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                dw = onehot.t() @ flat.detach()
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                N = self.cluster_size.sum()
                cluster = (self.cluster_size + self.eps) / (N + self.K * self.eps) * N
                self.embedding.copy_(self.ema_w / cluster.unsqueeze(1))

        commit_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
        z_q = z_e + (z_q - z_e).detach()
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, commit_loss, indices, perplexity


class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super().__init__()
        self.encoder, self.decoder, self.vq = encoder, decoder, vq

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices, perplexity = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, indices, perplexity


def train_step(model, x, optimizer):
    optimizer.zero_grad()
    x_rec, vq_loss, _, _ = model(x)
    loss = F.mse_loss(x_rec, x) + vq_loss     # MSE surrogate + VQ auxiliary terms
    loss.backward()
    optimizer.step()
    return loss
```
