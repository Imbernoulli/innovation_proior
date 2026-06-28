OK, let me think this through from scratch. I want an unsupervised model that learns a *useful* representation of data — images, speech, video — and by useful I mean it should hold onto the things that matter (an object spanning thousands of pixels, the words in a speech clip, the layout of a room) and not waste itself memorizing the local noise and texture that no one cares about. And I want the representation to be discrete. That second part isn't an aesthetic preference: language is already a sequence of symbols, speech transcribes to phonemes, an image is well summarized by a one-sentence caption. The natural state of "meaning" for these modalities is a handful of symbols, not a cloud of floats. Discrete codes are also what you'd want for downstream planning and reasoning — if it rains I take an umbrella; that's symbol-to-symbol, not float-to-float — and for compression.

So why doesn't everyone already use discrete latents? Let me lay out the tools I actually have and find where each one breaks.

The obvious frame is the variational autoencoder. I have a prior p(z), a decoder p(x|z), and an encoder q(z|x) that approximates the posterior, and I train by maximizing the evidence lower bound,

  log p(x) ≥ E_{q(z|x)}[log p(x|z)] − KL(q(z|x) || p(z)).

First term reconstructs, second term keeps the posterior near the prior. With Gaussian q and p this is lovely to train because of the reparameterization trick: write z = μ(x) + σ(x)·ε with ε∼N(0,I), and the randomness no longer sits on the parameters, so the gradient of the reconstruction term flows back into the encoder with low variance. That low variance is the whole reason continuous VAEs work as well as they do.

Now I want z discrete. The instant I make q(z|x) categorical and sample a symbol, I lose reparameterization — there's no smooth z = f(params, noise) for "pick category k." What do I have instead? Score-function estimators: differentiate E_q[·] by sampling and weighting by ∇log q. That's REINFORCE. It's unbiased but the variance is brutal; NVIL spends its whole design budget on baselines and variance-reduction just to make a single-sample discrete bound trainable, and VIMCO goes further by taking many samples and using the rest of the batch as a baseline for each one. Both work on small problems — MNIST, latents of dimension under ten — and neither closes the likelihood gap with a plain Gaussian VAE. The variance is the wall.

The other option that people reach for is to relax the discreteness: Gumbel-softmax / Concrete. Replace the hard categorical sample with a continuous, temperature-controlled approximation that *is* reparameterizable, then anneal the temperature down so it hardens into a real category. Clever, but look at what the temperature buys you: when it's high the sample is soft, gradients are low-variance but biased (you're not actually sampling a category); when it's low the thing approaches a real one-hot and the gradient variance shoots up. You're trading bias for variance along the anneal and never getting the clean low-variance signal the Gaussian trick gives. And again, it hadn't closed the gap. So relaxation isn't free either.

Let me set that aside and look at the *other* problem, because I suspect it's tangled with this one. The strongest likelihood models around are powerful autoregressive decoders — PixelCNN over an image grid, WaveNet over audio, both factorizing p(x) = Π_i p(x_i | x_{<i}) and modeling the data extremely well on their own. What happens if I use one as my VAE's decoder? I've seen this fail and I want to understand *why* precisely. Look back at the ELBO. The KL term KL(q(z|x)||p(z)) is minimized — driven all the way to zero — by setting q(z|x) = p(z), i.e. by making the posterior *ignore x entirely*. If the decoder is strong enough to model p(x) without any help from z, then the optimizer can drive KL to zero (free reduction of the objective) and lose almost nothing on the reconstruction term, because the decoder picks up the slack. So it does exactly that. The latent goes silent; the decoder does all the work. That's posterior collapse, and it's not a bug in the optimizer — the objective is *telling* it to throw the latent away. There's even an argument that, measured purely by log-likelihood, the best model is one with a powerful decoder and no informative latent at all. Which is a disaster if your entire goal is to *learn the latent*.

So now I see the two walls touch, but I should be careful not to blame the wrong term. The high-variance discrete-gradient problem comes from differentiating expectations through sampled categories, especially the reconstruction term. The KL term is a separate pressure: with a strong decoder it rewards q for becoming the prior and forgetting x. These are independent levers. The discrete-gradient one I will have to deal with directly. But the KL one — that pressure is created by the KL term being a *function of x* that the optimizer can lower by ignoring x. What if it weren't a function of x at all? Let me see what it takes to make the KL term contribute nothing to the gradient.

Suppose the posterior were *deterministic*: the encoder produces a vector and q(z|x) puts all its mass on one symbol — a one-hot. And suppose the prior p(z) were *uniform* over the K symbols and held fixed during the autoencoder phase. Then the KL is a finite sum I can just evaluate. q is one at the chosen symbol k* and zero elsewhere, p is 1/K everywhere, and 0·log 0 = 0, so only the k* term survives:

  KL(q || p) = Σ_z q(z) log( q(z)/p(z) ) = 1 · log( 1 / (1/K) ) = log K.

Let me actually put numbers on this to make sure I haven't fooled myself. For K = 2 I get log 2 = 0.693; for K = 8, log 8 = 2.079; for K = 512, log 512 = 6.238 — and computing the one-surviving-term sum directly gives exactly those same values. There's no x in any of them, and no encoder parameter either. With N independent latent positions it is N log K, still constant. So in the ELBO its gradient with respect to the encoder is identically zero: this term drops out of autoencoder training entirely. Good — that closes off the posterior-collapse route, because there is now no KL reward for pushing q(z|x) back to a data-independent prior; the decoder can be as expressive as I like and the objective still can't buy a free win by silencing the latent. The only remaining way for the encoder/decoder pair to improve the autoencoder objective is the data term. The fixed-uniform prior is purely a training device, though — I'm spending the prior's expressiveness to buy training stability, so if I ever want to *generate*, I'll have to learn a real prior over the resulting codes afterward. I'll come back to that.

But I've only relocated the hard part. If q is a deterministic one-hot, how do I produce it, and how on earth do I get gradients through it? Let me make the bottleneck concrete. Keep a codebook — an embedding table e ∈ R^{K×D}, K prototype vectors each of dimension D. The encoder outputs a continuous vector z_e(x) ∈ R^D. To discretize, take the nearest prototype:

  k = argmin_j ‖z_e(x) − e_j‖₂,   and pass z_q(x) = e_k to the decoder.

This is just vector quantization — represent a vector by the index of its nearest codebook entry. The posterior is q(z=k|x) = 1 for that k and 0 otherwise, deterministic and one-hot, exactly as I assumed above. Forward pass: encode, snap to nearest code, decode from the code. Clean.

Now the gradient. The data term is log p(x | z_q(x)) if I'm maximizing the ELBO, or -log p(x | z_q(x)) if I write the implementable loss to minimize. Either way, z_q = e_{argmin}. Let me confirm the gradient really is dead rather than just "small." If I build z_q by selecting a row of the embedding table with argmin and ask an autodiff engine whether z_q depends on z_e, it reports `requires_grad = False`: the only path from z_e to z_q ran through the argmin index, which carries no gradient. And the function is genuinely piecewise constant — if I nudge z_e by 1e-3 or 1e-2, the chosen index stays put (I checked: same argmin before and after the nudge), so ∂z_q/∂z_e = 0 in a neighborhood, not just at a measure-zero set. The decoder learns fine because it gets gradient with respect to its own input z_q, but the encoder gets nothing from reconstruction. Dead.

Let me stare at the geometry of what's happening at the quantizer, because the situation isn't as bad as "dead" suggests. z_e and z_q live in the *same* D-dimensional space — z_q is just the nearest codebook point to z_e. The decoder, by backprop, hands me ∇_{z_q} L: the direction in that space I should move the decoder's input to lower reconstruction loss. And z_e is sitting right there in the same space. So even though the map z_e → z_q has zero derivative, the gradient *at z_q* is still a plausible direction to move *z_e*: "to reconstruct better, the encoder's output should head this way," which on the next forward pass might even snap to a better code. The argmin is refusing to pass a signal that, geometrically, would be perfectly usable.

So I'll force it through. On the backward pass, pretend the quantizer was the identity: copy ∇_{z_q} L straight onto z_e. This is the straight-through idea — use the hard value going forward, pretend it's the identity going backward. How do I implement "forward = e_k, backward = identity onto z_e" in an autodiff graph? Try:

  z_q = z_e + sg[ e_k − z_e ],

where sg is stop-gradient (identity in the forward pass, zero derivative in the backward pass). Let me check both directions actually behave. Forward: sg evaluates to e_k − z_e, so z_q = z_e + (e_k − z_e) = e_k — and numerically, building it this way and comparing to e_k directly, they're equal to floating-point tolerance. Backward: I put an arbitrary downstream "decoder gradient" g on z_q and read off the gradient that lands on z_e. It comes back *exactly* equal to g, component for component — the sg branch contributes zero, so ∇_{z_e} z_q = I and the decoder's gradient is copied onto z_e unchanged, routed cleanly around the argmin. So the construction does what I wanted. Is it exact as an estimator of the true gradient? No — the gradient I'm applying to z_e was computed at z_q, and the two differ by the quantization residual e_k − z_e, so it's biased. But it's low-variance and cheap, which is the property I was missing with REINFORCE and never quite got from the Gumbel anneal. I could instead pass a subgradient through the quantization, but the straight copy is the simplest thing that could work, so I'll start there and only complicate it if it fails.

Now a gap opens up that I created by being clever. Trace where the codebook vectors e get their gradients. The reconstruction term: the straight-through trick routes its gradient *around* the quantizer, from z_q directly to z_e — which means the gradient skips the embeddings entirely. The e's appear in the forward pass (z_q = e_k) but in the backward pass the copy-trick bypasses them. So I should expect the codebook to get *no* gradient from reconstruction — and when I actually backprop a reconstruction surrogate through the straight-through bottleneck and inspect `e.grad`, it confirms what the algebra says only after I add a term that explicitly mentions e; with reconstruction alone the embeddings never move. If I do nothing, the prototypes sit wherever they were initialized while the encoder outputs drift around them. That's useless.

So the embeddings need their own training signal. What do I actually want the codebook to do? I want each prototype to sit where the encoder outputs that select it actually are — that's what makes z_q ≈ z_e, which is what makes the quantizer a small, faithful perturbation rather than a huge lossy jump. That's precisely vector quantization's own objective, and it's k-means: each prototype should be the mean of the encoder outputs assigned to it. The simplest way to push toward that with gradient descent is a squared-error term that drags the chosen e toward z_e:

  ‖ z_e(x) − e ‖₂².

But I have to be careful about *who* this term is allowed to move. I want it to move the dictionary toward the encoder, not drag the encoder toward the dictionary — the encoder is already being shaped by reconstruction, and I don't want this auxiliary term yanking it around too. So stop-gradient on z_e:

  ‖ sg[z_e(x)] − e ‖₂².

Let me verify the stop-gradient really partitions the update the way I'm claiming. With only this term active, I backprop and check `z_e.grad`: it comes back as `None` — z_e receives nothing, exactly because sg froze it. So this term moves e and only e; z_e is a frozen target. Each prototype slides toward the centroid of the encoder outputs that picked it. That's online k-means.

Is that enough? Let me imagine training and look for the next wall. The encoder output z_e and the codebook e live in a *dimensionless* space — there's no constraint fixing the scale of this latent. The decoder can absorb any global rescaling. So nothing stops z_e from growing. Worse, picture the dynamics: the codebook term moves e toward z_e, but if the encoder trains faster than the embeddings, z_e can run away from the prototype it just selected, the prototype chases but never catches up, the assignment flips around, and the whole thing fails to settle. The encoder isn't *committing* to any code; its output just inflates and oscillates. This is a real failure of the design as it stands, not a hypothetical — the codebook term alone gives the encoder no reason to stay anywhere near the dictionary.

So I need a term that makes the encoder commit — that keeps z_e close to the embedding it chose, rather than letting it sprint off. Same squared distance, but now the *other* operand is frozen: I want the encoder to move toward the code, not the code to move (the codebook term already handles the code's motion). Stop-gradient on e:

  β · ‖ z_e(x) − sg[e] ‖₂².

This is the commitment loss. Same check as before, mirror image: with only this term active, `e.grad` comes back `None` — the codebook receives nothing, the encoder receives all of it. So it pins z_e near its selected prototype, kills the runaway-scale problem, and stabilizes the assignments. The weight β controls how hard the encoder is made to commit relative to reconstruction; it should track the scale of the reconstruction loss. I'd expect to have to tune it. I'll fix β = 0.25 as a starting point — small enough that commitment is a gentle regularizer rather than competing with reconstruction — but I genuinely don't know yet how sensitive training is to it, and I'd want to sweep a range (say 0.1 to 2) to see whether the result is robust or whether β is a knob I'll be fighting.

Look at what the two stop-gradients did. There's really one quantity here, the squared gap ‖z_e − e‖² between the encoder output and its chosen code, and *two* things that ought to close it: the dictionary should come to the encoder, and the encoder should commit to the dictionary. If I wrote a single un-split term, both z_e and e would receive gradients from it simultaneously and chase each other on the same objective — unstable, and it muddies which part is doing the representation learning. By splitting it with stop-gradients I partition responsibility, and the two `None`-gradient checks above are exactly the confirmation that the partition is clean: sg[z_e] in one copy means "this term only updates the codebook," sg[e] in the other means "this term only updates the encoder." Two forces, two owners, one gap.

Putting it together, the training loss I minimize has three terms:

  L_min = -log p(x | z_q(x))  +  ‖ sg[z_e(x)] − e ‖₂²  +  β ‖ z_e(x) − sg[e] ‖₂².

Term one, the negative data log-likelihood or a reconstruction surrogate such as MSE, trains the decoder directly and the encoder through the straight-through copy; the embeddings get nothing from it. Term two, the codebook/VQ term, trains the embeddings only because the encoder side is frozen by sg. Term three, the commitment term, trains the encoder only because the codebook side is frozen by sg. (I confirmed all three routings above by reading off which of `z_e.grad`/`e.grad` come back nonzero versus `None`.) If I instead write the ELBO-style objective to maximize, the first sign flips back to +log p(x|z_q(x)); the auxiliary squared terms are still penalties. And the KL term that would normally sit in the ELBO is log K per latent position, constant, so it is gone from the gradients. With a whole field of N latents (a 32×32 grid for an image, a long sequence for audio), reconstruction is over the whole field and I average the codebook and commitment terms over positions, just as a reference implementation would average over tensor elements.

Let me sanity-check the likelihood story, since I claimed this is still a VAE. The marginal is log p(x) = log Σ_k p(x|z_k) p(z_k). The decoder is trained with z = z_q(x), the MAP code under the deterministic posterior, so once it has converged the intended approximation is that the sum is dominated by that code: log p(x) ≈ log p(x|z_q(x)) p(z_q(x)). Separately — and this part I can make rigorous rather than approximate — because every summand p(x|z_k)p(z_k) is nonnegative, dropping all summands except the z_q(x) one can only shrink the sum, and log is monotone, so

  log Σ_k p(x|z_k)p(z_k) ≥ log p(x|z_q(x))p(z_q(x))

for *any* code z_q(x), not just the MAP one. Let me check this isn't trivially vacuous by trying it on a random vector of nonnegative joint values: with K = 5 random joints, log of the full sum is −0.11, while the individual one-term logs range from −5.46 down to −0.75, and the bound −0.11 ≥ (each one-term value) holds in every one of the five cases — including the largest single term, where the gap is smallest. So it's a genuine, always-valid lower bound; the MAP claim (that the chosen summand actually *dominates* after training) is the stronger empirical statement that goes beyond what monotonicity alone gives me.

Now, there's a slicker way to handle the codebook than the squared-error term, and it's worth deriving because the squared-error version couples the dictionary update to whatever optimizer I'm running on the encoder. Go back to what term two is really doing: it's k-means. Fix the assignments for a moment; let {z_{i,1}, …, z_{i,n_i}} be the encoder outputs currently assigned to prototype e_i. The term is Σ_j ‖z_{i,j} − e_i‖², and minimizing over e_i has a closed form — set the derivative to zero, 2 Σ_j (e_i − z_{i,j}) = 0, which gives 2(n_i e_i − Σ_j z_{i,j}) = 0, so

  e_i = (1/n_i) Σ_j z_{i,j},

the mean of its assigned points. Let me make sure I haven't just found a stationary point that's a saddle or maximum. I take a handful of points, compute their mean, and evaluate the sum of squared distances there; then I perturb the prototype by small random offsets a couple thousand times and count how many beat the mean. Zero of them do — the mean is the strict minimizer — and the gradient I derived, 2(n·mean − Σ z), evaluates to norm ≈ 5e-7 at the mean, i.e. zero up to floating point. So the closed form is exactly the k-means update. I can't compute that mean exactly online because each minibatch only sees a few of the points, so make it an exponential moving average. Track a running count and a running sum per code,

  N_i ← γ N_i + (1−γ) n_i^{(t)},
  m_i ← γ m_i + (1−γ) Σ_j z_{i,j}^{(t)},
  e_i = m_i / N_i,

with γ around 0.99. Now e_i is just the EMA of the encoder outputs that land on it — the same k-means mean, accumulated online, and completely independent of the encoder/decoder optimizer (it's not a gradient step at all). One subtlety: a code that stops getting assignments has N_i → 0, and then m_i/N_i blows up or the code effectively dies. So smooth the counts a little, Laplace-style — replace N_i with (N_i + ε)/(Σ_i N_i + Kε)·Σ_i N_i — to keep things finite and give rarely-used codes a chance. This EMA variant should train faster since the dictionary tracks its centroids directly instead of crawling there by SGD; the squared-error term is the simpler default and the two are interchangeable in the loss.

One more thing the design needs and I've been deferring: generation. During autoencoder training I deliberately held the prior uniform — that's what made KL constant and killed collapse, and it keeps representation learning decoupled from prior learning. But a uniform prior over codes can't *generate*: sampling codes uniformly and decoding gives noise, because the real codes are heavily structured (neighboring positions in the latent grid are correlated, audio codes follow temporal regularities). So after the autoencoder is trained and the codes are fixed, fit a *separate*, expressive autoregressive model over the discrete codes — the same kind of model that was already strong over discrete observations. For images the codes form a 2D grid, so a PixelCNN over the latent grid (and since it's a single channel of indices, I only need spatial masking, not the cross-channel masking colors require); for audio, a WaveNet over the 1D code sequence. Then generate by ancestral-sampling a field of codes from this learned prior and decoding it to pixels or waveform. Fitting the prior after the fact, rather than jointly, is the simple choice; joint training could help and I'll leave it for later.

Let me write the bottleneck the way it actually goes in code. The forward computes pairwise distances to every code, takes the nearest, looks up the embedding, builds the two latent losses with the stop-gradients in the right places, and applies the straight-through copy so the decoder's gradient reaches the encoder.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.D = embedding_dim          # D: dimension of each code
        self.K = num_embeddings         # K: codebook size (K-way categorical)
        self.beta = commitment_cost     # weight on the commitment term
        # the codebook e in R^{K x D}; trained by the VQ (codebook) loss term
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):
        # z_e: encoder output, [B, D, H, W] -> flatten to [BHW, D]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, self.D)

        # nearest-neighbour code per vector: argmin_j ||z_e - e_j||^2,
        # expanded as ||z_e||^2 - 2 z_e . e + ||e||^2 for an efficient matmul
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)                      # one-hot posterior, as flat indices
        encodings = F.one_hot(idx, self.K).to(flat.dtype)
        indices = idx.view(z_e.shape[:-1])        # [B, H, W], matching Sonnet's shape
        z_q = self.embedding(idx).view(z_e.shape) # z_q(x) = e_k, snapped to nearest

        # codebook term: move e toward z_e (sg freezes the encoder so only e moves)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        # commitment term: make encoder commit to its code (sg freezes the codebook)
        commit_loss = F.mse_loss(z_q.detach(), z_e)
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through: forward returns e_k, backward copies grad to z_e unchanged
        z_q = z_e + (z_q - z_e).detach()

        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # back to [B, D, H, W]
        return z_q, vq_loss, indices, perplexity
```

And the EMA variant of the codebook update, replacing the codebook squared-error term with the online k-means mean, optimizer-independent and with Laplace-smoothed counts so rare codes don't die:

```python
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.D, self.K = embedding_dim, num_embeddings
        self.beta, self.decay, self.eps = commitment_cost, decay, eps
        # codebook is a buffer, not a Parameter: it is updated by EMA, not by SGD
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
        onehot = F.one_hot(idx, self.K).to(flat.dtype)     # assignments
        indices = idx.view(z_e.shape[:-1])
        z_q = self.embedding[idx].view(z_e.shape)

        if self.training:
            with torch.no_grad():
                # N_i <- gamma N_i + (1 - gamma) n_i  (counts per code)
                n = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                # m_i <- gamma m_i + (1 - gamma) sum of z_e assigned to code i
                dw = onehot.t() @ flat.detach()
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                # Laplace smoothing of the counts so unused codes don't blow up / die
                N = self.cluster_size.sum()
                cluster = ((self.cluster_size + self.eps)
                           / (N + self.K * self.eps) * N)
                # e_i = m_i / N_i
                self.embedding.copy_(self.ema_w / cluster.unsqueeze(1))

        # only the commitment term remains in the loss; the codebook is EMA-updated
        commit_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
        z_q = z_e + (z_q - z_e).detach()        # straight-through, as before
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, commit_loss, indices, perplexity
```

Dropping this into the autoencoder is just: encode to z_e, quantize, decode from z_q, and add the bottleneck's loss to the reconstruction loss.

```python
class Model(nn.Module):
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
    loss = F.mse_loss(x_rec, x) + vq_loss   # term 1 + (terms 2,3)
    loss.backward()                          # straight-through carries grad to encoder
    optimizer.step()
    return loss
```

So the causal chain, end to end: I wanted discrete, useful latents that survive a strong decoder, and the decision that opens the path is a deterministic one-hot posterior with a fixed uniform prior during autoencoder training, which I confirmed pins the KL at log K per latent position (0.69, 2.08, 6.24 for K = 2, 8, 512) — removing the KL pressure that silences latents under a powerful decoder. Making that one-hot concrete is vector quantization: encode, snap to the nearest codebook vector, decode. The snap kills the encoder's reconstruction gradient — verified: the quantized output reports no dependence on z_e and is constant under small perturbations — so I route the gradient around the argmin with a straight-through copy, which I checked copies the decoder gradient onto z_e identically while remaining biased but low-variance, the property the score-function and Gumbel routes never delivered cleanly. That copy then starves the codebook of gradient, so the codebook gets its own k-means term (whose mean-of-assigned-points minimizer I verified is the strict minimum) or, equivalently and optimizer-independently, an EMA of its assigned encoder outputs. Because the latent scale is unconstrained and the encoder can run away from its code, a commitment term pins the encoder to what it picked; the two stop-gradients split one gap into "dictionary moves" and "encoder commits," and I confirmed each term gives a `None` gradient to the side it's supposed to leave alone. Finally, since the prior was held uniform for training, I fit an expressive autoregressive prior over the learned codes afterward to generate.
