Self-supervised learning uses the same high-level recipe in every modality — corrupt the input, then predict the missing part — but in practice each domain has ended up with its own prediction target. BERT predicts sub-word ids from a closed vocabulary, wav2vec 2.0 and HuBERT have to learn or cluster discrete speech units, BEiT relies on an offline discrete visual tokenizer, and MAE regresses raw pixels. These targets are not only modality-specific, requiring bespoke machinery to manufacture; they are also local, carrying information about a single position and largely ignoring the surrounding context. The result is a zoo of incompatible pretraining objectives that share a slogan but not an algorithm.

The core failing is that the target itself is always hand-designed for the modality. What we need is a target that can be defined identically for speech, vision, and language, that needs no predefined vocabulary or codebook, and that is rich enough to replace the local targets above. The most general object available in any modality is the model's own latent representation: once input patches, audio frames, or tokens have passed through a Transformer, every position becomes a hidden vector. If we make the student predict those hidden vectors at masked positions, the objective becomes literally the same for all three modalities.

The method I propose is called data2vec. It uses a single Transformer architecture in a self-distillation setup. The student receives a masked version of the input; a teacher, built as an exponential moving average of the student's weights, receives the unmasked, full input. At each masked position, the student regresses the teacher's contextualized latent representation of that position. Because the teacher sees the complete input, its representation at position t is informed by self-attention over the entire sample, so the target is not local but contextualized and example-specific. The only modality-specific pieces left are the front-end feature encoder and the masking pattern, both of which can be borrowed from prior work.

The target is constructed from the teacher's top-K block features rather than only the final layer. For each block, we take the FFN output just before its last residual connection, normalize it, and average the normalized outputs across the top K blocks. Using multiple layers gives a richer, multi-scale target and avoids the top-layer over-specialization that hurts downstream transfer, especially in speech. The normalization serves two purposes: it prevents collapse to a constant target by forcing representations to have variance, and it equalizes the scales of different layers so no single layer dominates the average. Vision and NLP use a parameter-less LayerNorm over the feature dimension, while speech uses instance normalization over the time dimension because adjacent audio frames are highly correlated.

Training proceeds with a Smooth L1, or Huber, regression loss computed only at masked positions. The loss is summed over the feature dimension and scaled by one over the square root of the dimension to keep magnitudes stable across model sizes. The EMA teacher's decay is annealed linearly from a smaller value early in training to a larger value later: the teacher tracks the student quickly while the student is still random, then becomes more stable once training converges. If the decay is too low, student collapse can propagate into the teacher, so the annealing schedule and the normalization together keep the targets varied. For vision, the same augmented view is fed to both teacher and student; the only difference is the masking.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Data2Vec(nn.Module):
    def __init__(self, feature_encoder, encoder, dim, K, beta=2.0,
                 ema_decay=0.999, ema_end=0.9999, ema_anneal_steps=30000):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.encoder = encoder
        self.mask_emb = nn.Parameter(torch.zeros(dim))
        self.final_proj = nn.Linear(dim, dim)
        self.K, self.beta = K, beta
        self.ema = None
        self.ema_decay = ema_decay
        self.ema_end = ema_end
        self.ema_anneal_steps = ema_anneal_steps

    def set_num_updates(self, step):
        if step >= self.ema_anneal_steps:
            decay = self.ema_end
        else:
            decay = self.ema_decay + (self.ema_end - self.ema_decay) * step / self.ema_anneal_steps
        self.ema.set_decay(decay)
        self.ema.step(self.encoder)

    def forward(self, source, mask):
        feats = self.feature_encoder(source)

        x = feats.clone()
        x[mask] = self.mask_emb
        x, _ = self.encoder(x, return_layer_results=False)

        with torch.no_grad():
            self.ema.model.eval()
            _, layer_results = self.ema.model(feats, return_layer_results=True)
            blocks = [F.layer_norm(b.float(), b.shape[-1:]) for b in layer_results[-self.K:]]
            y = sum(blocks) / len(blocks)
            y = F.layer_norm(y.float(), y.shape[-1:])
            y = y[mask]

        x = self.final_proj(x[mask])

        d = x.size(-1)
        if self.beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(-1)
        else:
            loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.beta).sum(-1)
        return loss.sum() / math.sqrt(d)
```
