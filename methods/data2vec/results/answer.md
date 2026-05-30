# data2vec — A General Framework for Self-supervised Learning in Speech, Vision and Language

## Problem

Self-supervised learning uses the same high-level idea everywhere — corrupt the input, predict the missing part — but each modality invented its own *prediction target*: sub-word ids (NLP), learned discrete speech units (wav2vec 2.0, HuBERT), discrete visual tokens or pixels (BEiT, MAE). These targets are modality-specific and *local* (they describe a single position and ignore context). data2vec defines one objective that works identically for speech, vision, and language by removing the per-modality target machinery.

## Key idea

Predict the model's own *contextualized latent representation* of the full input, from a masked view, in a self-distillation setup. One standard Transformer is used as a **student** (encodes the masked input) and as an EMA **teacher** (encodes the unmasked full input). The student regresses, at masked positions only, the teacher's representation — which is contextualized because self-attention over the whole input makes each position's vector carry information from the entire sample. The target is built from a normalized average of the teacher's top-K block features (not just the top layer), giving a rich, open-vocabulary, example-specific target. Only the feature encoder and masking strategy are modality-specific.

## Final method

- **Teacher (EMA):** `Δ ← τΔ + (1−τ)θ`, with τ linearly annealed from τ0 to τe over τn updates, then constant. Feature encoder and positional encoder shared between teacher and student.
- **Target at masked step t:** per-block feature `a_t^l` = the FFN output (prior to the last residual) of teacher block l; normalize each → `â_t^l`; average the top K blocks: `y_t = (1/K) Σ_{l=L−K+1}^L â_t^l`; optionally normalize the average. Vision/NLP use parameter-less LayerNorm; speech uses instance norm over the sequence (adjacent frames correlate). Targets taken only at masked positions.
- **Objective:** Smooth L1 (Huber) regression of `y_t` by the student prediction `f_t(x)`:
  `L = ½(y−f)²/β` if `|y−f| ≤ β`, else `|y−f| − ½β`; summed over the feature dim, scaled by `1/√d`. (L2 also works; β tuned per modality.)
- **Masking / encoders (borrowed):** vision — ViT 16×16 patches, blockwise masking 60%; speech — conv encoder (16 kHz→50 Hz), span masking (p=0.065 starts, length 10, ~49%); text — BPE embeddings, BERT masking. The same modified image is used for teacher and student (no augmentation-invariance game).
- **Vision hyperparameters:** ViT-B (L=12, H=768) / ViT-L (L=24, H=1024); 196 tokens; β=2, K=6, τ=0.9998; Adam, cosine schedule, ViT-B 800 epochs, batch 2048, warmup 40 epochs to lr 0.002; stochastic depth 0.2; EMA in fp32.

## Why these choices

- **Latent target instead of discrete tokens:** no codebook/dVAE/clustering; targets are open-vocabulary and adapt to the example.
- **Teacher sees unmasked input:** makes the target contextualized and informative; masking the teacher hurts.
- **Average top-K layers (not top-1 like BYOL):** different layers carry different features; multi-layer targets are richer and consistently beat top-only; the top layer alone over-specializes.
- **FFN feature, not self-attention output:** the pre-residual self-attention output is biased toward other positions and gives an unusable target; the FFN feature also includes the position's own content.
- **Per-block + average normalization:** prevents collapse to a constant target and stops high-norm layers dominating the average.
- **Annealed EMA:** teacher moves fast while the student is random, slow once it is good; τ too low propagates student collapse into the teacher.
- **Smooth L1:** robust to outlier dimensions versus L2.

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Data2Vec(nn.Module):
    def __init__(self, feature_encoder, encoder, dim, K, beta=2.0,
                 ema_decay=0.999, ema_end=0.9999, ema_anneal_steps=30000):
        super().__init__()
        self.feature_encoder = feature_encoder   # modality-specific (patches / conv / embed)
        self.encoder = encoder                   # standard Transformer; returns per-block feats
        self.mask_emb = nn.Parameter(torch.zeros(dim))
        self.final_proj = nn.Linear(dim, dim)
        self.K, self.beta = K, beta
        self.ema = None                          # EMA copy of self.encoder, built lazily
        self.ema_decay, self.ema_end, self.ema_anneal_steps = ema_decay, ema_end, ema_anneal_steps

    def set_num_updates(self, step):             # anneal tau: fast early, slow late
        decay = self.ema_end if step >= self.ema_anneal_steps else \
            self.ema_decay + (self.ema_end - self.ema_decay) / self.ema_anneal_steps * step
        self.ema.set_decay(decay)
        self.ema.step(self.encoder)              # Delta <- tau*Delta + (1-tau)*theta

    def forward(self, source, mask):
        feats = self.feature_encoder(source)                 # [B, T, D]

        # student: MASKED input
        x = feats.clone()
        x[mask] = self.mask_emb
        x, _ = self.encoder(x, return_layer_results=False)

        # teacher (EMA): UNMASKED full input -> contextualized target
        with torch.no_grad():
            self.ema.model.eval()
            _, layer_results = self.ema.model(feats, return_layer_results=True)
            blocks = [F.layer_norm(b.float(), b.shape[-1:]) for b in layer_results[-self.K:]]
            y = sum(blocks) / len(blocks)                    # y_t = (1/K) sum LN(FFN_t^l)
            y = F.layer_norm(y.float(), y.shape[-1:])        # normalize the average (anti-collapse)
            y = y[mask]                                      # targets at masked positions only

        # student prediction at masked positions
        x = self.final_proj(x[mask])

        # Smooth L1 regression, summed over feature dim, scaled by 1/sqrt(d)
        d = x.size(-1)
        if self.beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(-1)
        else:
            loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.beta).sum(-1)
        return loss.sum() / math.sqrt(d)
```
