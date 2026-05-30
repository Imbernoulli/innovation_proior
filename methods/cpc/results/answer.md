# Contrastive Predictive Coding (CPC) and the InfoNCE loss

## Problem

Learn generic, unsupervised representations from high-dimensional sequential data (audio, images, text, agent observations) such that the slowly-varying high-level factors (phonemes, speaker, object class, sentiment) are linearly recoverable from a frozen representation. Predicting the raw future with a conditional generative model p(x_{t+k}|c_t) is wasteful: a high-dimensional observation carries far more bits than its high-level latent, so a reconstruction loss spends capacity on low-level detail and noise and can ignore the context entirely.

## Key idea

Do not model the future; model what the present and the future *share*. The information shared between context c and future x is the mutual information

  I(x; c) = Σ_{x,c} p(x,c) log [ p(x|c) / p(x) ],

whose integrand is the density ratio p(x|c)/p(x) — a much smaller object than either density. CPC learns that ratio directly with an unnormalized score and trains it by a contrastive (noise-contrastive) task, never reconstructing x.

## Architecture

- Encoder: z_t = g_enc(x_t), a non-linear (strided-conv) encoder, optionally at lower temporal resolution.
- Autoregressive summary: c_t = g_ar(z_{≤t}), e.g. a GRU, rolling the past latents into a context vector.
- Density-ratio score (unnormalized, need not integrate to 1), one matrix W_k per look-ahead step k:

  f_k(x_{t+k}, c_t) = exp( z_{t+k}^T W_k c_t )   ∝   p(x_{t+k}|c_t) / p(x_{t+k}).

The target x_{t+k} is encoded into z_{t+k}; it is never generated.

## InfoNCE loss

Form a set X of N samples: one positive future drawn from p(x_{t+k}|c_t) and N−1 negatives drawn from the marginal p(x_{t+k}). Minimize the categorical cross-entropy of identifying the positive:

  L_N = − E_X [ log ( f_k(x_{t+k}, c_t) / Σ_{x_j ∈ X} f_k(x_j, c_t) ) ].

In practice the N−1 negatives are the other minibatch elements' futures (in-batch negatives), so N = batch size.

## Two guarantees

**Optimal critic = density ratio.** The minimizer of L_N matches the true posterior over which slot is the positive. With the positive drawn from p(·|c) and the rest from the marginal,

  p(d=i|X,c) = [ p(x_i|c) ∏_{l≠i} p(x_l) ] / Σ_j [ p(x_j|c) ∏_{l≠j} p(x_l) ]
            = [ p(x_i|c)/p(x_i) ] / Σ_j [ p(x_j|c)/p(x_j) ],

so the optimal f_k(x,c) ∝ p(x|c)/p(x), independent of N.

**MI lower bound.** Insert the optimal f = r = p(x|c)/p(x) and keep the sampled denominator:

  log N − L_N^opt
    = E log[ N r_pos / (r_pos + Σ_neg r_j) ]
    = I(x_{t+k}; c_t)
      − E log[ (1/N)(r_pos + Σ_neg r_j) ].

The last expectation is nonnegative. Define the all-marginal candidate distribution m(X,c)=p(c)∏_j p(x_j), and the distribution q(X,c) where one uniformly chosen candidate is drawn from p(x|c) and the rest from p(x). Then (1/N)Σ_j r_j = q(X,c)/m(X,c), and by symmetry the expectation above is E_q log[q/m] = KL(q||m). Hence I(x_{t+k}; c_t) ≥ log N − L_N^opt, and the same lower bound holds for any f because a worse critic only raises L_N. Larger N raises the maximum certifiable value to log N nats. InfoNCE is also a lower bound on the Donsker–Varadhan / MINE estimator (writing f = e^F and dropping the positive term from the log-sum-exp), but keeping the positive in the denominator makes it far more stable when the target is easy to predict.

## Why it works

- Estimating a density ratio avoids modeling p(x) or p(x|c); only samples and a score are needed.
- Negative sampling (NCE / importance-sampling lineage) replaces the intractable normalization with a classification among samples.
- Predicting several steps ahead removes trivial local smoothness and forces slow, global features into the representation.
- After training, freeze the model and read off c_t (or z_t) as the representation; a linear probe recovers the high-level classes.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SequenceRepresentation(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        # g_enc: strided conv stack on the raw 16kHz waveform -> z_t (downsample 160)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        )
        # g_ar: autoregressive summary of z_{<=t} -> context c_t
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        # one linear predictor W_k per step: c_t (256) -> predicted latent (512)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for _ in range(timestep)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self._init_recurrent_weights()
        self.apply(self._weights_init)

    def _init_recurrent_weights(self):
        for names in self.gru._all_weights:
            for name in names:
                if "weight" in name:
                    nn.init.kaiming_normal_(
                        getattr(self.gru, name),
                        mode="fan_out",
                        nonlinearity="relu",
                    )

    @staticmethod
    def _weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(1, batch_size, 256, device=device)

    def forward(self, x, hidden):
        batch = x.size(0)
        t = torch.randint(self.seq_len // 160 - self.timestep, size=(1,), device=x.device).item()

        z = self.encoder(x).transpose(1, 2)            # (B, L, 512)

        # true future latents z_{t+1..t+K}
        encode_samples = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(1, self.timestep + 1):
            encode_samples[k - 1] = z[:, t + k, :].view(batch, 512)

        # context from the prefix
        output, hidden = self.gru(z[:, :t + 1, :], hidden)
        c_t = output[:, t, :].view(batch, 256)

        # predictions W_k c_t
        pred = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(self.timestep):
            pred[k] = self.Wk[k](c_t)

        nce = 0.0
        correct = None
        for k in np.arange(self.timestep):
            total = torch.mm(encode_samples[k], pred[k].t())   # (B, B) logits
            # diagonal = positive, off-diagonal = in-batch negatives (marginal)
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0),
                         torch.arange(0, batch, device=x.device))
            )
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce = -nce / (batch * self.timestep)           # InfoNCE; certifies <= log(batch) nats
        accuracy = correct.float() / batch             # farthest-step diagnostic
        return accuracy, nce, hidden

    def extract(self, x, hidden):
        z = self.encoder(x).transpose(1, 2)
        output, hidden = self.gru(z, hidden)
        return output, hidden                          # frozen representation c_t


class LinearProbe(nn.Module):
    def __init__(self, num_classes, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)


def train_step(model, batch, optimizer, hidden):
    optimizer.zero_grad()
    _, loss, hidden = model(batch, hidden)
    loss.backward()
    optimizer.step()
    return loss, hidden
```

The same recipe transfers across modalities by swapping g_enc and g_ar: a ResNet patch-encoder with a PixelCNN-style autoregressive model over a grid for images (predict latents of rows below from rows above), a 1-D-conv sentence encoder with a GRU predicting future sentence embeddings for text, and the agent's conv+LSTM encoder with the contrastive term added as an auxiliary loss for reinforcement learning.
