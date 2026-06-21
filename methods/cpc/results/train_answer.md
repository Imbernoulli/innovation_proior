The problem is to learn useful, generic representations from high-dimensional sequential data without labels. Supervised learning works but is expensive and task-specific; the real goal is an unsupervised signal that captures slowly-varying global structure—the phoneme, the object, the speaker, the topic—rather than local texture and noise. The natural starting point is prediction: force a model to predict the future from the past, and it must discover the regularities in the data. But the obvious implementation, a conditional generative model p(x_{t+k}|c_t) that reconstructs every sample of the future observation, is a poor fit for representation learning. A raw image or audio segment carries thousands of bits, while the high-level factor we care about is on the order of ten bits. A reconstruction loss therefore spends almost all of its capacity on low-level detail, and a powerful decoder can model the future while largely ignoring the context. Predicting only one step ahead is even worse, because local smoothness already gives the answer and the model never needs to learn global structure. The real target is not the future observation itself, but the information shared between the context and the future.

That shared information is mutual information, I(x;c), whose defining quantity is the density ratio p(x|c)/p(x). This ratio is a far smaller object than either full density: it cancels everything common between "x in general" and "x given this context" and keeps only what the context changes. If we can learn that ratio directly, without ever modeling p(x) or p(x|c), we capture exactly the shared bits and pay nothing for the low-level detail. The question is how to estimate a density ratio without normalizing densities. The answer comes from noise-contrastive estimation and importance sampling: instead of computing intractable normalizers, sample against them. Build a classification task with one true future paired with a context and several alternative futures drawn from the marginal distribution, then train a model to identify the real one. The optimum of this contrastive task is precisely the density ratio, independent of the number of negatives.

The method is Contrastive Predictive Coding, or CPC. It has three parts. First, a nonlinear encoder g_enc maps each observation x_t to a latent vector z_t, already at lower temporal resolution so local detail is compressed. Second, an autoregressive model g_ar, typically a GRU, rolls the prefix z_{≤t} into a single context vector c_t. Third, for each lookahead step k, a linear predictor W_k maps c_t to the latent space, and the unnormalized score f_k(x_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t) is interpreted as proportional to p(x_{t+k}|c_t)/p(x_{t+k}). The future is encoded, never reconstructed.

Training uses the InfoNCE loss. For each context c_t and each step k, we form a set of N candidates containing the true future x_{t+k} plus N−1 negatives drawn from the marginal. In practice these negatives are simply the futures of the other examples in the minibatch. The loss is the categorical cross-entropy of classifying the positive among the candidates, which is just a softmax over the scores f_k. Minimizing InfoNCE pushes the model toward the true density ratio and provides a lower bound on the mutual information: I(x_{t+k}; c_t) ≥ log N − L_N. Larger batch sizes raise the ceiling of certifiable information, and predicting several steps ahead forces the model off easy local smoothness and onto the slow global factors that persist across long horizons. After training, the encoder and context network are frozen and a linear probe on c_t recovers high-level labels.

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

        # g_enc: strided conv stack on raw waveform -> z_t, total downsample 160
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

        # one predictor W_k per lookahead step: c_t (256) -> predicted z (512)
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
        encoded_steps = self.seq_len // 160
        t = torch.randint(encoded_steps - self.timestep, size=(1,), device=x.device).item()

        z = self.encoder(x).transpose(1, 2)  # (B, L, 512)

        # true future latents z_{t+1..t+K}
        encode_samples = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(1, self.timestep + 1):
            encode_samples[k - 1] = z[:, t + k, :].view(batch, 512)

        # context from prefix z_{<=t}
        output, hidden = self.gru(z[:, :t + 1, :], hidden)
        c_t = output[:, t, :].view(batch, 256)

        # predictions W_k c_t
        pred = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(self.timestep):
            pred[k] = self.Wk[k](c_t)

        nce = 0.0
        correct = None
        for k in np.arange(self.timestep):
            # log-bilinear scores: (B, B), diagonal = positives, off-diagonal = negatives
            total = torch.mm(encode_samples[k], pred[k].t())
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0),
                         torch.arange(0, batch, device=x.device))
            )
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce = -nce / (batch * self.timestep)  # InfoNCE
        accuracy = correct.float() / batch
        return accuracy, nce, hidden

    def extract(self, x, hidden):
        z = self.encoder(x).transpose(1, 2)
        output, hidden = self.gru(z, hidden)
        return output, hidden


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
