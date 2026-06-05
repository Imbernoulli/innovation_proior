# Tacotron 2

## Problem

Synthesize speech from text that is hard to distinguish from a human, learning as
much of the pipeline as possible from data. WaveNet vocodes beautifully but
requires a hand-engineered linguistic front end (text analysis, lexicon, F₀,
durations); Tacotron learns the front end (char→spectrogram) but vocodes with
Griffin-Lim, which caps naturalness. The two failures are complementary.

## Key idea

Join a learned **char→mel-spectrogram** seq2seq front end to a **WaveNet vocoder
conditioned on the predicted mel**, with the mel spectrogram as the bridge:

- **Mel spectrogram as interface.** Computable from audio (so the two halves
  train separately), smooth, low-dimensional (~80 channels), phase-invariant
  within a frame (well-behaved MSE), and lossy in the perceptually-right way
  (emphasizes low frequencies, compresses highs).
- **Location-sensitive attention.** Additive attention augmented with features
  from the *cumulative* attention weights, biasing the alignment to advance
  monotonically and suppressing the skip/repeat/stall failure modes of TTS.
- **Pre-net bottleneck.** The autoregressive previous-frame input is squeezed
  through 2×256 ReLU layers (dropout left on at inference) so the decoder cannot
  coast on frame-to-frame continuity and is forced to use attention.
- **Stop token + residual post-net.** A sigmoid stop-token (threshold 0.5) ends
  generation; a 5-layer conv post-net predicts a residual refining the mel.
- **WaveNet vocoder** with a 10-component mixture-of-logistics output over 16-bit
  samples at 24 kHz, conditioned on the predicted mel (2 upsampling layers).

## Final structure / loss

Encoder: 512-dim char embedding → 3 conv layers (512 filters, width 5, BN, ReLU,
dropout 0.5) → BiLSTM (512, 256/direction). Attention energy:
e_{ij} = vᵀ tanh(W s_{i-1} + V h_j + U f_{ij}), f from a 32-filter length-31 conv
over the cumulative weights, all projected to 128-dim. Decoder: pre-net (2×256) →
LSTMCell(1024) attention RNN → location-sensitive context → LSTMCell(1024) decoder
RNN → linear to 80-dim mel and to a stop-token scalar. Loss = MSE(mel before
post-net) + MSE(mel after post-net) + BCE(stop token). Vocoder: 30 dilated conv
layers in 3 cycles (dilation 2^(k mod 10)), MoL negative-log-likelihood.
Mel: 50 ms window, 12.5 ms hop, Hann, 80 mel bins 125 Hz–7.6 kHz, clip 0.01, log.
No reduction factor; zoneout 0.1 on LSTMs.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_chars, d=512):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d),
                          nn.ReLU(), nn.Dropout(0.5)) for _ in range(3))
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
    def forward(self, chars):
        x = self.embed(chars).transpose(1, 2)
        for c in self.convs:
            x = c(x)
        return self.lstm(x.transpose(1, 2))[0]                  # (B, L, 512)

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attn_dim=128, n_filters=32, kernel=31):
        super().__init__()
        self.query  = nn.Linear(1024, attn_dim, bias=False)
        self.memory = nn.Linear(512, attn_dim, bias=False)
        self.loc_conv = nn.Conv1d(2, n_filters, kernel, padding=kernel // 2, bias=False)
        self.loc_proj = nn.Linear(n_filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, query, memory, proj_memory, attn_cat):    # attn_cat (B,2,L)
        loc = self.loc_proj(self.loc_conv(attn_cat).transpose(1, 2))
        e = self.v(torch.tanh(self.query(query).unsqueeze(1) + proj_memory + loc))
        weights = F.softmax(e.squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights

class Prenet(nn.Module):
    def __init__(self, n_mels=80, hidden=256):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_mels, hidden), nn.Linear(hidden, hidden)
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=True)  # dropout always on
        return F.dropout(F.relu(self.fc2(x)), 0.5, training=True)

class Decoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.prenet   = Prenet(n_mels)
        self.attn_rnn = nn.LSTMCell(256 + 512, 1024)
        self.attn     = LocationSensitiveAttention()
        self.dec_rnn  = nn.LSTMCell(1024 + 512, 1024)
        self.frame_proj = nn.Linear(1024 + 512, n_mels)
        self.stop_proj  = nn.Linear(1024 + 512, 1)
    def step(self, prev_mel, memory, proj_memory, state):
        h_a, c_a, h_d, c_d, context, cum_w, cur_w = state
        p = self.prenet(prev_mel)
        h_a, c_a = self.attn_rnn(torch.cat([p, context], -1), (h_a, c_a))
        context, cur_w = self.attn(h_a, memory, proj_memory,
                                   torch.stack([cur_w, cum_w], 1))
        cum_w = cum_w + cur_w
        h_d, c_d = self.dec_rnn(torch.cat([h_a, context], -1), (h_d, c_d))
        dc = torch.cat([h_d, context], -1)
        return (self.frame_proj(dc), self.stop_proj(dc),
                (h_a, c_a, h_d, c_d, context, cum_w, cur_w))

class Postnet(nn.Module):
    def __init__(self, n_mels=80, d=512):
        super().__init__()
        layers = [nn.Sequential(nn.Conv1d(n_mels, d, 5, padding=2),
                                nn.BatchNorm1d(d), nn.Tanh(), nn.Dropout(0.5))]
        for _ in range(3):
            layers.append(nn.Sequential(nn.Conv1d(d, d, 5, padding=2),
                                        nn.BatchNorm1d(d), nn.Tanh(), nn.Dropout(0.5)))
        layers.append(nn.Sequential(nn.Conv1d(d, n_mels, 5, padding=2),
                                    nn.BatchNorm1d(n_mels), nn.Dropout(0.5)))
        self.net = nn.Sequential(*layers)
    def forward(self, mel):
        return self.net(mel)                                    # residual

def feature_loss(mel_before, mel_after, stop_logits, mel_tgt, stop_tgt):
    mse = F.mse_loss(mel_before, mel_tgt) + F.mse_loss(mel_after, mel_tgt)
    return mse + F.binary_cross_entropy_with_logits(stop_logits, stop_tgt)

class WaveNetVocoder(nn.Module):                     # conditioned on predicted mel
    def __init__(self, n_mels=80, n_mix=10, channels=512):
        super().__init__()
        self.upsample = nn.ModuleList(                          # 2 layers (12.5ms hop)
            nn.ConvTranspose1d(n_mels, n_mels, 2 * f, stride=f) for f in (16, 16))
        self.dilated = nn.ModuleList(
            nn.Conv1d(channels, 2 * channels, 2, dilation=2 ** (k % 10),
                      padding=2 ** (k % 10)) for k in range(30))   # 3 cycles
        self.out = nn.Linear(channels, 3 * n_mix)               # mean, log-scale, weight
    def forward(self, mel, audio):
        ...                                                     # MoL negative log-likelihood
```
