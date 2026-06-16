Tacotron 2 synthesizes speech from text by joining a learned
character-to-mel-spectrogram sequence model to a WaveNet vocoder conditioned on
the predicted mel frames. WaveNet supplies the strong neural waveform generator
that earlier text-to-speech pipelines lacked, while the sequence model removes
the need for a hand-built linguistic front end with lexicon, duration, and F0
features.

The bridge is a mel spectrogram. It is computed directly from audio, so the
feature predictor can be trained on text and spectrogram pairs and the vocoder
can be trained separately on ground-truth-aligned predicted acoustic frames and
waveform samples. It is also smooth, low-dimensional, phase-invariant within a
frame, and weighted toward the frequencies most important for intelligible
speech.

The feature predictor uses a learned 512-dimensional character embedding, a
three-layer convolutional encoder stack with 512 width-5 filters per layer,
batch normalization, ReLU, and dropout 0.5, then a bidirectional LSTM with 512
total units. The decoder is autoregressive: the previous mel frame passes
through a two-layer 256-unit ReLU pre-net, the result is combined with an
attention context, and a stack of two 1024-unit unidirectional LSTMs predicts the
next 80-channel mel frame. A parallel scalar sigmoid predicts the stop token,
and inference ends when that probability exceeds 0.5. There is no reduction
factor; one decoder step predicts one mel frame.

Attention is additive attention extended with location features. The cumulative
attention weights from previous decoder steps are convolved with 32 one-
dimensional filters of length 31, projected into the 128-dimensional attention
space, and added inside the Bahdanau score:
e_ij = v^T tanh(W s_{i-1} + V h_j + U f_ij). This gives the alignment scoring
function a memory of what has already been consumed, suppressing stalls, skips,
and repeated text.

After the decoder emits the mel sequence, a five-layer convolutional post-net
with width-5 filters, 512 channels in the hidden layers, batch normalization, and
tanh on all but the last layer predicts a residual that is added to the decoder
mel. The mel objective is the sum of mean squared error before the post-net and
mean squared error after the post-net, with the stop token trained as a binary
completion prediction.

Mel features use a 50 ms STFT window, 12.5 ms hop, Hann window, 80 mel channels
from 125 Hz to 7.6 kHz, and a magnitude floor of 0.01 before log compression.
The WaveNet vocoder is trained on ground-truth-aligned predictions from the
feature network: the feature network runs in teacher-forcing mode so each
predicted frame aligns exactly with the target waveform samples. The vocoder
keeps the WaveNet structure of 30 dilated convolution layers in three dilation
cycles, with dilation 2^(k mod 10), uses two conditioning upsampling layers for
the 12.5 ms mel hop, and predicts a 10-component mixture of logistics over
16-bit 24 kHz waveform samples with negative log-likelihood loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_chars, d=512):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d, kernel_size=5, padding=2),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            for _ in range(3)
        ])
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)

    def forward(self, chars):
        x = self.embed(chars).transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        memory, _ = self.lstm(x.transpose(1, 2))
        return memory

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attn_dim=128, n_filters=32, kernel_size=31):
        super().__init__()
        self.query = nn.Linear(1024, attn_dim, bias=False)
        self.memory = nn.Linear(512, attn_dim, bias=False)
        self.location_conv = nn.Conv1d(
            1, n_filters, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.location = nn.Linear(n_filters, attn_dim, bias=False)
        self.energy = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, memory, projected_memory, cumulative_weights):
        loc = self.location_conv(cumulative_weights.unsqueeze(1)).transpose(1, 2)
        loc = self.location(loc)
        score = self.energy(
            torch.tanh(self.query(query).unsqueeze(1) + projected_memory + loc)
        ).squeeze(-1)
        weights = F.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights

class Prenet(nn.Module):
    def __init__(self, n_mels=80, hidden=256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_mels, hidden),
            nn.Linear(hidden, hidden),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.5, training=True)
        return x

class Decoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.prenet = Prenet(n_mels)
        self.attention_rnn = nn.LSTMCell(256 + 512, 1024)
        self.attention = LocationSensitiveAttention()
        self.decoder_rnn = nn.LSTMCell(1024 + 512, 1024)
        self.frame = nn.Linear(1024 + 512, n_mels)
        self.stop = nn.Linear(1024 + 512, 1)

    def step(self, previous_mel, memory, projected_memory, state):
        h_att, c_att, h_dec, c_dec, context, cumulative_weights = state
        prenet_out = self.prenet(previous_mel)
        h_att, c_att = self.attention_rnn(
            torch.cat([prenet_out, context], dim=-1), (h_att, c_att)
        )
        context, weights = self.attention(
            h_att, memory, projected_memory, cumulative_weights
        )
        cumulative_weights = cumulative_weights + weights
        h_dec, c_dec = self.decoder_rnn(
            torch.cat([h_att, context], dim=-1), (h_dec, c_dec)
        )
        decoder_context = torch.cat([h_dec, context], dim=-1)
        return (
            self.frame(decoder_context),
            self.stop(decoder_context),
            (h_att, c_att, h_dec, c_dec, context, cumulative_weights),
        )

class Postnet(nn.Module):
    def __init__(self, n_mels=80, channels=512):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),
            )
        ]
        for _ in range(3):
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),
            ))
        layers.append(nn.Sequential(
            nn.Conv1d(channels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Dropout(0.5),
        ))
        self.net = nn.Sequential(*layers)

    def forward(self, mel_before):
        return self.net(mel_before)

def apply_postnet(mel_before, postnet):
    residual = postnet(mel_before)
    return mel_before + residual

def feature_loss(mel_before, mel_after, stop_logits, mel_target, stop_target):
    mel_mse = F.mse_loss(mel_before, mel_target) + F.mse_loss(mel_after, mel_target)
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_target)
    return mel_mse + stop_loss

class WaveNetVocoder(nn.Module):
    def __init__(self, upsample_kernels, upsample_strides,
                 n_mels=80, n_mix=10, channels=512):
        super().__init__()
        self.conditioning_upsample = nn.ModuleList([
            nn.ConvTranspose1d(
                n_mels, n_mels, kernel_size=k, stride=s
            )
            for k, s in zip(upsample_kernels, upsample_strides)
        ])
        self.dilated = nn.ModuleList([
            nn.Conv1d(
                channels, 2 * channels, kernel_size=2,
                dilation=2 ** (k % 10), padding=2 ** (k % 10)
            )
            for k in range(30)
        ])
        self.output = nn.Linear(channels, 3 * n_mix)

    def forward(self, predicted_mel, audio_prefix):
        raise NotImplementedError
```
