Text-to-speech has been stuck on one trade-off: the more natural the output, the more hand-built machinery it takes to produce it. Concatenative unit-selection stitches recorded waveform snippets and sounds passable, but the joins leave boundary artifacts and the fixed inventory is inflexible. Statistical parametric synthesis instead generates smooth hand-designed acoustic features for a fixed vocoder, which removes the seams but sounds muffled and buzzy, because the features are a lossy summary and the vocoder is a rigid signal-processing model. Two recent systems each fixed one half of the pipeline and left the other broken. WaveNet models the raw waveform autoregressively, $p(x_t \mid x_{<t}, \text{conditioning})$, through a stack of dilated causal convolutions, and its audio quality starts to rival human speech, solving the vocoding half. But it is conditioned on linguistic features, predicted log-$F_0$, and phoneme durations, so producing those still demands a full text-analysis system, a pronunciation lexicon, and a duration model — exactly the hand-engineered front end we want to delete. Going the other way, Tacotron maps characters straight to a magnitude spectrogram with a sequence-to-sequence attention model, so the front end becomes one learned network with no linguistic features; but to get a waveform it runs Griffin-Lim phase estimation and an inverse STFT, a placeholder for a real neural vocoder whose characteristic artifacts cap the quality. So one system has a learned front end chained to a weak back end, and the other a strong back end chained to an expensive front end. The two failures are complementary, and that is the opening.

I propose Tacotron 2: join Tacotron's learned character-to-feature front end to WaveNet's neural waveform back end, with the entire design hinging on the acoustic representation that sits between them. What I want from that bridge is threefold: it should be cheap to compute directly from the waveform, so both halves can be trained from audio with no linguistic annotation; it should be low-dimensional and smooth, so the front end can predict it with a regression loss; and it should be lossy in the perceptually right way, keeping what matters and discarding what does not, so the front end is not burdened reproducing irrelevant detail. Raw waveform samples fail the second criterion badly — far too high-rate and high-entropy to predict frame-by-frame under squared error — and a linear-frequency STFT magnitude is computable from audio but high-dimensional and keeps fine high-frequency structure I do not need. A mel spectrogram fits all three: it warps the STFT magnitude's frequency axis onto an auditory scale, emphasizing the lower frequencies that carry most intelligibility and compressing the highs dominated by fricatives and noise bursts, and about 80 channels per frame suffice. It is computed from the waveform, so the two stages can train separately; it is smooth and phase-invariant within a frame, so a squared-error loss is well-behaved because phase is not in the target to wreck the regression; and it is far lower-level than linguistic features. It discards even more than a linear spectrogram, making waveform reconstruction a harder inverse problem, but it is still a much simpler conditioning signal than linguistic features plus $F_0$ plus durations, and a WaveNet conditioned on mel frames can serve as the vocoder. I compute it with a 50 ms window, 12.5 ms hop, Hann window, an 80-channel filterbank from 125 Hz to 7.6 kHz, clipped to a 0.01 floor and log-compressed to tame dynamic range.

The front-end network maps characters to mel frames with an encoder-decoder-with-attention. The encoder embeds characters into a 512-dimensional space, then runs them through three convolutional layers of 512 width-5 filters with batch norm, ReLU, and dropout 0.5, because speech depends on short character n-grams like digraphs and clusters and the convolutions capture that local context; a single bidirectional LSTM with 512 total units (256 each way) then gives every character position a representation that sees the whole input in both directions, and that is the encoded memory. The decoder generates mel frames autoregressively, one frame per step, and at each step computes a context vector summarizing the relevant encoder positions. The obvious choice is plain additive Bahdanau attention, $e_{ij} = v^\top \tanh(W s_{i-1} + V h_j)$, softmaxed into weights and summed against the memory. But text-to-speech alignment is mostly monotonic and forward-moving — speech consumes characters in order — and content-only attention has no memory of where it already attended, so it can stall on a character, skip one, or repeat a subsequence, which are precisely the catastrophic failure modes of TTS. So I extend the score with location features: I accumulate the prior alignment vectors, convolve the cumulative weights with 32 length-31 one-dimensional filters, project those features into the same 128-dimensional attention space as query and memory, and add them inside the tanh,
$$e_{ij} = v^\top \tanh(W s_{i-1} + V h_j + U f_{ij}),$$
where $f_{ij}$ are the location features from the convolved cumulative weights. Now the score at a position depends on how much attention that neighborhood has already received, which pushes the alignment to advance consistently and not revisit, directly suppressing skips and repeats.

The autoregressive input at each step is the previously predicted mel frame, and how it enters matters. Fed in at full width, the decoder can lean on the strong frame-to-frame continuity of speech and predict each frame locally from the last one while underusing the text, leaving the alignment too little pressure to become reliable. So the previous frame passes first through a small information bottleneck — a pre-net of two fully connected layers of 256 ReLU units — whose output is concatenated with the attention context and run through two unidirectional LSTM layers of 1024 units; the concatenation of the LSTM output and the attention context is then linearly projected to the target mel frame. I deliberately keep the pre-net dropout of 0.5 on at inference, so the autoregressive decoder still receives variation through the same bottleneck it learned with. There is no reduction factor: each decoder step emits exactly one frame, keeping the acoustic and attention time steps aligned one-to-one. Two outputs hang off that same concatenated decoder state. The first is the mel frame. The second answers how generation stops: rather than decoding for a fixed number of steps as Tacotron does, I add a stop token — project the decoder state plus context to a scalar, sigmoid it, and train it to predict the probability that the sequence is finished, stopping at inference at the first frame where it exceeds 0.5, so the model learns when to end on its own.

One refinement remains on the mel prediction. Because the decoder is autoregressive, while predicting a frame it cannot see future frames; but once the whole mel sequence exists, a convolutional pass can use both past and future local context to improve the reconstruction. So I run the decoder output through a five-layer convolutional post-net — 512 width-5 filters, batch normalization, tanh on all but the last layer — that predicts a residual added to the pre-post-net mel. The loss keeps both signals honest by summing the mean-squared error of the mel target against the prediction before the post-net and against the prediction after it; the mel target is smooth enough for squared error, and supervising both versions makes the residual refinement well-defined. Regularization elsewhere is dropout 0.5 on the convolutions and zoneout 0.1 on the LSTMs.

The vocoder is a WaveNet conditioned on the predicted mel frames instead of linguistic features. I keep the WaveNet stack of 30 dilated convolution layers in 3 dilation cycles, layer $k$ using dilation $2^{(k \bmod 10)}$. The conditioning must be upsampled from the mel frame rate to the sample rate, and because the 12.5 ms mel hop is coarser than WaveNet's original 5 ms conditioning, I use only 2 upsampling layers in the conditioning stack instead of 3. For the output I avoid a 256-way softmax over quantized amplitudes and instead model each 16-bit sample at 24 kHz with a 10-component mixture of logistic distributions: the stack output passes through ReLU then a linear projection giving each component's mean, log-scale, and mixture weight, trained by the negative log-likelihood of the true sample, because a continuous mixture fits high-bit-depth audio far better than a giant categorical. Since the bridge is computable from audio, the two halves train separately: first the feature-prediction network on characters and ground-truth mel frames with teacher forcing; then the WaveNet on the feature network's ground-truth-aligned predictions, running the feature network in teacher-forcing mode so each predicted frame is conditioned on the correct previous frame and stays exactly aligned with the waveform samples. This gives the vocoder the same kind of predicted mel it will see at inference without losing sample alignment during training.

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
