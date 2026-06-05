Let me lay out the trade-off the whole field is stuck on, because the design should resolve it. There are two halves to text-to-speech: turn text into some acoustic description, and turn that description into a waveform. Historically, the more natural the output, the more hand-built machinery it took. Concatenative unit-selection stitched recorded snippets — natural-ish but full of boundary artifacts and inflexible. Statistical parametric synthesis generated smooth hand-designed acoustic features for a fixed vocoder — no concatenation seams, but muffled and buzzy, because the features are a lossy summary and the vocoder is a rigid signal-processing model.

Two recent systems each fixed *one* half and left the other broken. WaveNet models the raw waveform autoregressively — p(x_t | x_{<t}, conditioning) through dilated causal convolutions — and the audio quality genuinely starts to rival human speech. That solves the vocoding half. But look at what it's *conditioned on*: linguistic features, predicted log-F₀, phoneme durations. To produce those you need a full text-analysis system, a pronunciation lexicon, and a duration model — exactly the hand-engineered front end the field wants to delete. So WaveNet has a beautiful back end chained to an expensive front end. Going the other way, Tacotron maps characters straight to a magnitude spectrogram with a seq2seq attention model — the front end is now one learned network, no linguistic features. But to get a waveform it runs Griffin-Lim phase estimation and an inverse STFT, which its own authors flagged as a placeholder: Griffin-Lim has characteristic artifacts and caps the quality well below a neural vocoder. So Tacotron has a learned front end chained to a weak back end.

The two failures are complementary, which tells me the move: take Tacotron's learned char→features front end and WaveNet's neural back end and join them. The only real design question is *what acoustic representation should sit between them*. Get that interface right and everything else follows.

What do I want from that intermediate? WaveNet currently takes linguistic features plus F₀ plus durations — rich, high-level, hand-derived. But I'm trying to *avoid* hand-derived features. So I want the bridge to be something (a) cheap to compute directly from the waveform, so I can train both halves from audio without any linguistic annotation, (b) low-dimensional and smooth, so the front end can actually predict it with a regression loss, and (c) lossy in the *right* way — keeping what's perceptually important and discarding what isn't, so the front end isn't burdened reproducing irrelevant detail. Raw waveform samples fail (b) badly — far too high-rate and high-entropy to predict frame-by-frame with squared error. A linear-frequency spectrogram (STFT magnitude) is computable from audio but high-dimensional and keeps fine high-frequency structure I don't need to model precisely.

A *mel* spectrogram fits all three. It warps the STFT magnitude's frequency axis onto an auditory scale: it emphasizes lower frequencies, which carry most of speech intelligibility, and compresses the highs, which are dominated by fricatives and noise bursts that don't need faithful modeling. Around 80 channels per frame is enough. It's trivially computed from the waveform (so the two stages train separately), it's smooth and phase-invariant *within* a frame (so a squared-error loss is well-behaved — phase isn't in the target to wreck the regression), and it's far lower-level and lower-dimensional than linguistic features. Yes, the mel transform discards even more than a linear spectrogram, so reconstructing a waveform from it is a harder inverse problem — but it's still a much simpler, lower-level conditioning than the linguistic-features-plus-F₀-plus-durations WaveNet was using, so a WaveNet conditioned on mel should manage it as a neural vocoder. Mel spectrogram it is, computed with a 50 ms window, 12.5 ms hop, Hann window, 80-channel filterbank from 125 Hz to 7.6 kHz, clipped to a 0.01 floor and log-compressed to tame dynamic range.

Now the front-end network: characters → mel frames. Encoder-decoder with attention. The encoder turns characters into a hidden sequence. Characters first go through a learned embedding (512-dim). Before recurrence I want some local context — speech depends on short character n-grams (digraphs, common letter clusters) — so a small stack of convolutions over the embedded characters: 3 conv layers, 512 filters, width 5 (each filter sees 5 characters), each with batch norm and ReLU. Then a single bidirectional LSTM (512 units, 256 per direction) gives each character position a representation that sees the whole input both ways. That's the encoded memory.

The decoder generates mel frames autoregressively, one frame at a time, and at each step it needs a context vector summarizing the relevant encoder positions — that's attention. The standard additive (Bahdanau) attention scores each encoder state against the current decoder state, e_{ij} = vᵀ tanh(W s_{i-1} + V h_j), softmax to weights, weighted sum. But text→speech has a special structure that pure content attention ignores: the alignment is essentially *monotonic and strictly advancing* — you speak the characters roughly in order, each consumed once. Content-only attention has no memory of where it already attended, so it can stall on a character, skip one, or repeat a subsequence — and those are exactly the catastrophic failure modes of TTS (a repeated or dropped syllable is glaring). I want to bias the alignment to keep moving forward. So use *location-sensitive* attention: feed the *cumulative* attention weights from previous steps in as an extra feature. Concretely, accumulate the prior alignment vectors, convolve them (32 filters, length 31) to get location features, project them, and add inside the tanh:

  e_{ij} = vᵀ tanh(W s_{i-1} + V h_j + U f_{ij}),

where f_{ij} are the location features from the convolved cumulative weights. Now the score at position j depends on how much attention that neighborhood has *already* received, which encourages the model to advance consistently and not revisit — directly suppressing the skip/repeat failures. Project query, memory, and location features all to a 128-dim attention space.

The decoder recurrence and what it emits. At each step the previous predicted mel frame is the autoregressive input. I'll pass it first through a small *pre-net* — 2 fully-connected layers of 256 ReLU units. This matters more than it looks: I found the pre-net acting as an *information bottleneck* is essential for learning attention. If the previous frame flowed in at full width, the decoder could lean on the strong frame-to-frame continuity of the acoustic signal and predict the next frame from the previous one alone, basically ignoring the text — and then attention never has to learn an alignment. Squeezing the autoregressive input through a narrow bottleneck starves that shortcut and forces the decoder to actually consult the encoded text via attention. The pre-net output is concatenated with the attention context and run through the recurrence: two uni-directional LSTM layers of 1024 units. The concatenation of the LSTM output and the attention context is then linearly projected to predict the target mel frame. I will *not* use a reduction factor — each decoder step emits exactly one frame, keeping it simple.

Two outputs hang off that same concatenated decoder state, not one. First the mel frame, as above. But how does generation *stop*? In Tacotron you decode for a fixed number of steps, which is wasteful and wrong for variable-length utterances. Instead I add a "stop token": project the same decoder-state-plus-context down to a scalar, sigmoid it, and train it to predict the probability that the sequence is finished; at inference, stop at the first frame where this exceeds 0.5. The model learns when to end on its own.

One more refinement on the mel prediction. A single linear projection produces a decent frame, but I can sharpen it. After the decoder emits the whole mel sequence, run it through a 5-layer convolutional *post-net* (512 filters, width 5, batch norm, tanh on all but the last layer) that predicts a *residual* to add to the pre-post-net prediction. So the decoder produces a coarse spectrogram and the post-net cleans it up. I train both versions: minimize the summed mean-squared error of the mel target against the prediction *before* the post-net and against the prediction *after* it — supervising both helps convergence (the pre-post-net prediction stays meaningful, and the residual is well-defined). I considered modeling the output with a mixture density network to avoid assuming constant variance, but those were harder to train and didn't sound better, so plain MSE it is.

Regularization: dropout 0.5 on the conv layers, zoneout 0.1 on the LSTMs. And one deliberate oddity — the pre-net dropout (0.5) is left *on at inference time*. Because the pre-net is the bottleneck the model relies on, keeping dropout active at generation injects variation and keeps the autoregressive process from collapsing into a degenerate fixed trajectory.

Now the vocoder: a WaveNet conditioned on the *predicted* mel frames instead of on linguistic features. Keep the WaveNet stack — 30 dilated convolution layers in 3 dilation cycles, layer k's dilation 2^(k mod 10). The conditioning has to be upsampled from the mel frame rate to the sample rate; because the mel hop is 12.5 ms (coarser than the original WaveNet's 5 ms conditioning), I use only 2 upsampling layers in the conditioning stack instead of 3. For the output distribution I won't use a 256-way softmax over quantized amplitudes; following the mixture-of-logistics idea, I model each 16-bit sample at 24 kHz with a 10-component mixture of logistic distributions — pass the stack output through ReLU then a linear projection to get each component's mean, log-scale, and mixture weight, and train by the negative log-likelihood of the true sample. A continuous mixture is a better fit for high-bit-depth audio than a giant categorical.

Because the bridge is the mel spectrogram — computable from audio — I train the two halves *separately*: the feature-prediction network on (characters, ground-truth mel) pairs, and the WaveNet vocoder on (mel, waveform) pairs, with no linguistic annotation anywhere.

Let me write it, mirroring how I'd build it. First the encoder and the location-sensitive attention.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):                            # chars -> conv n-gram context -> BiLSTM
    def __init__(self, n_chars, d=512):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d),
                          nn.ReLU(), nn.Dropout(0.5)) for _ in range(3))
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)  # 256 each way
    def forward(self, chars):
        x = self.embed(chars).transpose(1, 2)
        for c in self.convs:
            x = c(x)
        return self.lstm(x.transpose(1, 2))[0]       # (B, L, 512) encoder memory

class LocationSensitiveAttention(nn.Module):         # additive + cumulative-weight features
    def __init__(self, attn_dim=128, n_filters=32, kernel=31):
        super().__init__()
        self.query  = nn.Linear(1024, attn_dim, bias=False)   # from decoder LSTM state
        self.memory = nn.Linear(512, attn_dim, bias=False)    # from encoder states
        self.loc_conv = nn.Conv1d(2, n_filters, kernel, padding=kernel // 2, bias=False)
        self.loc_proj = nn.Linear(n_filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, query, memory, proj_memory, attn_cat):
        # attn_cat: (B, 2, L) = stacked [current weights, cumulative weights]
        loc = self.loc_proj(self.loc_conv(attn_cat).transpose(1, 2))          # (B, L, attn)
        e = self.v(torch.tanh(self.query(query).unsqueeze(1) + proj_memory + loc))  # energy
        weights = F.softmax(e.squeeze(-1), dim=1)                             # (B, L)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)          # (B, 512)
        return context, weights
```

The autoregressive decoder — pre-net bottleneck, two LSTMs, frame and stop-token projections — and the residual post-net:

```python
class Prenet(nn.Module):                             # info bottleneck; dropout ON at inference
    def __init__(self, n_mels=80, hidden=256):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_mels, hidden), nn.Linear(hidden, hidden)
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=True)   # always-on dropout
        return F.dropout(F.relu(self.fc2(x)), 0.5, training=True)

class Decoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.prenet = Prenet(n_mels)
        self.attn_rnn = nn.LSTMCell(256 + 512, 1024)             # prenet + context
        self.attn = LocationSensitiveAttention()
        self.dec_rnn  = nn.LSTMCell(1024 + 512, 1024)            # attn state + context
        self.frame_proj = nn.Linear(1024 + 512, n_mels)         # predict mel frame
        self.stop_proj  = nn.Linear(1024 + 512, 1)              # stop-token logit
    def step(self, prev_mel, memory, proj_memory, state):
        h_attn, c_attn, h_dec, c_dec, context, cum_w, cur_w = state
        p = self.prenet(prev_mel)
        h_attn, c_attn = self.attn_rnn(torch.cat([p, context], -1), (h_attn, c_attn))
        attn_cat = torch.stack([cur_w, cum_w], 1)               # current + cumulative weights
        context, cur_w = self.attn(h_attn, memory, proj_memory, attn_cat)
        cum_w = cum_w + cur_w
        h_dec, c_dec = self.dec_rnn(torch.cat([h_attn, context], -1), (h_dec, c_dec))
        dc = torch.cat([h_dec, context], -1)
        return (self.frame_proj(dc), self.stop_proj(dc),
                (h_attn, c_attn, h_dec, c_dec, context, cum_w, cur_w))

class Postnet(nn.Module):                            # predicts a residual to refine the mel
    def __init__(self, n_mels=80, d=512):
        super().__init__()
        layers = [nn.Sequential(nn.Conv1d(n_mels, d, 5, padding=2),
                                nn.BatchNorm1d(d), nn.Tanh(), nn.Dropout(0.5))]
        for _ in range(3):
            layers.append(nn.Sequential(nn.Conv1d(d, d, 5, padding=2),
                                        nn.BatchNorm1d(d), nn.Tanh(), nn.Dropout(0.5)))
        layers.append(nn.Sequential(nn.Conv1d(d, n_mels, 5, padding=2),
                                    nn.BatchNorm1d(n_mels), nn.Dropout(0.5)))  # no tanh last
        self.net = nn.Sequential(*layers)
    def forward(self, mel):                          # (B, n_mels, T)
        return self.net(mel)
```

The feature-prediction loss supervises the mel both before and after the post-net, plus the stop token:

```python
def feature_loss(mel_before, mel_after, stop_logits, mel_target, stop_target):
    mse = F.mse_loss(mel_before, mel_target) + F.mse_loss(mel_after, mel_target)
    stop = F.binary_cross_entropy_with_logits(stop_logits, stop_target)
    return mse + stop
# mel_after = mel_before + postnet(mel_before); inference stops when sigmoid(stop) > 0.5
```

And the WaveNet vocoder conditioned on the predicted mel, with a mixture-of-logistics output:

```python
class WaveNetVocoder(nn.Module):                     # 30 dilated layers, 3 cycles
    def __init__(self, n_mels=80, n_mix=10, channels=512):
        super().__init__()
        self.upsample = nn.ModuleList(                # 2 layers (12.5ms hop, not 5ms)
            nn.ConvTranspose1d(n_mels, n_mels, 2 * f, stride=f) for f in (16, 16))
        self.dilated = nn.ModuleList(
            nn.Conv1d(channels, 2 * channels, 2, dilation=2 ** (k % 10), padding=2 ** (k % 10))
            for k in range(30))
        self.out = nn.Linear(channels, 3 * n_mix)     # mean, log-scale, weight per component
    def forward(self, mel, audio):
        # condition on upsampled mel; autoregressive dilated stack; ReLU -> linear -> MoL
        ...
        # loss = negative log-likelihood of the 16-bit ground-truth sample under the MoL
```

The causal chain, start to end: TTS was split between a learned-front-end / weak-vocoder system (Tacotron, Griffin-Lim) and a great-vocoder / hand-engineered-front-end system (WaveNet on linguistic features), so I join the learned front end to the neural vocoder; the join hinges on the intermediate representation, and I pick the mel spectrogram because it's computable from audio, smooth, low-dimensional, phase-invariant within a frame, and lossy in the perceptually-right way; I predict it with a char-conv-BiLSTM encoder and an autoregressive LSTM decoder whose attention is *location-sensitive* — using cumulative attention weights to keep the alignment monotonic and kill skips/repeats — with a pre-net bottleneck that forces the decoder to actually use attention rather than coast on frame continuity, a learned stop token to end generation, and a residual post-net refining the frame under a summed-MSE loss; and I condition a WaveNet (mixture-of-logistics output) on those predicted mel frames as the neural vocoder, training the two halves separately.
