Let me lay out the trade-off the whole field is stuck on, because the design should resolve it. There are two halves to text-to-speech: turn text into some acoustic description, and turn that description into a waveform. Historically, the more natural the output, the more hand-built machinery it took. Concatenative unit-selection stitched recorded snippets - natural-ish but full of boundary artifacts and inflexible. Statistical parametric synthesis generated smooth hand-designed acoustic features for a fixed vocoder - no concatenation seams, but muffled and buzzy, because the features are a lossy summary and the vocoder is a rigid signal-processing model.

Two recent systems each fixed one half and left the other broken. WaveNet models the raw waveform autoregressively, p(x_t | x_{<t}, conditioning), through dilated causal convolutions, and the audio quality starts to rival human speech. That solves the vocoding half. But look at what it is conditioned on: linguistic features, predicted log-F0, phoneme durations. To produce those I still need a full text-analysis system, a pronunciation lexicon, and a duration model, exactly the hand-engineered front end the field wants to delete. So WaveNet has a strong back end chained to an expensive front end. Going the other way, Tacotron maps characters straight to a magnitude spectrogram with a seq2seq attention model; the front end is now one learned network, no linguistic features. But to get a waveform it runs Griffin-Lim phase estimation and an inverse STFT, and that is only a placeholder for a neural vocoder: Griffin-Lim has characteristic artifacts and caps the quality well below a neural vocoder. So Tacotron has a learned front end chained to a weak back end.

The two failures are complementary. One system has the back end I want and the front end I don't; the other has the front end I want and the back end I don't. So the obvious thing to try is to keep Tacotron's learned character-to-feature front end and swap its Griffin-Lim back end for WaveNet's neural one. That immediately raises the real design question, the one that decides whether the two halves can actually be trained and joined: what acoustic representation should sit on the wire between them?

Let me reason about what that intermediate has to be, from constraints rather than from the answer. WaveNet currently takes linguistic features plus F0 plus durations: rich, high-level, hand-derived signals. I'm trying to delete exactly those. So the bridge has to satisfy several things at once. (a) It must be cheap to compute directly from the waveform, because if it requires annotation I haven't removed the front end, I've renamed it; computability from audio also means I can train the two halves separately, each against ground truth. (b) It must be something a regression front end can actually predict frame-by-frame, which means low-dimensional and smooth, not high-rate and high-entropy. (c) It should be lossy in the right way - keep what's perceptually important, drop what isn't - so the front end isn't spending capacity reproducing detail the ear discards.

Run the candidates against those three. Raw waveform samples fail (b) hard: at 24 kHz that's 24000 scalar targets per second, with the entropy of the actual audio, and a squared-error regressor predicting one frame from the previous would be hopeless. A linear-frequency STFT magnitude passes (a) - it's a standard transform of the audio - but it's high-dimensional and keeps fine high-frequency structure I don't need the front end to nail precisely, so it's weak on (b) and (c). What I want is the STFT magnitude with its frequency axis warped onto a perceptual scale and its dimensionality cut: emphasize the low frequencies that carry most intelligibility, compress the highs that are mostly fricatives and noise bursts. That is the mel spectrogram. Around 80 channels per frame. Check it against the three constraints: (a) it's computed from the waveform by a fixed filterbank, so both stages can train against their own ground truth and I never need linguistic annotation; (b) it's smooth across frames and, crucially, phase has been discarded inside each frame, so the regression target has no phase wraparound to make squared error misbehave; (c) the mel warping is the perceptual lossiness I asked for. The one cost is that throwing away phase and warping the axis makes the waveform inverse harder than from a linear spectrogram - but that inverse is exactly what I'm handing to a neural vocoder, and an 80-channel mel frame is still a far lower-level, lower-dimensional conditioning signal than linguistic features plus F0 plus durations, which is the WaveNet conditioning I'm replacing. So I'll commit to a mel spectrogram on the wire and let the rest of the design follow from it. Concretely: 50 ms window, 12.5 ms hop, Hann window, 80-channel filterbank from 125 Hz to 7.6 kHz, magnitude clipped to a 0.01 floor and log-compressed to tame dynamic range.

Now the front-end network: characters to mel frames. Encoder-decoder with attention. The encoder turns characters into a hidden sequence. Characters first go through a learned embedding (512-dim). Before recurrence I want some local context, because speech depends on short character n-grams such as digraphs and common letter clusters, so I use a small stack of convolutions over the embedded characters: 3 conv layers, 512 filters, width 5, each with batch norm and ReLU. Then a single bidirectional LSTM, 512 units total and 256 in each direction, gives each character position a representation that sees the whole input both ways. That's the encoded memory.

The decoder generates mel frames autoregressively, one frame at a time, and at each step it needs a context vector summarizing the relevant encoder positions. The standard additive, Bahdanau-style score compares each encoder state with the current decoder state, e_ij = v^T tanh(W s_{i-1} + V h_j), then softmaxes the scores into weights and takes a weighted sum. The worry with pure content attention here is structural: TTS alignment is mostly monotonic and forward-moving, because speech usually consumes the characters in order, and content-only attention has no memory of where it already attended. Nothing in e_ij = v^T tanh(W s_{i-1} + V h_j) references the past trajectory, so two decoder steps with similar states will produce similar weight vectors - the model can stall on a character, skip one, or repeat a subsequence, and those are the catastrophic, intelligibility-destroying failure modes of TTS. To break that symmetry I want the score at position j to depend not only on what that character looks like but on how much attention its neighborhood has already received. So I keep a running sum of the prior alignment vectors, convolve that cumulative-attention vector with 32 length-31 one-dimensional filters to extract location features, project those into the same 128-dimensional attention space as the query and memory, and add them inside the tanh:

  e_{ij} = v^T tanh(W s_{i-1} + V h_j + U f_{ij}),

where f_{ij} are the location features from the convolved cumulative weights. Now the score at position j carries a record of how heavily that neighborhood has already been consumed. I expect that to bias the alignment toward advancing and away from revisiting; whether it actually suppresses the skip/repeat failures is an empirical question I'd want to confirm by looking at the learned alignment matrix on held-out utterances, not something I can prove from the score formula. But it's the right shape of fix: it gives the scoring function the memory that content attention structurally lacks. Project query, memory, and location features all to a 128-dim attention space.

At each decoder step the previous predicted mel frame is the autoregressive input. There's a tension I have to handle here. Mel frames are extremely autocorrelated - frame t looks a lot like frame t-1 - so if I feed the previous frame in at full 80-dim width, the decoder has a shortcut: predict the next frame as a small perturbation of the last one, mostly ignore the text, and still get a low training MSE under teacher forcing. The attention then never receives enough gradient pressure to learn a reliable alignment, and at inference, when the frames are the model's own and the shortcut compounds errors, it falls apart. So I deliberately starve that shortcut: the previous frame passes through a small information bottleneck first, a pre-net of 2 fully connected layers of 256 ReLU units, before it can influence the next prediction. The pre-net output is concatenated with the attention context and run through the recurrence, two unidirectional LSTM layers of 1024 units. The concatenation of the LSTM output and the attention context is then linearly projected to predict the target mel frame. I will not use a reduction factor; each decoder step emits exactly one frame, keeping the acoustic and attention time steps aligned one-to-one.

Before I trust this decoder I should make sure the pieces actually fit together dimensionally, because the attention has a query Linear of a fixed input width and it's easy to wire the order wrong. Let me trace one decoder step's shapes by hand, batch B, encoder length L. Encoder memory is (B, L, 512). The attention projects memory to (B, L, 128) once up front. Inside the step: the pre-net emits (B, 256); I concatenate it with the previous context (B, 512) to get (B, 768), which feeds the first LSTM cell (input 256+512, hidden 1024) and yields h_att = (B, 1024). Now h_att is what queries the attention - and the attention's query Linear is 1024 -> 128, so it consumes h_att exactly. That is a real ordering constraint I almost glossed: the attention RNN has to run before the attention so that its 1024-d state is available as the query; if I'd computed attention first I'd have no 1024-d query to give it. The location branch convolves the cumulative weights (B, L) -> (B, 32, L) -> project to (B, L, 128); query (B, 1, 128) broadcasts against projected memory (B, L, 128) and location (B, L, 128) inside the tanh, energy collapses to (B, L), softmax over L gives the weights, and the weighted sum over memory gives context (B, 512). I ran this trace in code and the softmax weights came out summing to 1.0 per row, and every concat lined up: the second LSTM takes h_att(1024) concatenated with context(512) = 1536, and the final frame and stop projections take h_dec(1024) concatenated with context(512) = 1536 -> 80 and -> 1. So the 256+512, 1024+512, and 1024->128 widths in the code aren't arbitrary; they're forced by this wiring, and the trace confirms they're mutually consistent.

Two outputs hang off that same concatenated decoder state, not one. First the mel frame, as above. But how does generation *stop*? In Tacotron you decode for a fixed number of steps, which is wasteful and wrong for variable-length utterances. Instead I add a "stop token": project the same decoder-state-plus-context down to a scalar, sigmoid it, and train it to predict the probability that the sequence is finished; at inference, stop at the first frame where this exceeds 0.5. The model learns when to end on its own.

One more refinement on the mel prediction. The decoder predicts frames autoregressively, so while it is decoding a frame it cannot use future frames - each frame is committed with only leftward context. Once the whole mel sequence exists, though, a convolutional pass over the sequence can use both past and future local context to clean up the acoustic reconstruction. So I run the decoder output through a 5-layer convolutional post-net, with 512 filters of width 5, batch normalization, and tanh on all but the last layer, and let that post-net predict a residual to add to the pre-post-net mel. The loss should keep both signals honest: minimize the summed mean-squared error of the mel target against the prediction before the post-net and against the prediction after it. The mel target is smooth and phase-free, so this squared-error objective is well-behaved; supervising both the pre- and post-net versions makes the residual refinement well-defined rather than letting the post-net drift.

Regularization: dropout 0.5 on the convolutional layers, zoneout 0.1 on the LSTMs. And one deliberate oddity: the pre-net dropout, also 0.5, stays on at inference time so the autoregressive decoder still gets variation through the same bottleneck it learned with during training.

Now the vocoder: a WaveNet conditioned on the predicted mel frames instead of on linguistic features. Keep the WaveNet stack: 30 dilated convolution layers in 3 dilation cycles, layer k's dilation 2^(k mod 10). The conditioning is at the mel frame rate and the output is at the sample rate, so it has to be upsampled, and I should work out the actual factor before deciding how many upsampling layers to spend on it. At 24 kHz a 12.5 ms hop is 0.0125 * 24000 = 300 samples per frame; equivalently the mel frame rate is 24000 / 300 = 80 frames per second. So the conditioning stack has to upsample each mel frame to exactly 300 audio samples. The original WaveNet used a 5 ms hop, 0.005 * 24000 = 120 samples, and stacked 3 upsampling layers to bridge that gap. My hop is 12.5 / 5 = 2.5x coarser, i.e. I have fewer mel frames to spread over the same audio, so I need less upsampling resolution, not more - it would be backwards to keep 3 layers here. A factor of 300 also factors comfortably into two transposed-convolution strides (for instance 12 and 25, or 15 and 20, both giving 300), so 2 upsampling layers suffice. I'll use 2. As a sanity check on whether 30 dilated layers see enough audio: the receptive field is 1 + sum over k of dilation_k * (kernel-1) with kernel 2 and dilations cycling 1,2,...,512 three times, which is 1 + 3 * 1023 = 3070 samples, about 3070 / 24000 = 128 ms of waveform context - comfortably longer than a phone or two, so the stack has the temporal reach to model local waveform structure conditioned on the mel.

For the output distribution I won't use a 256-way softmax over quantized amplitudes; following the mixture-of-logistics idea, I model each 16-bit sample at 24 kHz with a 10-component mixture of logistic distributions, pass the stack output through ReLU then a linear projection to get each component's mean, log-scale, and mixture weight, and train by the negative log-likelihood of the true sample. A continuous mixture is a better fit for high-bit-depth audio than a giant categorical: a 16-bit sample has 65536 levels, and a categorical over that is both enormous and blind to the fact that adjacent levels are nearly identical sounds, whereas a logistic mixture spends parameters on a smooth density instead of an unordered table.

Because the bridge is the mel spectrogram, computable from audio, I train the two halves separately. First train the feature-prediction network on characters and ground-truth mel frames with teacher forcing. Then train the WaveNet vocoder on the feature network's ground-truth-aligned predictions: run the feature network in teacher-forcing mode so each predicted frame is conditioned on the correct previous frame and stays exactly aligned with the waveform samples. There's a subtle reason this matters. If I trained the vocoder on the real mel frames, it would be conditioned at inference on the feature network's predicted frames, which carry the front end's systematic errors, and it would never have seen those during training - a train/test mismatch. Conditioning on teacher-forced predictions gives the vocoder the same kind of slightly-wrong mel it will actually receive at inference, while teacher forcing keeps each frame locked to its true waveform position so the sample alignment isn't lost.

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
        self.loc_conv = nn.Conv1d(1, n_filters, kernel, padding=kernel // 2, bias=False)
        self.loc_proj = nn.Linear(n_filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, query, memory, proj_memory, cumulative_weights):
        loc = self.loc_proj(
            self.loc_conv(cumulative_weights.unsqueeze(1)).transpose(1, 2)
        )                                                                    # (B, L, attn)
        e = self.v(torch.tanh(self.query(query).unsqueeze(1) + proj_memory + loc))  # energy
        weights = F.softmax(e.squeeze(-1), dim=1)                             # (B, L)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)          # (B, 512)
        return context, weights
```

The autoregressive decoder, with the pre-net bottleneck, two LSTMs, frame and stop-token projections, and the residual post-net:

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
        h_attn, c_attn, h_dec, c_dec, context, cum_w = state
        p = self.prenet(prev_mel)
        h_attn, c_attn = self.attn_rnn(torch.cat([p, context], -1), (h_attn, c_attn))
        context, cur_w = self.attn(h_attn, memory, proj_memory, cum_w)
        cum_w = cum_w + cur_w
        h_dec, c_dec = self.dec_rnn(torch.cat([h_attn, context], -1), (h_dec, c_dec))
        dc = torch.cat([h_dec, context], -1)
        return (self.frame_proj(dc), self.stop_proj(dc),
                (h_attn, c_attn, h_dec, c_dec, context, cum_w))

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
    def __init__(self, upsample_kernels, upsample_strides,
                 n_mels=80, n_mix=10, channels=512):
        super().__init__()
        self.upsample = nn.ModuleList(                # 2 layers (12.5ms hop, not 5ms)
            nn.ConvTranspose1d(n_mels, n_mels, k, stride=s)
            for k, s in zip(upsample_kernels, upsample_strides))
        self.dilated = nn.ModuleList(
            nn.Conv1d(channels, 2 * channels, 2, dilation=2 ** (k % 10), padding=2 ** (k % 10))
            for k in range(30))
        self.out = nn.Linear(channels, 3 * n_mix)     # mean, log-scale, weight per component
    def forward(self, mel, audio):
        # condition on upsampled mel; autoregressive dilated stack; ReLU -> linear -> MoL
        ...
        # loss = negative log-likelihood of the 16-bit ground-truth sample under the MoL
```
