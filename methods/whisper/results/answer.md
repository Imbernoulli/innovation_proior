# Whisper

## Problem

Build a speech recognizer that works reliably zero-shot — across recording
conditions, accents, domains, and languages — without per-deployment fine-tuning
of a decoder. Self-supervised encoders learn great representations but have no
decoder and need fine-tuning, which is both a usability burden and a robustness
hazard (it fits one dataset's quirks and fails to generalize).

## Key idea

**Scale weakly supervised multilingual/multitask training** to 680,000 hours with
an off-the-shelf sequence-to-sequence Transformer, and use a **single decoder told
its task via special tokens**:

- **Weak supervision at scale.** Harvest 680,000 hours of
  (audio, transcript) pairs from the web; rely on diversity plus filtering rather
  than gold-standard curation for robustness. Filter out machine-generated
  transcripts (all-caps/all-lower, no commas) and language-mismatched pairs;
  repurpose non-English-audio/English-text pairs as X→English translation data.
  The final mix includes 117,000 hours across 96 non-English languages and
  125,000 hours of X→English translation examples. Predict **raw, un-normalized
  text**, so no separate inverse-text-normalization stage is needed.
- **Off-the-shelf architecture** (no novel model), to isolate the effect of data
  scale: log-mel front end, conv stem (stride 2 → 20 ms frames), encoder-decoder
  Transformer, the decoder an audio-conditional LM with learned positions and tied
  input-output embeddings.
- **One model, many tasks via a token interface.** The decoder sequence is optional
  prior-text conditioning → `<|startoftranscript|>` → one of 99 language tokens
  → `<|transcribe|>`/`<|translate|>` → `<|notimestamps|>` or interleaved 20
  ms-quantized start/end time tokens around the text → `<|endoftranscript|>`.
  In no-speech segments, `<|nospeech|>` replaces the language token and the
  segment closes. Training masks only the optional prior text, so transcription,
  translation, language-ID, VAD, and timestamping all become token prediction;
  task and timestamp-mode tokens also serve as task specifiers.

## Training

30-second segments; 80-channel log-mel (25 ms window, 10 ms stride), globally
scaled to roughly [-1, 1]; speechless audio kept with a 10x subsample factor.
Next-token cross-entropy over the full sequence except the prepended conditioning
text. Gaussian fan-in initialization; AdamW (β₁=0.9, β₂=0.98, ε=1e-6, weight
decay 0.1), grad-norm clip 1.0, FP16, linear LR decay after 2048-update warmup,
batch 256, 2²⁰ updates (2–3 epochs), no augmentation/regularization. Sizes: Tiny
4/384/6 (39M), Base 6/512/8 (74M), Small 12/768/12 (244M), Medium 24/1024/16
(769M), Large 32/1280/20 (1550M). A brief fine-tune on
speaker-annotation-free transcripts addresses speaker-name guessing.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_x = nn.LayerNorm(d) if cross else None
        self.xatt = nn.MultiheadAttention(d, heads, batch_first=True) if cross else None
        self.ln2  = nn.LayerNorm(d)
        self.mlp  = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, xa=None, mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask)[0]
        if self.xatt is not None:
            x = x + self.xatt(self.ln_x(x), xa, xa)[0]
        return x + self.mlp(self.ln2(x))

def sinusoids(length, d, max_timescale=10000):
    inv = torch.exp(-torch.log(torch.tensor(max_timescale)) *
                    torch.arange(d // 2) / (d // 2 - 1))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, d, layers, heads, n_ctx=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer('pos', sinusoids(n_ctx, d))
        self.blocks = nn.ModuleList(ResidualAttentionBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, mel):
        x = F.gelu(self.conv1(mel)); x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.pos[:x.size(1)]
        for b in self.blocks:
            x = b(x)
        return self.ln(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab, d, layers, heads, n_ctx=448):
        super().__init__()
        self.token = nn.Embedding(vocab, d)
        self.pos   = nn.Parameter(torch.empty(n_ctx, d))     # learned positions
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(d, heads, cross=True) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, tokens, audio):
        x = self.token(tokens) + self.pos[:tokens.size(1)]
        mask = torch.triu(torch.full((tokens.size(1),) * 2, float('-inf')), 1)
        for b in self.blocks:
            x = b(x, audio, mask)
        return self.ln(x) @ self.token.weight.t()            # tied output

SOT, EOT, NOSPEECH = '<|startoftranscript|>', '<|endoftranscript|>', '<|nospeech|>'
TRANSCRIBE, TRANSLATE, NOTIME = '<|transcribe|>', '<|translate|>', '<|notimestamps|>'
TIME_STEP, MAX_TIME = 0.02, 30.0

def time_token(seconds):
    idx = round(max(0.0, min(float(seconds), MAX_TIME)) / TIME_STEP)
    return f'<|{idx * TIME_STEP:.2f}|>'                 # 20ms bins over a 30s window

def build_decoder_sequence(tok, seg, prev_text=None, p_cond=0.5):
    seq, lm = [], []                                  # lm[i]=1 where loss applies
    if prev_text is not None and torch.rand(1).item() < p_cond:
        c = tok.encode(prev_text); seq += c; lm += [0] * len(c)   # condition, don't predict
    seq.append(tok.special(SOT)); lm.append(1)         # predict it when prior text exists
    if seg.no_speech:                                # VAD replaces the language slot
        seq += [tok.special(NOSPEECH), tok.special(EOT)]; lm += [1, 1]; return seq, lm
    seq.append(tok.lang(seg.language)); lm.append(1)
    seq.append(tok.special(TRANSLATE if seg.task == 'translate' else TRANSCRIBE)); lm.append(1)
    if seg.use_timestamps:
        for u in seg.utterances:
            seq.append(tok.special(time_token(u.start))); lm.append(1)
            t = tok.encode(u.text); seq += t; lm += [1] * len(t)
            seq.append(tok.special(time_token(u.end))); lm.append(1)
    else:
        seq.append(tok.special(NOTIME)); lm.append(1)
        t = tok.encode(seg.text); seq += t; lm += [1] * len(t)
    seq.append(tok.special(EOT)); lm.append(1)
    return seq, lm

def loss_step(encoder, decoder, mel, seq, lm):
    audio  = encoder(mel)
    logits = decoder(seq[:, :-1], audio)
    tgt = seq[:, 1:].clone(); tgt[lm[:, 1:] == 0] = -100      # ignore conditioning context
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100)
```
