A speech recognizer should work the moment you point it at new audio — a different accent, a noisier room, an unfamiliar domain, another language — without anyone first fine-tuning it on that distribution. Today's strongest systems do the opposite. Self-supervised pre-training on raw audio has been scaled to something like a million hours of unlabeled speech, which dwarfs the thousand-odd hours of any academic labeled set, and it produces a genuinely excellent *encoder*. But the objective is purely unsupervised, so there is no decoder that turns those representations into text; to recognize speech at all you must fine-tune. That fine-tuning step is exactly what I want to remove, because it carries two costs. The first is usability: every new deployment distribution needs a skilled practitioner to run a fine-tuning job. The second is worse — robustness. A model fine-tuned on one dataset becomes very good at *that dataset's* idiosyncrasies, and there is a clean cautionary tale from vision where fine-tuning a classifier on ImageNet raised ImageNet accuracy by about nine points while not improving average accuracy on seven other datasets of the very same objects. The model learned the dataset's quirks, not the task. A per-dataset speech decoder has the identical exposure: it can look superhuman in-distribution and still botch basic cases out of distribution. So fine-tuning is not merely inconvenient; it actively trades away the generalization I care about. The alternatives that try to buy robustness honestly are too small: pooling seven clean, human-validated supervised corpora reaches only around five thousand hours, and the existing weakly-supervised harvests stop at tens of thousands of hours — still orders of magnitude below the unlabeled scale, and usually evaluated in-distribution rather than zero-shot anyway.

The moment the goal is phrased as "I do not want to fine-tune a decoder," the answer follows: train the decoder up front, supervised, so it is already good at producing text. That requires (audio, transcript) pairs, and a great many of them, because what buys robustness is *diversity* of supervised data, not a clever objective. The method is Whisper, and its central bet is that simple scaling of weak supervision has been underrated for speech. I relax the demand for gold-standard transcripts and scrape the web — captions and subtitles already paired with audio — all the way to 680,000 hours, letting raw diversity rather than curation do the work. Not all of that noise is equal, and the filtering reflects it. Diversity in *audio* quality is desirable: it is precisely what makes the model robust to recording conditions. Diversity in *transcript* quality is not, because some transcripts are simply wrong, and the single most dangerous kind is one that is itself machine-generated — the output of another ASR system. Training on those teaches the model to imitate another system's "transcript-ese," its normalization and missing punctuation, rather than how humans actually write speech. Much of this is detectable heuristically: existing ASR output tends to be all-uppercase or all-lowercase, or never emits commas, or strips complex punctuation, and an all-caps transcript is almost never human, so I drop pairs that look machine-generated. I also run an audio language detector and compare it against the transcript's language; if they disagree I discard the pair — *unless* the audio is non-English and the transcript is English, in which case I have accidentally found a translation pair and keep it as an X→English speech-translation example. Transcripts are fuzzy-deduplicated to cut boilerplate, and once an initial model exists I do a second pass that inspects the data sources with high error rate and removes the misaligned ones. The final mix includes 117,000 hours across 96 non-English languages and 125,000 hours of X→English translation data. One normalization decision is deliberate: most pipelines train on heavily normalized text — lowercased, no punctuation — and bolt on a separate inverse-text-normalization stage afterward, but a sequence-to-sequence model is expressive enough to just *learn* to emit the raw, naturalistic transcript, punctuation and casing and all, so I train it to predict raw un-normalized text and the entire inverse-text-normalization stage disappears.

I deliberately do *not* invent a new architecture, because the point is to isolate the effect of data scale and weak supervision, and a fancy model would confound that. I take an off-the-shelf encoder-decoder Transformer, well-validated to scale. Audio is resampled to 16 kHz, turned into an 80-channel log-mel spectrogram on 25 ms windows with a 10 ms stride, and globally scaled to roughly $[-1, 1]$ with about zero mean. Everything is broken into fixed 30-second segments, each paired with whatever transcript falls in its window — a fixed length is what lets me batch efficiently and gives a consistent context size — and speechless segments are kept at a 10x subsample factor to teach voice-activity detection. The encoder begins with a small convolutional stem: two width-3 conv layers with GELU, the second with stride 2 to halve the time resolution, after which sinusoidal position embeddings are added and the pre-activation residual blocks run, closing with a layer norm. The decoder uses *learned* position embeddings and ties its input and output token embeddings (the token-to-vector map and its inverse share weights, which saves parameters and regularizes); encoder and decoder share width and depth. Tokenization is byte-level BPE, reusing the GPT-2 tokenizer for English-only models and refitting the same-size vocabulary for multilingual so other languages do not fragment.

The part that actually has to be designed is the *interface*, because I want one model to transcribe, translate to English, detect language, decide whether there is speech at all, and optionally produce timestamps — a one-to-many mapping from the same audio. Here the decoder being a language model pays off: I express task requests and classification targets as ordinary special tokens in the same autoregressive sequence as the text. With probability $0.5$ I prepend the preceding segment's transcript as conditioning context so the audio-conditional LM can use transcript history to disambiguate hard audio, and I mask the loss over that conditioning text so the model conditions on it but is not trained to predict it. A `<|startoftranscript|>` token then marks where prediction begins. The first predicted content is *which language is being spoken*, one of 99 language tokens, because everything downstream depends on it; if there is no speech, that classification slot becomes `<|nospeech|>`, which is exactly how voice-activity detection falls out, and the segment closes. If there is speech, the next token specifies the task, `<|transcribe|>` or `<|translate|>`, followed by a format choice: `<|notimestamps|>` for plain text, or interleaved timestamp tokens for alignment, after which the output text begins and `<|endoftranscript|>` closes the segment. Timestamps are kept optional and cheap by quantizing time relative to the current 30-second chunk at the model's native resolution and adding a vocabulary token for each quantized time from 0.00 to 30.00 seconds; in timestamp mode a start-time token precedes each utterance's text and an end-time token follows it. The native resolution is not arbitrary — the front end's 10 ms stride times the stride-2 conv yields 20 ms per output frame, so quantizing to $0.02$ s matches exactly what the model can resolve. When a final utterance is only partially inside the chunk, in timestamp mode I predict only its start time, signaling that decoding should continue from an audio window aligned there; otherwise I truncate it. The training objective over this whole construction is next-token cross-entropy across the entire sequence *except* the prepended conditioning context, so a single decoder — prompted through a few special tokens — replaces language ID, voice-activity detection, multilingual transcription, X→English translation, and timestamped alignment, all as token prediction.

I train a suite of sizes, from Tiny (4 layers, width 384, 39M) up to Large (32 layers, width 1280, 1550M), because the scale question is itself part of the method. Initialization is Gaussian fan-in; optimization is AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-6}$, weight decay $0.1$, gradient-norm clipping at $1.0$, FP16 with dynamic loss scaling and activation checkpointing, a linear learning-rate decay to zero after a warmup over the first 2048 updates, batch size 256 segments, for $2^{20}$ updates — only two to three passes over the data. Because it is just a few epochs over a huge, diverse corpus, overfitting is not the concern, so I use *no* data augmentation or regularization and lean on the dataset's diversity for generalization. One behavioral quirk has to be fixed: many web transcripts name the speaker, so the model learns to *guess* speaker names from audio it cannot possibly identify them from, confidently emitting plausible wrong names; a brief fine-tune on the subset of transcripts without speaker annotations stops the decoder from treating speaker names as inferable. This is a cleanup, not the dataset-specific decoder fine-tuning I set out to avoid.

The system is the standard Transformer block and two towers, plus the multitask token-sequence construction that carries the task interface and the loss masking that goes with it.

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
