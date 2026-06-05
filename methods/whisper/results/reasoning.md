Let me start from what's actually broken about the best speech systems, because it points straight at what to build. The strongest recognizers come from self-supervised pre-training on raw audio — and that's been scaled to something like a million hours of unlabeled speech, dwarfing the thousand-odd hours of any academic labeled set. So the *encoder* is fantastic. But the objective is purely unsupervised, which means there's no decoder that turns those representations into text. To actually recognize speech you have to fine-tune. And that fine-tuning step is the thing I want to attack, because it carries two costs.

The first cost is usability: every new deployment distribution needs someone skilled to run a fine-tuning job. The second is subtler and worse — robustness. A model fine-tuned on one dataset gets very good at *that dataset's* idiosyncrasies. There's a clean cautionary tale from vision: fine-tuning a classifier on ImageNet raised ImageNet accuracy by about nine points while not improving average accuracy on seven other datasets of the very same objects. The model learned the dataset's quirks, not the task. A speech decoder fine-tuned per dataset has the identical exposure — it can look superhuman in-distribution and still botch basic cases out of distribution. So fine-tuning isn't just inconvenient; it actively trades away the generalization I care about.

So state the goal cleanly: I want a single system that works zero-shot — no per-deployment fine-tuning of a decoder — across recording conditions, accents, domains, and languages. The question is what training recipe produces that breadth. And the moment I phrase it as "I don't want to fine-tune a decoder," the answer has to be: train the decoder up front, supervised, so it's already good at producing text. Which means I need (audio, transcript) pairs, and I need a lot of them — because what buys robustness is *diversity* of supervised data, not a clever objective.

How much supervised data can I get? The clean, human-validated corpora are tiny — pooling seven of them gets you around five thousand hours, robust but minuscule next to a million unsupervised hours. The bottleneck is the demand for gold-standard transcripts. So relax it. The internet is full of audio already paired with transcripts — captions, subtitles — much of it noisy, but there's an enormous amount of it. Vision already learned this lesson: moving from curated datasets to much larger *weakly*-supervised ones improves robustness and generalization. So the bet is: scale weakly-supervised speech recognition far past the existing tens-of-thousands of hours — to hundreds of thousands — and let raw diversity, not curation, do the work. The whole hypothesis is that simple scaling of weak supervision has been underrated for speech.

Now the data is web-scraped, so I have to think about what filtering actually matters, because not all noise is equal. Diversity in *audio* quality is good — it's exactly what makes the model robust to recording conditions. Diversity in *transcript* quality is not good in the same way; some transcripts are just wrong. The single most dangerous kind of bad transcript is one that's itself machine-generated, the output of some other ASR system. If I train on those, I learn to imitate another model's "transcript-ese" — its normalization, its missing punctuation — rather than how humans actually write speech. I can detect a lot of this heuristically: existing ASR output tends to be all-uppercase or all-lowercase, or never emits commas, or strips complex punctuation. An all-caps transcript is almost never human. So I build heuristics to drop machine-generated transcripts. I also want the spoken language to actually match the transcript's language, so I run an audio language detector and compare; if they disagree, I throw the pair out — *unless* the audio is non-English and the transcript is English, in which case I've accidentally found a translation pair, and I keep it as an X→English speech-translation example. And I fuzzy-deduplicate transcripts to cut repeated boilerplate. After an initial model exists I can do another pass: look at which data *sources* give high error rate, inspect those, and remove the misaligned or low-quality ones.

There's a normalization decision hiding here that I want to make deliberately. Most ASR pipelines train on heavily *normalized* text — lowercased, no punctuation — and then bolt on a separate inverse-text-normalization stage to make the output readable. But I have a sequence-to-sequence model, and a seq2seq model is expressive enough to just *learn* to emit the raw, naturalistic transcript — punctuation, casing, and all. So I train it to predict raw text with no significant standardization, and the whole inverse-text-normalization stage disappears. That's a real simplification the architecture gives me for free.

Now the model. I am deliberately *not* going to invent a new architecture, because the point of this work is to isolate the effect of data scale and weak supervision; a fancy architecture would confound that. So I take an off-the-shelf encoder-decoder Transformer, which is well-validated to scale. The encoder reads audio and builds representations; the decoder is an audio-conditional language model that generates the text token by token, cross-attending to the encoder. The decoder being a language model is the key affordance — it's not just a recognizer, it can be *told* what to do.

Audio front end: resample everything to 16 kHz, compute an 80-channel log-mel spectrogram on 25 ms windows with a 10 ms stride, globally scaled to roughly [-1, 1] with about zero mean. I break audio into fixed **30-second segments**, each paired with whatever transcript falls in that window — a fixed-length chunk is what lets me batch efficiently and gives the model a consistent context size. Segments with no speech are kept (subsampled) and used to teach voice-activity detection. For the encoder, a small convolutional stem first: two conv layers, filter width 3, GELU, the second with stride 2 to halve the time resolution before the Transformer blocks. Then add sinusoidal position embeddings and run the encoder blocks; pre-activation residual blocks, with a final layer norm. The decoder uses *learned* position embeddings and ties its input and output token embeddings (the word-to-vector map and its inverse share weights — saves parameters and regularizes). Encoder and decoder share width and depth. Tokenization is byte-level BPE; I reuse the GPT-2 tokenizer for the English-only models and refit the vocabulary (same size) for multilingual so other languages don't fragment into too many pieces.

Now the part that actually has to be designed — the *interface*. I want one model to do not just transcription but the whole pile of speech tasks: transcribe, translate to English, detect language, decide if there's speech at all, and optionally produce timestamps. These are a one-to-many mapping from the same audio. With a single model, I need some way to *specify* which task to perform. And here the decoder-as-language-model pays off: I can specify everything as a sequence of special tokens that the decoder reads as its prefix, before it starts generating the answer. The task specification *is* part of the token stream.

Let me design that sequence carefully, in the order the model needs the information. I want the decoder to also be able to use *past text* as context — an audio-conditional LM should be allowed to condition on the transcript history to disambiguate hard audio — so with some probability I prepend the preceding segment's transcript as conditioning context (and I mask the loss over that conditioning text, since I'm not asking it to predict it). Then a `<|startoftranscript|>` token marks where prediction begins. The first thing to predict is *which language is being spoken* — a unique token per language — because everything downstream (and the user) needs to know that; if there's no speech, predict a `<|nospeech|>` token instead, which is exactly how voice-activity detection falls out. Next a task token: `<|transcribe|>` or `<|translate|>`. Then a choice about format: `<|notimestamps|>` if we don't want timestamps. After those tokens the task and format are fully specified, and the actual output text begins. Finally `<|endoftranscript|>`.

Timestamps deserve a moment. I want them optional and cheap, so I quantize time to the model's native resolution — 20 ms — and add a vocabulary token for each quantized time. When predicting timestamps, I interleave them with the text: a start-time token before each utterance's text, an end-time token after. The native 20 ms resolution comes from the front end: 10 ms stride times the stride-2 conv equals 20 ms per output frame, so quantizing to 20 ms matches what the model can actually resolve. When a final segment is only partially inside the 30-second chunk, in timestamp mode I predict only its start time, signaling that decoding should continue from an audio window aligned to that time; otherwise I truncate it. The training loss is next-token cross-entropy over the whole sequence *except* the prepended conditioning context, which is masked out.

What this buys me: one decoder, prompted by a few special tokens, replaces a whole pipeline — language ID, voice-activity detection, multilingual transcription, X→English translation, and timestamped alignment — all as token prediction. And because the supervised data spans 96+ languages and includes translation pairs, joint multilingual-multitask training is feasible; for large enough models there's no penalty and even a benefit to training all of it together.

Training details. A suite of sizes from tiny (4 layers, width 384) up to large (32 layers, width 1280, ~1.5B params), to study scaling. Optimize with AdamW (β₁=0.9, β₂=0.98, ε=1e-6, weight decay 0.1), gradient-norm clipping at 1.0, FP16 with dynamic loss scaling and activation checkpointing, a linear learning-rate decay to zero after a warmup over the first 2048 updates, batch size 256 segments, for about 2²⁰ updates — which is only two to three passes over the data. Because it's just a few epochs over a huge, diverse corpus, overfitting isn't the concern, so I use *no* data augmentation or regularization and lean on the dataset's diversity for generalization. One quirk I have to fix: many web transcripts name the speaker, so the model learns to *guess* speaker names from audio it can't possibly identify them from — it confidently emits plausible wrong names. I fix this with a brief fine-tune on the subset of transcripts that don't carry speaker annotations, which removes the behavior. (Note this is a behavioral cleanup, not the dataset-specific decoder fine-tuning I set out to avoid — the whole system is still used zero-shot on every benchmark.)

Let me write it, mirroring how I'd build it. First the standard Transformer block and the two towers, then the multitask token-sequence construction that is the real design.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):            # standard pre-norm block, GELU MLP 4x
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
        if self.xatt is not None:                   # decoder cross-attends to audio
            x = x + self.xatt(self.ln_x(x), xa, xa)[0]
        return x + self.mlp(self.ln2(x))

def sinusoids(length, d, max_timescale=10000):
    inv = torch.exp(-torch.log(torch.tensor(max_timescale)) *
                    torch.arange(d // 2) / (d // 2 - 1))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)

class AudioEncoder(nn.Module):                      # conv stem -> +sinusoids -> blocks -> LN
    def __init__(self, n_mels, d, layers, heads, n_ctx=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)   # stride 2 -> 20ms frames
        self.register_buffer('pos', sinusoids(n_ctx, d))
        self.blocks = nn.ModuleList(ResidualAttentionBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, mel):                         # (B, n_mels, n_frames)
        x = F.gelu(self.conv1(mel)); x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2) + self.pos[:x.size(-1)]
        for b in self.blocks:
            x = b(x)
        return self.ln(x)

class TextDecoder(nn.Module):                       # audio-conditional LM, tied embeddings
    def __init__(self, vocab, d, layers, heads, n_ctx=448):
        super().__init__()
        self.token = nn.Embedding(vocab, d)
        self.pos   = nn.Parameter(torch.empty(n_ctx, d))       # LEARNED positions
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(d, heads, cross=True) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, tokens, audio):
        x = self.token(tokens) + self.pos[:tokens.size(1)]
        mask = torch.triu(torch.full((tokens.size(1),) * 2, float('-inf')), 1)  # causal
        for b in self.blocks:
            x = b(x, audio, mask)
        return self.ln(x) @ self.token.weight.t()              # tied output projection
```

The real design is the decoder token sequence — task specification, conditioning, language/VAD, and interleaved timestamps — and which positions the loss covers:

```python
SOT, EOT, NOSPEECH = '<|startoftranscript|>', '<|endoftranscript|>', '<|nospeech|>'
TRANSCRIBE, TRANSLATE, NOTIME = '<|transcribe|>', '<|translate|>', '<|notimestamps|>'

def time_token(seconds):                            # quantize to native 20ms resolution
    return f'<|{round(seconds / 0.02) * 0.02:.2f}|>'

def build_decoder_sequence(tok, segment, prev_text=None, p_cond=0.5):
    seq, loss_mask = [], []                         # loss_mask: 1 where loss is applied
    if prev_text is not None and torch.rand(1).item() < p_cond:
        ctx = tok.encode(prev_text)                 # condition on prior transcript history
        seq += ctx; loss_mask += [0] * len(ctx)     # but DO NOT train to predict it
    seq.append(tok.special(SOT)); loss_mask.append(0)
    if segment.no_speech:                           # VAD falls out of language slot
        seq.append(tok.special(NOSPEECH)); loss_mask.append(1)
        seq.append(tok.special(EOT));      loss_mask.append(1)
        return seq, loss_mask
    seq.append(tok.lang(segment.language)); loss_mask.append(1)     # predict language
    seq.append(tok.special(TRANSLATE if segment.task == 'translate'
                           else TRANSCRIBE)); loss_mask.append(1)   # task token
    if segment.use_timestamps:
        for utt in segment.utterances:              # interleave start/end time with text
            seq.append(tok.special(time_token(utt.start))); loss_mask.append(1)
            toks = tok.encode(utt.text); seq += toks; loss_mask += [1] * len(toks)
            seq.append(tok.special(time_token(utt.end))); loss_mask.append(1)
    else:
        seq.append(tok.special(NOTIME)); loss_mask.append(1)
        toks = tok.encode(segment.text); seq += toks; loss_mask += [1] * len(toks)
    seq.append(tok.special(EOT)); loss_mask.append(1)
    return seq, loss_mask

def loss_step(encoder, decoder, mel, seq, loss_mask):
    audio = encoder(mel)
    logits = decoder(seq[:, :-1], audio)            # predict next token
    tgt = seq[:, 1:].clone()
    tgt[loss_mask[:, 1:] == 0] = -100               # ignore conditioning-context positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100)
```

And data-side, the filtering and 30-second segmentation that turn raw web pairs into examples:

```python
def keep_pair(audio, transcript, spoken_lang, transcript_lang):
    if looks_machine_generated(transcript):         # all-caps/all-lower, no commas, etc.
        return None
    if spoken_lang != transcript_lang:
        if transcript_lang == 'en':                 # non-en audio + en text = translation
            return ('translate', 'en')
        return None                                 # language mismatch -> drop
    return ('transcribe', transcript_lang)

def segment_audio(audio, transcript, length_s=30.0):
    # break into fixed 30s chunks paired with the transcript falling in each window;
    # keep speechless chunks (subsampled) as voice-activity-detection training data.
    ...
```

The causal chain, start to end: I want a recognizer that works zero-shot everywhere, and the fine-tuning step that today's unsupervised-encoder systems require is exactly what destroys that generalization and burdens deployment; so I train a *decoder* up front, supervised, which means I need huge, diverse labeled data, which I get by relaxing transcript quality and scraping the web at hundreds-of-thousands-of-hours scale (filtering out machine-generated and language-mismatched transcripts, repurposing non-English/English pairs as translation data, and predicting raw un-normalized text); I feed 30-second log-mel segments to an off-the-shelf encoder-decoder Transformer so I don't confound scale with architecture; and I make a *single* decoder cover transcription, translation, language-ID, voice-activity detection, and timestamping by specifying the task as a sequence of special tokens the audio-conditional LM reads before generating, training next-token cross-entropy over everything but the optional prepended text context.
