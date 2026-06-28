Let me start from what's actually broken about the best speech systems, because it points straight at what to build. The strongest recognizers come from self-supervised pre-training on raw audio — and that's been scaled to something like a million hours of unlabeled speech, dwarfing the thousand-odd hours of any academic labeled set. So the *encoder* is fantastic. But the objective is purely unsupervised, which means there's no decoder that turns those representations into text. To actually recognize speech you have to fine-tune. And that fine-tuning step is the thing I want to look at hard, because it carries two costs.

The first cost is usability: every new deployment distribution needs someone skilled to run a fine-tuning job. The second is subtler and worse — robustness. A model fine-tuned on one dataset gets very good at *that dataset's* idiosyncrasies. There's a clean cautionary tale from vision: fine-tuning a classifier on ImageNet raised ImageNet accuracy by about nine points while not improving average accuracy on seven other datasets of the very same objects. The model learned the dataset's quirks, not the task. A speech decoder fine-tuned per dataset has the identical exposure — it can look superhuman in-distribution and still botch basic cases out of distribution. So fine-tuning isn't just inconvenient; it actively trades away the generalization I care about.

So state the goal cleanly: I want a single system that works zero-shot — no per-deployment fine-tuning of a decoder — across recording conditions, accents, domains, and languages. The question is what training recipe produces that breadth. If the prohibition is "I don't want to fine-tune a decoder per dataset," then whatever decoder I ship has to already be good at producing text before deployment. There are two ways to get there. One is a cleverer self-supervised objective that somehow yields a usable decoder for free; the other is to just train the decoder up front, supervised, on (audio, transcript) pairs. The first is the more elegant research direction, but it keeps me inside the regime whose failure I just diagnosed — a learned objective tuned and then adapted. The second is dumber and needs a lot of labeled data, but it directly attacks the cost I identified. Let me follow the second and see how far the data requirement can be pushed, because what plausibly buys robustness here is *diversity* of supervised data, not a clever objective.

How much supervised data can I get? The clean, human-validated corpora are tiny — pooling seven of them gets you around five thousand hours, robust but minuscule next to a million unsupervised hours. The bottleneck is the demand for gold-standard transcripts. So relax it. The internet is full of audio already paired with transcripts — captions, subtitles — much of it noisy, but there's an enormous amount of it. Vision already learned an analogous lesson: moving from curated datasets to much larger *weakly*-supervised ones improves robustness and generalization. The bet I'm going to make, then, is to scale weakly-supervised speech recognition far past the existing tens-of-thousands of hours — I'll aim at 680,000 hours — and let raw diversity, not curation, do the work. I should be honest that this is a bet about whether weak supervision has been under-scaled for speech, not something I can prove before training; the rest of the design is what makes the bet executable.

Now the data is web-scraped, so I have to think about what filtering actually matters, because not all noise is equal. Diversity in *audio* quality is good — it's exactly what makes the model robust to recording conditions. Diversity in *transcript* quality is not good in the same way; some transcripts are just wrong. The single most dangerous kind of bad transcript is one that's itself machine-generated, the output of some other ASR system. If I train on those, I learn to imitate another model's "transcript-ese" — its normalization, its missing punctuation — rather than how humans actually write speech. I can detect a lot of this heuristically: existing ASR output tends to be all-uppercase or all-lowercase, or never emits commas, or strips complex punctuation. An all-caps transcript is almost never human. So I build heuristics to drop machine-generated transcripts. I also want the spoken language to actually match the transcript's language, so I run an audio language detector and compare; if they disagree, I throw the pair out — *unless* the audio is non-English and the transcript is English, in which case I've accidentally found a translation pair, and I keep it as an X→English speech-translation example. And I fuzzy-deduplicate transcripts to cut repeated boilerplate. After an initial model exists I can do another pass: look at which data *sources* give high error rate, inspect those, and remove the misaligned or low-quality ones.

There's a normalization decision hiding here that I want to make deliberately. Most ASR pipelines train on heavily *normalized* text — lowercased, no punctuation — and then bolt on a separate inverse-text-normalization stage to make the output readable. Do I need that two-stage design? The reason it exists is that older recognizers (CTC over a small character set, n-gram-rescored lattices) struggle to emit casing and punctuation, so the readable form is reconstructed afterward by a rule system. A sequence-to-sequence decoder has no such limitation — punctuation and casing are just more tokens it can be trained to predict in sequence, with no extra machinery. So if I train it on raw, naturalistic transcripts, the model should emit readable text directly and the entire inverse-text-normalization stage simply has nothing to do. I'll train on raw text with no significant standardization and drop that stage. The one thing I'd want to watch is whether raw text inflates word error rate at scoring time, since the eval transcripts use varied conventions — but that's a scoring problem to fix with a normalizer *at evaluation*, not a reason to normalize the *training* targets.

I am deliberately *not* going to invent a new architecture, because the point of this work is to isolate the effect of data scale and weak supervision; a fancy architecture would confound that. So I take an off-the-shelf encoder-decoder Transformer, which is well-validated to scale. The encoder reads audio and builds representations; the decoder is an audio-conditional language model that generates the text token by token, cross-attending to the encoder. That the decoder is itself a language model is going to matter more than it first appears — it means the decoder can be *conditioned* and *prompted*, not just run as a fixed recognizer — and I'll come back to exploit that.

For the audio front end, I resample everything to 16 kHz, compute an 80-channel log-mel spectrogram on 25 ms windows with a 10 ms stride, and globally scale it to roughly [-1, 1] with about zero mean. I break audio into fixed **30-second segments**, each paired with whatever transcript falls in that window — a fixed-length chunk is what lets me batch efficiently and gives the model a consistent context size. Segments with no speech are kept with a 10x subsample factor and used to teach voice-activity detection. For the encoder, a small convolutional stem first: two conv layers, filter width 3, GELU, the second with stride 2 to halve the time resolution before the Transformer blocks. Then add sinusoidal position embeddings and run the encoder blocks; pre-activation residual blocks, with a final layer norm. The decoder uses *learned* position embeddings and ties its input and output token embeddings (the word-to-vector map and its inverse share weights — saves parameters and regularizes). Encoder and decoder share width and depth. Tokenization is byte-level BPE; I reuse the GPT-2 tokenizer for the English-only models and refit the vocabulary (same size) for multilingual so other languages don't fragment into too many pieces.

Let me pin down the encoder's output time resolution, because several later choices depend on it and I keep wanting to quote "20 ms" without having checked the arithmetic. The mel front end emits one frame every 10 ms (the hop). The conv stem's second layer has stride 2, which halves that, so each encoder output frame covers 10 ms × 2 = 20 ms of audio. Over a 30-second segment that is 30 × 1000 / 20 = 1500 output frames. That number is not free-floating — it's the `n_ctx=1500` I have to size the encoder's positional buffer to, and it lines up exactly, so a 30-second window fills the encoder context with no waste and no truncation. Good; 20 ms is the model's genuine native time step, and I'll let that number drive the timestamp design rather than picking a resolution arbitrarily.

The part that actually has to be designed is the *interface*. I want one model to do not just transcription but the whole pile of speech tasks: transcribe, translate to English, detect language, decide if there's speech at all, and optionally produce timestamps. These are a one-to-many mapping from the same audio. With a single model, I need some way to *specify* which task to perform. One option is multiple output heads or a task-id fed in somewhere structural, but that re-introduces task-specific machinery — the very thing I'm trying to avoid — and it doesn't compose (how would "translate, with timestamps, in language X" be a single head?). Here is where the decoder-being-a-language-model pays off concretely: a task request, a language label, a "no speech" verdict, a "with/without timestamps" choice — all of these are just symbols, and a language model already consumes and emits symbols autoregressively. So I can express every one of them as ordinary special tokens living in the same sequence as the text, and the model conditions on the earlier ones to produce the later ones for free. No extra heads, and the choices compose because they're just a sequence.

I want the decoder to also be able to use *past text* as context — an audio-conditional LM should be allowed to condition on the transcript history to disambiguate hard audio — so with 50% probability I prepend the preceding segment's transcript as conditioning context, and I mask the loss over only that conditioning text. Then a `<|startoftranscript|>` token marks where prediction begins. The first predicted content should be *which language is being spoken* — one of 99 language tokens — because everything downstream depends on that. If there is no speech, that classification slot becomes `<|nospeech|>`, which is exactly how voice-activity detection falls out. If there is speech, the next token specifies the task, `<|transcribe|>` or `<|translate|>`, and then a choice about format: `<|notimestamps|>` if I want plain text, or timestamp tokens if I want alignment. After those tokens the task and desired format are fully specified, and the actual output text begins. Finally `<|endoftranscript|>` closes the segment.

Timestamps deserve a moment. I want them optional and cheap, so I quantize time relative to the current 30-second segment and add a vocabulary token for each quantized time. The natural quantization is the model's native step, which I just established is 20 ms — quantizing finer than the encoder can resolve would be inventing precision the representation doesn't have. So I add a time token for each 20 ms bin from 0.00 to 30.00 seconds; that's 30.00 / 0.02 + 1 = 1501 tokens covering the whole window inclusive of both ends. When predicting timestamps, I interleave them with the text: a start-time token before each utterance's text, an end-time token after. When a final segment is only partially inside the 30-second chunk, in timestamp mode I predict only its start time, signaling that decoding should continue from an audio window aligned to that time; otherwise I truncate it. The training loss is next-token cross-entropy over the whole sequence *except* the prepended conditioning context, which is masked out.

Before I trust this token-sequence design, let me actually trace the loss masking on a couple of hand-built sequences, because the off-by-one between "which positions I mask" and "which positions get a target" after the autoregressive shift is exactly the kind of thing I'd get subtly wrong. Take a transcribe segment with one prior-text token of conditioning prepended. The built sequence of ids is, schematically, `[ctx, SOT, LANG, TRANSCRIBE, NOTIME, w₁, w₂, EOT]` with loss mask `[0, 1, 1, 1, 1, 1, 1, 1]` — zero only on the conditioning token. The decoder is fed `seq[:, :-1]` and trained against `seq[:, 1:]`, with mask positions set to `-100` using `lm[:, 1:]`. Working that shift by hand: the input becomes `[ctx, SOT, LANG, TRANSCRIBE, NOTIME, w₁, w₂]` and the target becomes `[SOT, LANG, TRANSCRIBE, NOTIME, w₁, w₂, EOT]`. The shifted mask `lm[:, 1:]` is all ones here, so every target survives — and crucially the conditioning token `ctx` never appears as a target (it's only ever an input), while the *first* trained target is SOT, which the model predicts *from* the conditioning token. That's the intended behavior: the model learns to begin a transcript after seeing prior text, but is never asked to reproduce the prior text itself. So the masking does what the prose claims.

Now the case I'm actually worried about — no conditioning text at all. Then the sequence starts `[SOT, NOSPEECH, EOT]` for a silent segment, with mask `[1, 1, 1]`. After the shift, input is `[SOT, NOSPEECH]` and target is `[NOSPEECH, EOT]`. So SOT here is input-only and is *never* a target. That means SOT is predicted as an output only in the conditioned case, and serves as a pure delimiter in the unconditioned case — which is consistent, and it's why I can say SOT is "predicted when prior text exists." If I had instead wanted SOT itself supervised at position zero I'd have needed a BOS token before it; I don't, so this is fine. The trace also confirms voice-activity detection requires no special loss handling — a no-speech segment is just the three-token sequence `SOT → NOSPEECH → EOT`, trained exactly like any other. Good. Both walks behave, so I'll keep the construction.

So now one decoder, prompted and trained through a few special tokens, replaces a whole pipeline — language ID, voice-activity detection, multilingual transcription, X→English translation, and timestamped alignment — all as token prediction. For the multilingual and translation parts to be more than a vestigial side task, the data has to actually contain them at scale: of the 680,000 hours, 117,000 hours span 96 non-English languages and 125,000 hours are X→English translation pairs (which is where those repurposed language-mismatched pairs end up). That's a large enough fraction that joint multilingual-multitask training is a real bet rather than a starved afterthought.

I train a suite of sizes from tiny (4 layers, width 384) up to large (32 layers, width 1280, ~1.5B params), because the scale question is part of the method. I use Gaussian fan-in initialization and optimize with AdamW (β₁=0.9, β₂=0.98, ε=1e-6, weight decay 0.1), gradient-norm clipping at 1.0, FP16 with dynamic loss scaling and activation checkpointing, a linear learning-rate decay to zero after a warmup over the first 2048 updates, batch size 256 segments, for 2²⁰ updates — only two to three passes over the data. Because it's just a few epochs over a huge, diverse corpus, overfitting isn't the concern, so I use *no* data augmentation or regularization and lean on the dataset's diversity for generalization. One failure mode I should expect from web data: many transcripts name the speaker, so the model will learn to *guess* speaker names from audio it cannot possibly identify them from — it will confidently emit plausible wrong names. The fix is to take the subset of transcripts that carry no speaker annotation and briefly continue training on it, so the decoder stops treating a speaker name as something to infer from a 30-second window. I want to be clear with myself that this is a behavioral cleanup on a label artifact, not the dataset-specific decoder fine-tuning I set out to avoid — it touches no evaluation distribution.

I can write the system as the standard Transformer block and two towers, plus the multitask token-sequence construction that carries the task interface.

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
        x = x.transpose(1, 2)
        x = x + self.pos[:x.size(1)]
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
TIME_STEP, MAX_TIME = 0.02, 30.0

def time_token(seconds):
    idx = round(max(0.0, min(float(seconds), MAX_TIME)) / TIME_STEP)
    return f'<|{idx * TIME_STEP:.2f}|>'             # 20ms bins over a 30s window

def build_decoder_sequence(tok, segment, prev_text=None, p_cond=0.5):
    seq, loss_mask = [], []                         # loss_mask: 1 where loss is applied
    if prev_text is not None and torch.rand(1).item() < p_cond:
        ctx = tok.encode(prev_text)                 # condition on prior transcript history
        seq += ctx; loss_mask += [0] * len(ctx)     # but DO NOT train to predict it
    seq.append(tok.special(SOT)); loss_mask.append(1)   # predicted when prior text exists
    if segment.no_speech:                           # VAD replaces the language slot
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

The causal chain, start to end: I want a recognizer that works zero-shot everywhere, and the fine-tuning step that today's unsupervised-encoder systems require is exactly what destroys that generalization and burdens deployment; so I train a *decoder* up front, supervised, which means I need huge, diverse labeled data, which I get by relaxing transcript quality and scraping the web at 680,000-hour scale (filtering out machine-generated and language-mismatched transcripts, repurposing non-English/English pairs as translation data, and predicting raw un-normalized text); I feed 30-second log-mel segments to an off-the-shelf encoder-decoder Transformer so I don't confound scale with architecture; and I make a *single* decoder cover transcription, translation, language-ID, voice-activity detection, and timestamping by representing task, language/no-speech, timestamp mode, timestamps, and text as one special-token sequence — a construction I checked behaves correctly under the autoregressive loss shift — training next-token cross-entropy over everything but the optional prepended text context.
