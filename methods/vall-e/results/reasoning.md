Let me start from the failure I keep running into: zero-shot TTS. I want to hand a system the text to speak and a few seconds of some person's voice — a person it has never heard in training — and get back that text in that voice, naturally. The cascaded systems I have, mel-spectrogram acoustic model plus vocoder, are gorgeous on clean studio data but they fall apart on an unseen speaker; naturalness and speaker similarity both drop off a cliff. The patches the field uses are heavy: speaker adaptation means fine-tuning a model per speaker, and speaker encoding means training a separate speaker-verification-style encoder to spit out an embedding I condition on — and even then there's a stubborn gap between seen and unseen speakers. I want to ask whether there's a fundamentally different route that makes the gap go away instead of papering over it.

Where has the "generalize to anything unseen" problem actually been solved recently? Text. Large language models trained on enormous, diverse corpora got *in-context learning* almost for free: you don't fine-tune GPT-3 for a new task, you put a few examples in the prompt and it does the task. And the thing that drove that capability was scale — data went from gigabytes to a terabyte, and each jump bought generalization. Speech understanding shows the same data-scaling curve, going from hundreds of hours to over a million. Now look at TTS: the good multi-speaker systems are trained on hundreds of hours. Hundreds. That gap — that TTS is being trained on a thousandth of the data that gave text models their generalization — feels like it might *be* the whole problem. So the hypothesis I want to chase: stop designing speaker-specific machinery, and instead train a language model on a huge, diverse pile of speech, and let zero-shot generalization emerge the way it did for text.

But I can't just "train a language model on speech," because a language model needs *tokens*, and speech is a continuous waveform. The cascaded systems use mel-spectrograms as the intermediate, and mel frames are continuous vectors fit with an L1 or L2 regression loss. Two problems with that for my plan. First, a regression loss on continuous frames is sensitive to data quality and spends its capacity modeling short-range continuous detail — it overfits clean data and chokes on noisy data, which is exactly the kind of large, messy corpus I need to scale into. Second, and more basically, regression over continuous targets isn't a *language model* — there's no discrete next-token distribution to do in-context learning over, no way to sample diverse outputs, no prompting. So if I want the text-LM recipe, the very first thing I need is to turn speech into discrete tokens, like words.

How do I discretize speech? μ-law quantizes amplitude to 256 levels — WaveNet uses it — but it doesn't shorten the sequence at all, so an autoregressive model over μ-law tokens is hopelessly long and slow. Self-supervised units, the HuBERT or vq-wav2vec k-means codes, do shorten the sequence and they reconstruct *content* well, but they throw away speaker identity — which is fatal for me, because speaker identity is the entire point of zero-shot voice cloning — and they reconstruct at low audio quality. So neither is right. What I want is a discrete tokenization that keeps speaker timbre and acoustic environment, reconstructs at high quality, *and* shortens the sequence. That's exactly what a neural audio codec gives me. EnCodec encodes 24 kHz audio down to 75 frames per second — a 320× reduction — and each frame is a small set of discrete codes, and crucially it preserves the speaker and even the room, because it was trained to *reconstruct the waveform*, not to strip it to phonetic content. And it ships with its own decoder, so I get a high-quality vocoder for free; I never train one. So: EnCodec codes are my tokens. AudioLM already showed that you can run a language model over neural-codec tokens and get natural audio — but AudioLM is speech-to-speech, it continues audio, it has no text handle, so I can't tell it what to say. I want the same trick but text-controllable.

Now I have to look closely at what an EnCodec frame actually is, because the structure is going to dictate the whole model. EnCodec uses residual vector quantization: the frame's latent is quantized by a first codebook, the residual by a second, and so on, eight codebooks deep at 6 kbps. So one frame is not a single token — it's eight tokens, `c_{t,1} ... c_{t,8}`, and they are *not* exchangeable. The first codebook captures the dominant content, including speaker identity; each later codebook only refines the residual the earlier ones left behind, so importance falls off down the hierarchy. A ten-second clip is therefore a matrix `C` of shape `750 × 8` — 750 time frames, 8 codebooks deep. So my model isn't predicting a 1D token stream; it's predicting a 2D matrix. That's the crux. How do I factor `p(C | text, prompt)`?

The naive thing is to flatten the matrix into one long sequence — `c_{1,1}, c_{1,2}, ..., c_{1,8}, c_{2,1}, ...` — and run a single autoregressive model. But that multiplies my already-750-long sequence by 8 to 6000, and the per-step autoregression at audio time scales is the latency killer I was trying to escape. So flattening is out. The other extreme — predict all 8 × 750 codes fully in parallel, non-autoregressively — would be fast, but the time axis genuinely needs autoregression: I don't know in advance how *long* the output should be (speaking rate varies wildly across speakers, and I can't reliably train a duration predictor across thousands of diverse speakers), and an AR model decides its own length by emitting an end-of-sequence token. So neither pure-AR-over-flattened nor pure-NAR works cleanly.

The RVQ hierarchy gives me a cleaner split. The two axes of the matrix have completely different characters. Along the *time* axis I have genuine sequential dependency and unknown length — that wants autoregression. Along the *codebook* axis, the dependency is the residual hierarchy: codebook `j` depends on codebooks `< j`, but given all the earlier codebooks, predicting codebook `j` across *all 750 timesteps at once* is fine — there's no left-to-right dependency *within* a single codebook level that I can't already condition on. So split the problem by axis. Use an autoregressive model for the first codebook — the one that carries speaker identity and content, the one whose length I need the AR model to decide — and a non-autoregressive model for the remaining seven, each generated in one parallel shot conditioned on the codebooks below it. That gives me the best of both: AR handles length and the most important content; NAR collapses the time complexity of the other seven codebooks from O(T) to O(1) each. The full factorization is

  p(C | x, C̃) = p(c_{:,1} | x, C̃_{:,1}; θ_AR) · Π_{j=2}^{8} p(c_{:,j} | c_{:,<j}, x, C̃; θ_NAR),

where `x` is the phoneme sequence and `C̃` is the acoustic-prompt code matrix from the enrolled clip.

Now the AR model for the first codebook. It's a decoder-only Transformer. I embed the phoneme sequence `x` through a phoneme embedding and the first-codebook acoustic tokens through an acoustic embedding, concatenate them into one sequence (phonemes then audio codes, with an EOS after each), add sinusoidal position embeddings — computed *separately* for the phoneme part and the acoustic part, since they're two different streams — and run a causal Transformer. Each acoustic token `c_{t,1}` attends to the phonemes and all earlier acoustic tokens `c_{≤t,1}`, and the model is trained to maximize the next first-codebook token. The prediction head maps to the codebook size plus an EOS, and — a small efficiency I'll take — I tie the output projection weights to the acoustic embedding weights, since both live in the same code space.

The prompting mechanism for the AR model is where the in-context-learning idea becomes concrete, and it's beautifully cheap. I don't introduce any special "this is the prompt" token or separate prompt encoder during training. I just train an ordinary causal language model on `(phonemes, first-codebook codes)`. Because it's causal, *any prefix of the acoustic sequence automatically acts as a prompt for the rest* — that's just what autoregression is. So at inference, to clone a voice, I concatenate the phonemes of the enrolled clip and the phonemes of what I want to say into the phoneme prompt, and I feed the first-codebook codes of the enrolled clip as the acoustic prefix; then I let the AR model continue, generating first-codebook codes for the new text in the prefix's voice. No fine-tuning, no speaker embedding — the speaker information is carried in-context by the acoustic prefix, exactly like a few-shot text prompt. For decoding I sample rather than beam-search, because beam search collapses into loops and repetitions here, and sampling also gives me output diversity for the same text and speaker.

Now the NAR model for codebooks 2 through 8. Same Transformer architecture, but several differences forced by the non-autoregressive structure. First, since within a codebook level the tokens are predicted all at once and can't see each other, attention is *non-causal* — every position attends to every other. Second, conditioning on the lower codebooks: to predict codebook `i`, I take the codes from codebooks 1..`i−1`, embed each through its own acoustic embedding table, and *sum* them across codebooks to form the input acoustic representation — summing is the natural choice because RVQ codes are themselves additive (the quantized latent is the sum of the chosen codebook entries), so summing their embeddings mirrors the way the codec reconstructs the latent. Third, the speaker prompt: in the NAR setting I can't rely on a causal prefix, so I explicitly take the enrolled clip's full code matrix `C̃` (all eight codebooks), embed and sum across its codebooks, and prepend it as the acoustic prompt. The Transformer input is then the concatenation of the phoneme embeddings, the summed prompt embeddings, and the summed lower-codebook embeddings of the current utterance, with positions computed separately for prompt and sequence.

How does the NAR model know *which* codebook level it's currently predicting? I inject the stage index `i`. The clean way is adaptive layer normalization: instead of a fixed LayerNorm, scale and shift the normalized activations by parameters derived from the stage:

  AdaLN(h, i) = a_i · LayerNorm(h) + b_i,

where `a_i` and `b_i` come from a linear projection of a stage embedding for level `i`. So one shared NAR network handles all seven levels, told which level it's on through the AdaLN conditioning. In training I sample a random stage `i ∈ [2, 8]` each step and train the model to predict codebook `i` from the codebooks below it; at inference I run it seven times, levels 2 through 8 in order. And I share weights between the acoustic embedding layers and the prediction layers — the `j`-th prediction layer shares weights with the `(j+1)`-th acoustic embedding — because the code space is the same object whether I'm reading a code in or writing one out, so tying them saves parameters and ties the two views of each codebook together.

The division of labor now has a clean shape. The AR model decides the utterance length and lays down the speaker-and-content-bearing first codebook, with the speaker carried in by the acoustic prefix — that's where the flexibility lives. The NAR model fills in the seven refinement codebooks in parallel, anchored to the speaker by the explicit `C̃` prompt across all codebooks — that's where the speed lives, O(1) per level instead of O(T). Together: `p(C) = p(c_{:,1} | C̃_{:,1}, x; θ_AR) · Π_{j=2}^{8} p(c_{:,j} | c_{:,<j}, x, C̃; θ_NAR)`. Then hand the full 750×8 code matrix to EnCodec's decoder and out comes the waveform — the room and timbre of the prompt preserved, because the codes carry them and I never went through a speaker-stripping bottleneck.

The training is just language modeling, which is the entire point — it's simple and it scales. Tokenize 60K hours of audiobook speech with EnCodec, get phoneme alignments from an ASR model, and train the AR and NAR Transformers (both 12 layers, 16 heads, `d_model` 1024, FFN 4096) with cross-entropy. No adaptation, no speaker encoder, no vocoder training, no duration model. The only thing I'm betting on is that scale plus discrete codec tokens plus the AR-then-NAR factorization gives the in-context speaker generalization I couldn't get any other way.

I can now write the model skeleton so the tensors follow that factorization: the AR path returns logits only for first-codebook acoustic positions, and the NAR path returns logits only for the target utterance slots after conditioning on phonemes, the full enrolled-code prompt, and the lower generated codebooks.

```python
import math
import torch
import torch.nn as nn

NUM_AUDIO_TOKENS = 1024          # EnCodec codebook size
NUM_QUANTIZERS   = 8
EOS = NUM_AUDIO_TOKENS           # extra id for end-of-sequence (AR only)

class SinePositionalEmbedding(nn.Module):
    def forward(self, x):
        length, width = x.size(1), x.size(2)
        pos = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=x.dtype)
                        * (-math.log(10000.0) / width))
        pe = x.new_zeros(length, width)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

class AdaptiveLayerNorm(nn.Module):
    # AdaLN(h, i) = a_i * LayerNorm(h) + b_i ; a_i, b_i from the stage embedding
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model)
    def forward(self, h, stage_emb):
        if stage_emb.dim() == 2:
            stage_emb = stage_emb.unsqueeze(1)
        a, b = self.proj(stage_emb).chunk(2, dim=-1)
        return a * self.ln(h) + b

class NARBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, stage_emb):
        q = self.norm1(h, stage_emb)
        h = h + self.drop(self.attn(q, q, q, need_weights=False)[0])
        h = h + self.drop(self.ffn(self.norm2(h, stage_emb)))
        return h

class VALLE(nn.Module):
    def __init__(self, d_model=1024, nhead=16, nlayers=12, d_ff=4096,
                 n_phonemes=512, dropout=0.1):
        super().__init__()
        self.phoneme_emb = nn.Embedding(n_phonemes, d_model)
        self.phoneme_pos = SinePositionalEmbedding()

        # --- AR: first codebook only ---
        self.ar_audio_emb = nn.Embedding(NUM_AUDIO_TOKENS + 1, d_model)   # +1 for EOS
        self.ar_audio_pos = SinePositionalEmbedding()
        self.ar = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout=dropout,
                                       batch_first=True), nlayers)
        self.ar_predict = nn.Linear(d_model, NUM_AUDIO_TOKENS + 1)
        self.ar_predict.weight = self.ar_audio_emb.weight                 # tie embed <-> predict

        # --- NAR: codebooks 2..8, shared net, stage-conditioned via AdaLN ---
        self.nar_audio_embs = nn.ModuleList(
            [nn.Embedding(NUM_AUDIO_TOKENS, d_model) for _ in range(NUM_QUANTIZERS)])
        self.nar_prompt_pos = SinePositionalEmbedding()
        self.nar_audio_pos = SinePositionalEmbedding()
        self.nar_stage_emb = nn.Embedding(NUM_QUANTIZERS - 1, d_model)     # stages i=2..8
        self.nar = nn.ModuleList([NARBlock(d_model, nhead, d_ff, dropout)
                                  for _ in range(nlayers)])
        self.nar_predicts = nn.ModuleList(
            [nn.Linear(d_model, NUM_AUDIO_TOKENS) for _ in range(NUM_QUANTIZERS - 1)])
        for j in range(NUM_QUANTIZERS - 1):                                # predict layer j shares
            self.nar_predicts[j].weight = self.nar_audio_embs[j + 1].weight  # with (j+1)-th embedding

    def forward_ar(self, phoneme_prompt, codes1):
        # causal LM over [phonemes | first-codebook codes]; only acoustic slots are predicted
        x = self.phoneme_pos(self.phoneme_emb(phoneme_prompt))
        a = self.ar_audio_pos(self.ar_audio_emb(codes1))
        h = torch.cat([x, a], dim=1)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, device=h.device), diagonal=1).bool()  # no peeking ahead
        h = self.ar(h, mask=mask)
        return self.ar_predict(h[:, x.size(1):])                            # logits over c_{:,1} (+ EOS)

    def forward_nar(self, phonemes, prompt_codes, codes_lt_i, stage_i):
        # predict target codebook stage_i (2..8), non-causally, from lower codebooks and full prompt
        x = self.phoneme_pos(self.phoneme_emb(phonemes))
        prompt = sum(self.nar_audio_embs[j](prompt_codes[..., j]) for j in range(NUM_QUANTIZERS))
        prompt = self.nar_prompt_pos(prompt)
        y = sum(self.nar_audio_embs[j](codes_lt_i[..., j]) for j in range(stage_i - 1))  # sum lower
        y = self.nar_audio_pos(y)
        h = torch.cat([x, prompt, y], dim=1)
        stage_ids = torch.full((h.size(0),), stage_i - 2, device=h.device, dtype=torch.long)
        stage_emb = self.nar_stage_emb(stage_ids)                           # AdaLN tells it the stage
        for block in self.nar:
            h = block(h, stage_emb)
        target_h = h[:, -y.size(1):]                                        # discard phoneme/prompt slots
        return self.nar_predicts[stage_i - 2](target_h)

    @torch.no_grad()
    def generate_first_codebook(self, phoneme_prompt, acoustic_prefix, max_new_tokens, temperature=1.0):
        # sampling avoids the beam-search looping failure; acoustic_prefix is the enrolled speaker
        codes = acoustic_prefix
        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward_ar(phoneme_prompt, codes)[:, -1] / temperature
            next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            if (next_id == EOS).all():
                break
            generated.append(next_id.clamp_max(NUM_AUDIO_TOKENS - 1))
            codes = torch.cat([codes, next_id], dim=1)
        if not generated:
            return acoustic_prefix.new_empty(acoustic_prefix.size(0), 0)
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def synthesize(self, phoneme_prompt, prompt_codes, codec, max_new_tokens):
        # phoneme_prompt = enrolled transcription phonemes followed by requested-text phonemes
        codes1 = self.generate_first_codebook(phoneme_prompt, prompt_codes[..., 0], max_new_tokens)
        codes = [codes1]
        # NAR: greedily fill codebooks 2..8, each conditioned on all previous
        cur = codes1.unsqueeze(-1)
        for i in range(2, NUM_QUANTIZERS + 1):
            logits = self.forward_nar(phoneme_prompt, prompt_codes, cur, i)
            ci = logits.argmax(-1)
            cur = torch.cat([cur, ci.unsqueeze(-1)], dim=-1)
            codes.append(ci)
        C = torch.stack(codes, dim=-1)                                      # [T, 8]
        return codec.decode(C)                                              # off-the-shelf vocoder
```

The whole chain: zero-shot TTS fails because TTS is data-starved and mel-regression can't scale, so I borrow the text-LM recipe — but to run a language model on speech I need discrete tokens that keep the speaker, which is exactly what neural-codec RVQ codes give me, decoder included. Those codes form a `T × 8` matrix whose two axes differ: time wants autoregression and length-decision, codebook-depth wants conditioned parallel prediction. So I split it into an AR model for the first, identity-bearing codebook — where a plain causal LM makes any acoustic prefix a zero-shot voice prompt — and a NAR model for the other seven, conditioned on the lower codebooks via summed embeddings and told its stage through adaptive layer-norm, generated in parallel for speed. Train it as language modeling on 60K hours, decode the matrix with EnCodec, and the voice of a three-second unseen clip comes back speaking new words.
