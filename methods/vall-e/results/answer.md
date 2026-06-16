# VALL-E, distilled

VALL-E is a zero-shot text-to-speech system that treats TTS as **conditional codec language modeling**: it represents speech as discrete neural-audio-codec (EnCodec) tokens and trains language models to generate them, conditioned on a phoneme sequence and a ~3-second acoustic prompt. From seconds of an unseen speaker, it clones the voice — timbre, prosody, and acoustic environment — with no fine-tuning, no speaker encoder, and no separately trained vocoder.

## The problem

Cascaded mel-spectrogram TTS generalizes poorly to unseen speakers, and the fixes (per-speaker fine-tuning, speaker-encoder embeddings) are heavy and still leave a gap. Text language models, by contrast, generalize via in-context learning once trained at scale. VALL-E ports that recipe to speech — but a language model needs discrete tokens, and mel regression neither tokenizes speech nor scales to large noisy corpora.

## Key ideas

**Discrete codec tokens as the representation.** Use a pretrained EnCodec (6 kbps, 24 kHz) as the tokenizer: 75 Hz frames, `N_q = 8` residual-VQ codebooks of 1024 entries. A 10 s clip → a `750 × 8` code matrix `C`. These codes keep speaker identity and acoustic environment (unlike HuBERT/self-supervised units) and come with a high-quality decoder (no vocoder to train).

**Factor the `T × 8` matrix by axis (AR + NAR).** The codebook hierarchy is ordered — codebook 1 carries dominant content/speaker, later codebooks refine residuals. Time needs autoregression (unknown output length, decided by EOS); codebook depth allows conditioned parallel prediction. So:

```
p(C | x, C̃) = p(c_{:,1} | x, C̃_{:,1}; θ_AR) · Π_{j=2}^{8} p(c_{:,j} | c_{:,<j}, x, C̃; θ_NAR)
```

- **AR model** (decoder-only causal Transformer) generates codebook 1 from phonemes + the codebook-1 acoustic prefix. Trained as a plain causal LM, so *any acoustic prefix is automatically a prompt*; at zero-shot inference, the enrolled transcription phonemes are prepended to the target phonemes, and the enrolled first-codebook tokens are used as the acoustic prefix. Sampling-based decoding avoids beam-search loops and gives output diversity.
- **NAR model** (non-causal Transformer) generates codebooks 2–8, each in one parallel pass — O(1) instead of O(T) per level. Conditioned on lower codebooks by **summing their embeddings** (mirrors RVQ's additive codes) and on the speaker by the full 8-codebook prompt `C̃`. The current stage `i` is injected via **adaptive layer norm** `AdaLN(h, i) = a_i · LayerNorm(h) + b_i`. One shared net handles all 7 levels (random stage sampled per training step).

**Weight tying.** Output projection ties to the acoustic embedding; in the NAR model, prediction layer `j` shares weights with the `(j+1)`-th acoustic embedding.

**Decode** the full `750 × 8` matrix with EnCodec's decoder to the waveform.

## Setup

Train on LibriLight (60K hours, ~7K speakers; ASR-generated transcripts, 30 ms phoneme alignment). AR and NAR both: 12 layers, 16 heads, `d_model` 1024, FFN 4096, dropout 0.1. AdamW, warmup 32K steps to peak lr 5×10⁻⁴ then linear decay, 800K steps. NAR training uses a random 3 s segment of the same utterance as the acoustic prompt; AR training is plain causal first-codebook language modeling, so the prompt at inference is just a given prefix.

## Working code

```python
import math
import torch, torch.nn as nn

NUM_AUDIO_TOKENS = 1024
NUM_QUANTIZERS = 8
EOS = NUM_AUDIO_TOKENS

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
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(d, 2 * d)
    def forward(self, h, stage_emb):
        if stage_emb.dim() == 2:
            stage_emb = stage_emb.unsqueeze(1)
        a, b = self.proj(stage_emb).chunk(2, dim=-1)
        return a * self.ln(h) + b

class NARBlock(nn.Module):
    def __init__(self, d, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, h, stage_emb):
        q = self.norm1(h, stage_emb)
        h = h + self.drop(self.attn(q, q, q, need_weights=False)[0])
        h = h + self.drop(self.ffn(self.norm2(h, stage_emb)))
        return h

class VALLE(nn.Module):
    def __init__(self, d=1024, nhead=16, nlayers=12, d_ff=4096, n_phonemes=512, dropout=0.1):
        super().__init__()
        self.phoneme_emb = nn.Embedding(n_phonemes, d)
        self.phoneme_pos = SinePositionalEmbedding()
        # AR (codebook 1)
        self.ar_audio_emb = nn.Embedding(NUM_AUDIO_TOKENS + 1, d)
        self.ar_audio_pos = SinePositionalEmbedding()
        self.ar = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, d_ff, dropout=dropout, batch_first=True), nlayers)
        self.ar_predict = nn.Linear(d, NUM_AUDIO_TOKENS + 1)
        self.ar_predict.weight = self.ar_audio_emb.weight
        # NAR (codebooks 2..8)
        self.nar_audio_embs = nn.ModuleList([nn.Embedding(NUM_AUDIO_TOKENS, d) for _ in range(NUM_QUANTIZERS)])
        self.nar_prompt_pos = SinePositionalEmbedding()
        self.nar_audio_pos = SinePositionalEmbedding()
        self.nar_stage_emb = nn.Embedding(NUM_QUANTIZERS - 1, d)
        self.nar = nn.ModuleList([NARBlock(d, nhead, d_ff, dropout) for _ in range(nlayers)])
        self.nar_predicts = nn.ModuleList([nn.Linear(d, NUM_AUDIO_TOKENS) for _ in range(NUM_QUANTIZERS - 1)])
        for j in range(NUM_QUANTIZERS - 1):
            self.nar_predicts[j].weight = self.nar_audio_embs[j + 1].weight

    def forward_ar(self, phoneme_prompt, codes1):
        x = self.phoneme_pos(self.phoneme_emb(phoneme_prompt))
        a = self.ar_audio_pos(self.ar_audio_emb(codes1))
        h = torch.cat([x, a], dim=1)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, device=h.device), diagonal=1).bool()
        h = self.ar(h, mask=mask)
        return self.ar_predict(h[:, x.size(1):])

    def forward_nar(self, phonemes, prompt_codes, codes_lt_i, stage_i):
        x = self.phoneme_pos(self.phoneme_emb(phonemes))
        prompt = sum(self.nar_audio_embs[j](prompt_codes[..., j]) for j in range(NUM_QUANTIZERS))
        prompt = self.nar_prompt_pos(prompt)
        y = sum(self.nar_audio_embs[j](codes_lt_i[..., j]) for j in range(stage_i - 1))
        y = self.nar_audio_pos(y)
        h = torch.cat([x, prompt, y], dim=1)
        stage_ids = torch.full((h.size(0),), stage_i - 2, device=h.device, dtype=torch.long)
        stage_emb = self.nar_stage_emb(stage_ids)
        for block in self.nar:
            h = block(h, stage_emb)
        return self.nar_predicts[stage_i - 2](h[:, -y.size(1):])

    @torch.no_grad()
    def generate_first_codebook(self, phoneme_prompt, acoustic_prefix, max_new_tokens, temperature=1.0):
        codes, generated = acoustic_prefix, []
        for _ in range(max_new_tokens):
            logits = self.forward_ar(phoneme_prompt, codes)[:, -1] / temperature
            next_id = torch.multinomial(logits.softmax(-1), 1)
            if (next_id == EOS).all():
                break
            generated.append(next_id.clamp_max(NUM_AUDIO_TOKENS - 1))
            codes = torch.cat([codes, next_id], dim=1)
        return torch.cat(generated, dim=1) if generated else acoustic_prefix.new_empty(acoustic_prefix.size(0), 0)

    @torch.no_grad()
    def synthesize(self, phoneme_prompt, prompt_codes, codec, max_new_tokens):
        codes1 = self.generate_first_codebook(phoneme_prompt, prompt_codes[..., 0], max_new_tokens)
        cur = codes1.unsqueeze(-1)
        out = [codes1]
        for i in range(2, NUM_QUANTIZERS + 1):
            ci = self.forward_nar(phoneme_prompt, prompt_codes, cur, i).argmax(-1)
            cur = torch.cat([cur, ci.unsqueeze(-1)], dim=-1)
            out.append(ci)
        return codec.decode(torch.stack(out, dim=-1))                          # [T, 8] -> waveform
```
