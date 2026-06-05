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

- **AR model** (decoder-only causal Transformer) generates codebook 1 from phonemes + the codebook-1 acoustic prefix. Trained as a plain causal LM, so *any acoustic prefix is automatically a prompt* — this is the in-context voice cloning. Sampling-based decoding (beam search loops); gives output diversity.
- **NAR model** (non-causal Transformer) generates codebooks 2–8, each in one parallel pass — O(1) instead of O(T) per level. Conditioned on lower codebooks by **summing their embeddings** (mirrors RVQ's additive codes) and on the speaker by the full 8-codebook prompt `C̃`. The current stage `i` is injected via **adaptive layer norm** `AdaLN(h, i) = a_i · LayerNorm(h) + b_i`. One shared net handles all 7 levels (random stage sampled per training step).

**Weight tying.** Output projection ties to the acoustic embedding; in the NAR model, prediction layer `j` shares weights with the `(j+1)`-th acoustic embedding.

**Decode** the full `750 × 8` matrix with EnCodec's decoder to the waveform.

## Setup

Train on LibriLight (60K hours, ~7K speakers; ASR-generated transcripts, 30 ms phoneme alignment). AR and NAR both: 12 layers, 16 heads, `d_model` 1024, FFN 4096, dropout 0.1. AdamW, warmup 32K steps to peak lr 5×10⁻⁴ then linear decay, 800K steps. NAR acoustic prompt = a random 3 s segment of the same utterance; AR prompt is the natural causal prefix.

## Working code

```python
import torch, torch.nn as nn

NUM_AUDIO_TOKENS = 1024
NUM_QUANTIZERS = 8
EOS = NUM_AUDIO_TOKENS

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(d, 2 * d)
    def forward(self, h, stage_emb):
        a, b = self.proj(stage_emb).chunk(2, dim=-1)
        return a * self.ln(h) + b

class VALLE(nn.Module):
    def __init__(self, d=1024, nhead=16, nlayers=12, d_ff=4096, n_phonemes=512):
        super().__init__()
        self.phoneme_emb = nn.Embedding(n_phonemes, d)
        # AR (codebook 1)
        self.ar_audio_emb = nn.Embedding(NUM_AUDIO_TOKENS + 1, d)
        self.ar_pos = SinePositionalEmbedding(d)
        self.ar = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nhead, d_ff, batch_first=True), nlayers)
        self.ar_predict = nn.Linear(d, NUM_AUDIO_TOKENS + 1)
        self.ar_predict.weight = self.ar_audio_emb.weight
        # NAR (codebooks 2..8)
        self.nar_audio_embs = nn.ModuleList([nn.Embedding(NUM_AUDIO_TOKENS, d) for _ in range(NUM_QUANTIZERS)])
        self.nar_stage_embs = nn.ModuleList([nn.Embedding(1, d) for _ in range(NUM_QUANTIZERS - 1)])
        self.nar = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nhead, d_ff, batch_first=True), nlayers)
        self.nar_ln = AdaptiveLayerNorm(d)
        self.nar_predicts = nn.ModuleList([nn.Linear(d, NUM_AUDIO_TOKENS) for _ in range(NUM_QUANTIZERS - 1)])
        for j in range(NUM_QUANTIZERS - 1):
            self.nar_predicts[j].weight = self.nar_audio_embs[j + 1].weight

    def forward_ar(self, phonemes, codes1):
        h = torch.cat([self.phoneme_emb(phonemes), self.ar_pos(self.ar_audio_emb(codes1))], dim=1)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, device=h.device), diagonal=1).bool()
        return self.ar_predict(self.ar(h, mask=mask))

    def forward_nar(self, phonemes, prompt_codes, codes_lt_i, stage_i):
        x = self.phoneme_emb(phonemes)
        prompt = sum(self.nar_audio_embs[j](prompt_codes[..., j]) for j in range(NUM_QUANTIZERS))
        y = sum(self.nar_audio_embs[j](codes_lt_i[..., j]) for j in range(stage_i - 1))
        h = self.nar(torch.cat([x, prompt, y], dim=1))
        h = self.nar_ln(h, self.nar_stage_embs[stage_i - 2].weight)
        return self.nar_predicts[stage_i - 2](h)

    @torch.no_grad()
    def synthesize(self, phonemes, prompt_codes, codec):
        codes1 = sample_ar(self.forward_ar, phonemes, prompt_codes[..., 0])   # AR, until EOS
        cur = codes1.unsqueeze(-1)
        out = [codes1]
        for i in range(2, NUM_QUANTIZERS + 1):
            ci = self.forward_nar(phonemes, prompt_codes, cur, i).argmax(-1)  # NAR, greedy
            cur = torch.cat([cur, ci.unsqueeze(-1)], dim=-1)
            out.append(ci)
        return codec.decode(torch.stack(out, dim=-1))                          # [T, 8] -> waveform
```
