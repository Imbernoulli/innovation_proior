The problem I keep failing on is zero-shot text-to-speech: hand a system the text to speak and a few seconds of a speaker it has never heard, and get that text back in that voice — timbre, prosody, even the room — naturally, with no per-speaker fine-tuning. The cascaded systems I have, a mel-spectrogram acoustic model followed by a vocoder, are gorgeous on clean studio data but collapse on an unseen speaker; both naturalness and speaker similarity fall off a cliff. The field's patches are heavy and only paper over the gap rather than closing it: speaker *adaptation* fine-tunes a separate model per speaker, and speaker *encoding* trains a dedicated speaker-verification-style encoder whose embedding conditions one shared synthesizer — and even then a stubborn gap remains between seen and unseen speakers. The deeper observation is where generalization-to-anything-unseen *was* recently solved: text. Large language models trained on enormous, diverse corpora acquired in-context learning almost for free — you prompt GPT-3 with a few exemplars instead of retraining it — and the thing that bought that capability was sheer data scale, gigabytes climbing to a terabyte. Speech understanding shows the same data-scaling curve, hundreds of hours rising past a million. Yet the strong multi-speaker TTS systems are trained on mere hundreds of hours, a thousandth of what gave text models their generalization. That gap feels like it might *be* the whole problem, so the bet I want to make is to stop building speaker-specific machinery and instead train a language model on a huge, diverse pile of speech and let zero-shot generalization emerge the way it did for text.

The obstacle is that a language model needs discrete tokens, and speech is a continuous waveform. Mel-spectrograms — continuous frames fit with an L1/L2 regression loss — fail me twice: regression on continuous frames is sensitive to data quality and spends capacity on short-range continuous detail, so it overfits clean data and chokes on exactly the large noisy corpus I need; and regression over continuous targets is not a language model at all, with no discrete next-token distribution to prompt over, no way to sample diverse outputs. So the first move is to make speech discrete like words. μ-law quantizes amplitude to 256 levels but never shortens the sequence, so autoregression over it is hopelessly long; HuBERT/vq-wav2vec self-supervised units do shorten the sequence and reconstruct content, but they throw away speaker identity — fatal for voice cloning — and reconstruct at low quality. What fits is a neural audio codec: EnCodec compresses 24 kHz audio to 75 frames per second (a 320× reduction), each frame a small set of discrete codes, and because it is trained to *reconstruct the waveform* it preserves speaker timbre and acoustic environment rather than stripping them; it also ships its own decoder, so I get a high-quality vocoder for free and never train one. AudioLM already showed a language model over neural-codec tokens yields natural audio, but it is speech-to-speech with no text handle — I cannot tell it what to say. I want that trick made text-controllable.

I propose VALL-E, which treats text-to-speech as conditional codec language modeling. The structure of an EnCodec frame dictates the whole design. EnCodec uses residual vector quantization: the frame's latent is quantized by a first codebook, its residual by a second, and so on, $N_q = 8$ codebooks deep at 6 kbps. One frame is therefore not a single token but eight non-exchangeable tokens $c_{t,1}\dots c_{t,8}$, where the first codebook carries the dominant content including speaker identity and each later codebook only refines the residual the earlier ones left, so importance falls off down the hierarchy. A ten-second clip becomes a $750 \times 8$ matrix $C$. The model must predict a 2D matrix, and the question is how to factor $p(C \mid x, \tilde{C})$ where $x$ is the phoneme sequence and $\tilde{C}$ is the acoustic-prompt code matrix of the enrolled clip. Flattening the matrix into one length-6000 stream and running a single autoregressive model multiplies an already-750-long sequence by eight and reinstates exactly the per-step latency I am trying to escape; predicting all $8 \times 750$ codes fully in parallel is fast but breaks on the time axis, where output length is genuinely unknown (speaking rate varies wildly and a duration predictor across thousands of diverse speakers is unreliable) and an autoregressive model is what naturally decides its own length via an end-of-sequence token. So neither extreme works. What does work is splitting the matrix by axis, because the two axes have completely different character. The time axis has real sequential dependency and unknown length — that wants autoregression. The codebook axis has the residual hierarchy, where codebook $j$ depends on codebooks $< j$ but, given all the lower codebooks, predicting codebook $j$ across *all 750 timesteps at once* is fine because there is no left-to-right dependency within a single level. So I use an autoregressive (AR) model for the first codebook — the identity- and content-bearing one whose length I need decided — and a non-autoregressive (NAR) model for the remaining seven, each generated in one parallel shot conditioned on the codebooks below it. The factorization is

$$p(C \mid x, \tilde{C}) = p(c_{:,1} \mid x, \tilde{C}_{:,1};\, \theta_{\mathrm{AR}}) \cdot \prod_{j=2}^{8} p(c_{:,j} \mid c_{:,<j},\, x,\, \tilde{C};\, \theta_{\mathrm{NAR}}),$$

which gives AR's length-decision and most-important content together with NAR collapsing the time complexity of the other seven codebooks from $O(T)$ to $O(1)$ per level.

The AR model is a decoder-only causal Transformer. I embed the phonemes through a phoneme embedding and the first-codebook tokens through an acoustic embedding, concatenate them (phonemes then audio codes, EOS after each), add sinusoidal positions computed *separately* for the two streams since they are different streams, and run a causal Transformer in which each $c_{t,1}$ attends to the phonemes and all earlier $c_{\le t,1}$. The prediction head maps to the codebook size plus EOS, and I tie its weight to the acoustic embedding since both live in the same code space. The prompting mechanism is where in-context learning becomes concrete and cheap: I introduce no special prompt token and no prompt encoder during training — I just train an ordinary causal language model on $(\text{phonemes}, \text{first-codebook codes})$, and because it is causal, *any prefix of the acoustic sequence automatically acts as a prompt for the rest*. At inference, to clone a voice, I concatenate the enrolled clip's phonemes with the phonemes of what I want to say, feed the enrolled clip's first-codebook codes as the acoustic prefix, and let the model continue — the speaker is carried in-context by the acoustic prefix, exactly like a few-shot text prompt, with no fine-tuning and no speaker embedding. I decode by sampling rather than beam search, because beam search collapses into loops and repetitions here, and sampling also yields output diversity for the same text and speaker.

The NAR model for codebooks 2 through 8 shares the Transformer architecture but with differences forced by being non-autoregressive. Because within a level all tokens are predicted at once and cannot see one another, attention is *non-causal* — every position attends to every other. To predict codebook $i$, I take codes from codebooks $1\dots i-1$, embed each through its own table, and *sum* them across codebooks; summing is the natural choice because RVQ codes are themselves additive — the quantized latent is the sum of the chosen codebook entries — so summing the embeddings mirrors how the codec reconstructs the latent. For the speaker prompt I cannot rely on a causal prefix, so I take the enrolled clip's full eight-codebook matrix $\tilde{C}$, embed and sum across its codebooks, and prepend it as the acoustic prompt; the Transformer input is the concatenation of phoneme embeddings, the summed prompt embeddings, and the summed lower-codebook embeddings of the current utterance, with positions computed separately for prompt and sequence. The model is told which level it is currently predicting through adaptive layer normalization: instead of a fixed LayerNorm, scale and shift the normalized activations by parameters derived from the stage,

$$\mathrm{AdaLN}(h, i) = a_i \cdot \mathrm{LayerNorm}(h) + b_i,$$

where $a_i$ and $b_i$ come from a linear projection of a stage embedding for level $i$. One shared NAR network thus handles all seven levels; in training I sample a random stage $i \in [2, 8]$ each step and predict codebook $i$ from the codebooks below it, and at inference I run it seven times for levels 2 through 8 in order. I again share weights, here so that the $j$-th prediction layer ties to the $(j{+}1)$-th acoustic embedding, since reading a code in and writing one out address the same code space. The division of labor is clean: the AR model decides utterance length and lays down the speaker-and-content-bearing first codebook with the speaker carried in by the acoustic prefix — that is where flexibility lives — while the NAR model fills the seven refinement codebooks in parallel, anchored to the speaker by the explicit $\tilde{C}$ prompt across all codebooks — that is where speed lives. The full $750 \times 8$ matrix then goes to EnCodec's decoder, and out comes the waveform with the prompt's room and timbre intact, because the codes carry them and nothing ever passed through a speaker-stripping bottleneck.

The training is just language modeling, which is the entire point — simple and scalable. I tokenize 60K hours of audiobook speech (LibriLight, ~7K speakers) with EnCodec, get phoneme alignments from an ASR model, and train both Transformers (12 layers, 16 heads, $d_{\mathrm{model}}=1024$, FFN 4096, dropout 0.1) with cross-entropy: AdamW, 32K warmup steps to a peak learning rate $5\times 10^{-4}$ then linear decay, 800K steps total. NAR training uses a random 3-second segment of the same utterance as the acoustic prompt; AR training is plain causal first-codebook language modeling, so the inference prompt is simply a given prefix. No adaptation, no speaker encoder, no vocoder training, no duration model — the only bet is that scale plus discrete codec tokens plus the AR-then-NAR factorization delivers the in-context speaker generalization I could not get any other way, so the voice of a three-second unseen clip comes back speaking new words.

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
