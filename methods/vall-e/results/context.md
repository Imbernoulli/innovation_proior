# Context

The goal is **zero-shot text-to-speech**: given the text to speak and a *short* recording of a target speaker never seen in training, synthesize that text in that speaker's voice — preserving timbre, prosody, even the acoustic environment — without any per-speaker fine-tuning, speaker-embedding extractor, or hand-designed adaptation. This is the landscape as it stands at the end of 2022.

## Research question

Cascaded TTS systems map text → mel-spectrogram (acoustic model) → waveform (vocoder). They sound excellent when trained on clean, studio-quality, single- or few-speaker corpora. But they generalize poorly: for a speaker unseen in training, naturalness and speaker similarity collapse. The standard fixes are heavy — speaker *adaptation* (fine-tune the model on the new speaker's data) or speaker *encoding* (a separately trained speaker-verification-style encoder produces an embedding that conditions the synthesizer). Both demand extra machinery and still leave a generalization gap between seen and unseen speakers.

The precise problem: build a TTS system that, from a ~3-second enrolled clip, produces high-quality personalized speech for *arbitrary unseen speakers*, with no fine-tuning, no pre-designed acoustic features, and no specialized speaker encoder — and that improves as you scale training data, the way text language models do.

What a solution must achieve:
- **Strong zero-shot generalization** to unseen speakers from seconds of reference audio.
- **An intermediate representation that scales to messy, large data.** A mel-spectrogram trained under an L1/L2 regression loss overfits to clean data and wastes capacity on short-range continuous detail; it cannot absorb tens of thousands of hours of noisy, loosely-transcribed audio.
- **A formulation that admits in-context learning / prompting**, so a single pretrained model handles new speakers by conditioning, not retraining.
- **Speaker, prosody, and acoustic-environment fidelity** carried by the reference, plus output diversity.

## Background

Two currents converge here: the scaling lesson from text language models, and the maturation of discrete audio representations.

**The scaling lesson.** Text language models improved monotonically as their training corpora grew — from ~16 GB to ~1 TB of text — and large autoregressive LMs (GPT-3 class) acquired **in-context learning**: a frozen model performs a new task from a few exemplars in its prompt, no weight updates. Speech understanding showed the same data-scaling trend (hundreds → tens of thousands → over a million hours). Meanwhile, advanced multi-speaker TTS was still trained on only hundreds of hours — plausibly the root of its poor generalization. The hypothesis: replicate the text-LM recipe in speech — train a language model on *vastly* more, *diverse*, semi-supervised speech data — and zero-shot generalization may follow.

**Why mel-spectrograms block this.** Regression on continuous mel frames (L1/L2) is sensitive to data quality and spends modeling capacity on short-range continuous structure; it does not benefit from — and is destabilized by — large noisy corpora. To borrow the language-model recipe, speech needs to become a sequence of **discrete tokens**, like text.

**Speech quantization, and why neural codec codes win.** Audio is stored as 16-bit samples (65,536 values) at >10 kHz — far too long and high-cardinality to model directly. μ-law (WaveNet) quantizes amplitude to 256 values but does not shorten the sequence, so AR synthesis stays slow. Self-supervised discrete units (vq-wav2vec, HuBERT k-means codes) reconstruct *content* and shorten sequences, but discard speaker identity and reconstruct at low quality. **Neural audio codec codes** from EnCodec's residual vector quantizer are the right representation: they (1) retain abundant speaker and acoustic information, so reconstruction preserves identity even for unseen speakers; (2) come with an off-the-shelf decoder to a high-quality waveform, no separate vocoder training; and (3) shorten the sequence (320× downsampling → 75 Hz).

**The structure of RVQ codes — load-bearing.** EnCodec encodes each frame as `N_q=8` codes (for the 6 kbps, 24 kHz setting) from a residual vector quantizer: the first codebook captures the dominant acoustic content (including speaker identity), and each subsequent codebook models the residual the previous ones missed — so importance decreases down the hierarchy. A 10-second clip becomes a `750 × 8` matrix of codes (`750 = 24000·10/320`). The codebooks are *ordered*: the first carries the dominant content, and codebook `j` models the residual left by codebooks `< j`.

**AudioLM — the immediate ancestor.** AudioLM trains language models over both self-supervised semantic tokens and neural-codec acoustic tokens to do high-quality **speech-to-speech** continuation, demonstrating that LM-style modeling of codec tokens yields natural audio and needs no separately trained vocoder. *Limitation:* it is speech-to-speech (continuation/conditioned on audio), with no explicit text control — you cannot tell it *what* to say.

## Baselines

**YourTTS (the SOTA zero-shot baseline).** A flow-based multi-speaker TTS system extending VITS, conditioned on a speaker embedding from a pretrained speaker encoder, trained on a combination of VCTK/LibriTTS-scale data. It is the strongest prior zero-shot TTS. *Gap:* it relies on a speaker-encoder embedding and a few-hundred-hour training set; speaker similarity for genuinely unseen speakers is limited, and it does not scale its representation to large noisy corpora.

**Speaker-adaptation TTS.** Fine-tune a multi-speaker model on the target speaker's recordings. *Gap:* requires per-speaker fine-tuning data and a separate model per speaker to host — operationally heavy.

**Speaker-encoding TTS.** A pretrained speaker encoder (often a speaker-verification network) produces an embedding that conditions one shared TTS model. *Gap:* a generalization gap between seen and unseen speakers; quality hinges on the encoder, which may need to be increasingly complex.

**Cascaded acoustic-model-plus-vocoder TTS (e.g. FastSpeech-class, Tacotron-class).** Predict mel-spectrograms then vocode. *Gap:* mel regression overfits to clean data, doesn't scale to noisy corpora, and offers no in-context-learning route to new speakers.

**AudioLM (speech LM over codec tokens).** As above — the methodological template (LM over discrete codec codes), but speech-to-speech, not text-controllable.

## Evaluation settings

- **Training data:** LibriLight — 60K hours of unlabeled English audiobook speech, ~7,000 speakers. Audio-only, so transcriptions are produced by an ASR/alignment model; phoneme alignments at a 30 ms frameshift. (Hundreds of times larger than prior TTS corpora, which used ≤600 hours.)
- **Tokenizer:** EnCodec at 6 kbps / 24 kHz — 8 RVQ codebooks of 1024 entries each, 75 Hz frame rate.
- **Test sets (all speakers unseen in training):** LibriSpeech test-clean and VCTK. Two prompting regimes: synthesize given text from a separate 3-second enrolled clip, and continuation from the first 3 seconds of an utterance.
- **Objective metrics:** speaker similarity via a SOTA speaker-verification model (WavLM-TDNN, cosine in [-1,1]); robustness via word error rate from an ASR model (HuBERT-Large, CTC, no LM fusion) on the synthesized audio.
- **Subjective metrics:** comparative MOS (CMOS, naturalness, range −3..+3) and similarity MOS (SMOS, 1..5) by crowdsourced native listeners.

## Code framework

The scaffold is a discrete-token TTS harness: text→phonemes, audio→codec tokens via a pretrained codec, a generic conditional sequence model over those tokens, and a codec decoder back to waveform. The codec, phonemizer, embeddings, Transformer layers, and cross-entropy loss already exist; the *modeling structure over the multi-codebook token matrix* is the empty slot.

```python
import torch, torch.nn as nn

# --- available primitives ---
phonemize = ...                       # text -> phoneme id sequence
codec = PretrainedNeuralCodec(...)    # waveform -> code matrix [T, Q]; codes -> waveform
N_CODEBOOKS = 8
VOCAB = 1024                          # entries per codebook

class TTSModel(nn.Module):
    """Conditioned on phonemes x and an acoustic prompt (codes of an enrolled clip),
       produce the full code matrix C [T, Q] for the text."""
    def __init__(self, d_model=1024, nhead=16, nlayers=12, d_ff=4096):
        super().__init__()
        self.phoneme_emb = nn.Embedding(N_PHONEMES, d_model)
        # TODO: how to model the [T, Q] code matrix
        pass

    def forward(self, phonemes, prompt_codes, target_codes=None):
        # TODO: condition on (phonemes, prompt_codes); model the code matrix; return loss / codes
        raise NotImplementedError

    @torch.no_grad()
    def synthesize(self, phonemes, prompt_codes):
        # TODO: produce code matrix C, then waveform = codec.decode(C)
        raise NotImplementedError
```
