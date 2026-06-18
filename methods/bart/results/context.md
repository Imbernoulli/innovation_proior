# Context: one pretraining objective for both understanding and generation

## Research question

Self-supervised pretraining has transformed NLP, but the most successful recipes
split along a fault line. The masked-language-model family is excellent at
*understanding* tasks (classification, span extraction) because it builds
representations conditioned on context from both sides — but it predicts masked
tokens independently and non-autoregressively, so it does not naturally *generate*
text. The left-to-right language-model family generates fluently but conditions
each token only on its left context, so it is weaker at understanding tasks that
need the whole sentence. Each recent variant — better mask distributions, new
prediction orders, extra context windows — tends to improve one family of end
tasks while remaining awkward for the other. The precise question is whether a
single self-supervised pretraining scheme can be strong across the *full* range of
end tasks at once: classification, span extraction, summarization, dialogue,
abstractive question answering, and even machine translation — without giving up
the understanding quality of masked models or the generation quality of
autoregressive ones.

## Background

**Self-supervised denoising.** The dominant pretraining objective is a denoising
autoencoder in disguise: corrupt text by masking a random subset of words, then
reconstruct the originals. Recent gains came from refining the *corruption* — which
tokens to mask and how they cluster (SpanBERT, Joshi et al. 2019, masks contiguous
spans), the *order* in which masked tokens are predicted (XLNet, Yang et al. 2019,
predicts in a permuted order so each prediction sees both sides), and the
*context* available when predicting (UniLM, Dong et al. 2019, mixes attention masks
so some predictions are left-only). All share a structural constraint: the
corruption is applied *in place*, so the model's input and output are token-aligned
and the same length. That alignment limits the kinds of corruption that are even
expressible.

**The encoder/decoder split.** A bidirectional Transformer encoder (BERT, Devlin
et al. 2018) sees both sides at every layer — ideal for understanding — but its
masked-token predictions are conditionally independent given the input, so it
cannot roll out text. A left-to-right Transformer decoder (GPT, Radford et al.
2018; Radford et al. 2019) predicts autoregressively and generates naturally, but
each position is blind to its right context. ELMo (Peters et al. 2018) glued a
left-only and a right-only language model together but never trained interactions
between the two directions.

**Seq2seq Transformers.** The original encoder-decoder Transformer (Vaswani et al.
2017) was built for machine translation: a bidirectional encoder reads the source,
and an autoregressive decoder writes the target, attending back to the encoder via
cross-attention at every decoder layer. The source and target need not be the same
string or even the same length.

**Diagnostic findings already on the table.** Several observations frame the
design. Left-only decoders do poorly on span-extraction tasks because future
context is needed for the classification decision — bidirectionality matters for
understanding. Conversely, objectives lacking any left-to-right autoregressive
component are weaker at generation. Fixed-width masked spans still expose the
number of hidden tokens, so they do not test whether a model can infer missing
length from context. And it is established that data scale, batch size, and
optimization details matter as much as the objective itself, so comparisons must
control for them (Liu et al. 2019). For machine translation, pretraining the
*encoder* with learned representations helps, but gains from putting a pretrained
language model in the *decoder* have been limited (Edunov et al. 2019), and the
biggest MT gains have required pretraining on both source and target languages
(MASS, Song et al. 2019; XLM, Lample & Conneau 2019), which needs monolingual data
for every language of interest.

## Baselines

**BERT (masked language model).** Bidirectional encoder, 15% of tokens replaced
with `[MASK]`, predicted independently. Strong on understanding; cannot generate
autoregressively. Gap: no decoder, so generation tasks require bolting on extra
machinery.

**GPT (left-to-right language model).** Autoregressive decoder, generates fluently.
Gap: left-only context hurts understanding tasks like span extraction.

**UniLM (multitask masked LM).** A single masked model trained with a mixture of
attention masks — some bidirectional, some left-only, some prefix — so it can serve
both understanding and generation. Gap: its predictions are still conditionally
independent, and pretraining still mismatches generation because the model is not
trained to decode an uncorrupted target autoregressively.

**MASS (masked seq2seq).** An encoder-decoder where a contiguous span (≈50% of
tokens) is masked in the source and the decoder predicts exactly those masked
tokens. Closest in shape to a denoising seq2seq. Gap: encoder and decoder see
*disjoint* token sets (encoder gets the unmasked tokens, decoder produces the
masked ones), which weakens it for discriminative tasks where the decoder should
also see the whole input.

**XLNet (permuted LM).** Predicts tokens autoregressively in a permuted order so
each prediction can condition on both sides. Gap: the decoding order at
pretraining time is permuted, not the simple left-to-right order used at generation
time, so it does not cleanly match the generation setting.

## Evaluation settings

Discriminative: SQuAD v1.1/v2.0 span extraction (exact-match/F1) and the GLUE suite
(MNLI, SST-2, QQP, QNLI, STS-B, RTE, MRPC, CoLA). Generative: CNN/DailyMail and
XSum summarization (ROUGE-1/2/L; CNN/DM extractive-leaning, XSum highly
abstractive), ConvAI2 persona-conditioned dialogue (F1, perplexity), and ELI5
long-form abstractive question answering (ROUGE; answers only weakly constrained by
the question). Machine translation: WMT'16 Romanian-English augmented with
back-translation (BLEU). Pretraining corpora range from BookCorpus + Wikipedia
(~16GB, 1M steps for ablations) up to the 160GB news/books/stories/web mixture used
at large scale; tokenization via byte-level BPE. Generation uses beam search (beam
5) with length tuning; ablation comparisons report perplexity to compare
objectives directly.

## Code framework

The substrate is a standard seq2seq Transformer (encoder + decoder with
cross-attention), an Adam optimizer with warmup/decay, and a cross-entropy /
label-smoothed loss. What is *not* fixed is how to corrupt the input document
before the encoder sees it, and how to read out a representation for discriminative
finetuning from a model that has a decoder. The scaffold leaves those slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Seq2SeqTransformer(nn.Module):
    """Bidirectional encoder + autoregressive decoder with cross-attention (exists)."""
    def __init__(self, vocab, d_model, enc_layers, dec_layers, n_heads, ffn): ...
    def encode(self, src_tokens): ...
    def decode(self, prev_output_tokens, enc_out):  # causal self-attn + cross-attn
        ...
    def forward(self, src_tokens, prev_output_tokens):
        return self.decode(prev_output_tokens, self.encode(src_tokens))  # -> logits

# --- corruption: TO DECIDE ---
def corrupt(document):
    # TODO: what transformation(s) to apply to the document before the encoder?
    pass

def pretraining_loss(model, document):
    # TODO: define the self-supervised objective relating corrupt(document) to document
    pass

# --- finetuning readout: TO DECIDE ---
class ClassificationHead(nn.Module):
    def __init__(self, d_model, inner, n_classes, dropout): ...
    def forward(self, features): pass   # which decoder position represents the sequence?

# --- training loop / optimizer (exist) ---
def train(model, data, *, batch_size, total_steps, warmup, peak_lr):
    pass
```
