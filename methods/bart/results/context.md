# Context: one pretraining objective for both understanding and generation

## Research question

Self-supervised pretraining has transformed NLP, and the most successful recipes
fall into two families. The masked-language-model family builds representations
conditioned on context from both sides and is used for *understanding* tasks
(classification, span extraction); it predicts masked tokens independently and
non-autoregressively. The left-to-right language-model family predicts each token
from its left context and is used to *generate* text. Recent variants adjust the
mask distribution, the prediction order, or the available context. The question is
whether a single self-supervised pretraining scheme can serve the *full* range of
end tasks — classification, span extraction, summarization, dialogue, abstractive
question answering, and machine translation.

## Background

**Self-supervised denoising.** The dominant pretraining objective is a denoising
autoencoder: corrupt text by masking a random subset of words, then reconstruct the
originals. Recent variants refine the *corruption* — which tokens to mask and how
they cluster (SpanBERT, Joshi et al. 2019, masks contiguous spans), the *order* in
which masked tokens are predicted (XLNet, Yang et al. 2019, predicts in a permuted
order so each prediction sees both sides), and the *context* available when
predicting (UniLM, Dong et al. 2019, mixes attention masks so some predictions are
left-only). In these schemes the corruption is applied *in place*: the model's
input and output are token-aligned and the same length.

**The encoder/decoder split.** A bidirectional Transformer encoder (BERT, Devlin
et al. 2018) sees both sides at every layer, and its masked-token predictions are
conditionally independent given the input. A left-to-right Transformer decoder
(GPT, Radford et al. 2018; Radford et al. 2019) predicts autoregressively. ELMo
(Peters et al. 2018) concatenates a left-only and a right-only language model
trained separately.

**Seq2seq Transformers.** The original encoder-decoder Transformer (Vaswani et al.
2017) was built for machine translation: a bidirectional encoder reads the source,
and an autoregressive decoder writes the target, attending back to the encoder via
cross-attention at every decoder layer. The source and target need not be the same
string or the same length.

**Diagnostic findings already on the table.** Several observations frame the
design. Left-only decoders condition each position only on its left context;
bidirectionality is used for the classification decision in span-extraction tasks.
Objectives with a left-to-right autoregressive component are used for generation.
Fixed-width masked spans expose the number of hidden tokens. Data scale, batch
size, and optimization details affect results as much as the objective, so
comparisons control for them (Liu et al. 2019). For machine translation, pretraining
the *encoder* with learned representations helps; gains from putting a pretrained
language model in the *decoder* have been limited (Edunov et al. 2019), and the
largest MT gains have come from pretraining on both source and target languages
(MASS, Song et al. 2019; XLM, Lample & Conneau 2019), which uses monolingual data
for every language of interest.

## Baselines

**BERT (masked language model).** Bidirectional encoder, 15% of tokens replaced
with `[MASK]`, predicted independently given the input.

**GPT (left-to-right language model).** Autoregressive decoder; each position
conditions on its left context.

**UniLM (multitask masked LM).** A single masked model trained with a mixture of
attention masks — some bidirectional, some left-only, some prefix — so it serves
both understanding and generation. Predictions are conditionally independent given
the input.

**MASS (masked seq2seq).** An encoder-decoder where a contiguous span (≈50% of
tokens) is masked in the source and the decoder predicts exactly those masked
tokens. The encoder gets the unmasked tokens and the decoder produces the masked
ones, so the two see disjoint token sets.

**XLNet (permuted LM).** Predicts tokens autoregressively in a permuted order so
each prediction can condition on both sides; the pretraining decoding order is the
permutation rather than left-to-right.

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
label-smoothed loss. The open slots are how to corrupt the input document before
the encoder sees it, and how to read out a representation for discriminative
finetuning from a model that has a decoder.

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
