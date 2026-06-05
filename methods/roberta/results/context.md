# Research question

A wave of self-training methods — ELMo (Peters et al. 2018), GPT (Radford et al.
2018), BERT (Devlin et al. 2018), cross-lingual LM pretraining (Lample & Conneau
2019), XLNet (Yang et al. 2019) — has produced large, rapid gains on language
understanding tasks. Each new method reports a new objective or architecture and
a new state of the art. But the gains are hard to attribute. Pretraining is
expensive, so each method is tuned only lightly; the corpora differ in size and
are often private; and many design choices are changed at once. The precise
question is whether the reported improvements come from the headline modeling
idea (a new pretraining objective, a new architecture) or from mundane,
under-reported choices: how long the model trains, how big the batches are, how
much and which data it sees, how the masking is generated, how the input is
formatted. A solution would isolate these factors under a single, controlled,
well-tuned implementation and make the original masked-language-model recipe a
properly tuned comparator, so the field can stop comparing under-trained
baselines against well-trained successors.

# Background

**Masked language modeling as a pretraining objective.** The dominant
understanding-pretraining recipe takes a Transformer (Vaswani et al. 2017),
feeds it a concatenation of two text segments delimited by special tokens
(`[CLS] x_1 … x_N [SEP] y_1 … y_M [EOS]`, with `M+N` under a maximum length `T`),
and trains it with two losses. The masked language model (MLM) loss selects 15%
of input tokens; of those, 80% are replaced with a `[MASK]` token, 10% are left
unchanged, and 10% are replaced with a random vocabulary token; the model
predicts the original token at each selected position under a cross-entropy
loss. Because attention is bidirectional, every token's representation is
conditioned on both sides — this is the point of masking rather than
left-to-right prediction. The second loss, next-sentence prediction (NSP), is a
binary classifier on whether the two segments were consecutive in the source
text (positive: consecutive; negative: segments from different documents; sampled
50/50); it was introduced to help sentence-pair tasks such as natural language
inference that require reasoning about a relationship between two spans.

**How masking was implemented.** The original implementation masks *once*, during
data preprocessing — a single static mask per sequence. To avoid the model seeing
the identical mask every epoch, the training data is duplicated 10 times with 10
different masks, and training runs 40 epochs, so each particular mask is seen 4
times. This couples the masking pattern to the number of epochs.

**The optimization recipe.** The standard configuration uses Adam (Kingma & Ba
2014) with β₁=0.9, β₂=0.999, ε=1e-6, and L2 weight decay 0.01; the learning rate
is warmed up over the first 10,000 steps to a peak of 1e-4 and then linearly
decayed; dropout 0.1 everywhere including attention; GELU activations (Hendrycks
& Gimpel 2016). The reference run is 1,000,000 updates with minibatches of 256
sequences of maximum length 512.

**Diagnostic observations already on the table.** Several pre-existing findings
frame the problem. Training neural machine translation with very large
minibatches improves both optimization speed and final quality when the learning
rate is scaled up appropriately (Ott et al. 2018), and masked-LM training has
been shown amenable to large-batch training (You et al. 2019, scaling to tens of
thousands of sequences). Increasing pretraining data improves end-task
performance (Baevski et al. 2019). And several groups have already questioned
whether the NSP loss is necessary at all (Lample & Conneau 2019; Yang et al.
2019; Joshi et al. 2019), in tension with the original report that removing NSP
hurts. Separately, subword tokenization via byte-pair encoding (Sennrich et al.
2016) — building a vocabulary of subword units from corpus statistics, typically
10K–100K units — had a recent variant (Radford et al. 2019) that operates on
*bytes* rather than unicode characters, letting a 50K-unit vocabulary encode any
text with no unknown tokens and no language-specific preprocessing.

**Data sources.** The original 16GB corpus is BookCorpus (Zhu et al. 2015) plus
English Wikipedia. Larger and more diverse web corpora exist or can be built:
the CommonCrawl News collection (Nagel 2016), an open recreation of the WebText
corpus from Reddit-shared URLs with at least three upvotes (Gokaslan & Cohen
2019), and a CommonCrawl subset filtered to a story-like style (Trinh & Le 2018).

# Baselines

**BERT (Devlin et al. 2018).** A deep bidirectional Transformer pretrained with
MLM + NSP on 16GB of text, then finetuned per task with a small head. Base is
L=12, H=768, A=12, 110M parameters; large is L=24, H=1024, A=16, 355M. It set the
prevailing state of the art on GLUE, SQuAD, and related tasks. Its reported recipe
— 1M steps, batch 256, static masking, segment-pair input, both losses, 30K
character-level BPE vocabulary — is the object under examination. The open gap:
the recipe was never systematically tuned, so it is unknown how much of the
headroom above BERT belongs to BERT itself if trained better.

**XLNet (Yang et al. 2019).** A permutation-based autoregressive pretraining
objective intended to fix MLM's pretrain/finetune mismatch (the `[MASK]` token
never appears at finetuning) while keeping bidirectional context. It reported
beating BERT on GLUE, SQuAD, and RACE. But it changes many things at once: a new
objective, roughly 10× more pretraining data than BERT, batches 8× larger for
half the steps (so 4× as many sequences seen). Its headline win over BERT thus
conflates objective with budget. Gap: the contribution of the objective alone is
not isolated.

**Other post-BERT directions.** Multi-task finetuning (Liu et al. 2019),
entity-aware pretraining (Sun et al. 2019), span-based masking (Joshi et al.
2019, SpanBERT), and other autoregressive variants (Song et al. 2019; Chan et al.
2019) each report improvements. Across these, bigger models on more data is a
recurring driver of gains, raising the same attribution problem.

# Evaluation settings

Three benchmarks are the natural yardstick. GLUE (Wang et al. 2019) is nine
language-understanding datasets framed as single-sentence or sentence-pair
classification (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI), with a
held-out test server and leaderboard; development-set results come from
single-task finetuning on each task's own training data. SQuAD v1.1 (Rajpurkar et
al. 2016) asks the model to extract an answer span from a context paragraph;
v2.0 (Rajpurkar et al. 2018) adds unanswerable questions, handled by an extra
binary answerability classifier whose loss is summed with the span loss. RACE
(Lai et al. 2017) is large-scale multiple-choice reading comprehension (~28K
passages, ~100K questions) from Chinese English exams, with notably long
contexts; each question has four candidate answers and the system picks one.
Finetuning uses Adam with linear warmup and decay; metrics are accuracy
(GLUE/RACE) and exact-match/F1 (SQuAD).

# Code framework

The substrate is a sequence-modeling toolkit: a data pipeline, a bidirectional
Transformer encoder, Adam, mixed-precision training, and distributed
data-parallel with gradient accumulation. The encoder and training machinery are
available; the remaining choices are text encoding, sequence construction, token
masking, the masked-token projection head, and the optimizer settings.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """Stack of bidirectional self-attention blocks."""
    def __init__(self, vocab_size, n_layers, hidden, n_heads, ffn, dropout): ...
    def forward(self, tokens):  # -> per-token hidden states [B, T, H]
        ...

def adam(params, *, lr, betas, eps, weight_decay):
    ...

def linear_warmup_decay(step, warmup, total, peak_lr):
    ...

class Tokenizer:
    def __init__(self, encoder_json, vocab_bpe):
        # TODO: choose the subword scheme and vocabulary.
        pass
    def encode(self, text): pass

def build_pretraining_inputs(documents, max_len=512, sep_id=2):
    # TODO: decide how documents are segmented and packed.
    pass

def mask_tokens(tokens, mask_id, vocab_size, *, ignore_index=-100, p=0.15,
                special_ids=()):
    # TODO: decide when masks are sampled and how selected tokens are corrupted.
    pass

class MLMHead(nn.Module):
    """Project hidden states back to the vocabulary for selected positions."""
    def __init__(self, hidden, vocab_size, embed_weight):
        pass
    def forward(self, features, masked_tokens=None):
        pass

def mlm_loss(logits, labels, masked_tokens=None):
    # TODO: masked-token cross-entropy.
    pass

class ClassificationHead(nn.Module):
    """Finetuning head over the first-token representation."""
    def __init__(self, hidden, inner, n_classes, dropout): ...
    def forward(self, features): ...

def make_optimizer(params, peak_lr):
    # TODO: set Adam's betas, epsilon, weight decay, and clipping policy.
    pass
```
