# Context: Transfer learning for NLP, circa 2018-2019

## Research question

Pre-training a large neural network on a data-rich unsupervised task and then fine-tuning
it on a downstream task has become the dominant recipe in NLP, and it works remarkably
well. But the recipe has splintered into a confusing landscape of incompatible choices.
Different efforts vary, often simultaneously, the **pre-training objective** (causal
language modeling vs. masked/denoising vs. permutation vs. span masking), the **model
architecture** (encoder-only, decoder-only, encoder-decoder, single-stack-with-prefix),
the **unlabeled corpus** (Wikipedia, BooksCorpus, WebText, CC-News, Common Crawl
variants), the **fine-tuning procedure** (full fine-tune, adapter modules, gradual
unfreezing, multi-task), and the **task interface** (a classification head, a span-pointer
head, a per-token tagging head, a generative decoder). Because each new result changes
several of these axes at once, it is impossible to attribute a reported improvement to any
single decision, or to understand how the choices interact.

The precise problem: there is no controlled testbed in which one can vary a *single* axis
(objective, architecture, data, fine-tuning strategy, scale) while holding everything else
fixed and measure the effect across many tasks. The obstacle is structural — the **output
interface differs from task to task** (a classification softmax, a span pointer, per-token
logits, a generative decoder), which entangles the architecture and loss with the task and
prevents free substitution of the other components. Any controlled study would also need a
large, clean, public unlabeled corpus and a fixed experimental pipeline against which to
run the sweep. Solving this would let the field measure which ideas actually matter and
combine the winners.

## Background

**The Transformer.** The self-attention Transformer (Vaswani et al., 2017) replaced
recurrence with attention. Its primary primitive is self-attention: each output position
is a weighted average of all positions, with weights computed from query-key dot products,
split across multiple heads. The original form is an **encoder-decoder**: a fully-visible
(bidirectional) encoder over the source, and a causally-masked, autoregressive decoder
that cross-attends to the encoder, trained for sequence-to-sequence problems like
translation. Position is injected via additive sinusoidal or learned **absolute** position
signals, because attention itself is order-invariant. Standard details that became
folklore: scale query-key dot products by 1/√d_k before softmax (to keep logits at unit
scale), feed-forward sublayers four times wider than the model dimension, post-norm
LayerNorm with learned scale and bias, residual connections, dropout.

**Two ways the single stack got specialized.** It became common to drop one half of the
encoder-decoder and use a single Transformer stack with a particular attention mask:
- A **causal** mask throughout turns the stack into a left-to-right language model (the
  decoder-only form), used for next-token prediction and generation.
- A **fully-visible** mask turns it into a bidirectional encoder for classification and
  span tasks.
- A hybrid — fully-visible over a prefix, causal over the continuation — gives a
  "prefix language model" that can both encode context bidirectionally and generate.

**Position: relative beats absolute.** Absolute position embeddings tie representations to
absolute indices and generalize poorly to sequence lengths not seen in training. Shaw et
al. (2018) introduced **relative position representations**: the signal depends on the
offset between the key and the query rather than their absolute positions, learned as a
vector per offset injected into the attention computation. This improved translation and
is more robust to length.

**Normalization simplifications.** LayerNorm re-centers and re-scales activations with a
learned gain and bias. Zhang & Sennrich (2019, RMSNorm) observed that the re-centering
contributes little and proposed normalizing by the root-mean-square only (no mean
subtraction, no bias), which is cheaper and just as stable. Separately, applying
normalization to the *input* of each sublayer with the residual path left un-normalized
("pre-norm") makes deep Transformers easier and more stable to train than the original
post-norm placement.

**Why unsupervised pre-training, and the rise of denoising.** Unlabeled text is
essentially free and enormous (the Common Crawl web archive produces ~20 TB/month), and
neural networks scale: bigger models on more data tend to do better. Early NLP
pre-training used a **causal language modeling** objective (predict the next token). It was
then observed repeatedly (Devlin et al., 2018; Voita et al., 2019; Lample & Conneau, 2019)
that **denoising / masked-language-modeling** objectives — corrupt some input tokens and
train the model to reconstruct them, using bidirectional context — transfer better than
causal LM, and denoising quickly became standard. Span-level masking (Joshi et al., 2019,
SpanBERT) — corrupting contiguous spans rather than independent tokens — was found to
improve over token-level masking.

**Scaling is real.** A consistent empirical finding across vision and language is that
performance improves predictably with more parameters and more data, motivating both large
models and large corpora — provided the corpus is large and diverse enough that it need
not be repeated many times (small corpora repeated many times degrade downstream results).

**Common Crawl as raw material, and its problems (diagnostic).** Common Crawl's web-text
is mostly *not* useful natural language: it is dominated by boilerplate (menus, policy
notices), placeholder text, error messages, source code, duplicated passages, and
offensive content. Prior uses applied limited, non-public, or domain-narrow filtering.
This is the pre-method state of affairs that any data study must first clean up.

## Baselines

These are the prior approaches a unified study would compare against or build its
variants from. For each: the core idea, the mechanics, and the gap it leaves.

**Causal language model pre-training (Dai & Le 2015; Radford et al. 2018, GPT; Howard &
Ruder 2018, ULMFiT; Peters et al. 2018, ELMo).** Pre-train a (typically decoder-only or
recurrent) network to predict the next token, then fine-tune. Mechanics: maximize
Σ log p(x_t | x_{<t}). Gap: only left-to-right context, so token representations cannot use
the right-hand side; empirically transfers worse than denoising; and as a *task interface*
it forces everything into next-token prediction over a concatenation, with no
bidirectional encoding of the context.

**BERT / masked language modeling (Devlin et al., 2018).** An encoder-only Transformer
with a fully-visible mask. Corrupt 15% of tokens — of those, 80% replaced by a special
`[MASK]` token, 10% by a random token, and 10% left unchanged — and train the encoder to
predict the original tokens at the corrupted positions; a special `[CLS]` token's output
feeds a classification head. Mechanics: per-position softmax over the vocabulary at
masked positions, bidirectional context. Gaps: it is **encoder-only**, producing a
fixed-length per-token or per-sequence prediction, so it cannot natively do
**generative** tasks (translation, abstractive summarization); the random/unchanged
corruption heuristics exist only to reduce the encoder's pretrain/finetune `[MASK]`
mismatch; and each downstream task needs its own bespoke head.

**Decoder-only LM as a universal interface (Radford et al., 2019, GPT-2).** A single
causal stack; tasks are framed as text continuation and evaluated **zero-shot** by priming
the model with a prompt (e.g. a document followed by "TL;DR:" to elicit a summary).
Mechanics: autoregressive sampling from p(output | prompt). Gaps: fully causal masking
cripples the representation of the context/prefix (every prefix token sees only its left
context), and it targets zero-shot LM behavior rather than fine-tuned transfer with a
proper encoder.

**Prefix language model / unified single stack (Liu et al., 2018; Dong et al., 2019,
UniLM).** A single Transformer stack with a fully-visible mask over the input prefix and a
causal mask over the generated continuation, so it can both encode context bidirectionally
and generate. Mechanics: like a decoder-only LM but with bidirectional prefix attention.
Gap: it has no explicit, separate encoder-decoder cross-attention; the input and output
share one stack and one set of parameters.

**MASS (Song et al., 2019).** A seq2seq masked objective for encoder-decoder models: mask
a contiguous fragment of the input and have the decoder generate it. Mechanics: corrupt a
span, reconstruct. Gap: still reconstructs against the full uncorrupted notion and is one
specific point in a large design space of denoising variants.

**Prior task-unification frameworks.** decaNLP (McCann et al., 2018) casts every task as
**question answering** but *mandates* simultaneous multi-task training and an explicit
question/answer format. Keskar et al. (2019) cast tasks as **span extraction** by
appending candidate outputs and pointing at the correct input span. Gaps: the QA framing
forces a rigid format and joint training; the span-extraction framing cannot express
**generative** tasks where outputs cannot be enumerated (translation, summarization). Both
point toward, but do not reach, a fully general text-in/text-out interface.

**Fine-tuning and multi-task strategies (Houlsby et al., 2019, adapters; Howard & Ruder
2018, gradual unfreezing).** Adapters insert small trainable modules and freeze the rest,
to update fewer parameters; gradual unfreezing thaws layers progressively. Gap: they trade
some quality for parameter efficiency relative to plain full fine-tuning, and how they
compare under a controlled pipeline is unknown.

## Evaluation settings

The natural yardstick is a diverse suite of pre-existing English benchmarks, all available
through TensorFlow Datasets:
- **GLUE** and **SuperGLUE** classification meta-benchmarks, each an average over
  constituent tasks: linguistic acceptability (CoLA), sentiment (SST-2),
  paraphrase/similarity (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE,
  CB), coreference (WNLI, WSC, DPR), sentence completion (COPA), word-sense (WiC), and QA
  (MultiRC, ReCoRD, BoolQ). STS-B is a regression task (similarity scored 1-5). WNLI's
  validation set is adversarial w.r.t. its training set and is conventionally omitted.
- **CNN/Daily Mail** abstractive summarization, scored with ROUGE (ROUGE-1/2/L-F).
- **SQuAD** extractive question answering, scored with exact-match and F1.
- **WMT** English→German, English→French, and English→Romanian translation, scored with
  BLEU (SacreBLEU, "exp" smoothing, "intl" tokenization). EnFr is high-resource; EnRo is
  a standard low-resource benchmark.

Protocol: pre-train on the unlabeled corpus, then **separately fine-tune on each task**;
report results on validation sets (to avoid selecting on test); select the best checkpoint
by validation performance; use **greedy decoding** at inference. To gauge significance,
train the base configuration multiple times and treat its run-to-run variance as
applicable to each variant.

## Code framework

What already exists before the method: a deep-learning framework with autodiff and an
optimizer, a tokenizer, a generic Transformer self-attention block, a maximum-likelihood
(cross-entropy) loss, and a teacher-forced training loop. The contribution will be filled
into the empty slots below.

```python
import torch, torch.nn as nn

# ---- Already exists: tokenizer, optimizer, cross-entropy, training loop ----
tokenizer = load_subword_tokenizer()                    # SentencePiece/WordPiece/etc.
optimizer_cls = load_optimizer_class()                  # chosen by the training setup
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)         # teacher-forced max-likelihood

# ---- Slot 1: the task interface ----------------------------------------------
def to_model_io(task, example):
    # TODO: present a downstream task (classification, regression, QA,
    # summarization, translation) to the model. How should the obstacle above --
    # the per-task output interface -- be handled so the other axes can vary?
    pass

# ---- Slot 2: the self-supervised pre-training transformation -----------------
def build_pretraining_example(token_ids):
    # TODO: corrupt an unlabeled token sequence into (input, target) so the model
    # learns generalizable knowledge with no labels. What corruption? What target?
    pass

# ---- Slot 3: how position enters attention -----------------------------------
def position_signal(query_len, key_len, n_heads):
    # TODO: self-attention is order-invariant; inject position. Absolute? relative?
    # how parameterized, how shared across layers/heads, how to generalize to length?
    pass

# ---- Slot 4: normalization for a sublayer ------------------------------------
class SubLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        # TODO: which normalization, where placed relative to the residual path?
        pass
    def forward(self, x):
        pass

# ---- Slot 5: the sequence model itself ---------------------------------------
class SequenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: one stack or two? what attention mask(s)? cross-attention or not?
        # share parameters? tie input/output embeddings?
        pass
    def forward(self, input_ids, target_ids):
        # TODO: produce next-token logits over the target under teacher forcing
        pass

# ---- Training loop (already exists) ------------------------------------------
def train_step(model, batch):
    logits = model(batch["input_ids"], batch["target_ids"])
    loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
    loss.backward()
```
