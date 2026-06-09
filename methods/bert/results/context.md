# Context: pre-training general-purpose language representations

## Research question

Across natural language processing, almost every task — sentence-level ones like
natural language inference and paraphrase detection, and token-level ones like
named-entity recognition and extractive question answering — is starved for
labeled data, while unlabeled text is essentially infinite. The field has
established that pre-training a model on unlabeled text and then transferring it
helps broadly. The open question is how to extract the *most* general, reusable
representation from that unlabeled text.

A representation is most useful for understanding tasks when each token's vector
encodes the token's meaning **in its full context — both the words to its left
and the words to its right**, at every layer of the network. For span-extraction
question answering, the representation of a candidate answer word is only as good
as the context it can see, and the words that disambiguate it lie on both sides.
The pain point is concrete: the prevailing way to pre-train on unlabeled text is
a language model — predict the next word — and a next-word predictor is
intrinsically one-directional. The representations it produces only ever encode
left context. Whatever pre-training recipe one adopts has to be learnable from
unlabeled text alone and transferable to both sentence-level and token-level
tasks with essentially no task-specific architecture engineering.

## Background

**Word representations as the substrate.** Modern NLP rests on pre-trained word
representations. Non-neural class-based and distributional methods (Brown et al.
1992) gave way to neural word embeddings: word2vec (Mikolov et al. 2013) trains
vectors by discriminating a true center word from corrupted ones using
surrounding context; GloVe (Pennington et al. 2014) factorizes co-occurrence
statistics. These give one fixed vector per word type — they cannot represent
that "bank" means different things in different sentences. Generalizations to
sentence and paragraph embeddings followed (skip-thought, Kiros et al. 2015,
which generates neighboring sentences; Logeswaran & Lee 2018, which *ranks* the
true next sentence against distractors).

**Contextual representations.** The decisive shift was making a word's vector a
function of the whole sentence. A language model does this for free: its hidden
state at position t is a contextual summary of the prefix. context2vec (Melamud
et al. 2016) learns a representation of a word from both its left and right
context with LSTMs, but the model is feature-based and not deeply bidirectional.

**The Transformer (Vaswani et al. 2017).** A sequence model built entirely from
attention, no recurrence. Each layer: multi-head self-attention followed by a
position-wise feed-forward network, each wrapped in a residual connection and
layer normalization. Self-attention computes, for every pair of positions,
softmax(QKᵀ/√d_k)V — the 1/√d_k factor keeps the dot products from growing with
dimension and pushing the softmax into saturation; multiple heads let the model
attend to several relations at once; the feed-forward inner dimension is
conventionally 4× the hidden size. Self-attention is **all-to-all** — a position
can attend to any other position in either direction. To use a Transformer as a
left-to-right language model, one imposes a causal (triangular) attention mask so
position t cannot attend to positions > t.
Position information is injected through positional encodings (sinusoidal in the
original, but a learned position-embedding table is an equally valid option).

**The two transfer recipes that exist.** Both pre-train on unlabeled text with a
language-model objective, then apply the result to downstream tasks; they differ
in *how* they apply it.

**Empirical observations that motivate the problem.** Two diagnostic facts about
the existing systems frame everything that follows. First, the systems that
encode both directions do so only *shallowly*: they train a left-to-right model
and a right-to-left model separately and concatenate them, so no single hidden
unit at any internal layer is ever jointly conditioned on both sides — the
joining happens only at the very top. Second, on token-level benchmarks the
purely left-to-right fine-tuning systems visibly underperform, and the
explanation is mechanical: a token's hidden state has seen nothing to its right,
which is exactly the information a span predictor needs.

## Baselines

**ELMo (Peters et al. 2018) — feature-based, shallowly bidirectional.**
Train a forward LSTM language model that predicts xₜ from x_{<t} (maximizing
Σₜ log p(xₜ | x₁,…,x_{t−1})) and, **independently**, a backward LSTM language
model that predicts xₜ from x_{>t}. For a stack of L layers, each token then has
2L+1 vectors (the context-independent input plus the forward and backward hidden
states at each layer). A downstream model collapses these into one vector via a
learned, task-specific softmax-weighted sum, scaled by a learned coefficient, and
feeds it as a *frozen additional feature* into a task-specific architecture. This
substantially advanced QA, NLI, NER, and sentiment. Gap it leaves: the two
directions are trained as separate language models and only **concatenated**;
there is never a single representation jointly conditioned on both sides, and
within any layer a forward unit still sees only the left and a backward unit only
the right. And because the features are frozen, every task still needs its own
architecture built on top.

**OpenAI GPT (Radford et al. 2018) — fine-tuning, unidirectional.**
Pre-train a multi-layer Transformer **with a causal attention mask** as a
left-to-right language model on a large text corpus, maximizing
Σₜ log p(xₜ | x_{<t}). Then transfer by fine-tuning: initialize the whole network
from the pre-trained weights, add a single linear output head, and train *all*
parameters end-to-end on the downstream task. Because almost no parameters are
task-specific, this is clean and strong — it set the prior state of the art on
the GLUE sentence-level tasks. Gap it leaves: the causal mask makes every
token's representation depend on left context only. This is forced by the
language-model objective, not chosen — and it is exactly wrong for token-level
tasks, where both-sided context is essential, and leaves performance on the table
even for sentence-level tasks.

**What both inherit.** Both recipes take the unidirectionality (or
shallow-bidirectionality) of the language-model objective as given: GPT trains a
single left-to-right model under a causal mask, ELMo trains separate left-to-right
and right-to-left models. Neither departs from the next-word objective that the
field has settled on for learning from unlabeled text.

## Evaluation settings

The natural yardsticks already exist as standard, fixed benchmarks. **GLUE**
(Wang et al. 2018) bundles diverse sentence- and pair-level understanding tasks
with canonical train/dev/test splits and a held-out-label evaluation server:
MNLI (3-way entailment), QQP and MRPC (paraphrase), QNLI (answer-sentence
selection), RTE (entailment, small), SST-2 (sentiment), CoLA (linguistic
acceptability), STS-B (similarity regression). **SQuAD v1.1** (Rajpurkar et al.
2016) is extractive QA: given a question and a Wikipedia passage that contains
the answer, predict the answer span, scored by Exact Match and F1. **SQuAD v2.0**
adds unanswerable questions, so a system must also decide that no span answers.
**SWAG** (Zellers et al. 2018) is grounded commonsense inference: pick the most
plausible continuation of a sentence among four. **CoNLL-2003 NER** (Tjong Kim
Sang & De Meulder 2003) is token-level entity tagging, scored by entity F1.
Pre-training corpora are likewise standard: BooksCorpus (800M words) and English
Wikipedia, both available as large unlabeled text; a document-level corpus
(rather than a shuffled-sentence one like the Billion Word Benchmark) is needed
to obtain long contiguous spans.

## Code framework

The starting harness is a Transformer encoder (self-attention is already
all-to-all unless a mask is imposed), a subword tokenizer with a fixed vocabulary,
an embedding lookup, an Adam optimizer with warmup, and a standard training loop.
The data-to-target transformation, input-marking convention, and output mapping
are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB_SIZE   = 30000          # subword (WordPiece-style) vocabulary
MAX_LEN      = 512
H, L, A      = 768, 12, 12    # hidden size, layers, attention heads
FFN          = 4 * H

# Transformer encoder layer from the existing sequence-modeling toolkit.
# Multi-head self-attention with 1/sqrt(d_k) scaling, FFN of width 4H, residual
# + LayerNorm. A mask, if supplied, restricts which positions can interact.
class TransformerEncoderLayer(nn.Module):
    def __init__(self, h=H, a=A, ffn=FFN):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, a, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, ffn), nn.GELU(), nn.Linear(ffn, h))
    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))

# Subword tokenizer with a fixed vocabulary.
def tokenize(text): ...           # returns subword ids

# Define the unlabeled-text training signal.
def build_pretraining_example(corpus):
    # TODO: turn raw unlabeled text into (inputs, targets) for self-supervision.
    # What objective gives useful contextual representations, and how are raw
    # text spans assembled into model inputs and targets?
    pass

# Represent the input sequence.
class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, H)
        self.pos = nn.Embedding(MAX_LEN, H)
        # TODO: add any extra input-indicator table the objective requires.
        self.ln  = nn.LayerNorm(H)
    def forward(self, ids, indicator_ids=None):
        # TODO: combine token, position, and any objective-required indicators.
        pass

# Network being pre-trained, plus whatever prediction interface the objective needs.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = InputEmbedding()
        self.encoder = nn.ModuleList(TransformerEncoderLayer() for _ in range(L))
        # TODO: map encoder outputs into the pre-training targets.
    def forward(self, ids, indicator_ids=None, pad_mask=None):
        # TODO: embed -> encoder -> produce whatever the
        # objective in build_pretraining_example needs to score.
        pass

# Optimizer + training loop scaffold.
def pretrain(model, data):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                           weight_decay=0.01)                 # + warmup, lin decay
    for batch in data:
        loss = ...   # TODO: loss returned by the self-supervised objective
        loss.backward(); opt.step(); opt.zero_grad()

# Fine-tuning scaffold.
def finetune(pretrained, task_head, task_data):
    # TODO: reuse the pretrained encoder, attach one small task head, train all.
    pass
```
