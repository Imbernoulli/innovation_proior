## Research question

Full-network pretraining of Transformer encoders has driven a rapid run of gains
on language understanding — on the Chinese-English-exam reading task RACE, machine
accuracy moved from 44.1% at the task's introduction to 83.2% in the best
published system. A consistent lesson across these results is that *larger*
networks help: more layers, wider hidden size, more heads tend to improve
downstream accuracy. The natural next move is simply to keep scaling. But two
things stand in the way. First, hardware memory is finite; state-of-the-art
models already have hundreds of millions to billions of parameters, so naively
widening a model hits memory limits, and distributed training slows down because
communication cost is proportional to the parameter count. Second — and more
troubling — scaling is not even reliably *helping*: when the optimization budget
(steps, learning rate) is held fixed and the hidden size of a large BERT model is
simply doubled, the bigger model performs *worse* on downstream tasks, with no
sign of overfitting. So the precise question is: is better NLP as easy as bigger
models, and if not, can the parameters be reorganized so that a model with *fewer*
parameters than the current large one trains more stably and scales further?

## Background

**Full-network pretrain-then-finetune.** The field shifted from pretraining word
embeddings — static (Mikolov et al. 2013; Pennington et al. 2014) or contextual
(Peters et al. 2018) — to pretraining the entire network and finetuning it per
task (Dai & Le 2015; Radford et al. 2018; Devlin et al. 2018). Within this
paradigm, larger hidden size / more layers / more heads has repeatedly improved
accuracy (Devlin et al. 2018 stop at hidden size 1024, presumably for cost).

**Where the parameters live in a Transformer encoder.** Two pools dominate. The
token embedding matrix has size V×E (vocabulary times embedding dimension), and
when the embedding dimension is tied to the hidden size H (as is standard, E≡H),
widening H widens this matrix proportionally to the vocabulary — tens of thousands
of rows — even though most rows are updated only sparsely. The per-layer
Transformer parameters (attention projections plus a feed-forward block whose
inner dimension is conventionally 4H) are replicated independently across all L
layers, so the depth multiplies this pool L-fold.

**Memory and communication.** Prior remedies attack memory but not communication:
gradient checkpointing recomputes activations to make memory sublinear at the
cost of an extra forward pass (Chen et al. 2016); reversible layers reconstruct a
layer's activations from the next so intermediate activations needn't be stored
(Gomez et al. 2017); model parallelism splits a giant model across devices
(Raffel et al. 2019; Shoeybi et al. 2019). None of these *reduce* the parameter
count, so distributed communication stays expensive.

**Cross-layer parameter sharing, prior art.** Sharing weights across Transformer
layers was explored for encoder-decoder tasks: the Universal Transformer
(Dehghani et al. 2018) reports that a recurrent, weight-shared Transformer *beats*
a vanilla one on language modeling and subject-verb agreement; the Deep
Equilibrium Model (Bai et al. 2019) shows a shared-layer Transformer can reach a
fixed point where a layer's input and output embeddings coincide. These exist
outside the pretrain/finetune setting.

**Inter-sentence objectives and a known weakness.** BERT pairs masked language
modeling with next-sentence prediction (NSP): a binary classifier on whether two
segments are consecutive in the corpus, with positives drawn as adjacent segments
and negatives as segments from two *different* documents, sampled 50/50. NSP was
meant to help sentence-pair tasks like natural language inference. But subsequent
work found its effect unreliable and removed it with no loss, even gains (Yang et
al. 2019; Liu et al. 2019). Coherence and discourse-ordering objectives
have a long line of study (Hobbs 1979; Grosz et al. 1995; skip-thought, Kiros et
al. 2015; discourse-marker and sentence-ordering objectives, Jernite et al. 2017;
Nie et al. 2019).

**Regularization at scale.** Dropout is standard in these models; whether it
helps when a very large model is *underfitting* its data (high train loss, no
overfitting even after a million steps) is an open empirical question, with prior
hints that combining certain normalizers and dropout can interact badly (Li et
al. 2019).

## Baselines

**BERT (Devlin et al. 2018).** Bidirectional Transformer encoder, GELU
activations, masked language modeling + NSP, WordPiece vocabulary of 30K with the
embedding dimension tied to hidden size. Base: L=12, H=768, A=12, 108M params;
large: L=24, H=1024, A=16, 334M. The reference point. Gaps it leaves: the
embedding matrix scales with H; the per-layer parameters scale with depth; NSP is
a weak auxiliary task; and simply widening it (a 2048-hidden "xlarge") *degrades*
accuracy under a fixed budget.

**XLNet (Yang et al. 2019) and RoBERTa (Liu et al. 2019).** Both keep the
embedding-tied-to-hidden convention, both drop NSP, and both report strong
results partly by training on much more data. They establish that NSP can be
removed and that more data helps, but they do not address the parameter-count and
scaling-stability problem — they make the model *better*, not *lighter*.

## Evaluation settings

The standard yardstick is three benchmarks. GLUE (Wang et al. 2018) — nine
understanding tasks (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI), with
single-task finetuning and a held-out test server; MNLI reported on the matched
split. SQuAD v1.1 (Rajpurkar et al. 2016), span extraction from a Wikipedia
paragraph with 100K answerable questions; v2.0 (Rajpurkar et al. 2018) adds 50K
unanswerable ones, handled with a separate answerability classifier trained
jointly with the span loss. RACE (Lai et al. 2017), four-way multiple-choice
reading comprehension from Chinese English exams, with the passage, question, and
each candidate answer concatenated and a representation of the first token used to
score each option. Pretraining corpora are BookCorpus (Zhu et al. 2015) plus
English Wikipedia (~16GB); tokenization via SentencePiece (Kudo & Richardson
2018), vocabulary 30K. Metrics: accuracy (GLUE/RACE), exact-match/F1 (SQuAD).
Development-set medians over five runs are reported for high-variance GLUE tasks.

## Code framework

The substrate is a Transformer-encoder pretraining harness: token, position, and
segment embeddings; a stack of self-attention + feed-forward blocks; a masked-LM
head; an optional inter-sentence classification head; and a large-batch optimizer
(LAMB, You et al. 2019). Several of the design choices are left open as slots to
be filled.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- embedding stem: token, position, and segment embeddings ---
class EmbeddingStem(nn.Module):
    def __init__(self, vocab_size, hidden, embedding_width=None,
                 max_positions=512, type_vocab_size=2, dropout=0.1):
        # TODO: define the embedding parameterization.
        pass

    def forward(self, input_ids, token_type_ids=None):
        pass

# --- one Transformer encoder block (attention + 4H feed-forward, GELU) ---
class EncoderBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden); self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Linear(4*hidden, hidden))

    def forward(self, x, padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

# --- encoder stack ---
class EncoderStack(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, dropout):
        # TODO: define how the encoder's per-layer parameters are organized.
        pass

    def forward(self, x, padding_mask=None):
        pass

# --- objectives ---
def gather_positions(sequence_output, positions):
    pass

class MLMHead(nn.Module):
    def __init__(self, hidden, embedding_width, vocab_size):
        # TODO: choose how hidden states are projected to vocabulary logits.
        pass

    def forward(self, sequence_output, positions, embedding_table):
        pass

class SentencePairHead(nn.Module):
    def __init__(self, hidden):
        pass

    def forward(self, sequence_output):
        pass

def inter_sentence_examples(seg_a, seg_b):
    # TODO: define what makes a positive vs a negative pair for the inter-sentence task.
    pass

def sample_ngram_length(max_n=3):
    pass

def total_loss(mlm_logits, mlm_labels, mlm_weights, pair_logits, pair_labels):
    pass
```
