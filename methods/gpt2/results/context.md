# Context: getting one model to do many language tasks without task-specific training

## Research question

Machine-learning systems for language excel as narrow experts: collect a large
supervised dataset for a task, train a model with a task-specific architecture,
and it performs well on that task's distribution — but it is brittle off-
distribution and useless for any other task. The dominant transfer recipe softens
this — pretrain a representation, then *fine-tune* with supervised data and a
task-specific head per task — but it still requires labeled data and a separate
adaptation for every task. The precise question is whether a single model, trained
once with no supervision, can perform a wide range of language tasks *zero-shot* —
with no parameter updates and no architecture changes per task — simply by being
prompted appropriately. If so, what must the training data, the input
representation, and the model be so that the tasks emerge on their own?

## Background

**Language modeling as distribution estimation.** A language model estimates the
joint distribution over a sequence of symbols, factorized autoregressively as
p(x) = Πᵢ p(sᵢ | s₁,…,sᵢ₋₁) (Jelinek & Mercer 1980; Bengio et al. 2003). This
gives tractable sampling and tractable conditionals p(s_{n−k},…,s_n | s₁,…,s_{n−k−1}).
Self-attention architectures (the Transformer, Vaswani et al. 2017) have made
these conditionals far more expressive than recurrent models.

**Task conditioning.** A single task is estimating p(output | input). A *general*
system that should do many tasks ought to condition on the task as well:
p(output | input, task). Task conditioning has been done architecturally (task-
specific encoders/decoders, Kaiser et al. 2017) or algorithmically (the inner/outer
optimization loop of MAML, Finn et al. 2017). But language itself is a flexible
medium for specifying a task, its input, and its output all as one symbol sequence
(McCann et al. 2018): a translation example can be written as the sequence
("translate to french", english text, french text); a reading-comprehension example
as ("answer the question", document, question, answer). McCann et al. showed a
single model could be trained to infer and perform many tasks given examples in
this format — but with explicit supervision.

**An empirical observation about scale.** In preliminary toy experiments,
large enough language models trained on text containing task demonstrations begin
to perform some of those tasks without explicit supervision, just far more slowly
than models given explicit supervision — a faint signal whose mechanism and limits
were not understood.

**Pretraining lineage.** Word vectors transferred to task-specific architectures
(Mikolov et al. 2013; Collobert et al. 2011); then contextual recurrent
representations (Dai & Le 2015; Peters et al. 2018); then self-attention blocks
that suggested task-specific architectures were no longer necessary (Radford et al.
2018; Devlin et al. 2018) — though all of these still needed supervised fine-tuning
to perform a task. A separate line had shown language models doing specific tasks
with little or no supervision: commonsense reasoning (Schwartz et al. 2017) and
sentiment analysis (Radford et al. 2017).

**Data quality and diversity.** Most language models had been trained on a single
domain (news, Wikipedia, or fiction books). Diverse, near-unlimited text exists in
web scrapes like Common Crawl, but those have severe quality problems — much of the
content is unintelligible — and curating them is the bottleneck. Reddit karma (an
upvote count) is a cheap human signal of whether a linked page is interesting or
useful.

**Input representation.** A general model should be able to assign a probability to
*any* string. Standard pipelines lowercase, tokenize, and emit out-of-vocabulary
tokens, which restricts the modelable strings and couples evaluation to
preprocessing. UTF-8 bytes can represent any string (Gillick et al. 2015), but
byte-level language models are not competitive with word-level ones on large
benchmarks (Al-Rfou et al. 2018). Byte-pair encoding (BPE, Sennrich et al. 2015)
is a practical middle ground: it merges frequent symbol pairs, behaving word-like
for common sequences and character-like for rare ones, with a vocabulary of
typically 32K–64K. Reference BPE operates on Unicode *code points*, which would
require a base vocabulary of over 130,000 to cover all of Unicode before any merges.

**Optimization and depth.** Very deep residual networks train more stably when the
normalization is placed on the input of each residual block rather than on the
residual path (pre-activation residual networks, He et al. 2016), keeping a clean
identity path through depth. Layer normalization (Ba et al. 2016) normalizes each
activation vector to zero mean and unit variance and applies a learned gain and
bias.

## Baselines

**GPT (Radford et al. 2018).** A left-to-right Transformer language model,
pretrained on a book corpus, then *fine-tuned* with a supervised head per task.
Core idea: generative pretraining followed by discriminative fine-tuning. Gap: it
still needs labeled data and per-task adaptation; it does not perform tasks
zero-shot.

**Task-specific supervised state of the art.** Per-task systems with engineered
architectures (reading-comprehension readers, translation models, summarization
models). Gap: each is built and trained for one task and one dataset; none is a
single general model.

**Word-level vs byte-level language models.** Word-level models are accurate but
require lossy preprocessing and an out-of-vocabulary mechanism; raw byte-level
models are fully general but underperform. Gap: neither is simultaneously general
and competitive.

## Evaluation settings

Because the model is to be tested zero-shot, the yardstick is a broad battery of
existing datasets used *without* fine-tuning: language-modeling perplexity/bits-
per-character on Penn Treebank, WikiText-2/103, enwik8, text8, and the One Billion
Word Benchmark; long-range-dependency prediction on LAMBDA (predict the final word
given ≥50 tokens of context) and the Children's Book Test (cloze accuracy on common
nouns and named entities); reading comprehension on CoQA; translation on WMT-14
French-English; summarization on CNN/Daily Mail; and open-domain question answering
on Natural Questions. Metrics are each dataset's standard metric, computed from the
language model's probabilities. (Datasets overlapping the training text, such as
Wikipedia, are handled by removing that source from training.)

## Code framework

The substrate is a decoder-only Transformer language model and an autoregressive
cross-entropy objective. What is *not* fixed: the data the model trains on, how raw
text becomes tokens, and the internal arrangement and initialization of the stack.
The scaffold leaves those slots.

```python
import tensorflow as tf, numpy as np

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x + 0.044715*tf.pow(x, 3))))

def norm(x, scope):  # layer normalization: zero mean, unit var, learned gain/bias
    ...

def attn(x, scope, n_state, *, past, hparams):   # causal self-attention (exists)
    ...

def mlp(x, scope, n_state, *, hparams):          # position-wise FFN, 4x inner (exists)
    ...

def block(x, scope, *, past, hparams):
    # TODO: arrange normalization, attention, MLP, and the residual adds
    pass

# --- tokenizer: TO DECIDE ---
def encode(text):
    # TODO: how to turn ANY string into tokens with no preprocessing and no UNK,
    #       while keeping a compact vocabulary and not wasting slots?
    pass

# --- data: TO DECIDE ---
def build_corpus():
    # TODO: what text should the model see so that diverse task demonstrations occur
    #       naturally, while staying high quality and not contaminating evaluation?
    pass

def model(hparams, X, past=None, scope='model'):
    # token + position embeddings -> stack of blocks -> logits via the tied embedding matrix
    # TODO: weight initialization for the stack
    pass

def lm_loss(logits, X):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=X[:, 1:], logits=logits[:, :-1]))   # next-token prediction
```
