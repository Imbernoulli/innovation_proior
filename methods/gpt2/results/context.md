# Context: one language model, many tasks, no per-task training

## Research Question

Language systems are typically built as narrow experts. A dataset is collected for
one task, a model or head is trained for that task, and the resulting system is
judged on held-out examples from the same distribution. Transfer learning changes
this pattern: a large unlabeled corpus initializes a model, and then each
downstream task is approached with labeled examples, task-specific formatting, and
supervised adaptation.

The question is whether a single left-to-right language model can be trained once
on unlabeled text and then perform many language tasks with no parameter updates
and no architecture changes, where the task is invoked only by the text placed in
the context. Answering it requires specifying the training data, the text
representation, and the model so that this zero-shot setting can be evaluated.

## Background

A language model estimates a distribution over a sequence of symbols. With the
usual left-to-right factorization,

```text
p(x) = product_{i=1}^n p(s_i | s_1, ..., s_{i-1}).
```

This objective gives conditionals of later tokens given earlier tokens. A single
supervised task can be written as `p(output | input)`, while a general system can
be written as `p(output | input, task)`. Prior multitask and meta-learning systems
encode the task in the architecture or optimization procedure, and natural language
itself can also describe the task: a translation example, a question answering
example, or a summary request can each be written as one sequence of tokens.

The corpus is then the main empirical lever. Single domains such as news,
Wikipedia, or books supply one style of text. Web-scale text is broad, and raw web
scrapes also carry many low-quality documents, so corpus construction involves a
quality filter and attention to overlap between the training text and evaluation
benchmarks.

The input representation is a separate design choice. A general model assigns
probability to strings, and several encodings are available. Byte-level modeling
covers any string. Word-level vocabularies are compact for common words. Subword
methods such as byte-pair encoding sit between these extremes, operating over a
base alphabet of symbols that frequent adjacent pairs are merged from.

## Baselines

The immediate predecessor is a decoder-only Transformer language model trained on
unlabeled books and then fine-tuned with supervised examples and task-specific
input transformations. It shows that a single Transformer stack can transfer
across many tasks.

The Transformer supplies the causal self-attention machinery: each position
attends only to previous positions, the attention scores are scaled by
`1/sqrt(d_k)`, and each block combines attention, a position-wise feed-forward
network, residual connections, and layer normalization. The original arrangement
places normalization after the residual addition.

Byte-pair encoding supplies an open-vocabulary subword mechanism by iteratively
merging frequent adjacent symbols. In its standard form it operates over
character/code-point strings. Raw byte-level modeling operates directly over the
256 byte values.

## Evaluation Settings

The evaluation is zero-shot: no fine-tuning, no task-specific learned heads, and
no benchmark training split used for adaptation. The language model is conditioned
by text prompts or by scoring candidate continuations.

Tests include language-modeling transfer on Penn Treebank, WikiText-2,
WikiText-103, enwik8, text8, LAMBADA, the Children's Book Test, and the One
Billion Word Benchmark; task behavior on reading comprehension, summarization,
translation, and open-domain question answering; and contamination checks that
look for overlap between the training corpus and benchmark test text.

## Code Framework

The scaffold starts with a decoder-only Transformer and a next-token objective.
The open slots are the corpus, the tokenizer, and the exact residual block and
initialization choices.

```python
import tensorflow as tf

def norm(x, scope, axis=-1, epsilon=1e-5):
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        return (x - u) * tf.rsqrt(s + epsilon) * g + b

def attention_mask(nd, ns, *, dtype):
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    return tf.cast(i >= j - ns + nd, dtype)

def block(x, scope, *, past, hparams):
    # TODO: decide where normalization belongs and how residuals are initialized.
    pass

def encode(text):
    # TODO: encode any Unicode string without an unknown token.
    pass

def build_corpus():
    # TODO: collect diverse high-quality text while controlling benchmark overlap.
    pass

def lm_loss(logits, tokens):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tokens[:, 1:], logits=logits[:, :-1]))
```
