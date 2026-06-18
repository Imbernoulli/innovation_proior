# Context: one language model, many tasks, no per-task training

## Research Question

Language systems are still built as narrow experts. A dataset is collected for one
task, a model or head is trained for that task, and the resulting system is judged
on held-out examples from the same distribution. Transfer learning weakens this
dependence but does not remove it: a large unlabeled corpus initializes a model,
then each downstream task still needs labeled examples, task-specific formatting,
and supervised adaptation.

The question is whether a single left-to-right language model can be trained once
on unlabeled text and then perform many language tasks with no parameter updates
and no architecture changes. The task must be invoked only by the text placed in
the context. A successful answer has to specify the training data, the text
representation, and the model changes needed to make that zero-shot setting
plausible rather than a toy observation.

## Background

A language model estimates a distribution over a sequence of symbols. With the
usual left-to-right factorization,

```text
p(x) = product_{i=1}^n p(s_i | s_1, ..., s_{i-1}).
```

This objective already gives conditionals of later tokens given earlier tokens. A
single supervised task can be written as `p(output | input)`, while a general
system needs `p(output | input, task)`. Prior multitask and meta-learning systems
often encode the task in the architecture or optimization procedure, but natural
language itself can also describe the task. A translation example, a question
answering example, or a summary request can all be written as one sequence of
tokens.

The remaining difficulty is empirical. The corpus must contain many natural task
demonstrations, not only one domain such as news, Wikipedia, or books. Web-scale
text is attractive because it is broad, but raw web scrapes are noisy and contain
many low-quality documents. Any corpus construction strategy must therefore find
a cheap quality filter and must also avoid obvious overlap with evaluation
benchmarks.

The input representation has a separate constraint. A general model should assign
probability to any string without lowercasing, task-specific tokenization, or an
unknown-token escape hatch. Bytes satisfy coverage but make sequences long and
low-level. Word-level vocabularies are compact for common words but lose rare
strings. Subword methods sit between these extremes, yet a fully general subword
system must handle Unicode without spending the entire vocabulary on base
symbols.

## Baselines

The immediate predecessor is a decoder-only Transformer language model trained on
unlabeled books and then fine-tuned with supervised examples and task-specific
input transformations. It shows that a single Transformer stack can transfer
widely, but it still depends on labeled data and adaptation for every task.

The Transformer supplies the causal self-attention machinery: each position can
attend only to previous positions, the attention scores are scaled by
`1/sqrt(d_k)`, and each block combines attention, a position-wise feed-forward
network, residual connections, and layer normalization. The original arrangement
places normalization after the residual addition; deeper stacks raise stability
questions for that choice.

Byte-pair encoding supplies a practical open-vocabulary subword mechanism by
iteratively merging frequent adjacent symbols. In its standard form it operates
over character/code-point strings, so using it for an arbitrary Unicode model
raises the base-vocabulary problem. Raw byte-level modeling avoids that base
vocabulary issue but has been weaker than word-level modeling at large scale.

## Evaluation Settings

The evaluation must be zero-shot: no fine-tuning, no task-specific learned heads,
and no benchmark training split used for adaptation. The language model can only
be conditioned by text prompts or by scoring candidate continuations.

Useful tests include language-modeling transfer on Penn Treebank, WikiText-2,
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
