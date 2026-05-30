# GPT-2: language models as unsupervised multitask learners

## Problem

ML language systems are narrow experts that need a supervised dataset and a
task-specific architecture per task; even pretrain-then-finetune needs labeled data
and per-task adaptation. GPT-2 asks whether one model, trained once with no
supervision, can perform many language tasks **zero-shot** — no parameter updates,
no architecture changes — purely from the language-modeling objective.

## Key idea

A single task is estimating p(output | input); a general system should model
p(output | input, task). Because a task, its input, and its output can all be
written as one symbol sequence (e.g. "translate to french", english, french), this
conditional is just an ordinary conditional of an autoregressive language model
p(x) = Πᵢ p(sᵢ | s₁…sᵢ₋₁). The supervised objective equals the unsupervised one
evaluated on a subset of the sequence, so the language-modeling optimum *contains*
the task optima — and a sufficiently large model trained on enough varied text
should learn to do the tasks implicitly. Scale unlocks it.

Three things make this work:

1. **WebText data.** Diverse, high-quality text so task demonstrations occur
   naturally: all outbound links posted to Reddit with ≥3 karma (a cheap human
   quality filter), text extracted with Dragnet + Newspaper, deduplicated, ~8M
   documents / ~40GB; **Wikipedia removed** to avoid contaminating zero-shot
   evaluation.

2. **Byte-level BPE.** Encode *any* string with no preprocessing and no unknown
   token, with a compact vocabulary. BPE over bytes (base 256, not Unicode code
   points which would need >130K base symbols), but **forbid merges across
   character categories** (with a space exception) so the vocabulary isn't wasted
   on punctuation variants like "dog", "dog.", "dog!". Vocabulary 50,257.

3. **Deep pre-LN Transformer.** Decoder-only Transformer, largely as in GPT, with:
   layer normalization moved to the **input** of each sub-block (clean identity
   residual, like pre-activation ResNets); an **additional layer norm after the
   final block** (the residual-sum output is otherwise unnormalized); residual-layer
   weights scaled at init by **1/√N** (N = number of residual layers) to control
   variance accumulation along the residual path; context **1024** tokens; batch
   size 512.

Model sizes (log-uniform): 117M (12 layers, d_model 768), 345M (24, 1024), 762M
(36, 1280), 1542M = GPT-2 (48, 1600). Learning rate tuned per size on a 5%
held-out WebText sample; all sizes still underfit WebText.

## Code

```python
import tensorflow as tf, numpy as np

def gelu(x):
    return 0.5*x*(1 + tf.tanh(np.sqrt(2/np.pi)*(x + 0.044715*tf.pow(x, 3))))

def norm(x, scope, axis=-1, epsilon=1e-5):
    with tf.variable_scope(scope):
        n = x.shape[-1].value
        g = tf.get_variable('g', [n], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        return (x - u) * tf.rsqrt(s + epsilon) * g + b

def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        h = gelu(conv1d(x, 'c_fc', n_state))               # 4x inner width
        return conv1d(h, 'c_proj', x.shape[-1].value)

def block(x, scope, *, past, hparams):                     # PRE-norm
    with tf.variable_scope(scope):
        a, present = attn(norm(x, 'ln_1'), 'attn', hparams.n_embd, past=past, hparams=hparams)
        x = x + a                                          # clean identity residual
        m = mlp(norm(x, 'ln_2'), 'mlp', 4*hparams.n_embd, hparams=hparams)
        x = x + m
        return x, present

def model(hparams, X, past=None, scope='model'):
    with tf.variable_scope(scope):
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],   # tied embedding
                              initializer=tf.random_normal_initializer(stddev=0.02))
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past))
        for layer in range(hparams.n_layer):               # residual weights init * 1/sqrt(N)
            h, _ = block(h, 'h%d' % layer, past=None, hparams=hparams)
        h = norm(h, 'ln_f')                                # extra final layer norm
        logits = tf.matmul(tf.reshape(h, [-1, hparams.n_embd]), wte, transpose_b=True)
        return logits

def lm_loss(logits, X):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=X[:, 1:], logits=logits[:, :-1]))           # next-token prediction
```

```python
# byte-level BPE: base 256, no cross-category merges (space excepted), vocab 50,257
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
# a category-splitting regex runs BEFORE merges so "dog"/"dog."/"dog!" share "dog"
```

The hyperparameters are n_vocab=50257, n_ctx=1024, with base
n_embd=768/n_head=12/n_layer=12, the pre-LN block above, the extra `ln_f`, and
logits tied to the `wte` embedding.
