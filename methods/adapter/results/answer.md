# Adapter Tuning

## Problem

Adapt one pre-trained Transformer to many downstream NLP tasks without storing and training a full
copy of the backbone for every task. Full fine-tuning trains 100% of the model per task; the target
is a frozen shared backbone plus a small private trainable delta for each task.

## Method

Freeze the pre-trained Transformer weights w and define a task-specific function
psi_{w,v}(x) whose new parameters v are small and whose initialization satisfies
psi_{w,v_0}(x) approximately equals phi_w(x). The trainable task parameters are:

- bottleneck adapters inserted inside every Transformer layer,
- layer-normalization affine parameters,
- the task head on the pooled [CLS] representation.

Each adapter receives a d-dimensional sub-layer output, compresses it to bottleneck width m, applies
GeLU, projects back to d, and adds the original input:

```text
Adapter(x) = x + W_up GeLU(x W_down + b_down) + b_up
W_down in R^{d x m}, W_up in R^{m x d}, m << d
```

The parameter count per adapter, including biases, is:

```text
d*m + m + m*d + d = 2md + d + m.
```

The internal residual skip is what makes near-zero projection weights produce an approximate
identity rather than an approximate zero map. A zero-mean truncated Gaussian with standard
deviation 1e-2 is used for the main training setup; the code path below uses a conservative 1e-3
default. Both are small relative to the backbone initialization and keep the new branch near zero at
the start.

Placement is fixed by the BERT post-LN block. There are two adapters per Transformer layer: one
after the attention output projection and dropout, and one after the feed-forward output projection
and dropout. In both cases the adapter is before the residual add and layer normalization:

```text
attention_output = Adapter(dropout(Project(SelfAttention(x))))
y = LayerNorm(attention_output + x)

ffn_output = Adapter(dropout(Project(FeedForward(y))))
z = LayerNorm(ffn_output + y)
```

With L Transformer layers, the bottleneck modules add `2L * (2md + d + m)` adapter parameters per
task, plus layer-normalization affine parameters and the classifier head.

## Code

```python
import tensorflow as tf


def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
  """Bottleneck adapter with an internal residual skip."""
  with tf.variable_scope("adapters"):
    in_size = input_tensor.get_shape().as_list()[1]
    w1 = tf.get_variable(
        "weights1", [in_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b1 = tf.get_variable(
        "biases1", [1, hidden_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(input_tensor, w1, [[1], [0]]) + b1
    net = gelu(net)

    w2 = tf.get_variable(
        "weights2", [hidden_size, in_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b2 = tf.get_variable(
        "biases2", [1, in_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(net, w2, [[1], [0]]) + b2

  return net + input_tensor


def transformer_layer(layer_input, attention_heads, intermediate_output,
                      hidden_size, hidden_dropout_prob, initializer_range,
                      adapter_fn=feedforward_adapter):
  with tf.variable_scope("attention/output"):
    attention_output = tf.layers.dense(
        attention_heads, hidden_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    attention_output = dropout(attention_output, hidden_dropout_prob)
    attention_output = adapter_fn(attention_output)
    attention_output = layer_norm(attention_output + layer_input)

  with tf.variable_scope("output"):
    layer_output = tf.layers.dense(
        intermediate_output, hidden_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    layer_output = dropout(layer_output, hidden_dropout_prob)
    layer_output = adapter_fn(layer_output)
    layer_output = layer_norm(layer_output + attention_output)
    return layer_output


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps):
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  learning_rate = tf.train.polynomial_decay(
      learning_rate, global_step, num_train_steps, end_learning_rate=0.0,
      power=1.0, cycle=False)
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
    warmup_progress = (
        tf.cast(global_steps_int, tf.float32) / tf.cast(warmup_steps_int, tf.float32))
    warmup_lr = init_lr * warmup_progress
    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (1.0 - is_warmup) * learning_rate + is_warmup * warmup_lr

  train_vars = []
  for collection in ["adapters", "layer_norm", "head"]:
    train_vars += tf.get_collection(collection)

  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      adapter_weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  grads = tf.gradients(loss, train_vars)
  grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
  train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
  return tf.group(train_op, [global_step.assign(global_step + 1)])
```
