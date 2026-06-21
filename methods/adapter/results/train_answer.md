The problem is how to adapt one large pre-trained Transformer, such as BERT, to many downstream NLP tasks without paying the storage and training cost of a full backbone copy for every task. Full fine-tuning gives each task its own complete set of backbone weights, so serving N tasks requires roughly N times the original model size and makes it impossible to add a new task without storing another full checkpoint. Feature-based transfer keeps the backbone frozen and only trains a small head on top, which is compact and extensible, but the task cannot modify the internal Transformer computation, so it usually lags full fine-tuning. What is needed is a middle path: a frozen shared backbone plus a small, task-specific, trainable delta that is inserted deep enough inside the network to reshape its behavior.

The key insight is that the delta must start as nearly an identity. Because the new parameters are randomly initialized inside an already trained network, an arbitrary initial transform would push every following layer off its pre-trained distribution. If the delta is written as a residual correction to the original sub-layer output, and its projection weights are initialized near zero, then at the start of training the network's forward pass is almost unchanged. As training proceeds, the correction branch learns a task-specific transformation while the backbone remains fixed. This gives both extensibility, because each task has private parameters, and compactness, because the delta is small.

The method is called Adapter, specifically bottleneck adapter tuning for Transformers. It freezes all pre-trained weights and inserts small trainable adapter modules inside every Transformer layer, one after the multi-head attention output projection and one after the feed-forward output projection. Each adapter is a bottleneck network that projects a d-dimensional sub-layer output down to a much smaller width m, applies a GeLU nonlinearity, projects back to d, and adds the original input as a residual skip. Including biases, one adapter has 2md + d + m parameters. With m chosen much smaller than d, the total per-task parameter count is a small fraction of the backbone.

The placement respects the post-layer-normalization BERT block. After self-attention is projected back to the model dimension d and dropout is applied, the adapter transforms that projected output before the residual addition and layer normalization. The same happens after the intermediate feed-forward layer is projected back to d. The adapter therefore modifies the sub-layer contribution that enters the residual stream, not the whole layer at once. This preserves the residual skeleton while giving the task fine-grained control over attention and feed-forward behavior at every layer. In addition to the adapters, the layer-normalization affine parameters and the task-specific classification head are trained, because once the adapters begin shifting activation distributions, the cheap scale and shift parameters help re-center and re-scale them, and the head must be new for each task's label space.

The initialization is crucial. The adapter returns x plus the bottleneck branch. If the down-projection and up-projection weights are sampled from a zero-mean truncated Gaussian with a small standard deviation such as 1e-3, and the biases start at zero, the branch output is initially near zero and the adapter is near the identity. This means the adapted network starts close to the original pre-trained network and only gradually learns a task-specific correction. Because the original backbone weights are frozen, training one task never overwrites another task's private adapters, layer-norm settings, or head, so tasks can be added sequentially without revisiting old data.

The optimization is restricted to only the task-specific variables. During training, gradients are computed only for the adapter variables, the layer-normalization variables, and the task head. The original BERT weights receive no updates. This keeps the per-task training cost low and guarantees that the shared backbone stays identical across tasks. The training schedule follows the standard BERT fine-tuning recipe: Adam-style optimization with a linear warmup over the first 10% of steps and linear decay to zero afterward, gradient clipping, and a small weight-decay rate applied only to the trainable variables.

```python
import tensorflow as tf


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
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


def transformer_layer(layer_input, attention_output, intermediate_output,
                      hidden_size, hidden_dropout_prob, initializer_range,
                      adapter_fn=feedforward_adapter):
    with tf.variable_scope("attention/output"):
        attention_output = tf.layers.dense(
            attention_output, hidden_size,
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


def create_training_op(loss, init_lr, num_train_steps, num_warmup_steps):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate, global_step, num_train_steps, end_learning_rate=0.0,
        power=1.0, cycle=False)
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        warmup_progress = (
            tf.cast(global_steps_int, tf.float32) /
            tf.cast(warmup_steps_int, tf.float32))
        warmup_lr = init_lr * warmup_progress
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_lr)

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
