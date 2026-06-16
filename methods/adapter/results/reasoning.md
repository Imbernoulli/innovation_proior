Let me start from the deployment constraint, because it is the thing that makes ordinary transfer feel wasteful. I have one large pre-trained BERT checkpoint, and tasks arrive one after another. Full fine-tuning is the strong recipe, but it makes a complete private copy of the backbone for each task. After N tasks I am not serving one reusable representation; I am serving N independent checkpoints. Storage grows like N|w|, training a new task moves all of w again, and there is no clean way to say that old tasks are preserved except by keeping their entire copies.

Feature-based transfer avoids moving w: write the pre-trained network as phi_w(x), freeze it, and train a downstream function chi_v(phi_w(x)). That keeps the backbone shared, but now the task can only read the representation that phi_w already emits. It cannot reconfigure the computation inside the Transformer. Fine-tuning goes to the other extreme: train w itself, so the task can alter every layer, but the task-specific delta is the whole backbone. I want the middle shape: a function psi_{w,v}(x) where w is copied from pre-training and then frozen, v is small, and many tasks cost |w| plus one small v per task. If v for task 5 is disjoint from v for tasks 1 through 4, then training task 5 cannot overwrite them. That gives me the online property for free.

This immediately forces an initialization condition. I am going to insert new randomly initialized parameters into a deep pre-trained network. If those new parameters produce an arbitrary transform at step zero, each following layer receives activations from the wrong distribution. The network begins training already damaged. So v_0 has to make psi_{w,v_0}(x) approximately equal phi_w(x). The extra path must begin as almost an identity and become active only as training finds a useful task-specific correction.

Where can such a correction live? A vision residual-module idea gives the right abstraction: make the private task parameters write into the frozen backbone, not just into a head. But BERT is not a convolutional ResNet. Each Transformer layer has an attention sub-layer and a feed-forward sub-layer. In the post-LN BERT block, each sub-layer produces something in the model dimension d, applies dropout, adds the residual stream, and then layer-normalizes. The slot has to preserve the d-dimensional interface, and it should sit where it can change the sub-layer contribution without breaking the residual skeleton.

A full dense d-to-d correction is too expensive. At d = 768, one d by d matrix is already about 590k weights. Two of those per layer over 12 layers would no longer feel like a tiny task delta. I need a d-to-d function whose parameter count is linear in d times some much smaller width, not quadratic in d.

The economical shape is to squeeze first. Project the d-dimensional vector down to m dimensions, apply the same kind of smooth nonlinearity BERT already uses, then project back to d. The down matrix contributes dm weights, the down bias contributes m, the up matrix contributes md, and the up bias contributes d, so one module has 2md + d + m parameters. If m is much smaller than d, I get a real d-to-d nonlinear correction with a small parameter count, and m becomes the single knob for the accuracy/compactness trade-off.

The bottleneck alone still does not satisfy the initialization condition. If the module replaces x by Up(GeLU(Down(x))), then making the weights tiny gives me a near-zero output, not the original x. The residual idea has to be inside the module itself: return x + Up(GeLU(Down(x))). Now a near-zero branch gives me a near-identity function. With a zero-mean truncated Gaussian at a small scale and zero biases, the correction branch starts near zero, so the frozen network's forward pass is almost unchanged at the beginning. A 1e-2 standard deviation is small enough for the training setup; an even more conservative 1e-3 default in code keeps the same invariant. The important point is that the injected branch starts small enough that the original computation is not knocked off its pre-trained path.

Now the placement becomes precise. In the attention block, self-attention is projected back to hidden size, then dropout is applied. I put the small residual bottleneck on that projected sub-layer output, before adding the original layer input and before layer normalization:

```text
attention_output = Adapter(dropout(Project(SelfAttention(layer_input))))
attention_output = LayerNorm(attention_output + layer_input)
```

The feed-forward block has the same pattern. After the intermediate GeLU feed-forward layer is projected back to hidden size and dropped out, the second small bottleneck transforms that projected output before the residual add:

```text
layer_output = Adapter(dropout(Project(Intermediate(attention_output))))
layer_output = LayerNorm(layer_output + attention_output)
```

That means two serial bottleneck modules per Transformer layer: one after attention, one after feed-forward. They are not parallel side branches around the whole layer, and they are not placed after layer normalization. They modify the sub-layer contribution that will be added into the residual stream.

What gets trained? The frozen-backbone argument says all original BERT weights stay fixed. The task head must be new, because label spaces and losses differ across tasks. I also want the layer-normalization affine parameters to move. They are only scale and shift, 2d parameters per layer norm, but once the small modules begin changing sub-layer outputs, those affine parameters give each task a cheap way to re-center and rescale the activations. They are not expressive enough to be the whole adaptation mechanism, but they are a useful companion to the nonlinear bottlenecks. So the trainable set is exactly the bottleneck variables, the layer-norm variables, and the head variables.

Let me check the parameter accounting in the full Transformer. One bottleneck is 2md + d + m. Each layer has two of them, so L layers add 2L(2md + d + m) bottleneck parameters per task. Then I add the layer-norm affine variables and the classifier head. For BERT-scale d and m in the tens or hundreds, that is a small fraction of the backbone rather than another full checkpoint. The online property is also exact: each task has private bottlenecks, private layer-norm settings, and a private head, while the shared backbone is unchanged.

The final shape is the small function and the two insertion calls inside the Transformer layer, with optimization restricted to the three task-specific collections.

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

The chain is tight: many tasks force a frozen shared w plus a small private v; a deep frozen backbone forces psi_{w,v_0} to begin near phi_w; near-identity forces an internal skip plus near-zero projection weights; a cheap d-to-d correction forces the d-to-m-to-d bottleneck with 2md + d + m parameters; the BERT post-LN block fixes the insertion point after each sub-layer projection and before residual plus layer norm; the task-specific trainable set is the adapter variables, layer-norm affine variables, and head variables, leaving the backbone untouched.
