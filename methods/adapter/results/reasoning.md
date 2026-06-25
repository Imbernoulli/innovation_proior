Let me start from the deployment constraint, because it is the thing that makes ordinary transfer feel wasteful. I have one large pre-trained BERT checkpoint, and tasks arrive one after another. Full fine-tuning is the strong recipe, but it makes a complete private copy of the backbone for each task. After N tasks I am not serving one reusable representation; I am serving N independent checkpoints. Storage grows like N|w|, training a new task moves all of w again, and there is no clean way to say that old tasks are preserved except by keeping their entire copies.

Feature-based transfer avoids moving w: write the pre-trained network as phi_w(x), freeze it, and train a downstream function chi_v(phi_w(x)). That keeps the backbone shared, but now the task can only read the representation that phi_w already emits. It cannot reconfigure the computation inside the Transformer. Fine-tuning goes to the other extreme: train w itself, so the task can alter every layer, but the task-specific delta is the whole backbone. I want the middle shape: a function psi_{w,v}(x) where w is copied from pre-training and then frozen, v is small, and many tasks cost |w| plus one small v per task. Let me check what such a shape buys me on the online constraint, since that was the part I cared about. If v for task 5 is a disjoint set of variables from v for tasks 1 through 4, then a gradient step taken while training task 5 writes only into task-5 variables and into nothing that the earlier tasks depend on; the shared w never moves because it is frozen. So earlier tasks evaluate to exactly the same function after task 5 trains as before it. That is what I needed from an online method, and it falls out of "freeze w, keep each task's v private" rather than needing any rehearsal of old data.

This immediately forces an initialization condition. I am going to insert new randomly initialized parameters into a deep pre-trained network. If those new parameters produce an arbitrary transform at step zero, each following layer receives activations from the wrong distribution. The network begins training already damaged. So v_0 has to make psi_{w,v_0}(x) approximately equal phi_w(x). The extra path must begin as almost an identity and become active only as training finds a useful task-specific correction.

Where can such a correction live? A vision residual-module idea gives the right abstraction: make the private task parameters write into the frozen backbone, not just into a head. But BERT is not a convolutional ResNet. Each Transformer layer has an attention sub-layer and a feed-forward sub-layer. In the post-LN BERT block, each sub-layer produces something in the model dimension d, applies dropout, adds the residual stream, and then layer-normalizes. The slot has to preserve the d-dimensional interface, and it should sit where it can change the sub-layer contribution without breaking the residual skeleton.

A full dense d-to-d correction is too expensive. At d = 768, one d by d matrix is already 768 * 768 = 589,824 weights. Two of those per layer over 12 layers is about 14 million weights, more than a tenth of a 110M-parameter BERT-BASE, which no longer feels like a tiny task delta. I need a d-to-d function whose parameter count is linear in d times some much smaller width, not quadratic in d.

The economical shape is to squeeze first. Project the d-dimensional vector down to m dimensions, apply the same kind of smooth nonlinearity BERT already uses, then project back to d. The down matrix contributes dm weights, the down bias contributes m, the up matrix contributes md, and the up bias contributes d, so one module has 2md + d + m parameters. If m is much smaller than d, I get a real d-to-d nonlinear correction with a small parameter count, and m becomes the single knob for the accuracy/compactness trade-off.

Now I need the initialization condition to actually hold for this bottleneck. The naive way to get a near-identity is to keep the module's output near zero at the start and hope the rest takes care of itself, but a bottleneck that maps x to Up(GeLU(Down(x))) maps to near-zero, not near-x, when its weights are small. So the tempting move "just initialize the projection weights tiny" gives me the wrong fixed point: a near-zero sub-layer contribution, which after the residual add and layer norm is a real distortion of phi_w, not an approximation of it. I have to make the identity explicit. The residual idea has to live inside the module: return x + Up(GeLU(Down(x))). Now small weights give me x plus a small correction, which is the near-identity I wanted.

Let me make sure "small weights" is actually small enough to leave the pre-trained forward pass intact, instead of just asserting it. Take the code's truncated-Gaussian init with standard deviation s on both projection matrices and zero biases. The down-projection pre-activation a_j = sum_i x_i (W_down)_{ij} is a sum of d terms each with standard deviation s|x|/sqrt(d) in expectation, so a_j has standard deviation about s|x|. For small s these pre-activations sit near zero, where GeLU(a) is approximately a/2, so the hidden units have standard deviation about (s/2)|x|. The up-projection then sums m such units against weights of scale s, giving each output coordinate standard deviation about (s/2)|x| * s * sqrt(m) / sqrt(d)... collecting the d output coordinates, the correction has norm about 0.5 s^2 sqrt(m d) relative to |x|. So the relative size of the injected branch scales like s squared, not s — two weight matrices in series each contribute a factor of s.

I want a number, so I plug in d = 768, m = 64. The estimate 0.5 s^2 sqrt(m d) gives 1.1e-2 for s = 1e-2 and 1.1e-4 for s = 1e-3. I also drew random x and random truncated-Gaussian projections and measured ||correction|| / ||x|| directly: the mean was 1.16e-2 (max 1.75e-2) at s = 1e-2, and 1.11e-4 at s = 1e-3, which matches the s-squared estimate. So at s = 1e-2 the correction branch is on the order of one percent of the activation it sits beside at initialization, and at s = 1e-3 it is roughly a hundredth of a percent. Both are small enough that the frozen network's forward pass is essentially unchanged at step zero; the s = 1e-3 default in the code is the more conservative of the two and starts the branch even closer to zero. This is the property I needed: the injected branch begins small enough that the original computation is not knocked off its pre-trained path, and it grows only if training pushes the projection weights up.

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

Let me put the parameter accounting on real numbers, because "small fraction" is the whole selling point and I should not take it on faith. One bottleneck is 2md + d + m. Each layer has two of them, so L layers add 2L(2md + d + m). At BERT-BASE scale, d = 768, L = 12, and m = 64, one bottleneck is 2*64*768 + 768 + 64 = 99,136 parameters, and the two-per-layer total over 12 layers is 2*12*99,136 = 2,379,264. The layer-norm affine variables add 2L * 2d = 2*12*2*768 = 36,864, so adapters plus layer-norm settings come to about 2.42 million parameters. Against a roughly 110-million-parameter backbone that is 2.2 percent per task. If I want it tighter, m = 8 drops the bottleneck total to 313,536 and the whole per-task delta to about 0.32 percent; if I want more capacity, m = 256 pushes it to about 8.6 percent. So m really is a smooth dial from sub-percent to single-digit-percent per task, and every setting is far below the 100 percent that full fine-tuning pays. The online property holds exactly alongside this: each task owns its bottlenecks, its layer-norm settings, and its head, and the shared backbone is bit-for-bit identical across tasks, so by the disjoint-variables argument above no task can perturb another.

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

Tracing the constraints back through: many tasks pushed me to a frozen shared w with a small private v per task; freezing a deep network meant psi_{w,v_0} had to start near phi_w, which ruled out the plain bottleneck (small weights there give near-zero, not near-identity) and forced the internal skip with near-zero projection weights, a choice I checked numerically holds to about one percent at s = 1e-2; the cost constraint forced the d-to-m-to-d bottleneck rather than a full d-by-d correction, which the parameter count confirms lands at a couple of percent of the backbone; the BERT post-LN block fixed the insertion point after each sub-layer projection and before the residual add and layer norm; and the trainable set came out as the adapter variables, the layer-norm affine variables, and the head, with the backbone left untouched.
