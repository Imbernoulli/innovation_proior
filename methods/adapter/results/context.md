## Research question

A large pre-trained Transformer such as BERT transfers well to downstream NLP tasks, and the dominant recipe copies the entire checkpoint and tunes all of it for each task. Serving N tasks then means storing N full sets of backbone weights. The question is how to adapt one pre-trained Transformer to many downstream tasks at a small per-task parameter cost, training tasks one at a time without revisiting earlier datasets, while reaching downstream quality close to full fine-tuning.

## Background

**Feature-based transfer and fine-tuning.** A pre-trained network can be written as phi_w(x). Feature-based transfer keeps w fixed and trains a downstream model chi_v(phi_w(x)) on top of the upstream representation. Fine-tuning instead copies w and optimizes it on the downstream task; the per-task delta is the full backbone: |delta| = |w|.

**Small residual modules in vision.** Multi-domain image classification can use small per-domain residual modules inside a shared convolutional backbone. The convolutional parameterization is specific to vision, and a text Transformer has a different sub-layer layout.

**Normalization as modulation.** Conditional batch normalization, FiLM, and self-modulation adapt networks by learning per-condition affine normalization parameters that rescale and shift existing features pointwise.

**Multi-task and continual learning.** Multi-task learning shares parameters across tasks with simultaneous access to all task data. Continual learning handles streams of tasks, where modifying a shared network can cause catastrophic forgetting.

**Transformer block structure.** In the post-layer-normalization BERT block, each layer has two primary sub-layers: multi-head attention and a position-wise feed-forward network. Each sub-layer projects back to the model dimension d, applies dropout, adds the residual stream, then applies layer normalization:

```text
attention path:      h = Attention(x) projected to d;      y = LayerNorm(x + dropout(h))
feed-forward path:   h = FeedForward(y) projected to d;    z = LayerNorm(y + dropout(h))
```

The block operates throughout at the model dimension d, with a residual add and layer normalization at the end of each sub-layer path.

## Baselines

- **Full fine-tuning of BERT.** Copy all pre-trained weights, attach a linear classifier to the pooled [CLS] representation, and tune the whole model. Trains 100% of the backbone per task and stores a full copy per task.
- **Feature-based transfer.** Freeze the representation model and train a task model on top of its embeddings or internal representations, keeping the upstream model fixed.
- **Top-layer-only fine-tuning.** Freeze lower layers and tune the top k Transformer layers; smaller k trains fewer weights. Stores a full checkpoint per task.
- **Layer-normalization-only tuning.** Train only normalization affine parameters and the task head, giving pointwise scale and shift.

## Evaluation settings

- **GLUE** text classification and inference tasks, including MNLI, CoLA, SST-2, MRPC, STS-B, QQP, QNLI, and RTE, using the benchmark's task metrics.
- **Additional public classification tasks** with diverse dataset sizes, label counts, and text lengths, plus **SQuAD v1.1** extractive question answering.
- **Base models** are public pre-trained BERT BASE and LARGE checkpoints. Classification uses a linear head on the pooled [CLS] token representation.
- **Training protocol** follows BERT fine-tuning: Adam-style optimization, linear warmup over the first 10% of steps, linear decay to zero, batch size 32, and hyperparameter selection on validation performance.
- **Comparison axis** is downstream task quality against the number of trainable task-specific parameters.

## Code framework

The available implementation frame is a TensorFlow 1 BERT encoder: dense attention and feed-forward projections, dropout, residual addition, layer normalization, a pooled [CLS] classifier head, and an optimizer with warmup and linear decay. The frame is left incomplete where the task-specific construction would go.

```python
import tensorflow as tf


def task_module(input_tensor):
  """TODO: task-specific construction goes here."""
  return input_tensor


def transformer_layer(layer_input, attention_output, intermediate_output,
                      hidden_size, hidden_dropout_prob, task_module_fn=None):
  with tf.variable_scope("attention/output"):
    attention_output = tf.layers.dense(attention_output, hidden_size)
    attention_output = dropout(attention_output, hidden_dropout_prob)
    # TODO: decide whether a task-specific transform belongs here.
    attention_output = layer_norm(attention_output + layer_input)

  with tf.variable_scope("output"):
    layer_output = tf.layers.dense(intermediate_output, hidden_size)
    layer_output = dropout(layer_output, hidden_dropout_prob)
    # TODO: decide whether a task-specific transform belongs here.
    layer_output = layer_norm(layer_output + attention_output)
    return layer_output


def create_training_op(loss, learning_rate, num_train_steps, num_warmup_steps):
  """TODO: collect only the per-task variables to train."""
  train_vars = []
  grads = tf.gradients(loss, train_vars)
  optimizer = make_warmup_decay_adam(learning_rate, num_train_steps, num_warmup_steps)
  return optimizer.apply_gradients(zip(grads, train_vars))
```
