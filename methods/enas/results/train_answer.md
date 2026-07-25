Neural architecture search framed as reinforcement learning gives a clean way to automate model design: a controller samples a discrete architecture, the child network is trained, its validation performance becomes a reward, and the controller is updated toward better designs. The catch is that every sampled child is initialized from scratch, trained to convergence, scored once, and then thrown away. In the NASNet-scale setting this means tens of thousands of GPU-hours. Tricks like training each child for fewer epochs or evaluating fewer candidates only weaken the signal, because the controller still needs reliable validation rewards to discriminate architectures.

Existing ideas do not remove that bottleneck cleanly. Evolutionary search can copy parent weights into mutated children, but the benefit is local to each parent-child pair. Hypernetworks generate child weights from an architecture encoding, yet the generation process constrains the weights to a low-rank subspace, so the search may reward architectures that look good under generated weights rather than under ordinary training. Performance predictors and progressive searches reduce how long each child is trained, but they still leave a per-candidate training cost. The deeper assumption to discard is that every architecture owns an independent set of weights.

The method I propose is Efficient Neural Architecture Search, or ENAS. The central move is to view the entire search space as one large directed acyclic graph and every candidate architecture as a subgraph of it. All parameters are placed on the supergraph once; when the controller samples a subgraph, that subgraph uses only the corresponding slice of the shared weights. Two architectures that pick the same operation or edge therefore share the same parameters. This collapses the expensive inner loop into ordinary training of a single weight collection, with one sampled subgraph back-propagated per minibatch.

ENAS alternates two cheap optimization problems. First, fix the controller policy and train the shared weights omega on the training set. The objective is the expected training loss of an architecture sampled from the current controller, and in practice we use a one-sample Monte Carlo estimate: sample one architecture per minibatch, run its active subgraph, and backpropagate through only the active parameters. Over many minibatches the different subgraphs average out, so omega learns to serve the whole distribution of architectures. Second, fix omega and train the controller parameters theta by REINFORCE on validation reward. Because the architecture tokens are discrete, we use the score-function gradient, subtracting an exponential moving-average baseline to reduce variance. For image classification the reward is validation accuracy; for language modeling lower perplexity is better, so we use a constant divided by validation perplexity. An entropy bonus and temperature on the controller logits keep exploration alive.

The controller is an LSTM that emits architecture decisions in order: for a recurrent cell it chooses each node’s predecessor and activation; for a macro convolutional space it chooses each layer’s operation and binary skip connections, with a KL penalty pulling skip density toward a target such as 0.4; for a micro convolutional space it designs normal and reduction cells by sampling two predecessors and two operations per internal node. The recurrent cell uses highway gates so every sampled path has both a carry and a transform route, and unused nodes are averaged to form the cell output. After search, we sample several architectures from the trained controller, score them with the shared weights, keep the best, and train only that one architecture from scratch. The full from-scratch training is moved from every candidate to one final architecture, reducing the search cost from tens of thousands of GPU-hours to a single-GPU run.

```python
import tensorflow.compat.v1 as tf

REWARD_CONSTANT = 80.0


def build_controller_loss(sample_log_probs, sample_entropy, valid_loss,
                          baseline, entropy_weight, baseline_decay):
  valid_ppl = tf.exp(tf.stop_gradient(valid_loss))
  reward = REWARD_CONSTANT / valid_ppl
  reward += entropy_weight * tf.stop_gradient(sample_entropy)

  baseline_update = tf.assign_sub(
      baseline, (1.0 - baseline_decay) * (baseline - reward))
  with tf.control_dependencies([baseline_update]):
    reward = tf.identity(reward)

  # sample_log_probs stores cross-entropy terms, i.e. -log pi(arc; theta).
  return tf.reduce_sum(sample_log_probs) * (reward - baseline)


def shared_recurrent_step(x_t, prev_state, sample_arc, w_input, w_skip):
  h_gate = tf.matmul(tf.concat([x_t, prev_state], axis=1), w_input)
  h, gate = tf.split(h_gate, 2, axis=1)
  funcs = tf.stack([tf.tanh(h), tf.nn.relu(h), tf.identity(h), tf.sigmoid(h)])
  state = prev_state + tf.sigmoid(gate) * (funcs[sample_arc[0]] - prev_state)
  nodes, used, offset = [state], [], 1

  for node_id, node_weights in enumerate(w_skip, start=1):
    prev_id = sample_arc[offset]
    func_id = sample_arc[offset + 1]
    prev = tf.stack(nodes, axis=0)[prev_id]
    w = node_weights[func_id, prev_id]      # shared edge/operation weights
    h_gate = tf.matmul(prev, w)
    h, gate = tf.split(h_gate, 2, axis=1)
    funcs = tf.stack([tf.tanh(h), tf.nn.relu(h), tf.identity(h), tf.sigmoid(h)])
    candidate = funcs[func_id]
    state = prev + tf.sigmoid(gate) * (candidate - prev)
    nodes.append(state)
    used.append(tf.one_hot(prev_id, depth=len(w_skip) + 1, dtype=tf.int32))
    offset += 2

  unused = tf.equal(tf.reduce_sum(tf.stack(used), axis=0), 0)
  return tf.reduce_mean(tf.boolean_mask(tf.stack(nodes), unused), axis=0)


def train_search(sess, child_train_op, controller_train_fn, epochs):
  for _ in range(epochs):
    for _ in training_minibatches():
      sess.run(child_train_op)       # updates omega through one sampled subgraph
    controller_train_fn(sess)        # updates theta from validation rewards
```
