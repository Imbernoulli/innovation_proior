Efficient Neural Architecture Search (ENAS) turns neural architecture search from
"train every sampled child from scratch" into "train one shared supergraph and
sample subgraphs from it."

All candidate child architectures are subgraphs of a single directed acyclic
graph. The controller is an LSTM policy pi(m; theta) that samples the active
edges and operations. The child weights omega live on the supergraph; when a
child m is sampled, it uses only the omega entries on its active subgraph. Two
children that select the same operation/edge share the same parameters.

The training objective splits into two alternating updates:

1. Fix theta and train omega on training minibatches:
   E_{m ~ pi(m; theta)}[L(m; omega)].
   A one-sample Monte Carlo gradient is used in practice: sample one child
   architecture per minibatch, backpropagate through its active subgraph, and
   update the shared weights.
2. Fix omega and train theta on validation reward:
   E_{m ~ pi(m; theta)}[R(m, omega)].
   Since m is discrete, use REINFORCE with a moving-average baseline:
   grad_theta E[R] ~= (R - b) grad_theta log pi(m; theta).
   The reward is validation accuracy for CIFAR-10 and a decreasing transform of
   validation perplexity for Penn Treebank. Entropy and tempered/scaled logits
   keep the controller from collapsing too early.

After search, sample several architectures from the trained controller, score
them with the shared weights on a validation minibatch, keep the best one, and
train only that selected architecture from scratch. This moves the expensive
full training run from every sampled candidate to one final architecture. The
search cost drops from tens of thousands of GPU-hours to less than 16 hours on
one GTX 1080Ti, more than a 1000x reduction.

Search spaces:

- Recurrent cell: an N-node DAG. The controller chooses each node's predecessor
  and activation; each predecessor/activation choice selects a corresponding
  shared matrix. Highway gates stabilize the recurrent transitions, and loose
  ends are combined as the cell output.
- Convolutional macro space: each layer samples an operation from conv,
  separable-conv, max-pool, and avg-pool choices, plus binary skip connections
  from earlier layers. A KL penalty pulls skip probabilities toward a target
  density such as rho = 0.4.
- Convolutional micro space: each internal cell node samples two previous nodes
  and two operations, adds the two results, and concatenates unused nodes at the
  cell output. A reduction cell uses the same decisions with stride 2.

Implementation core:

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
