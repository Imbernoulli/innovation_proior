Neural architecture search framed as reinforcement learning gives a clean way to automate model design: a controller samples a discrete architecture, the child network is trained, its validation performance becomes a reward, and the controller is updated toward better designs. The catch is that every sampled child is initialized from scratch, trained to convergence, scored once, and then thrown away. In the NASNet-scale setting this means tens of thousands of GPU-hours. Tricks like training each child for fewer epochs or evaluating fewer candidates only weaken the signal, because the controller still needs reliable validation rewards to discriminate architectures.

Existing ideas do not remove that bottleneck cleanly. Evolutionary search can copy parent weights into mutated children, but the benefit is local to each parent-child pair. Hypernetworks generate child weights from an architecture encoding, yet the generation process constrains the weights to a low-rank subspace, so the search may reward architectures that look good under generated weights rather than under ordinary training. Performance predictors and progressive searches reduce how long each child is trained, but they still leave a per-candidate training cost. The deeper assumption to discard is that every architecture owns an independent set of weights.

The method I propose is Efficient Neural Architecture Search, or ENAS. The central move is to view the entire search space as one large directed acyclic graph and every candidate architecture as a subgraph of it. All parameters are placed on the supergraph once; when the controller samples a subgraph, that subgraph uses only the corresponding slice of the shared weights. Two architectures that pick the same operation or edge therefore share the same parameters. This collapses the expensive inner loop into ordinary training of a single weight collection, with one sampled subgraph back-propagated per minibatch.

ENAS alternates two cheap optimization problems. First, fix the controller policy and train the shared weights omega on the training set. The objective is the expected training loss of an architecture sampled from the current controller, and in practice we use a one-sample Monte Carlo estimate: sample one architecture per minibatch, run its active subgraph, and backpropagate through only the active parameters. Over many minibatches the different subgraphs average out, so omega learns to serve the whole distribution of architectures. Second, fix omega and train the controller parameters theta by REINFORCE on validation reward. Because the architecture tokens are discrete, we use the score-function gradient, subtracting an exponential moving-average baseline to reduce variance. For image classification the reward is validation accuracy; for language modeling lower perplexity is better, so we use a constant divided by validation perplexity. An entropy bonus and temperature on the controller logits keep exploration alive.

The controller is an LSTM that emits architecture decisions in order: for a recurrent cell it chooses each node’s predecessor and activation; for a macro convolutional space it chooses each layer’s operation and binary skip connections, with a KL penalty pulling skip density toward a target such as 0.4; for a micro convolutional space it designs normal and reduction cells by sampling two predecessors and two operations per internal node. The recurrent cell uses highway gates so every sampled path has both a carry and a transform route, and unused nodes are averaged to form the cell output. After search, we sample several architectures from the trained controller, score them with the shared weights, keep the best, and train only that one architecture from scratch. The full from-scratch training is moved from every candidate to one final architecture, reducing the search cost from tens of thousands of GPU-hours to a single-GPU run.

```python
import tensorflow.compat.v1 as tf

REWARD_CONSTANT = 80.0


def activation(fn_id, h):
    """Activation menu indexed by fn_id: tanh, relu, identity, sigmoid."""
    return tf.stack([
        tf.tanh(h),
        tf.nn.relu(h),
        tf.identity(h),
        tf.sigmoid(h)
    ])[fn_id]


class SharedRecurrentCell(object):
    """Recurrent cell whose weights are shared across sampled DAG topologies."""

    def __init__(self, hidden_size, num_nodes, num_functions, scope="child"):
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_functions = num_functions
        with tf.variable_scope(scope):
            self.w_input = tf.get_variable(
                "w_input", [2 * hidden_size, 2 * hidden_size])
            self.w_skip = []
            for node in range(1, num_nodes):
                with tf.variable_scope("node_%d" % node):
                    self.w_skip.append(tf.get_variable(
                        "w",
                        [num_functions, node, hidden_size, 2 * hidden_size]))

    def __call__(self, x_t, prev_state, sample_arc):
        h_and_gate = tf.matmul(tf.concat([x_t, prev_state], axis=1),
                               self.w_input)
        h, gate = tf.split(h_and_gate, 2, axis=1)
        first_fn = sample_arc[0]
        state = prev_state + tf.sigmoid(gate) * (
            activation(first_fn, h) - prev_state)
        nodes = [state]
        used = []
        offset = 1

        for node in range(1, self.num_nodes):
            prev_id = sample_arc[offset]
            fn_id = sample_arc[offset + 1]
            used.append(tf.one_hot(prev_id, depth=self.num_nodes,
                                   dtype=tf.int32))
            prev = tf.stack(nodes, axis=0)[prev_id]
            w = self.w_skip[node - 1][fn_id, prev_id]
            h_and_gate = tf.matmul(prev, w)
            h, gate = tf.split(h_and_gate, 2, axis=1)
            state = prev + tf.sigmoid(gate) * (
                activation(fn_id, h) - prev)
            nodes.append(state)
            offset += 2

        used = tf.reduce_sum(tf.stack(used), axis=0)
        loose = tf.boolean_mask(tf.stack(nodes), tf.equal(used, 0))
        return tf.reduce_mean(loose, axis=0)


def build_controller_loss(sample_log_probs, sample_entropy, valid_loss,
                          baseline, entropy_weight=1e-4,
                          baseline_decay=0.99):
    """REINFORCE loss for the LSTM controller using a moving-average baseline."""
    valid_ppl = tf.exp(tf.stop_gradient(valid_loss))
    reward = REWARD_CONSTANT / valid_ppl
    reward += entropy_weight * tf.stop_gradient(sample_entropy)

    baseline_update = tf.assign_sub(
        baseline, (1.0 - baseline_decay) * (baseline - reward))
    with tf.control_dependencies([baseline_update]):
        reward = tf.identity(reward)

    advantage = reward - baseline
    # sample_log_probs stores cross-entropy terms: -log pi(arc; theta)
    return tf.reduce_sum(sample_log_probs) * advantage, reward


def build_shared_weight_train_op(child_loss, child_variables, global_step,
                                 learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = tf.gradients(child_loss, child_variables)
    grads, _ = tf.clip_by_global_norm(grads, 0.25)
    return optimizer.apply_gradients(zip(grads, child_variables),
                                     global_step=global_step)


def train_search(sess, child_train_op, controller_train_fn, num_epochs):
    """Alternate training the shared weights and the controller."""
    for _ in range(num_epochs):
        # Train omega: one sampled subgraph per minibatch.
        for _ in training_minibatches():
            sess.run(child_train_op)
        # Train theta: validation rewards via REINFORCE.
        controller_train_fn(sess)
    return best_arc_by_shared_validation_reward(sess)
```
