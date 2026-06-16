Let me start from the expensive part, because the rest of the machinery is not where the pain lives. I have an LSTM controller with parameters theta. It defines a distribution pi(m; theta) over child architectures m by emitting a sequence of discrete decisions: which operation, which previous node, which skip edge. I sample a child, train that child, measure validation performance, and use that scalar as a reward. REINFORCE can update theta even though the architecture decisions are discrete. This is a good formulation. The bad part is that the reward for one sample is bought by training one whole child network from random initialization to convergence, reading off one validation number, and then throwing all of those weights away.

That is where the GPU-hours disappear. The controller is small. Sampling tokens is cheap. The policy-gradient update is cheap. The waste is that every candidate relearns basic filters, recurrent transitions, embeddings, and skip-path behavior from scratch, even though many candidates reuse the same kinds of local computations. If I keep scoring children this way, architecture search is always a nested loop where the inner loop is "train a complete neural network." Reducing the number of samples or training each child less is just starving the search. I need to remove the repeated from-scratch training itself.

The assumption to attack is that each architecture owns an independent set of weights. That assumption is natural if I think of each candidate as a separate model. But it is not inevitable. Transfer learning already tells me that weights learned in one setting can be useful in another. Weight inheritance in evolutionary search tells me that a mutated child does not have to begin from random noise; it can inherit the parent's parameters. The inheritance version is local, though. It only helps when the new child is near a previous parent. I want the stronger version: one weight store that can serve the whole search space.

This starts to make sense once I stop picturing the search space as a bag of unrelated networks. A cell search space is a fixed menu of possible decisions. At a node I choose a previous node and an operation. At a convolutional layer I choose an operation and a set of previous layers for skip connections. Across candidates, the menu is the same; only the active choices differ. If I draw every possible local computation and every possible edge, I get one large directed acyclic graph. A child architecture is not a separate object anymore. It is a subgraph selected from this larger graph.

Now parameter sharing becomes the natural representation. Put parameters on the edges and local computations of the big DAG. When the controller samples an architecture, the sampled tokens choose which edges and operations are active, and therefore which subset of parameters omega is used. If two children select the same edge or operation slot, they use the same weights for that slot. All child architectures are subgraphs of a single supergraph, and all of them share the one parameter collection omega. I no longer train a fresh child to convergence just to score it. I train omega as the weights of the supergraph, and a sampled child receives the relevant slice of omega automatically.

This creates two coupled optimization problems. The controller parameters theta decide which subgraph is sampled. The shared weights omega determine how well the sampled subgraph performs. I cannot update them as if they were one ordinary differentiable model, because theta controls discrete choices. But I can alternate.

For omega, I hold theta fixed. The controller policy pi(m; theta) is just a sampler over architectures. The objective for the shared weights is the expected training loss of a sampled child,

  minimize_omega E_{m ~ pi(m; theta)}[L(m; omega)].

For a sampled m, the loss is an ordinary cross-entropy or language-model loss through the active subgraph, so it is differentiable in omega. The Monte Carlo gradient is

  grad_omega E[L(m; omega)] ~= (1/M) sum_i grad_omega L(m_i; omega),  m_i ~ pi(m; theta).

This is unbiased because theta is fixed and the only derivative is with respect to omega. The obvious worry is variance: one architecture touches only part of the shared DAG. But increasing M would start making each shared-weight step expensive again. The aggressive choice is M = 1: sample one architecture per minibatch, backprop through only its active subgraph, and move omega. Over an epoch, different minibatches activate different subgraphs, so the architecture sampling noise averages together with the usual minibatch noise. That keeps the shared-weight phase cheap: one child subgraph per minibatch, not many trained children per controller sample.

For theta, I hold omega fixed. Now the objective is expected validation reward,

  maximize_theta E_{m ~ pi(m; theta)}[R(m, omega)].

The reward is measured after executing the sampled subgraph with the current shared weights. It is not differentiable through the sampled tokens, so I use the score-function identity:

  grad_theta E[R] = E[R(m, omega) * grad_theta log pi(m; theta)].

The estimator has high variance, so I subtract a moving baseline b:

  E[(R - b) * grad_theta log pi] = E[R * grad_theta log pi] - b * grad_theta sum_m pi(m; theta)
                                = E[R * grad_theta log pi].

The baseline update can be a simple exponential moving average, b <- b - (1 - decay) * (b - R). If I store the controller's decision terms as cross-entropies, sample_log_probs = -sum log pi(m; theta), then the sign in code looks slightly different from the math: minimizing sample_log_probs * (R - b) raises the probability of an architecture when its reward is above the baseline and lowers it when the reward is below the baseline. That is the same REINFORCE ascent step, written with cross-entropy terms.

The data split matters. Omega should be trained on the training split; otherwise the weights do not learn the task. Theta should be rewarded on validation data; otherwise the controller can select architectures that exploit training-set fit rather than generalization. For language modeling, lower perplexity is better, so I convert validation perplexity into a reward like c / valid_ppl. For image classification, validation accuracy already has the right direction. In both cases the reward is computed on a validation minibatch with omega fixed.

Now I need the controller's sampling spaces to line up with this shared-weight view.

For a recurrent cell, I use a DAG with N computation nodes. The first node consumes the current input x_t and the previous cell state h_{t-1}, then applies a sampled activation. For a later node ell, the controller samples a previous index j < ell and an activation f_ell. The node computes a transformed state from h_j. To make deep recurrent transitions trainable, I use a highway gate rather than a bare transformation:

  c_ell = sigmoid(h_j W^{(c)}_{ell,j})
  h_ell = c_ell * f_ell(h_j W^{(h)}_{ell,j}) + (1 - c_ell) * h_j.

The gate gives every sampled recurrent path an easy carry route and a controlled transform route. The shared-weight interpretation is exact: every ordered pair (j, ell) and activation choice indexes a specific slice or matrix in omega, and the sampled previous index decides which slice is active. Nodes that are not selected as inputs by later nodes are combined as the cell output, so a sampled topology has a well-defined output even when the graph branches.

The controller for this cell is an LSTM. It emits decisions in order. For predecessor choices it uses attention-like logits over previous controller hidden states; for operation choices it uses a softmax over the activation menu. After sampling a choice, it feeds an embedding of that choice into the next controller step. The first step receives an empty learned embedding. Temperature and a scaled tanh on logits keep early distributions from becoming too sharp.

For a macro convolutional search space, each layer has two kinds of sampled decisions. First, choose the operation: 3x3 conv, 5x5 conv, 3x3 separable conv, 5x5 separable conv, 3x3 average pool, or 3x3 max pool. Second, choose skip connections from previous layers. The selected previous outputs are incorporated into the current layer, with projection and normalization where shapes need to be stabilized. Each layer-operation branch has its own parameters in omega, and every child that selects that branch uses the same parameters.

The skip-connection sampling needs a guardrail. If the controller can freely turn every skip on or off, it can drift into too sparse or too dense wiring while the shared weights are still noisy. I keep a prior skip probability rho, around 0.4, and penalize the KL divergence between the controller's skip probabilities and that target. In implementation terms this belongs in the controller objective as an added penalty term, not as a positive reward for large divergence. The reward can still be validation accuracy plus entropy encouragement; the KL term pulls the skip distribution toward the target density.

For a micro convolutional search space, I search cells rather than the whole network. A cell has two input nodes from the preceding cells. Each new internal node samples two previous nodes and two operations from identity, 3x3 separable conv, 5x5 separable conv, 3x3 average pool, and 3x3 max pool; the two operation outputs are added. Loose ends are concatenated to form the cell output. A reduction cell comes from the same decision space, but operations that touch the cell inputs use stride 2 so spatial resolution is halved. The controller samples the normal cell and the reduction cell in sequence, so the final architecture can be built by stacking the sampled cells.

The whole training loop is now simple. I train omega for a pass through the training data, sampling one architecture per minibatch and backpropagating only through its active subgraph. Then I freeze omega and train theta for a fixed number of controller steps, each time sampling an architecture, scoring it on validation data, updating the moving baseline, and applying the REINFORCE loss. I repeat the two phases. At the end I sample several architectures from the trained controller, evaluate each with the shared weights on a validation minibatch, keep the best one, and train that single architecture from scratch as the final model. I have moved the expensive from-scratch training from "every controller sample" to "one final selected architecture."

When I wire this up, the child owns shared variables indexed by the sampled arc, the controller owns `sample_arc`, `sample_entropy`, and cross-entropy decision terms, and the two train ops are separate.

```python
import tensorflow.compat.v1 as tf

REWARD_CONSTANT = 80.0


def activation(fn_id, h):
  return tf.stack([tf.tanh(h), tf.nn.relu(h), tf.identity(h), tf.sigmoid(h)])[fn_id]


class SharedRecurrentCell(object):
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
              "w", [num_functions, node, hidden_size, 2 * hidden_size]))

  def __call__(self, x_t, prev_state, sample_arc, is_training):
    h_and_gate = tf.matmul(tf.concat([x_t, prev_state], axis=1), self.w_input)
    h, gate = tf.split(h_and_gate, 2, axis=1)
    first_fn = sample_arc[0]
    state = prev_state + tf.sigmoid(gate) * (activation(first_fn, h) - prev_state)
    nodes = [state]
    used = []
    offset = 1

    for node in range(1, self.num_nodes):
      prev_id = sample_arc[offset]
      fn_id = sample_arc[offset + 1]
      used.append(tf.one_hot(prev_id, depth=self.num_nodes, dtype=tf.int32))
      prev = tf.stack(nodes, axis=0)[prev_id]
      w = self.w_skip[node - 1][fn_id, prev_id]
      h_and_gate = tf.matmul(prev, w)
      h, gate = tf.split(h_and_gate, 2, axis=1)
      state = prev + tf.sigmoid(gate) * (activation(fn_id, h) - prev)
      nodes.append(state)
      offset += 2

    used = tf.reduce_sum(tf.stack(used), axis=0)
    loose = tf.boolean_mask(tf.stack(nodes), tf.equal(used, 0))
    return tf.reduce_mean(loose, axis=0)


def build_shared_weight_train_op(child_loss, child_variables, global_step, lr):
  optimizer = tf.train.GradientDescentOptimizer(lr)
  grads = tf.gradients(child_loss, child_variables)
  grads, _ = tf.clip_by_global_norm(grads, 0.25)
  return optimizer.apply_gradients(zip(grads, child_variables),
                                   global_step=global_step)


def build_controller_train_op(sample_log_probs, sample_entropy, valid_loss,
                              baseline, controller_variables,
                              entropy_weight=1e-4, baseline_decay=0.99):
  valid_ppl = tf.exp(tf.stop_gradient(valid_loss))
  reward = REWARD_CONSTANT / valid_ppl
  reward += entropy_weight * tf.stop_gradient(sample_entropy)
  baseline_update = tf.assign_sub(
      baseline, (1.0 - baseline_decay) * (baseline - reward))
  with tf.control_dependencies([baseline_update]):
    reward = tf.identity(reward)

  advantage = reward - baseline
  loss = tf.reduce_sum(sample_log_probs) * advantage
  optimizer = tf.train.AdamOptimizer(3.5e-4)
  return optimizer.minimize(loss, var_list=controller_variables), reward


def alternating_search(session, child_train_op, controller_train_fn,
                       num_epochs):
  for _ in range(num_epochs):
    for _ in training_minibatches():
      session.run(child_train_op)      # one sampled child subgraph updates omega
    controller_train_fn(session)       # validation rewards update theta
  return best_arc_by_shared_validation_reward(session)
```

The cost came from training every sampled child independently; the search space can be represented as one DAG whose children are subgraphs; putting parameters on the DAG lets all children share omega; omega can be trained by ordinary backprop through one sampled subgraph per minibatch; theta can be trained separately by REINFORCE from validation rewards; and only the final selected architecture needs its own full from-scratch training run. That is how the search cost collapses from hundreds of GPUs for days to a single-GPU run while preserving the RL controller's ability to choose discrete architectures.
