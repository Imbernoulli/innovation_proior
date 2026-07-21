Let me start from the expensive part, because the rest of the machinery is not where the pain lives. I have an LSTM controller with parameters theta. It defines a distribution pi(m; theta) over child architectures m by emitting a sequence of discrete decisions: which operation, which previous node, which skip edge. I sample a child, train that child, measure validation performance, and use that scalar as a reward. REINFORCE can update theta even though the architecture decisions are discrete. The formulation is fine as far as it goes. The bad part is that the reward for one sample is bought by training one whole child network from random initialization to convergence, reading off one validation number, and then throwing all of those weights away.

That is where the GPU-hours disappear. The controller is small. Sampling tokens is cheap. The policy-gradient update is cheap. The waste is that every candidate relearns basic filters, recurrent transitions, embeddings, and skip-path behavior from scratch, even though many candidates reuse the same kinds of local computations. If I keep scoring children this way, architecture search is always a nested loop where the inner loop is "train a complete neural network." Reducing the number of samples or training each child less is just starving the search. I need to remove the repeated from-scratch training itself.

Let me put a rough number on what I am paying so I know whether the inner loop is even the right thing to attack. Say the search visits on the order of 10^3 to 10^4 distinct child architectures, and each one is trained for a few GPU-hours to reach a scoreable state. That is roughly 10^4 GPU-hours of pure child-training, which is the same order as the tens of thousands of GPU-hours the baseline reports. The controller updates and the sampling are negligible against this. So almost the entire cost is "train a fresh network per sample," and any win has to come from there. Cutting the sample count by 2x only buys 2x; I want orders of magnitude.

The assumption to attack is that each architecture owns an independent set of weights. That assumption is natural if I think of each candidate as a separate model. But it is not inevitable. Transfer learning already tells me that weights learned in one setting can be useful in another. Weight inheritance in evolutionary search tells me that a mutated child does not have to begin from random noise; it can inherit the parent's parameters. The inheritance version is local, though. It only helps when the new child is near a previous parent. I want the stronger version: one weight store that can serve the whole search space.

This starts to make sense once I stop picturing the search space as a bag of unrelated networks. A cell search space is a fixed menu of possible decisions. At a node I choose a previous node and an operation. At a convolutional layer I choose an operation and a set of previous layers for skip connections. Across candidates, the menu is the same; only the active choices differ. If I draw every possible local computation and every possible edge, I get one large directed acyclic graph. A child architecture is not a separate object anymore. It is a subgraph selected from this larger graph.

So suppose I put the parameters on the edges and local computations of the big DAG instead of on each child. When the controller samples an architecture, the sampled tokens choose which edges and operations are active, and therefore which subset of parameters omega is used. If two children select the same edge or operation slot, they use the same weights for that slot. Check the bookkeeping on the smallest case. Take a 3-node recurrent DAG. Node 1 is fixed. Node 2 must pick a predecessor in {1} and an activation; node 3 must pick a predecessor in {1, 2} and an activation. The set of distinct ordered (predecessor, node) edges that could ever be active is (1->2), (1->3), (2->3) — three edges — and each can carry any of, say, 4 activations. If I allocate one weight matrix per (predecessor, node, activation) slot, that is 3 * 4 = 12 matrices total, and that allocation is independent of how many children I sample. Any child is just a choice of one incoming edge and one activation per node, i.e. it lights up node-2's slot from {1->2} x 4 and node-3's slot from {1->3, 2->3} x 4. Two children that both route 2->3 with tanh hit the identical matrix. So the count of stored parameters is set by the supergraph, not by the search, and children overlap exactly where their local choices coincide.

So I no longer train a fresh child to convergence just to score it. I train omega as the weights of the supergraph, and a sampled child receives the relevant slice of omega automatically.

This creates two coupled optimization problems. The controller parameters theta decide which subgraph is sampled. The shared weights omega determine how well the sampled subgraph performs. I cannot update them as if they were one ordinary differentiable model, because theta controls discrete choices. But I can alternate.

For omega, I hold theta fixed. The controller policy pi(m; theta) is just a sampler over architectures. The objective for the shared weights is the expected training loss of a sampled child,

  minimize_omega E_{m ~ pi(m; theta)}[L(m; omega)].

For a sampled m, the loss is an ordinary cross-entropy or language-model loss through the active subgraph, so it is differentiable in omega. The Monte Carlo gradient is

  grad_omega E[L(m; omega)] ~= (1/M) sum_i grad_omega L(m_i; omega),  m_i ~ pi(m; theta).

Let me check that this is actually a valid estimator and not just an analogy, because the expectation is over m, and m does not depend on omega when theta is fixed. Write the expectation explicitly: E_m[L(m; omega)] = sum_m pi(m; theta) L(m; omega). Differentiate in omega: grad_omega sum_m pi(m; theta) L(m; omega) = sum_m pi(m; theta) grad_omega L(m; omega) = E_m[grad_omega L(m; omega)], because pi does not depend on omega. So the average of per-sample gradients grad_omega L(m_i; omega) with m_i ~ pi is an unbiased estimate of the true gradient. Good — the "theta fixed" condition is exactly what makes pulling the gradient inside the sum legal, and there is no extra score-function term here. The obvious worry is variance: one architecture touches only part of the shared DAG. But increasing M would start making each shared-weight step expensive again. The aggressive choice is M = 1: sample one architecture per minibatch, backprop through only its active subgraph, and move omega. Over an epoch, different minibatches activate different subgraphs, so the architecture sampling noise averages together with the usual minibatch noise. That keeps the shared-weight phase cheap: one child subgraph per minibatch, not many trained children per controller sample.

For theta, I hold omega fixed. Now the objective is expected validation reward,

  maximize_theta E_{m ~ pi(m; theta)}[R(m, omega)].

The reward is measured after executing the sampled subgraph with the current shared weights. It is not differentiable through the sampled tokens, so I use the score-function identity:

  grad_theta E[R] = E[R(m, omega) * grad_theta log pi(m; theta)].

The estimator has high variance, so I subtract a baseline b. That costs no bias for the usual REINFORCE reason: E[b * grad_theta log pi] = b * grad_theta (sum_m pi(m; theta)) = b * grad_theta 1 = 0 for any b that does not depend on m, so E[(R - b) * grad_theta log pi] = E[R * grad_theta log pi]. That leaves b free to cut variance — a moving average of recent rewards is the cheap choice, b <- b - (1 - decay) * (b - R).

Now I have to be careful about the sign when this hits code, because the standard estimator ascends and an optimizer descends, and I plan to store the controller's decision terms as cross-entropies rather than log-probs. Let sample_log_probs = -sum log pi(m; theta), i.e. the cross-entropy of the sampled tokens. The code minimizes loss = sample_log_probs * (R - b) = -(R - b) * sum log pi(m; theta). Take the gradient the optimizer follows, which is grad_theta loss, and the optimizer moves theta in the -grad_theta loss direction. grad_theta loss = -(R - b) * grad_theta sum log pi. So the update direction is -grad_theta loss = +(R - b) * grad_theta sum log pi. Now read off the two cases. If R > b, the coefficient (R - b) is positive, so theta moves along +grad_theta log pi, which is the direction that increases log pi of the sampled architecture — its probability goes up. If R < b, (R - b) is negative, so theta moves along -grad_theta log pi and the architecture's probability goes down. That is precisely REINFORCE ascent on E[R], recovered through a minimized cross-entropy loss: the odd-looking "minimize sample_log_probs * (R - b)" is correct, not a bug.

The data split matters. Omega should be trained on the training split; otherwise the weights do not learn the task. Theta should be rewarded on validation data; otherwise the controller can select architectures that exploit training-set fit rather than generalization. For language modeling, lower perplexity is better, so I convert validation perplexity into a reward via c / valid_ppl: a decreasing transform of perplexity, so any positive c rescales the reward without changing which architecture looks better. For image classification, validation accuracy already increases with quality, so it can be used directly. In both cases the reward is computed on a validation minibatch with omega fixed.

Now I need the controller's sampling spaces to line up with this shared-weight view.

For a recurrent cell, I use a DAG with N computation nodes. The first node consumes the current input x_t and the previous cell state h_{t-1}, then applies a sampled activation. For a later node ell, the controller samples a previous index j < ell and an activation f_ell. The node computes a transformed state from h_j. A bare transformation here worries me: a deep sampled chain of these nodes is just a deep recurrent path, and stacking many learned linear-plus-nonlinear maps tends to attenuate or blow up the signal before omega has learned anything, which would make most sampled topologies unscoreable early. So instead of h_ell = f_ell(h_j W) I use a highway/gated form:

  c_ell = sigmoid(h_j W^{(c)}_{ell,j})
  h_ell = c_ell * f_ell(h_j W^{(h)}_{ell,j}) + (1 - c_ell) * h_j.

Check the gate at its two extremes elementwise. If the gate saturates to c_ell -> 0, then h_ell -> (1 - 0) * h_j = h_j: the node passes its input straight through, so a long sampled path can carry state with no transformation and no attenuation. If c_ell -> 1, then h_ell -> 1 * f_ell(h_j W^{(h)}) + 0 = f_ell(h_j W^{(h)}): the node becomes a pure transform. For intermediate c_ell it is a convex combination of carry and transform, elementwise. So every sampled recurrent path has an easy identity route available and a controlled transform route, and the network can choose per-unit how much to transform — the gate itself supplies that stability, not the particular activation a node happens to sample. The shared-weight interpretation lines up too: every ordered pair (j, ell) and activation choice indexes a specific slice or matrix in omega, and the sampled previous index decides which slice is active.

I still need a sampled topology to have a well-defined output even when several nodes branch. The rule is that nodes which are not selected as a predecessor by any later node are "loose ends" and get averaged to form the cell output. Let me trace this on a 3-node cell to be sure it is well defined and not vacuous. Suppose node 2 picks predecessor 1, and node 3 also picks predecessor 1. Then node 1 is used (by both 2 and 3), but neither node 2 nor node 3 is anyone's predecessor — both are loose ends — so the output is mean(h_2, h_3): defined, and it uses both leaves. Now a chain: node 2 picks 1, node 3 picks 2. Then 1 and 2 are used, only node 3 is loose, output = h_3: still defined, the end of the chain. The degenerate worry would be a topology with no loose end, but the last node N can never be selected as anyone's predecessor (there is no later node), so it is always loose; the set of loose ends is never empty. Good — the output is always defined and never empty.

The controller for this cell is an LSTM. It emits decisions in order. For predecessor choices it uses attention-like logits over previous controller hidden states; for operation choices it uses a softmax over the activation menu. After sampling a choice, it feeds an embedding of that choice into the next controller step. The first step receives an empty learned embedding. Temperature and a scaled tanh on logits keep early distributions from becoming too sharp.

For a macro convolutional search space, each layer has two kinds of sampled decisions. First, choose the operation: 3x3 conv, 5x5 conv, 3x3 separable conv, 5x5 separable conv, 3x3 average pool, or 3x3 max pool. Second, choose skip connections from previous layers. The selected previous outputs are incorporated into the current layer, with projection and normalization where shapes need to be stabilized. Each layer-operation branch has its own parameters in omega, and every child that selects that branch uses the same parameters.

The skip-connection sampling needs a guardrail. If the controller can freely turn every skip on or off, it can drift into too sparse or too dense wiring while the shared weights are still noisy. I keep a prior skip probability rho, around 0.4, and penalize the KL divergence between the controller's skip probabilities and that target. Check the direction this actually pushes: for a single Bernoulli skip with controller probability q against target rho, KL(q || rho) = q log(q/rho) + (1-q) log((1-q)/(1-rho)), which is >= 0 and equals 0 only at q = rho; its derivative in q is log(q/rho) - log((1-q)/(1-rho)), negative for q < rho and positive for q > rho. So adding this KL as a penalty to a minimized controller loss pulls q toward rho from either side. That means it belongs in the controller objective as an added penalty term, not as a positive reward (a positive coefficient on the divergence would push q away from rho, the wrong way). The reward itself can still be validation accuracy plus entropy encouragement; the KL term pulls the skip distribution toward the target density.

For a micro convolutional search space, I search cells rather than the whole network. A cell has two input nodes from the preceding cells. Each new internal node samples two previous nodes and two operations from identity, 3x3 separable conv, 5x5 separable conv, 3x3 average pool, and 3x3 max pool; the two operation outputs are added. Loose ends are concatenated to form the cell output. A reduction cell comes from the same decision space, but operations that touch the cell inputs use stride 2 so spatial resolution is halved. The controller samples the normal cell and the reduction cell in sequence, so the final architecture can be built by stacking the sampled cells.

The whole training loop is now simple. I train omega for a pass through the training data, sampling one architecture per minibatch and backpropagating only through its active subgraph. Then I freeze omega and train theta for a fixed number of controller steps, each time sampling an architecture, scoring it on validation data, updating the moving baseline, and applying the REINFORCE loss. I repeat the two phases. At the end I sample several architectures from the trained controller, evaluate each with the shared weights on a validation minibatch, keep the best one, and train that single architecture from scratch as the final model. I have moved the expensive from-scratch training from "every controller sample" to "one final selected architecture."

Let me come back to the cost estimate from the start with this structure in place. The shared-weight phase trains omega for roughly one pass over the data per epoch — that is one network's worth of training, amortized across all the subgraphs the controller samples along the way, not one network per candidate. The controller phase only runs the cheap subgraphs forward to read validation rewards; no child is trained there at all. The only full from-scratch training left is the single final architecture. So instead of ~10^3-10^4 full training runs I pay one supergraph training plus one final training run — the inner loop that dominated the original estimate is gone. That is consistent with the search collapsing from tens of thousands of GPU-hours to a single-GPU run measured in hours, though the exact wall-clock factor I would only trust after actually running it.

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

Checking `SharedRecurrentCell.__call__` against the two-leaf case above (num_nodes = 3, both node 1 and node 2 read node 0): used sums one_hot(0,3) twice to [2, 0, 0], tf.equal(..., 0) masks in nodes 1 and 2, and the output is their mean — the same case worked by hand, and node N-1 is always in that mask since nothing later can select it as a predecessor, so the mean is never over an empty tensor. The controller op assembles reward = REWARD_CONSTANT / valid_ppl + entropy bonus, updates the baseline under a control dependency so advantage = reward - baseline uses the freshly-updated baseline, and minimizes sample_log_probs * advantage — which, by the sign trace above, is REINFORCE ascent.

What made this work is that the search space was never really a bag of independent networks — it was one DAG that every child subgraphs, so a single shared weight collection can serve the whole controller distribution instead of one per candidate.
