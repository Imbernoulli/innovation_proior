# MeshGraphNets: learning mesh-based simulation with graph networks

## Problem

Learn a fast, accurate, general forward model for mesh-based physical simulation — cloth, hyper-elastic solids, incompressible and compressible flow — that operates natively on irregular and time-varying meshes, captures both the internal (PDE) dynamics and external (contact/collision) dynamics, and keeps the mesh's defining advantage, adaptive resolution, alive at inference without invoking the original solver.

## Key idea

Make the graph **be the simulation mesh**, and run message passing in two coupled spaces:
- **Mesh-space** edges `E^M` come from the mesh connectivity and carry the *reference* displacement `u_ij = u_i − u_j`. Message passing along them acts as a learnable discretized differential operator over the material domain, and the reference coordinate encodes the rest state for free (so elastics/cloth are modeled well).
- **World-space** edges `E^W` are added by spatial proximity (`|x_i − x_j| < r_W`, excluding mesh-connected pairs) for Lagrangian systems, to capture contact and (self-)collision — interactions that are non-local in the mesh.

All positional information enters only as **relative edge features**, making the model spatially equivariant by construction. The network is an **Encode-Process-Decode** graph net; outputs are interpreted as derivatives and advanced by a forward-Euler integrator. Trained with one-step supervision plus **training noise** (with target adjustment) for stable long rollouts, and **online zero-mean/unit-variance normalization**. Adaptivity is made learnable by predicting a **sizing field** with the same architecture and applying a generic, domain-independent local remesher.

## Architecture

Encoder: MLPs `ε^M, ε^W, ε^V` map raw mesh-edge, world-edge and node features into latent vectors of size 128. Mesh-edge features: `u_ij, |u_ij|, x_ij, |x_ij|`. World-edge features: `x_ij, |x_ij|`. Node features: dynamical quantities `q_i` + one-hot node type `n_i`.

Processor: `L = 15` message-passing blocks, each with its own parameters, generalizing the GN block to multiple edge sets:

```
e'^M_ij ← f^M(e^M_ij, v_i, v_j)
e'^W_ij ← f^W(e^W_ij, v_i, v_j)
v'_i    ← f^V(v_i, Σ_j e'^M_ij, Σ_j e'^W_ij)
```

with `f^M, f^W, f^V` MLPs, **sum** aggregation, and residual connections.

Decoder: MLP `δ^V` maps the final node latent to outputs `p_i` (derivatives + optional direct predictions such as pressure/stress).

Integrator (forward Euler, Δt = 1):
- first-order: `q^{t+1}_i = p_i + q^t_i`
- second-order: `q^{t+1}_i = p_i + 2 q^t_i − q^{t−1}_i`

All MLPs: 2 hidden layers, ReLU, width/output 128, LayerNorm on every output **except** the decoder. Inputs and targets normalized online to zero mean, unit variance.

## Training

One-step L2 loss between the network output and the normalized target derivative, masked to predicted (normal/outflow) nodes. Adam, learning rate exponentially decayed from `1e-4` to `1e-6` over `5e6` steps; the first ~1000 steps only accumulate normalization statistics (no gradient updates).

**Training noise** (key to rollout stability): add zero-mean fixed-variance noise to the most recent dynamical value of predicted nodes so the training input distribution matches the noisy rollout distribution. Adjust the targets so the integrated output cancels the noise:
- first-order: subtract the noise from the target (`gamma = 1.0`);
- second-order (cloth): position and velocity inputs share the same noise and cannot both be corrected — blend `ẍ̃ = γ ẍ̃^P + (1−γ) ẍ̃^V` with `γ = 0.1` (mostly velocity correction). Noise scales are found by scanning around the one-step error: e.g. cloth `3e-3` on position, incompressible flow `2e-2` on momentum.

## Learned adaptive remeshing

The sizing field `S(u) ∈ R^{2×2}` (the only domain-specific part of remeshing) defines edge validity: an edge is valid iff `u_ij^T S_i u_ij ≤ 1`. Predict `S` per node with the same architecture (L2 on ground-truth sizing). At test time, predict next state `M̂^{t+1}` and sizing `Ŝ^{t+1}`, then apply a generic local remesher `M^{t+1} = R(M̂^{t+1}, Ŝ^{t+1})`:
- **split** an edge if `u_ij^T S_ij u_ij > 1`, `S_ij = (S_i + S_j)/2` (refine);
- **collapse** an edge if it creates no new invalid edge (coarsen);
- **flip** an edge per the anisotropy-aware Delaunay criterion (Bossen & Heckbert 1996), `(u_jk × u_ik) u_il^T S_A u_jl < u_jk^T S_A u_ik (u_il × u_jl)` with `S_A = (S_i + S_j + S_k + S_ℓ)/4`.

Order: split (descending metric) → flip → collapse (ascending metric) → flip.

**Estimating sizing targets** when no ground-truth sizing exists: for each node, `S_i = argmax Σ_{j∈N_i} u_ij^T S_i u_ij` s.t. `∀j: u_ij^T S_i u_ij ≤ 1`, i.e. the minimum-area zero-centred ellipse containing the neighbour displacements `{u_ij}`, solved with Welzl's Minidisk algorithm.

## Code

```python
import collections, functools
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GraphNetBlock(snt.AbstractModule):
  """Multi-edge interaction block with residual connections."""

  def __init__(self, model_fn, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn

  def _update_edge_features(self, node_features, edge_set):
    sender = tf.gather(node_features, edge_set.senders)
    receiver = tf.gather(node_features, edge_set.receivers)
    features = tf.concat([sender, receiver, edge_set.features], axis=-1)
    with tf.variable_scope(edge_set.name + '_edge_fn'):
      return self._model_fn()(features)

  def _update_node_features(self, node_features, edge_sets):
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:                              # SUM aggregation per edge set
      features.append(tf.math.unsorted_segment_sum(
          edge_set.features, edge_set.receivers, num_nodes))
    with tf.variable_scope('node_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _build(self, graph):
    new_edge_sets = [es._replace(features=self._update_edge_features(graph.node_features, es))
                     for es in graph.edge_sets]
    new_node_features = self._update_node_features(graph.node_features, new_edge_sets)
    new_node_features += graph.node_features                # residual connections
    new_edge_sets = [es._replace(features=es.features + old.features)
                     for es, old in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode graph net."""

  def __init__(self, output_size, latent_size, num_layers, message_passing_steps,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  def _make_mlp(self, output_size, layer_norm=True):
    widths = [self._latent_size] * self._num_layers + [output_size]
    net = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      net = snt.Sequential([net, snt.LayerNorm()])
    return net

  def _encoder(self, graph):
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edge_sets = [es._replace(features=self._make_mlp(self._latent_size)(es.features))
                       for es in graph.edge_sets]
    return MultiGraph(node_latents, new_edge_sets)

  def _decoder(self, graph):
    with tf.variable_scope('decoder'):
      return self._make_mlp(self._output_size, layer_norm=False)(graph.node_features)

  def _build(self, graph):
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn)(latent_graph)
    return self._decoder(latent_graph)
```

```python
class Normalizer(snt.AbstractModule):
  """Online zero-mean / unit-variance feature normalizer."""

  def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer'):
    super(Normalizer, self).__init__(name=name)
    self._max_accumulations = max_accumulations
    self._std_epsilon = std_epsilon
    with self._enter_variable_scope():
      self._acc_count = tf.Variable(0., trainable=False)
      self._num_accumulations = tf.Variable(0., trainable=False)
      self._acc_sum = tf.Variable(tf.zeros(size), trainable=False)
      self._acc_sum_squared = tf.Variable(tf.zeros(size), trainable=False)

  def _build(self, data, accumulate=True):
    op = tf.no_op()
    if accumulate:
      op = tf.cond(self._num_accumulations < self._max_accumulations,
                   lambda: self._accumulate(data), tf.no_op)
    with tf.control_dependencies([op]):
      return (data - self._mean()) / self._std()

  @snt.reuse_variables
  def inverse(self, data):
    return data * self._std() + self._mean()

  def _accumulate(self, data):
    return tf.group(
        tf.assign_add(self._acc_sum, tf.reduce_sum(data, axis=0)),
        tf.assign_add(self._acc_sum_squared, tf.reduce_sum(data**2, axis=0)),
        tf.assign_add(self._acc_count, tf.cast(tf.shape(data)[0], tf.float32)),
        tf.assign_add(self._num_accumulations, 1.))

  def _mean(self):
    return self._acc_sum / tf.maximum(self._acc_count, 1.)

  def _std(self):
    var = self._acc_sum_squared / tf.maximum(self._acc_count, 1.) - self._mean()**2
    return tf.maximum(tf.sqrt(var), self._std_epsilon)
```

```python
class NodeType:                       # one-hot node-type categories
  NORMAL, OBSTACLE, AIRFOIL, HANDLE, INFLOW, OUTFLOW, WALL_BOUNDARY = range(7)
  SIZE = 9


def triangles_to_edges(faces):
  """Bidirectional, deduplicated mesh edges from triangle faces."""
  edges = tf.concat([faces[:, 0:2], faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  unique = tf.bitcast(tf.unique(packed)[0], tf.int32)
  senders, receivers = tf.unstack(unique, axis=1)
  return (tf.concat([senders, receivers], 0), tf.concat([receivers, senders], 0))


class ClothModel(snt.AbstractModule):
  """Lagrangian, second-order cloth. (World edges are added as a second EdgeSet identically.)"""

  def __init__(self, learned_model, name='Model'):
    super(ClothModel, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = Normalizer(size=3, name='output_normalizer')
      self._node_normalizer = Normalizer(size=3 + NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = Normalizer(size=7, name='edge_normalizer')  # x_ij+|x_ij|+u_ij+|u_ij|

  def _build_graph(self, inputs, is_training):
    velocity = inputs['world_pos'] - inputs['prev|world_pos']          # h=1 velocity estimate
    node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)
    senders, receivers = triangles_to_edges(inputs['cells'])
    rel_world = tf.gather(inputs['world_pos'], senders) - tf.gather(inputs['world_pos'], receivers)
    rel_mesh = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
    edge_features = tf.concat([
        rel_world, tf.norm(rel_world, axis=-1, keepdims=True),
        rel_mesh, tf.norm(rel_mesh, axis=-1, keepdims=True)], axis=-1)
    mesh_edges = EdgeSet('mesh_edges', self._edge_normalizer(edge_features, is_training),
                         receivers, senders)
    return MultiGraph(self._node_normalizer(node_features, is_training), [mesh_edges])

  @snt.reuse_variables
  def loss(self, inputs):
    graph = self._build_graph(inputs, is_training=True)
    out = self._learned_model(graph)
    target_acc = inputs['target|world_pos'] - 2*inputs['world_pos'] + inputs['prev|world_pos']
    target = self._output_normalizer(target_acc)
    mask = tf.equal(inputs['node_type'][:, 0], NodeType.NORMAL)
    err = tf.reduce_sum((target - out)**2, axis=1)
    return tf.reduce_mean(err[mask])

  def _update(self, inputs, out):
    acc = self._output_normalizer.inverse(out)
    return 2*inputs['world_pos'] + acc - inputs['prev|world_pos']      # integrate twice

  def _build(self, inputs):
    return self._update(inputs, self._learned_model(self._build_graph(inputs, is_training=False)))


class CFDModel(snt.AbstractModule):
  """Eulerian, first-order fluid on a fixed mesh (no world edges)."""

  def __init__(self, learned_model, name='Model'):
    super(CFDModel, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = Normalizer(size=2, name='output_normalizer')
      self._node_normalizer = Normalizer(size=2 + NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = Normalizer(size=3, name='edge_normalizer')  # u_ij+|u_ij|

  def _build_graph(self, inputs, is_training):
    node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
    node_features = tf.concat([inputs['velocity'], node_type], axis=-1)
    senders, receivers = triangles_to_edges(inputs['cells'])
    rel_mesh = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
    edge_features = tf.concat([rel_mesh, tf.norm(rel_mesh, axis=-1, keepdims=True)], axis=-1)
    mesh_edges = EdgeSet('mesh_edges', self._edge_normalizer(edge_features, is_training),
                         receivers, senders)
    return MultiGraph(self._node_normalizer(node_features, is_training), [mesh_edges])

  @snt.reuse_variables
  def loss(self, inputs):
    graph = self._build_graph(inputs, is_training=True)
    out = self._learned_model(graph)
    target = self._output_normalizer(inputs['target|velocity'] - inputs['velocity'])
    node_type = inputs['node_type'][:, 0]
    mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL), tf.equal(node_type, NodeType.OUTFLOW))
    err = tf.reduce_sum((target - out)**2, axis=1)
    return tf.reduce_mean(err[mask])

  def _update(self, inputs, out):
    return inputs['velocity'] + self._output_normalizer.inverse(out)   # integrate once
```

```python
def add_noise(frame, field, scale, gamma):
  """One-step training noise with derived target adjustment."""
  noise = tf.random.normal(tf.shape(frame[field]), stddev=scale)
  mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
  noise = tf.where(mask, noise, tf.zeros_like(noise))                  # corrupt predicted nodes only
  frame[field] += noise
  frame['target|' + field] += (1.0 - gamma) * noise                   # gamma=1.0 first-order; 0.1 cloth
  return frame


PARAMETERS = {
    'cfd':   dict(noise=2e-2, gamma=1.0, field='velocity',  history=False, size=2, batch=2),
    'cloth': dict(noise=3e-3, gamma=0.1, field='world_pos', history=True,  size=3, batch=1),
}


def build_and_optimize(params):
  learned_model = EncodeProcessDecode(output_size=params['size'], latent_size=128,
                                      num_layers=2, message_passing_steps=15)
  model = params['model'].Model(learned_model)
  step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(1e-4, step, decay_steps=int(5e6), decay_rate=0.1) + 1e-6
  loss_op = model.loss(next_inputs)
  train_op = tf.train.AdamOptimizer(lr).minimize(loss_op, global_step=step)
  train_op = tf.cond(tf.less(step, 1000),                              # warm up normalizer stats only
                     lambda: tf.group(tf.assign_add(step, 1)),
                     lambda: tf.group(train_op))
  return train_op
```

```python
def remesh(mesh, sizing):
  """Generic, domain-independent local remesher R(M, S)."""
  split_invalid_edges(mesh, sizing)        # size = u_ij^T ((S_i+S_j)/2) u_ij > 1, descending
  flip_anisotropic_delaunay(mesh, sizing)  # Bossen-Heckbert criterion
  collapse_safe_edges(mesh, sizing)        # only if no new invalid edge is created, ascending
  flip_anisotropic_delaunay(mesh, sizing)  # final cleanup
  return mesh


def estimate_sizing(mesh, i):
  """Min-area origin-centred ellipse containing neighbour displacements (Welzl Minidisk)."""
  return min_area_enclosing_ellipse([mesh.u[j] - mesh.u[i] for j in mesh.neighbors(i)])
```

At inference, each step: build the graph from `M^t`, run Encode-Process-Decode, integrate to get the next state (and optionally a sizing field), then `M^{t+1} = R(M̂^{t+1}, Ŝ^{t+1})`; iterate. The same model rolls out stably for thousands of steps and generalizes to larger, differently-shaped meshes than seen in training because every position is encoded relatively.
