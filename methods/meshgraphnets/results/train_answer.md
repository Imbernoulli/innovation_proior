I want to learn to advance a physical simulation in time — cloth, a deforming elastic plate, water past a cylinder, air over a wing — fast enough to be useful, accurate enough to trust, and general enough to span all of those without rebuilding the model each time. The engineers who do this for real do it on meshes, and for one concrete reason: a mesh lets you put resolution exactly where the physics is hard — tiny oriented elements at an airfoil tip or along a sharp cloth crease, big lazy elements in the smooth empty regions. A uniform grid can only match that accuracy by spending orders of magnitude more cells everywhere. So adaptive discretization is the property I refuse to give up, and it is precisely the property the existing learned simulators give up. The grid camp puts a CNN on a regular grid: hardware-friendly, but it spends uniform resolution and cannot represent an adaptive or deforming mesh at all, undersampling the small high-gradient regions (wakes, boundary layers) that matter most. The particle camp throws the mesh away entirely and connects a free cloud of points by spatial proximity, $|x_i - x_j| < R$. That model already is a graph and already learns fluids, so it is the one worth dismantling carefully.

The radius graph fails in two structural ways. First, it has no rest state. Cloth springs back because the restoring force depends on how far the current spacing of two material points deviates from their natural spacing in the undeformed fabric — but a purely spatial graph only sees where points are right now, so it cannot represent "this edge is stretched 5% past its rest length," cannot represent the elastic force, and on cloth and elastics accumulates error until the rollout diverges. Second, a single radius is the wrong tool for an irregular mesh: on the airfoil mesh, edge lengths span $2\cdot10^{-4}\,\text{m}$ to $3.5\,\text{m}$, so any fixed $R$ drowns the dense wing region in hundreds of neighbours while leaving the far field disconnected. The mesh already carries exactly what the radius graph reconstructs badly — a reference (material-space) coordinate $u_i$ on every node and edges connecting genuine material neighbours — so reconstructing a worse version of that from spatial proximity is throwing away the right answer.

I propose MeshGraphNets. The central move is to make the graph be the simulation mesh: nodes are mesh nodes, edges are mesh edges, and message passing runs along the material connectivity. On each mesh edge I place the reference displacement $u_{ij} = u_i - u_j$, whose magnitude is invariant to how the sheet is currently bent — so the rest state, the very thing the radius graph could not see, is now the most natural feature I have, present on every edge by construction. This is not merely a workaround: a finite-element solver approximates the spatial differential operators of the PDE (gradient, Laplacian) by local stencils over the mesh edges, and the internal dynamics of every system here are governed by such operators — elastic forces are derivatives of displacement, viscous and pressure terms are derivatives of velocity. A learned function that takes an edge's two endpoint features and its relative coordinate, produces a message, and sums those messages at the node has exactly the shape of a discretized differential operator. One round is a first-order operator over the 1-ring; stacking rounds composes higher-order operators reaching further. Mesh-space message passing is a learnable stencil — the same computational structure as the solver, not an analogy to it.

Pure mesh-space, however, discards the one thing the radius graph got right: contact and collision. When the flag folds onto itself, two points adjacent in space are many mesh edges apart along the cloth; when the cloth lands on a separate obstacle, the two meshes share no mesh edges at all, so mesh-space messages can never couple them. So I keep both spaces. Internal dynamics live on mesh edges $E^M$ from the mesh connectivity; external dynamics live on world edges $E^W$ added by spatial proximity, $|x_i - x_j| < r_W$, with $r_W$ on the order of the smallest mesh edge so it catches genuine near-contact without smearing the internal dynamics, and excluding any pair already joined by a mesh edge so the two edge sets do not redundantly double up. The encoder turns a mesh $M^t$ into a multigraph $G = (V, E^M, E^W)$; for an Eulerian fluid on a fixed mesh there is no deforming surface and no contact, so the world edges are simply dropped.

All positional information enters only as relative edge features, never as absolute node coordinates — physics is translation-invariant, and handing the network absolute positions forces it to learn that invariance from data, which it does not, it overfits. A mesh edge therefore carries $u_{ij}$ and $|u_{ij}|$ (the rest geometry, the stencil's spatial argument) together with the current world displacement $x_{ij} = x_i - x_j$ and $|x_{ij}|$ (how stretched the edge is right now — the deformation that drives the elastic force). A world edge carries only $x_{ij}$ and $|x_{ij}|$, since the reference coordinate of an arbitrarily far material point is meaningless across a contact. Node features carry the remaining dynamical quantities $q_i$ plus a one-hot node type distinguishing normal, kinematic/handle, obstacle, inflow/outflow and wall nodes. With everything positional relative, translating the scene leaves every message unchanged — the model is spatially equivariant by construction, which is what buys generalization to new positions and shapes.

The processor generalizes the standard graph-network block to two edge sets, each with its own edge function and its own aggregation, with the node update consuming both aggregates:

$$e'^{M}_{ij} \leftarrow f^M(e^M_{ij}, v_i, v_j), \qquad e'^{W}_{ij} \leftarrow f^W(e^W_{ij}, v_i, v_j), \qquad v'_i \leftarrow f^V\!\Big(v_i, \sum_j e'^{M}_{ij}, \sum_j e'^{W}_{ij}\Big).$$

The aggregation must be a sum, not a mean or a max: physical fluxes are additive — the net force on a node is the sum of neighbour forces, the net mass flux the sum across its edges — and sum is the permutation-invariant aggregator that matches "add up contributions." A mean would wash out node degree (a node with more neighbours genuinely receives more total flux); a max throws away all but one contribution. Each $f$ is an MLP, and every block carries a residual connection, because I stack many of them and residuals let a deep stack train without the signal degrading. Each block propagates information one hop and physics needs multi-hop coupling within a timestep, so too few blocks leaves the receptive field too small to enforce constraints; more blocks help with diminishing returns once the field covers the interaction range. The knee sits at $L = 15$ blocks, which buys essentially all the accuracy for all these systems at reasonable cost, and each block has its own parameters — successive stencil applications do different work (first-order then composed higher-order operators), so tying them would under-fit.

This wraps into Encode-Process-Decode. The encoder is MLPs $\varepsilon^M, \varepsilon^W, \varepsilon^V$ mapping the raw mesh-edge, world-edge and node features into latent vectors of width 128; the processor is the $L$ blocks; the decoder is one MLP $\delta^V$ reading the final node latent into the output $p_i$. The output is not the next state but a derivative: predicting the absolute next position would force the network to relearn "things mostly stay where they are" every step and leave small errors unconstrained, so I predict the change and integrate it with plain forward Euler at $\Delta t = 1$ (the network absorbs the real timestep into its learned scale). A first-order system integrates once, $q^{t+1}_i = p_i + q^t_i$; a second-order system (cloth, where $p_i$ is an acceleration) integrates twice, which in discrete form is $q^{t+1}_i = p_i + 2 q^t_i - q^{t-1}_i$, and needs one step of history, $h=1$, to form the velocity estimate $\dot x^t_i = x^t_i - x^{t-1}_i$. Extra output channels can carry quantities read out directly without integration — pressure in a fluid, von-Mises stress in the plate. Every MLP is two hidden layers, ReLU, width and output 128, with LayerNorm on every output except the decoder: LayerNorm keeps activations well-scaled across fifteen blocks (which matters for training a deep stack), but the decoder must emit a raw physical derivative on its own scale, and forcing that through LayerNorm would fight the integrator.

The piece that actually decides whether any of this works is the gap between training and rollout. I train on single-step prediction, but at test time I feed the model its own output for hundreds or thousands of steps. During training every input is a clean ground-truth state; during rollout, after a few steps the input is the model's own slightly-wrong prediction — off the data manifold, in a region never seen in training — so the error compounds geometrically until the rollout blows up. Rather than the expensive, ill-conditioned route of training through a long rollout, I make the training input distribution look like the rollout distribution: add zero-mean fixed-variance noise to the most recent value of the dynamical variable, only on the predicted (normal) nodes, so the model learns to map a slightly-wrong input to the right next step — exactly the correction skill rollout demands. The noise scale tracks the model's own one-step error (the standard deviation of the per-step target), scanned per dataset around that scale; it lands around $3\cdot10^{-3}$ on position for cloth and $2\cdot10^{-2}$ on momentum for the incompressible flow.

Noise forces a target adjustment, and the signs must be right or the model learns to propagate noise instead of correcting it. For a first-order quantity: if a node's true position is $x^t = 2$, noise makes the input $\tilde x^t = 2.1$, and the true next position is $3$ (true velocity target $1$), then leaving the target at $1$ teaches the model to integrate $2.1 + 1 = 3.1$ and carry the error forward. Subtracting the noise sets the target to $0.9$, so $2.1 + 0.9 = 3.0$ — the model absorbs the noise. The adjusted first-order target is the true derivative minus the input noise. The second-order cloth case cannot be fully satisfied: noise added to the position also corrupts the velocity estimate, since velocity is a difference of positions, so one acceleration target must correct two corrupted inputs. With $x^{t-1}=1.4, x^t=2, x^{t+1}=3$ the true acceleration is $0.4$; adding $0.1$ of noise gives $\tilde x^t = 2.1$ and $\dot{\tilde x}^t = 0.7$. Making the position come out right needs $\ddot{\tilde x}^P = 0.2$ (next position exactly $3$, but next velocity $0.9$ not $1.0$); making the velocity come out right needs $\ddot{\tilde x}^V = 0.3$ (next velocity exactly $1.0$, but next position $3.1$). Because position and velocity share the same noise, no single target fixes both, so I blend them, $\ddot{\tilde x} = \gamma\,\ddot{\tilde x}^P + (1-\gamma)\,\ddot{\tilde x}^V$, and sweep; the best is $\gamma = 0.1$, mostly correcting velocity, lightly correcting position. (For more than one step of history the noise is added as a random walk whose per-step variance makes the last-step variance match the target noise variance.)

Two more training details handle heterogeneous features and warm-up. Inputs and targets span millimetre displacements, large momenta and a one-hot type vector, so I normalize every input and target feature to zero mean and unit variance, accumulating the statistics online during training and stopping after a million updates to avoid numerical drift; the decoder output emerges normalized, so I invert the output normalizer before integrating. The loss is a plain L2 between the network's normalized output and the normalized target derivative, masked to the predicted nodes. I train with Adam, learning rate exponentially decaying from $10^{-4}$ toward $10^{-6}$, and for the first thousand steps I apply no gradients — just let the normalizers accumulate statistics so that when training proper begins the features are already well-scaled.

The last piece delivers on the constraint I refused to give up: adaptivity at inference. Classical adaptive remeshing splits the job — decide where to be fine or coarse, then mechanically make the mesh that way. The "make it" part (split an oversize edge, collapse an edge if it is safe, flip an edge to keep good aspect ratios) is generic, domain-independent machinery. The "where" part is the only domain-specific piece, classically a sizing field $S(u) \in \mathbb{R}^{2\times2}$ giving the maximum allowed edge length in each direction; an edge is valid exactly when $u_{ij}^\top S_i\, u_{ij} \le 1$, and because $S$ is a tensor it can demand short edges across a bend and long edges along it — anisotropic, oriented resolution, exactly what a fold or a boundary layer wants. The obvious approach — call the original simulator's remesher each step — reintroduces the hand-built solver I was replacing, since the remesher belongs to that solver and needs the domain-specific sizing field. So I learn the sizing field too: the same architecture, with a decoder output producing a sizing tensor $S_i$ per node, supervised by L2 against the ground-truth sizing. At test time each step predicts both the next state $\hat M^{t+1}$ and the next sizing $\hat S^{t+1}$ and hands both to a generic local remesher $M^{t+1} = R(\hat M^{t+1}, \hat S^{t+1})$ — no simulator in the loop, the domain knowledge distilled into the learned sizing field.

The generic remesher is concrete. Split an edge when it is invalid under the averaged tensor, $u_{ij}^\top S_{ij}\, u_{ij} > 1$ with $S_{ij} = (S_i + S_j)/2$, inserting a node whose attributes average the endpoints. Collapse an edge only if doing so creates no new invalid edge, so resolution is never coarsened past requirement. Flip an edge under the anisotropy-aware Delaunay test: with the four nodes $i,j,k,\ell$ around the edge and $S_A = (S_i + S_j + S_k + S_\ell)/4$, flip when $(u_{jk}\times u_{ik})\,u_{i\ell}^\top S_A u_{j\ell} < u_{jk}^\top S_A u_{ik}\,(u_{i\ell}\times u_{j\ell})$. The order is: split every splittable edge in descending metric order (refine), flip everything that should flip, collapse everything safe in ascending metric order (coarsen), then a final flip cleanup — domain-independent throughout, only $S$ carrying the physics. Dynamic meshes have no node correspondence between steps, so to build history and targets I interpolate dynamical quantities from the neighbouring meshes onto the current mesh by barycentric interpolation in mesh-space. And when the data exposes no ground-truth sizing field, I estimate it: the sizing field that would have caused exactly the observed transition under a near-optimal remesher makes every resulting edge valid but as long as possible, which for each node is $S_i = \arg\max \sum_{j\in N_i} u_{ij}^\top S_i\, u_{ij}$ subject to $u_{ij}^\top S_i\, u_{ij} \le 1$ for all neighbours. Geometrically the constraint says every neighbour displacement lies inside the ellipse defined by $S_i$, and the objective picks the tightest such origin-centred ellipse — the minimum-area zero-centred ellipse enclosing the points $\{u_{ij}\}$, solved efficiently by Welzl's Minidisk algorithm. So even with no sizing labels, compatible targets are recovered from mesh transitions alone.

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

At inference, each step builds the graph from $M^t$, runs Encode-Process-Decode, integrates to the next state (and optionally a sizing field), then applies $M^{t+1} = R(\hat M^{t+1}, \hat S^{t+1})$, and iterates. The same model rolls out stably for thousands of steps and generalizes to larger, differently-shaped meshes than seen in training, because every position is encoded relatively. Each design choice is forced by removing one failure: dropping mesh edges and $u_{ij}$ erases the rest state and mismatches the four-orders-of-magnitude edge lengths, so the radius graph is unstable on irregular meshes and diverges on cloth; dropping world edges leaves the cloth unable to couple to a separate obstacle and self-contact needing too many hops; using absolute rather than relative encoding overfits the training positions. MeshGraphNets removes all three at once.
