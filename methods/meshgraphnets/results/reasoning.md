Let me start from the thing that actually hurts. I want to learn to advance a physical simulation in time — cloth, a deforming elastic plate, water past a cylinder, air over a wing — fast enough to be useful and accurate enough to trust, and I want it to work across all of those without rebuilding the model each time. The engineers who do this for real do it on meshes, and they do it on meshes for one concrete reason: a mesh lets you put resolution exactly where the physics is hard. Tiny oriented elements at the airfoil tip and along a sharp crease in cloth; big lazy elements in the smooth empty regions. A uniform grid can only match that accuracy by spending orders of magnitude more cells everywhere. So if I'm going to learn physics and I care about the airfoil tip and the cloth fold, I have to keep the mesh and its adaptivity. That's the constraint I refuse to give up, and it's the constraint everyone else seems to give up.

Because what does the learned-physics literature actually do? Two camps. The grid camp puts a CNN on a regular grid — popular because of hardware support, but it spends uniform resolution and can't represent an adaptive or a deforming mesh at all. The particle camp throws the mesh away entirely and works on a free cloud of points connected by spatial proximity. The particle approach is the more interesting one to me because it's already a graph and already learns, so let me think hard about why it isn't just the answer.

The particle model — call it the radius-graph simulator — builds its graph each step by connecting any two particles within a fixed radius R of each other in space, then runs message passing and reads out an acceleration. It's a clean encode-process-decode pipeline and it genuinely learns fluids. So why not just run it on my mesh nodes? Let me actually try to picture it on cloth. A flag flaps; two parts of the sheet fold over and nearly touch. In world space those two points are within R, so the radius graph connects them — fine, that's collision, I want that. But now picture the resting, undeformed sheet. Two material points an inch apart along the fabric: how does the radius graph know they're an inch apart in the *cloth*? It doesn't. It only sees where they are in space right now. The fabric has a rest length between every pair of neighbours, and the restoring force — the entire reason cloth springs back instead of crumpling to a point — is a function of how much the current spacing deviates from that rest spacing. A purely spatial graph has erased the rest state. It can't represent "this edge is stretched 5% past its natural length," so it can't represent the elastic force, so on cloth and elastics it accumulates error and the rollout eventually diverges. That's the first wall, and it's structural, not a tuning problem.

There's a second problem with the radius graph that's about the mesh being irregular. My airfoil mesh has edges from `2·10⁻⁴ m` near the wing to `3.5 m` out in the far field. Pick any single radius R. Near the wing, R covers hundreds of nodes — the graph is absurdly dense, message passing drowns. In the far field, R covers nobody — the graph is disconnected, information can't propagate. A fixed radius is fundamentally the wrong tool for a discretization whose whole point is that spacing varies by four orders of magnitude. So even setting aside the rest-state issue, radius connectivity and adaptive meshes are at war.

The mesh already encodes the rest-state geometry and the right local neighbourhood — that's literally what a finite-element mesh *is*. Each node has a reference coordinate `u_i` in the undeformed material domain, and the mesh edges connect material neighbours. Why am I throwing that away to reconstruct a worse version of it from spatial proximity? I shouldn't build the graph from spatial radius at all. I should make the graph *be the simulation mesh*: nodes are mesh nodes, edges are mesh edges. Then message passing happens along the material connectivity, and crucially I can put the *reference* displacement `u_ij = u_i - u_j` on each edge. Distance in `u` is invariant to how the cloth is currently bent — it's the rest geometry — so the rest state is encoded for free, on every edge, by construction. The thing the radius graph couldn't see is now the most natural feature I have.

And there's a deeper reason message passing on mesh edges is the right primitive, not just a workaround. What is a finite-element solver actually doing? It approximates the spatial differential operators of the PDE — the gradient, the Laplacian — by local stencils over the mesh edges. The internal dynamics of basically every physical system I listed are governed by such differential operators: elastic forces are derivatives of displacement, viscous and pressure terms are derivatives of the velocity field. A learned function that takes an edge's endpoint features and the relative coordinate `u_ij` and produces a message, then sums those messages at a node — that's exactly the shape of a discretized differential operator. Message passing along mesh edges *is* a learnable stencil. One round computes a first-order operator over the 1-ring; stacking rounds reaches further and composes higher-order operators. So mesh-space message passing isn't an analogy to the PDE, it's the same computational structure. Good. That's the spine.

But wait — I just talked myself into mesh-only message passing, and that throws out the one thing the radius graph got right. Collision and contact. When the flag folds onto itself, two points that are adjacent in *space* are far apart along the *cloth* — many mesh edges away. Mesh-space message passing would need a huge number of hops to connect them, and worse, when cloth lands on a separate obstacle (a sphere) the two meshes aren't connected by mesh edges *at all*, so mesh-space messages can *never* couple them. So pure mesh-space can't do collision. Let me not lose the radius graph's virtue; let me keep both. Internal dynamics live in mesh-space, along `u`-edges. External dynamics — contact, (self-)collision — live in world-space, and *there* the spatial-proximity idea is exactly right. So I'll have two kinds of edges: the mesh edges `E^M` that come from the mesh's own connectivity, and **world edges** `E^W` that I add by spatial proximity in `x`. I'll add a world edge between `i` and `j` when `|x_i - x_j| < r_W`, with `r_W` on the order of the *smallest* mesh edge length so it captures genuine near-contact without smearing the internal dynamics, and I'll exclude any pair that's already connected by a mesh edge so the two edge sets don't redundantly double up. So the encoder turns the mesh `M^t` into a multigraph `G = (V, E^M, E^W)`. (For an Eulerian fluid on a fixed mesh there's no deforming surface and no contact, so I just drop the world edges — they only exist for Lagrangian systems.)

Now the features. I learned the hard way from the radius-graph people that *absolute* positions are poison for generalization: physics is translation-invariant, and if I hand the network absolute node coordinates it has to learn that invariance from data, it doesn't, it overfits to the positions it saw. So put positions in only as *relative* edge features, and let nodes carry only the dynamical quantities. Concretely on a mesh edge `e^M_ij` I encode the reference displacement `u_ij = u_i - u_j` and its norm `|u_ij|` (the rest geometry / the stencil's spatial argument), and I also encode the *current* world-space displacement `x_ij = x_i - x_j` and its norm `|x_ij|` (so the edge sees how stretched it currently is — that's the deformation that drives the elastic force). On a world edge `e^W_ij` I encode `x_ij` and `|x_ij|` only — a world edge is about spatial closeness, the reference coordinate of an arbitrarily-far material point is meaningless there. Node features `v_i` carry the remaining dynamical quantities `q_i` plus a one-hot **node type** vector so the model can tell a normal simulated node from a kinematic handle, an obstacle, an inflow/outflow or a wall. Everything positional is relative, so the whole thing is spatially equivariant by construction; translate the scene and every message is unchanged. That should give me the generalization-to-new-positions-and-shapes that absolute encoding kills.

Let me write down the message-passing block precisely, because I want it to generalize the standard graph-network block to my two edge sets. The standard GN block does: per-edge update `e'_k = φ^e(e_k, v_{sender}, v_{receiver})`, then aggregate the updated edges into each node `ē'_i = ρ({e'_k : receiver = i})`, then per-node update `v'_i = φ^v(ē'_i, v_i)`. I have two edge sets, so each gets its own edge function and its own aggregation, and the node update consumes both aggregates:

  e'^M_ij ← f^M(e^M_ij, v_i, v_j)
  e'^W_ij ← f^W(e^W_ij, v_i, v_j)
  v'_i  ← f^V(v_i, Σ_j e'^M_ij, Σ_j e'^W_ij)

The aggregation has to be a sum, not a mean or a max. Physical fluxes are additive — the net force on a node is the sum of forces from its neighbours, the net mass flux is the sum of fluxes across its edges — and sum is the permutation-invariant aggregator that matches "add up contributions." A mean would wash out node degree (a node with more neighbours genuinely receives more total flux); a max throws away all but one contribution. Sum it is. Each `f` is an MLP, and I want a residual connection across the block — `v'_i` and `e'` add to their inputs — because I'm going to stack a lot of these blocks and residual connections are what let deep stacks train without the signal degrading.

How many blocks? Each block propagates information one hop, and physics needs multi-hop coupling — a constraint at one node has to be felt several nodes away within a single timestep, a pressure disturbance has to travel. Too few blocks and the receptive field is too small to propagate constraints; the model literally can't see far enough to be consistent. More blocks always help in principle but cost compute linearly and give diminishing returns once the receptive field covers the relevant interaction range. Scanning this, there's a clear knee: around fifteen message-passing blocks buys essentially all the accuracy for all these systems at a reasonable cost. So `L = 15`, and — important — each block has its *own* parameters, not shared, because successive "stencil applications" are doing different work (first-order then composed higher-order operators) and tying them would under-fit.

Wrap this in encode-process-decode. The encoder is three MLPs: `ε^M` for mesh-edge features, `ε^W` for world-edge features, `ε^V` for node features, each mapping the concatenated raw features into a latent vector of size 128. The processor is the `L` blocks above. The decoder is one MLP `δ^V` that reads the final node latent into the output `p_i`. What should `p_i` be? Not the next state directly. If I predict the absolute next position the network has to relearn "things mostly stay where they are" every step, and small errors are unconstrained. Predict the *change* — a derivative — and integrate it. So `p_i` is interpreted as a (higher-order) derivative of the dynamical quantity, and I apply a plain forward-Euler integrator with Δt = 1 (the network can absorb the real Δt into its learned scale). For a first-order system, integrate once: `q^{t+1}_i = p_i + q^t_i`. For a second-order system (cloth, where `p_i` is an acceleration), integrate twice, which in discrete form is `q^{t+1}_i = p_i + 2 q^t_i - q^{t-1}_i`. And `p_i` can carry extra channels for quantities I want to read out directly without integrating — pressure in a fluid, von-Mises stress in the plate — those are just direct predictions. Then I update the mesh nodes with `q^{t+1}` to get `M^{t+1}`. For the second-order case I need one step of history to form the current velocity estimate `ẋ^t_i = x^t_i - x^{t-1}_i`, so `h = 1` for cloth; everything else needs no history, `h = 0`.

Let me pin the MLP shapes. All these encoder/processor/decoder MLPs are two-hidden-layer, ReLU, with layer and output size 128 — enough capacity, and the model turns out insensitive to width and depth so I don't need to fuss. Every MLP output is normalized by a LayerNorm — *except* the decoder `δ^V`. Why except the decoder? LayerNorm stabilizes the latent message-passing by keeping activations well-scaled across fifteen blocks, which matters a lot for training a deep stack; but the decoder's job is to emit a raw physical derivative on its own scale, and forcing that through a LayerNorm would fight the integrator. So LayerNorm everywhere internal, none on the final readout.

Now I have to confront the thing that actually decides whether any of this works: I train on single-step prediction, but at test time I roll out for hundreds, even thousands of steps, feeding the model its own output as the next input. This is where one-step-trained simulators die. Think about the distribution mismatch. During training, every input is a clean ground-truth state. During rollout, after a few steps the input is the model's own slightly-wrong prediction — it's off the data manifold, in a region the model never saw in training, so it makes a larger error, which pushes the next input further off-manifold, and the error compounds geometrically until the rollout blows up. The model is never trained on the kind of corrupted input it actually has to consume.

Hit a wall here — let me stop and think about the right fix rather than reaching for a longer rollout loss (training through a long rollout is expensive, the gradients are nasty, and it couples timesteps I'd rather keep decoupled). The cleaner fix: deliberately corrupt the training inputs so the *training* input distribution looks like the *rollout* input distribution. Add zero-mean random noise of a fixed variance to the most recent value of the dynamical variable at training time. Now the model learns to map a slightly-wrong input to the right next step — exactly the correction skill it needs during rollout. How much noise? The natural scale is the model's own one-step error, which tracks the standard deviation of the per-step target; so I look at that and scan the noise magnitude around it on a log scale, a couple of values per decade, per dataset. For the cloth flag it lands around `3e-3` on position; for the cylinder fluid around `2e-2` on momentum; and so on. Only corrupt the nodes the model actually predicts — not the boundary / kinematic nodes, those are given.

But adding noise creates a subtle bookkeeping problem with the targets, and I have to get the signs right or I'll teach the model to *propagate* noise instead of *correct* it. Take a first-order system. Suppose a node's true position is `x^t = 2`, and after noise the input becomes `x̃^t = 2.1`. The true next position is `x^{t+1} = 3`, so the true velocity target is `ẋ = 1`. If I leave the target at 1, the model integrates `x̃^t + ẋ = 2.1 + 1 = 3.1` — it learns to carry the +0.1 error forward. Wrong. I want the model, *given the noisy input*, to land on the correct next state. So I adjust the target by subtracting the noise: target becomes `0.9`, and then `x̃^t + 0.9 = 2.1 + 0.9 = 3.0 = x^{t+1}`. The model learns to absorb the noise. In general the adjusted target is the true derivative minus the input noise — that's the rule for any first-order quantity.

The second-order (cloth) case is genuinely harder, and I have to be honest that I can't satisfy everything. Here the model takes the position `x^t` and the velocity estimate `ẋ^t = x^t - x^{t-1}`, and outputs acceleration `ẍ`, integrated twice. I add noise to the *position* `x^t`, which automatically corrupts the velocity estimate too, since velocity is a difference of positions. So a single noise injection corrupts both the position input and the velocity input, and I'd like the adjusted acceleration target to correct both. Let me do the arithmetic and see if I can. Say `x^{t-1} = 1.4`, `x^t = 2`, `x^{t+1} = 3`. Then `ẋ^t = 0.6`, `ẋ^{t+1} = 1`, and the true acceleration is `ẍ = ẋ^{t+1} - ẋ^t = 0.4`. Add 0.1 of noise to the position: `x̃^t = 2.1`, so `ẋ̃^t = 2.1 - 1.4 = 0.7`. Now I want an adjusted acceleration target. Option P: make the *position* come out right. Set `ẍ̃^P = 0.2`, so the next velocity is `ẋ̃^{t+1} = ẋ̃^t + ẍ̃^P = 0.7 + 0.2 = 0.9`, and the next position is `x̃^{t+1} = x̃^t + ẋ̃^{t+1} = 2.1 + 0.9 = 3.0 = x^{t+1}`. Position fixed — but the predicted next velocity is 0.9, not the true 1.0. Option V: make the *velocity* come out right. Set `ẍ̃^V = 0.3`, so `ẋ̃^{t+1} = 0.7 + 0.3 = 1.0 =` the true next velocity — but then the next position is `2.1 + 1.0 = 3.1 ≠ 3`. So because position and velocity inputs share the same noise, no single acceleration target corrects both at once; the strong coupling makes it impossible to satisfy both. Fine — treat the choice as a knob. Blend the two corrections with a weight `γ ∈ [0,1]`: `ẍ̃ = γ ẍ̃^P + (1-γ) ẍ̃^V`. Sweep it; the best is `γ = 0.1`, i.e. mostly correct the velocity, lightly correct the position. (And if I ever take more than one step of history, I add the noise as a random walk with per-step variance chosen so the variance at the last step matches the target noise variance — so multi-step history is corrupted consistently rather than all-at-once.)

One more training-side thing, and it's small but it matters for heterogeneous features. My inputs and targets are on wildly different scales — millimetre displacements next to large momenta next to a one-hot type vector. Unnormalized, training is slow and the loss is dominated by the big-magnitude channels. So normalize every input and target feature to zero mean and unit variance. I don't have global dataset statistics handy and I don't want a separate pass, so accumulate the mean and variance *online* during training, element-wise, and stop accumulating after a large number of updates (a million) to avoid numerical drift once the estimate has converged. The decoder output comes out normalized, so I invert the output normalizer before integrating — the integrator needs the raw physical derivative. The loss itself is then simply an L2 between the network's normalized output and the normalized target derivative, masked to the predicted (normal) nodes only. Train with Adam, learning rate decaying exponentially from `1e-4` toward `1e-6`, and — a nice trick — don't apply gradients for the first thousand steps, just let the normalizers accumulate statistics so that when training proper begins the features are already well-scaled.

Now the last piece, and it's the one that makes this actually deliver on the promise I refused to give up at the start: adaptivity at test time. So far I predict the next state on a *fixed* mesh. But the whole reason to use a mesh is to move resolution around as the simulation evolves — refine where a new cloth fold appears, coarsen where the fabric flattens. Classical adaptive remeshing splits the job in two: decide *where* to be fine or coarse, then mechanically *make* the mesh that way. The "make it" part — split an oversize edge, collapse an edge if it's safe, flip an edge to keep good aspect ratios — is generic, domain-independent machinery. The "where" part is the only thing that needs domain knowledge, and it's classically encoded as a **sizing field**: a tensor `S(u) ∈ R^{2×2}` at each point saying the maximum allowed edge length in each direction. An edge `u_ij` is valid exactly when `u_ij^T S_i u_ij ≤ 1`; bigger than that, it's too long and must be split. Because `S` is a tensor and not a scalar, it can demand short edges across a bend and long edges along it — anisotropic, oriented resolution, which is precisely what you want for a fold or a boundary layer.

Here's the trap I have to avoid. The obvious thing is: at each rollout step, predict the next state, then call the original simulator's remesher to adapt the mesh. But the remesher needs the sizing field, and the sizing field is the *domain-specific heuristic* — and the remesher is part of the original solver. If I call it every step, I've reintroduced the hand-built, system-specific solver I was trying to replace. That defeats the entire point.

The only domain-specific thing is the sizing field, so *learn the sizing field too*. I already have a network architecture that maps a mesh to per-node outputs — use the *same* architecture, add a decoder output that produces a sizing tensor `S_i` at each node, and supervise it with an L2 loss against the ground-truth sizing field. Then at test time, each step I predict both the next state `M̂^{t+1}` and the next sizing field `Ŝ^{t+1}`, and I hand both to a *generic, domain-independent* local remesher `R` to get the adapted mesh: `M^{t+1} = R(M̂^{t+1}, Ŝ^{t+1})`. No simulator in the loop. The domain knowledge has been distilled into the learned sizing field; the remesher is universal.

Let me make the generic remesher concrete for triangular meshes, because "generic" has to mean I actually wrote down the three operations and their conditions. Split: an edge between `i` and `j` should be split when it's invalid under the averaged sizing tensor, `u_ij^T S_ij u_ij > 1` with `S_ij = (S_i + S_j)/2`; a split inserts a new node whose attributes — position, sizing tensor — are the average of the two endpoints. Collapse: an edge should be collapsed (removing a node, coarsening) only if doing so creates no new invalid edge, so I never coarsen past the required resolution. Flip: an edge should be flipped to improve aspect ratio when the anisotropy-aware Delaunay criterion is met — with the four nodes `i, j, k, ℓ` around the edge and `S_A = (S_i + S_j + S_k + S_ℓ)/4`, flip when `(u_jk × u_ik) u_il^T S_A u_jl < u_jk^T S_A u_ik (u_il × u_jl)` — this is the standard pliant anisotropic-Delaunay test and it optimizes the directional aspect ratio under the metric. I sequence them: first split every splittable edge in descending order of `u_ij^T S_ij u_ij` to refine, then flip everything that should be flipped, then collapse everything I safely can in ascending order of the same metric to coarsen as much as possible, then a final flip pass to clean up. Domain-independent throughout — only `S` carried the physics.

There's a wrinkle I should handle: the dynamic meshes have no node correspondence between steps — after remeshing, node `i` at time `t` may not exist at `t+1`. To build per-node history and targets I interpolate the dynamical quantities from the neighbouring meshes (`M^{t-1}` and `M^{t+1}`) onto the current mesh `M^t` using barycentric interpolation in mesh-space. And there's a second wrinkle: what if the training data doesn't expose a ground-truth sizing field at all — the simulator never wrote one out? Then I have to *estimate* it from the meshes themselves. Given two consecutive meshes `M^t` and `M^{t+1}`, I want the sizing field `S` that would have caused exactly that transition under the remesher: `M^{t+1} = R(M^t, S)`. Assume the remesher is near-optimal — every resulting edge is valid (size ≤ 1) but as long as possible (right at the boundary). For each node `i` with mesh neighbours `N_i`, that's the optimization

  S_i = argmax  Σ_{j ∈ N_i} u_ij^T S_i u_ij   s.t.  ∀ j ∈ N_i:  u_ij^T S_i u_ij ≤ 1.

Stare at the constraint geometrically. `u_ij^T S_i u_ij ≤ 1` for a positive-definite `S_i` says the point `u_ij` lies inside the ellipse defined by `S_i`. Doing this for all neighbours says: find the ellipse (centred at the origin) that contains all the neighbour displacement points `{u_ij}`. The "argmax of the sizes subject to validity" then picks the *tightest* such ellipse — the minimum-area zero-centred ellipse containing the points. And that's a classic computational-geometry problem: smallest enclosing ellipse, solvable efficiently by Welzl's Minidisk algorithm. So even with no sizing labels, I can recover compatible targets from mesh transitions alone.

Let me also sanity-check that this whole construction explains, in advance, the failure modes I started from — not as new experiments, just as consequences. The radius-graph model has no `u` and no mesh edges, so it has no rest-state feature and a fixed radius mismatched to the four-orders-of-magnitude edge lengths: it should be unstable on irregular and dynamic meshes and should diverge on cloth — consistent with what's documented. Mesh-only (no world edges) can't connect the cloth mesh to a separate obstacle mesh and needs too many hops for self-contact, so collision-heavy cases should get noticeably worse — consistent. Absolute (non-relative) encoding should overfit the positions and generalize badly — consistent. Each of my design choices is forced by removing one of these failures.

Now let me write the actual code, faithful to the structure I derived: a core encode-process-decode graph net with a multi-edge GN block, plus per-domain wrappers that build the right graph and integrate the right order, plus the online normalizer and the noise-injecting data pipeline.

```python
import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

# A typed edge set (mesh edges, world edges, ...) and a graph with several of them.
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GraphNetBlock(snt.AbstractModule):
  """One message-passing block: per-edge-set update, sum-aggregate, node update, residual."""

  def __init__(self, model_fn, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn  # builds a fresh MLP

  def _update_edge_features(self, node_features, edge_set):
    # e'_ij = f(e_ij, v_i, v_j): concat endpoint node features with the edge.
    sender = tf.gather(node_features, edge_set.senders)
    receiver = tf.gather(node_features, edge_set.receivers)
    features = tf.concat([sender, receiver, edge_set.features], axis=-1)
    with tf.variable_scope(edge_set.name + '_edge_fn'):
      return self._model_fn()(features)

  def _update_node_features(self, node_features, edge_sets):
    # v'_i = f(v_i, sum_j e'^M_ij, sum_j e'^W_ij): SUM aggregation per edge set.
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(
          edge_set.features, edge_set.receivers, num_nodes))
    with tf.variable_scope('node_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _build(self, graph):
    # update every edge set, then nodes
    new_edge_sets = [es._replace(features=self._update_edge_features(graph.node_features, es))
                     for es in graph.edge_sets]
    new_node_features = self._update_node_features(graph.node_features, new_edge_sets)
    # residual connections across the block (essential for stacking ~15 blocks)
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old.features)
                     for es, old in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encoder (per node/edge-set MLP) -> L message-passing blocks -> decoder MLP."""

  def __init__(self, output_size, latent_size, num_layers, message_passing_steps,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  def _make_mlp(self, output_size, layer_norm=True):
    # two-hidden-layer ReLU MLP of width 128; LayerNorm on every output EXCEPT the decoder.
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    # encode node features and each edge set's features into latent vectors of size 128
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edge_sets = [es._replace(features=self._make_mlp(self._latent_size)(es.features))
                       for es in graph.edge_sets]
    return MultiGraph(node_latents, new_edge_sets)

  def _decoder(self, graph):
    # read out the per-node derivative p_i; NO layer norm here (raw physical scale)
    with tf.variable_scope('decoder'):
      return self._make_mlp(self._output_size, layer_norm=False)(graph.node_features)

  def _build(self, graph):
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)
    for _ in range(self._message_passing_steps):     # L=15 separate-parameter blocks
      latent_graph = GraphNetBlock(model_fn)(latent_graph)
    return self._decoder(latent_graph)
```

```python
# Online zero-mean / unit-variance normalizer, statistics accumulated during training.
class Normalizer(snt.AbstractModule):
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
    if accumulate:  # stop after a million updates, once stats have converged
      op = tf.cond(self._num_accumulations < self._max_accumulations,
                   lambda: self._accumulate(data), tf.no_op)
    with tf.control_dependencies([op]):
      return (data - self._mean()) / self._std()

  @snt.reuse_variables
  def inverse(self, data):              # un-normalize the decoder output before integrating
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
# Cloth: a Lagrangian, second-order system. Mesh edges carry BOTH reference (u) and
# world (x) displacements; node features are velocity-estimate + node type; output is
# acceleration, integrated TWICE. (A separate world-edge set would be added the same way.)
class ClothModel(snt.AbstractModule):
  def __init__(self, learned_model, name='Model'):
    super(ClothModel, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = Normalizer(size=3, name='output_normalizer')
      self._node_normalizer = Normalizer(size=3 + NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = Normalizer(size=7, name='edge_normalizer')  # x_ij(3)+|x_ij|(1)+u_ij(2)+|u_ij|(1)

  def _build_graph(self, inputs, is_training):
    velocity = inputs['world_pos'] - inputs['prev|world_pos']          # h=1 history -> velocity estimate
    node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)          # relative encoding: NO absolute position
    senders, receivers = triangles_to_edges(inputs['cells'])          # graph = the simulation mesh
    rel_world = tf.gather(inputs['world_pos'], senders) - tf.gather(inputs['world_pos'], receivers)
    rel_mesh = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
    edge_features = tf.concat([
        rel_world, tf.norm(rel_world, axis=-1, keepdims=True),        # current stretch (drives elastic force)
        rel_mesh,  tf.norm(rel_mesh,  axis=-1, keepdims=True)], -1)   # rest geometry (the stencil argument)
    mesh_edges = EdgeSet('mesh_edges', self._edge_normalizer(edge_features, is_training),
                         receivers, senders)
    return MultiGraph(self._node_normalizer(node_features, is_training), [mesh_edges])

  @snt.reuse_variables
  def loss(self, inputs):
    graph = self._build_graph(inputs, is_training=True)
    out = self._learned_model(graph)
    # second-order target: discrete acceleration = x^{t+1} - 2 x^t + x^{t-1}
    target_acc = inputs['target|world_pos'] - 2*inputs['world_pos'] + inputs['prev|world_pos']
    target = self._output_normalizer(target_acc)
    mask = tf.equal(inputs['node_type'][:, 0], NodeType.NORMAL)        # supervise predicted nodes only
    err = tf.reduce_sum((target - out)**2, axis=1)
    return tf.reduce_mean(err[mask])

  def _update(self, inputs, out):
    acc = self._output_normalizer.inverse(out)                        # back to physical scale
    return 2*inputs['world_pos'] + acc - inputs['prev|world_pos']     # integrate twice (forward Euler, dt=1)

  def _build(self, inputs):
    return self._update(inputs, self._learned_model(self._build_graph(inputs, is_training=False)))
```

```python
# Fluid (Eulerian, first-order, fixed mesh): no world edges, mesh edges carry only u_ij;
# node features are the field value + node type; output is the field's change, integrated ONCE,
# plus a direct (un-integrated) pressure prediction.
class CFDModel(snt.AbstractModule):
  def __init__(self, learned_model, name='Model'):
    super(CFDModel, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = Normalizer(size=2, name='output_normalizer')
      self._node_normalizer = Normalizer(size=2 + NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = Normalizer(size=3, name='edge_normalizer')   # u_ij(2)+|u_ij|(1)

  def _build_graph(self, inputs, is_training):
    node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
    node_features = tf.concat([inputs['velocity'], node_type], axis=-1)
    senders, receivers = triangles_to_edges(inputs['cells'])
    rel_mesh = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
    edge_features = tf.concat([rel_mesh, tf.norm(rel_mesh, axis=-1, keepdims=True)], -1)
    mesh_edges = EdgeSet('mesh_edges', self._edge_normalizer(edge_features, is_training),
                         receivers, senders)
    return MultiGraph(self._node_normalizer(node_features, is_training), [mesh_edges])

  @snt.reuse_variables
  def loss(self, inputs):
    graph = self._build_graph(inputs, is_training=True)
    out = self._learned_model(graph)
    target = self._output_normalizer(inputs['target|velocity'] - inputs['velocity'])  # predict the change
    node_type = inputs['node_type'][:, 0]
    mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL), tf.equal(node_type, NodeType.OUTFLOW))
    err = tf.reduce_sum((target - out)**2, axis=1)
    return tf.reduce_mean(err[mask])

  def _update(self, inputs, out):
    return inputs['velocity'] + self._output_normalizer.inverse(out)   # integrate once
```

```python
# Data pipeline: one-step targets + training noise with the target-adjustment derived above.
def add_noise(frame, field, scale, gamma):
  noise = tf.random.normal(tf.shape(frame[field]), stddev=scale)
  mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
  noise = tf.where(mask, noise, tf.zeros_like(noise))                  # only corrupt predicted nodes
  frame[field] += noise                                               # corrupt the most-recent value
  # target adjusted so the integrated output cancels the injected noise:
  #   first-order:        gamma = 1.0  -> target += 0      (full position correction)
  #   second-order cloth: gamma = 0.1  -> target += 0.9*noise (mostly velocity correction)
  frame['target|' + field] += (1.0 - gamma) * noise
  return frame

# per-domain settings (noise scale found by scanning around the one-step error; gamma swept for cloth)
PARAMETERS = {
    'cfd':   dict(noise=2e-2, gamma=1.0, field='velocity',  history=False, size=2, batch=2),
    'cloth': dict(noise=3e-3, gamma=0.1, field='world_pos', history=True,  size=3, batch=1),
}

# assemble: Encode-Process-Decode with latent 128, 2-layer MLPs, 15 message-passing steps; Adam,
# exponential LR decay 1e-4 -> 1e-6; the first ~1000 steps only accumulate normalizer statistics.
learned_model = EncodeProcessDecode(output_size=params['size'], latent_size=128,
                                    num_layers=2, message_passing_steps=15)
```

```python
# Test-time adaptivity: same network also predicts a per-node sizing tensor S; a generic,
# domain-independent local remesher then refines/coarsens the mesh. No simulator in the loop.
def remesh(mesh, sizing):                       # R(M, S)
  # split every invalid edge (size > 1, descending), where size = u_ij^T ((S_i+S_j)/2) u_ij
  split_invalid_edges(mesh, sizing)
  flip_anisotropic_delaunay(mesh, sizing)       # Bossen-Heckbert flip criterion
  collapse_safe_edges(mesh, sizing)             # collapse only if it creates no new invalid edge
  flip_anisotropic_delaunay(mesh, sizing)       # final cleanup pass
  return mesh

def estimate_sizing(mesh, node_i):
  # when no ground-truth sizing exists: tightest origin-centred ellipse containing the
  # neighbour displacements u_ij  ==  argmax sum_j u_ij^T S u_ij  s.t.  u_ij^T S u_ij <= 1
  return min_area_enclosing_ellipse([mesh.u[j] - mesh.u[node_i] for j in mesh.neighbors(node_i)])
```

The causal chain, start to finish: I refuse to give up adaptive meshes, which rules out grids; the radius-particle alternative erases the rest state and mismatches irregular spacing, which forces me to make the graph *be the mesh* and put the reference displacement `u_ij` on every edge — recovering the rest state and turning message passing into a learnable differential-operator stencil; that loses contact, so I add proximity-based world edges and keep both spaces; physics is translation-invariant, so all positions go in as relative edge features and the model is equivariant by construction; the GN block generalizes to two edge sets with sum aggregation and residuals, stacked fifteen deep for multi-hop coupling; outputs are derivatives integrated by forward Euler; one-step training would compound error in rollout, so I inject training noise and adjust the targets so the model learns to *correct* it (a `γ` blend where position and velocity can't both be fixed); online normalization handles the heterogeneous scales; and to keep adaptivity alive without the original solver, I learn the *sizing field* — the only domain-specific part — with the same network, hand it to a generic split/flip/collapse remesher, and estimate sizing targets as a smallest-enclosing-ellipse problem when no labels exist.
