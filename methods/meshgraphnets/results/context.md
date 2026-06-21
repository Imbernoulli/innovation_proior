# Context: learning to simulate physics directly on irregular, adaptive meshes

## Research question

Can a neural network learn to advance a mesh-based physical simulation forward in time — fast, accurately, and across very different physics (cloth, elastic solids, incompressible and compressible flow) — while keeping the single property that makes meshes the tool of choice in engineering: the freedom to put resolution exactly where the physics needs it?

Mesh-based finite-element / finite-volume simulation is the workhorse of structural mechanics, aerodynamics, electromagnetics, geophysics and acoustics. Its great advantage over a regular grid is **adaptive discretization**: a mesh can place tiny, oriented elements at an airfoil tip or along a sharp crease in cloth and coarse elements in the smooth, empty regions, achieving accuracy a uniform grid would need orders of magnitude more cells to match. But high-resolution simulations are slow, and classical solvers must be tuned per system and often scale poorly on hardware accelerators.

A learned surrogate that runs substantially faster than the classical solver, generalizes to new shapes and scales, and is differentiable (so it could later drive design optimization or control) would broaden the range of problems on which neural simulators are useful. The hard part is the discretization: most prior learned simulators throw away the mesh — they work on a regular grid, or on a free particle cloud — and with it they throw away adaptivity. A solution would have to (a) operate natively on an irregular, possibly time-varying mesh, (b) capture both the *internal* dynamics that a PDE imposes between neighbouring material points and the *external* dynamics (contact, collision) that couple points which are far apart in the material but touching in space, and (c) keep resolution adaptive at run time without secretly calling the original solver.

## Background

**The governing equations live on a mesh.** A simulation mesh `M = (V, E)` has nodes `V` carrying a reference (material / mesh-space) coordinate `u_i` that spans the undeformed domain, plus the dynamical quantities `q_i` we want to predict. *Eulerian* systems (fluids) evolve continuous fields — velocity / momentum, density, pressure — sampled at fixed mesh nodes. *Lagrangian* systems (cloth, deforming solids) move and deform: each node additionally has a world-space position `x_i` in 3D, so the same material point lives at `u_i` in the rest configuration and at `x_i(t)` in space. Finite-element discretizations approximate the spatial differential operators (gradient, Laplacian) of the PDE by local stencils over the mesh edges; the time stepping is a numerical integrator. Internal forces (elasticity, viscosity, pressure gradients) are local in the *mesh*; contact and (self-)collision are local in *space* but can be arbitrarily far apart in the mesh.

**Adaptive meshing via a sizing field.** Classical adaptive remeshing (Narain, Samii & O'Brien 2012; Bossen & Heckbert 1996; Wicke et al. 2010) splits the job in two: decide *where* the mesh should be fine or coarse, then *make it so*. The "where" is encoded by a **sizing field** — a symmetric tensor field `S(u) ∈ R^{2×2}` over the domain that specifies the maximum allowed edge length in each direction. An edge `u_ij = u_i − u_j` is judged *valid* when its size under the metric is at most one, `s(i,j)² = u_ij^T ((S_i + S_j)/2) u_ij ≤ 1`; an oversize edge must be split. Because `S` is a tensor, it can demand short edges across a fold and long edges along it (anisotropy). The "make it so" is a generic loop of three local operations — **split** an invalid edge (refine), **collapse** an edge if doing so creates no new invalid edge (coarsen), and **flip** an edge to keep element aspect ratios sensible under an anisotropy-aware Delaunay criterion (Bossen & Heckbert 1996). Crucially, only the sizing field needs domain knowledge (curvature heuristics for cloth, gradient heuristics for CFD); the split/flip/collapse machinery is domain-independent.

**Graphs as a substrate for learned physics.** A mesh is already a graph. The Graph Network formalism (Battaglia et al. 2018), building on the original graph neural network (Scarselli et al. 2008) and on Interaction Networks (Battaglia et al. 2016), defines a learnable "GN block" that maps a graph to a graph through three steps: a per-edge update `e'_k = φ^e(e_k, v_{r_k}, v_{s_k})` computed from an edge and its two endpoint nodes; a permutation-invariant aggregation of incoming edge messages at each node `ē'_i = ρ^{e→v}({e'_k : r_k = i})`; and a per-node update `v'_i = φ^v(ē'_i, v_i)`. Stacking such blocks performs *message passing*: information and constraints propagate one hop per block, so K blocks couple nodes up to K hops apart. Sanchez-Gonzalez et al. (2018) showed GN blocks can act as learnable physics engines for articulated systems and control.

**Motivating diagnostic findings about the existing learned simulators.** Two failure modes are well documented and set up the problem:
- A *fixed-radius particle graph* (the connectivity used by particle-based learned simulators) computes edges from spatial proximity `|x_i − x_j| < R`. On an irregular mesh this oversamples dense regions and undersamples sparse ones; the receptive field is mismatched to the discretization, producing instability and high rollout error on irregular and dynamically-changing meshes. A purely spatial graph also has *no notion of a rest state*: it cannot tell how far apart two material points are in the undeformed sheet, so it struggles to model restoring forces in materials like cloth and elastics, and error accumulates until the rollout diverges.
- A *regular-grid CNN*, the most popular learned-physics architecture, must spend uniform resolution everywhere. On aerodynamics it can capture large-scale flow but undersamples the small, important wake region around a wing tip even when given several times more cells than a mesh would use over a far smaller region; it also tends to develop fluctuations over long rollouts.

**Spatial equivariance from relative encoding.** Physical laws are translation-invariant. Sanchez-Gonzalez et al. (2020) observed that feeding *relative* positional displacements (and their magnitudes) as edge features — rather than absolute node coordinates — bakes translation invariance into the model and greatly improves generalization to new positions and configurations.

## Baselines

**Particle-based graph simulator (GNS; Sanchez-Gonzalez et al. 2020).** State is a particle cloud `X = (x_1,...,x_N)`; an encoder builds a latent graph by connecting particles within a fixed connectivity radius `R` (recomputed each step by nearest-neighbour search), with node embeddings `v_i = ε^v(x_i)` and edge embeddings `e_ij = ε^e(r_ij)` where `r_ij` is the relative displacement and its magnitude. A processor applies `M` GN blocks without global features and with residual connections; a decoder MLP `δ^v` reads out per-particle acceleration, integrated by forward Euler. All MLPs are two-hidden-layer, size 128, ReLU, LayerNorm except the decoder; inputs/targets are normalized online to zero mean and unit variance; training is one-step supervised with random-walk input noise `σ = 0.0003` to keep the training input distribution close to the noisy distribution the model sees during rollout. **Gap:** the radius graph is spatial-only — no rest-state geometry, so materials with a reference configuration (cloth, elastics) are modeled poorly; and the fixed radius is mismatched to irregular and adaptive meshes, where it over/undersamples and destabilizes.

**Graph convolutional surrogate (GCN; Kipf & Welling 2017, applied to flow by Belbute-Peres et al. 2020).** A GCN updates each node by a linear transform of the (normalized) average of its neighbours' features, `H' = σ(ÂHW)` — it does **not** compute a learned function on each edge from the pair of endpoint features. Belbute-Peres et al. embed a differentiable aerodynamics solver in a GCN pipeline for steady-state super-resolution. **Gap:** without per-edge message computation and relative encoding, the model is poorly suited to learning local physical laws from the pair geometry; it is prone to overfitting and (without the solver in the loop) cannot produce stable dynamical rollouts on rich flow.

**Grid convolutional surrogate (CNN/U-Net; Thuerey et al. 2020).** A U-Net regresses flow fields on a regular grid. **Gap:** uniform resolution; cannot represent adaptive or Lagrangian deforming meshes, and undersamples small high-gradient regions (wakes, boundary layers) even at high cell counts.

**Classical adaptive remesher in the loop (Narain et al. 2012; ArcSim).** The sizing-field + split/flip/collapse machinery above. **Gap (for a learned simulator):** the sizing field requires domain-specific heuristics and the remesher is part of the original simulator; calling it at every step of a learned rollout reintroduces exactly the hand-built, system-specific solver the surrogate was meant to replace.

## Evaluation settings

The natural yardsticks are trajectory-prediction tasks across distinct PDEs, each generated by a different established solver:
- **Cloth** (free-flapping flag; cloth draped on a moving sphere) on triangular surface meshes, including dynamically remeshed ones — simulated with ArcSim (Narain et al. 2012).
- **Hyper-elastic structural mechanics** (a plate deformed by a kinematic actuator) on a tetrahedral mesh — simulated with COMSOL, quasi-static.
- **Incompressible flow** (water past a cylinder) on a fixed 2D triangular mesh — COMSOL.
- **Compressible flow** (aerodynamics around an airfoil cross-section) on an irregular 2D triangular mesh whose edge lengths span `2·10⁻⁴ m` to `3.5 m` — SU2.

Each dataset has ~1000 train / 100 valid / 100 test trajectories of 250–600 steps. Node features include a one-hot **node type** distinguishing normal, kinematic/handle, obstacle, inflow/outflow and wall nodes (kinematic nodes follow scripted motion and are not predicted). Metrics: one-step error and multi-step rollout error (root-mean-square error of position for Lagrangian systems, of momentum/velocity for Eulerian systems), and rollout stability over hundreds–thousands of steps. Wall-clock cost per step is compared against the ground-truth solver. Baselines for comparison: GNS (and a GNS+mesh-position hybrid), GCN, and a grid U-Net.

## Code framework

The primitives that already exist: a graph data structure with typed edge sets; standard MLP / LayerNorm building blocks; an online feature normalizer; a forward-Euler state update; the Adam optimizer with exponential learning-rate decay; a generic local remesher that, *given a sizing field*, applies split/flip/collapse; and a dataset pipeline that streams trajectories of meshes and adds training noise. The contribution slots in below are left empty.

```python
import collections
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class OnlineNormalizer(snt.AbstractModule):
  """Zero-mean, unit-variance normalizer with statistics accumulated online."""
  # exists already: accumulate mean/std during training, normalize and inverse.
  ...


def make_mlp(latent_size, num_layers, output_size, layer_norm=True):
  """Two-hidden-layer ReLU MLP, optionally followed by LayerNorm."""
  widths = [latent_size] * num_layers + [output_size]
  net = snt.nets.MLP(widths, activate_final=False)
  if layer_norm:
    net = snt.Sequential([net, snt.LayerNorm()])
  return net


class GraphModel(snt.AbstractModule):
  """Maps an input graph to per-node outputs."""

  def __init__(self, output_size, latent_size, num_layers, num_steps):
    super(GraphModel, self).__init__()
    # store sizes; the internal computation is the open problem.
    ...

  def _build(self, graph):
    # TODO: the graph computation we will design here, mapping the input
    #       graph to per-node outputs.
    pass


class Simulator(snt.AbstractModule):
  """Turns a raw mesh state into a graph, runs the model, integrates one step."""

  def _build_graph(self, inputs):
    # TODO: how to turn a mesh state into graph nodes and edge sets
    #       (which features go on nodes, which on edges, what connectivity).
    pass

  def loss(self, inputs):
    # supervise the per-node output against the (normalized) ground-truth target.
    pass

  def _update(self, inputs, network_output):
    # forward-Euler integration of the decoded output into the next state.
    pass


def remesh(mesh, sizing_field):
  """Generic domain-independent local remesher: split / flip / collapse edges
  so the mesh satisfies the given sizing field. Already available."""
  ...


def learner(model):
  ds = load_trajectories(...)            # stream meshes
  ds = add_targets_and_noise(ds, ...)    # one-step targets + training noise
  loss_op = model.loss(next(ds))
  step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(1e-4, step, decay_steps=int(5e6), decay_rate=0.1) + 1e-6
  train_op = tf.train.AdamOptimizer(lr).minimize(loss_op, global_step=step)
  # train loop ...
```
