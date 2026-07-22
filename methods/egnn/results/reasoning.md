Let me start from the failure mode. I have a set of points in space — atoms of a molecule, particles in an N-body box, nodes of a graph I've placed in some latent geometry — and each carries a position `x_i ∈ R^n` and maybe a velocity, plus some scalar features `h_i`. The thing I want to predict is tied to that geometry: sometimes a scalar like energy, sometimes a vector like the future position or a velocity. And the physics doesn't care about my coordinate frame. If I slide the whole system by `g`, rotate it by some orthogonal `Q`, or reflect it, an energy is unchanged and a predicted displacement just comes out rotated and slid the same way. On top of that the points have no canonical order, so relabelling them must relabel my outputs identically. So the symmetry I'm up against is the Euclidean group `E(n)` — translations plus the orthogonal group `O(n)` of rotations and reflections — together with permutations.

I could try to *learn* this invariance from data augmentation: rotate every training example a thousand ways and hope the network figures it out. But that's wasteful and never exact. The whole lesson of the last decade is the opposite: bake the symmetry into the architecture. CNNs are translation equivariant by construction, so they don't need translation augmentation; GNNs are permutation equivariant by construction. What I want is the analogous thing for the *continuous* spatial symmetry, the rotations and reflections and translations, on top of permutation equivariance. Formally I want a layer `phi` such that for the right output-side action `S_g`, `phi(T_g x) = S_g(phi(x))` for every group element. And I want two regimes at once: my scalar features should be **invariant** (rotate the input, the scalar is unchanged), and my coordinate outputs should be **equivariant** (rotate the input by `Q`, the output coordinates come out rotated by the same `Q`).

The natural substrate is message passing on a graph, because that already hands me permutation equivariance for free. The Gilmer-style layer is

  m_ij = phi_e(h_i, h_j, a_ij),  m_i = sum_{j} m_ij,  h_i^{l+1} = phi_h(h_i, m_i),

with `phi_e`, `phi_h` MLPs. Permute the nodes and everything permutes along, because the aggregation is a symmetric sum. Good — half my problem is already solved. But this layer is completely blind to where the points actually are; it never touches `x_i`. The naive patch is to just feed the coordinate in as a feature: stuff `x_i` into `h_i`. I want to be honest about whether that actually breaks anything before I throw it out, so let me push a number through. Take `x = (1, 0)` in the plane and a fixed feature map `phi(v) = relu(W v)` with some `W` standing in for the first layer of the edge MLP; rotating by 90 degrees sends `x` to `Q x = (0, 1)`. With a particular `W` I get `phi(x) = (1.76, 0.98, 1.87)` and `phi(Q x) = (0.40, 2.24, 0.00)` — completely different vectors. An invariant feature should have come out identical; it didn't. So feeding the raw coordinate as a scalar feature is genuinely not invariant: any MLP that treats the coordinate as an ordinary input value does not commute with `Q`, and the rotation leaks straight into the features. Wall. The coordinate has to enter the computation only through quantities that the group leaves alone, or through operations that *commute* with the group.

So what does the group leave alone? Let me write down the two facts that the orthogonal group and translations actually give me, because the whole design is going to hang off them. First, an orthogonal `Q` preserves inner products: `(Qa)^T(Qb) = a^T Q^T Q b = a^T I b = a^T b`, so it preserves norms too. Second, a shared translation cancels in a difference: `(a+g) - (b+g) = a - b`. Put those together on a pair of points. The relative difference `x_i - x_j` is unchanged by translation (the `g`'s cancel) but rotates with `Q`. The distance `||x_i - x_j||` is unchanged by translation *and* by `Q` (norm-preserving). So the squared distance `||x_i - x_j||^2` is a genuine `E(n)`-invariant scalar built purely from the geometry — it survives translation, rotation, and reflection untouched.

That immediately tells me how to make the *message* invariant. SchNet already saw this: if I feed only the distance into the edge function, the message can't possibly depend on the global pose. Take the GNN edge function and hand it the squared relative distance alongside the scalar features:

  m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij).

If `h_i` is itself invariant (it will be — I'll check by induction), then every argument of `phi_e` is invariant, so `m_ij` is invariant. Translate and rotate the inputs: `||Q x_i + g - (Q x_j + g)||^2 = ||Q(x_i - x_j)||^2 = (x_i - x_j)^T Q^T Q (x_i - x_j) = ||x_i - x_j||^2`, exactly unchanged. So far this is SchNet's trick repackaged — invariant message passing — and if I stop here I get exactly SchNet's ceiling: every layer is invariant, the network can only ever emit invariant scalars. That's fine for energy. But I also want to *output a vector* — an updated position, a velocity. SchNet can't, because once the spatial info collapses to a scalar distance at layer one, the directional, type-1 structure is gone forever. So invariant message passing alone is not enough; I need a channel that carries and *transforms* the coordinate equivariantly through the layers. Wall again, but a productive one: it tells me I need a second update, on `x` itself, that is equivariant rather than invariant.

How do I update a coordinate equivariantly using cheap operations? The expensive answer is the one Tensor Field Networks and the SE(3)-Transformer take: represent everything in higher-order steerable types, build the layer out of spherical harmonics and Clebsch–Gordan coefficients so it commutes with `SO(3)` by construction. That genuinely works and it's expressive. But the spherical harmonics have to be recomputed for every relative geometry, it's heavy, and the whole apparatus is welded to three dimensions — there's no clean version in `n` dimensions, and I explicitly want my graph-autoencoder use to live in `n > 3`. So steerable kernels are off the table for what I'm after. I want something that is equivariant the way the relative-difference vector itself is equivariant, with no harmonics.

Stare at the relative difference `x_i - x_j` again. It's translation-invariant and it rotates with `Q` — that is, it transforms as a type-1 vector, *exactly* the way I want my coordinate output to transform. So a difference vector is already a little equivariant building block. The question is how to combine a bunch of them into an update to `x_i` without wrecking that property. If I take a weighted sum `sum_j w_ij (x_i - x_j)` and I rotate the input, the sum becomes `sum_j w_ij (Q x_i - Q x_j) = Q sum_j w_ij (x_i - x_j)` — *provided* the weights `w_ij` don't themselves change under the rotation. That proviso is the whole game. If the weights are invariant scalars, `Q` factors straight out of the sum and the combination is equivariant. And I already have an invariant scalar per edge sitting right there: the message `m_ij`. So let the weight be a scalar function of the message, `phi_x(m_ij)` with `phi_x: R^nf -> R^1`, and update

  x_i^{l+1} = x_i^l + C sum_{j≠i} (x_i^l - x_j^l) phi_x(m_ij).

The coordinate moves along a weighted combination of the directions to all the other points, where the weight on each direction is a learned invariant scalar read off the (invariant) edge message. The displacement is type-1 because it's a sum of type-1 difference vectors with type-0 weights; adding it to `x_i` (also type-1) keeps the result type-1. And it's translation-equivariant because the difference vectors are translation-invariant and I add the displacement to `x_i`, which carries the translation. This is cheap — one scalar MLP and a weighted sum of vectors — and it works in any `n`, because nothing about it references three dimensions.

Crucially, `phi_x` must output a *scalar*, not a vector or a matrix. If I let it output something with directional structure, that thing would itself have to transform under `Q`, and then `Q` would no longer factor cleanly out of the product — I'd be back to needing the steerable machinery to make the weight transform correctly. The scalarness is exactly what lets an ordinary MLP supply the weight while equivariance is handled entirely by the difference vector it multiplies. Equivariance lives in the vectors, learning lives in the invariant scalars, and they meet only through a product (vector)·(scalar).

Now let me actually prove the whole layer is equivariant, end to end, because I want to be sure the invariance of `h` and the equivariance of `x` are mutually consistent through a stack — it's an inductive claim. I'll write the layer as the four equations: the edge message (1), the coordinate update (2), the aggregation `m_i = sum_{j≠i} m_ij` (3), and the node update `h_i^{l+1} = phi_h(h_i^l, m_i)` (4). I want to show `Q x^{l+1} + g, h^{l+1} = EGCL(Q x^l + g, h^l)` — feeding the transformed inputs gives the transformed coordinates and the *same* features.

Assume the induction hypothesis: `h^l` is invariant (at `l = 0` this just means I don't encode absolute position or orientation into `h^0`, which I control). Feed `Q x_i + g`. Equation (1): the only place `x` enters is the squared distance, and I just showed `||Q x_i + g - (Q x_j + g)||^2 = ||x_i - x_j||^2`. The other arguments `h_i, h_j, a_ij` are invariant by hypothesis. So `m_ij` is unchanged — invariant. Now equation (2), the one that has to come out equivariant. Substitute the transformed coordinates:

  Q x_i + g + C sum_{j≠i} (Q x_i + g - [Q x_j + g]) phi_x(m_ij).

Inside the sum the two `g`'s cancel, leaving `Q x_i - Q x_j = Q(x_i - x_j)`. The scalar `phi_x(m_ij)` is unchanged because `m_ij` is. So the expression is

  Q x_i + g + C sum_{j≠i} Q(x_i - x_j) phi_x(m_ij)
  = Q x_i + g + Q · C sum_{j≠i} (x_i - x_j) phi_x(m_ij)        [Q is linear, pull it out]
  = Q ( x_i + C sum_{j≠i} (x_i - x_j) phi_x(m_ij) ) + g
  = Q x_i^{l+1} + g.

So the output coordinate is exactly the untransformed update with `Q` applied and `g` added — equivariant, precisely as required. Equations (3) and (4) depend only on `m_ij` and `h^l`, both invariant, so `m_i` is invariant and `h_i^{l+1} = phi_h(h_i^l, m_i)` is invariant — which re-establishes the induction hypothesis for the next layer. By induction a whole stack of these layers is `E(n)`-equivariant on `x` and invariant on `h`.

The algebra is short enough that I distrust it a little — pulling `Q` out of a sum is exactly the kind of step where a sign or an index error hides. Let me put numbers in and watch the output actually rotate. Two points in the plane, `x_1 = (1, 0)` and `x_2 = (0, 2)`; a 90-degree rotation `Q = [[0,-1],[1,0]]` and a translation `g = (5, -3)`. The invariant scalar first: `||x_1 - x_2||^2 = ||(1,-2)||^2 = 5`, and on the transformed points `x_1' = Q x_1 + g = (5,-2)`, `x_2' = Q x_2 + g = (3,-3)`, so `||x_1' - x_2'||^2 = ||(2,1)||^2 = 5` — the distance survives, as it must. Now the coordinate update on node 1 with a single neighbour and some fixed invariant weight `w = phi_x(m_12) = 0.37` (the *same* number in both frames, since it reads off the invariant message). In the original frame `x_1^{new} = x_1 + (x_1 - x_2) w = (1,0) + (1,-2)(0.37) = (1.37, -0.74)`. Equivariance predicts the transformed update should be `Q x_1^{new} + g = Q(1.37,-0.74) + (5,-3) = (0.74, 1.37) + (5,-3) = (5.74, -1.63)`. Running the update directly in the transformed frame: `x_1' + (x_1' - x_2') w = (5,-2) + (2,1)(0.37) = (5.74, -1.63)`. The two agree to the digit. So the factoring isn't just symbolically clean — the layer really does carry a rotation+translation of the input through to the same rotation+translation of the output. Both regimes, on the same graph, with no spherical harmonics and no commitment to `n = 3`.

Let me pin down the constant `C`. The coordinate update sums over `M - 1` difference vectors (every other node, if fully connected). A bare sum grows in magnitude with the number of nodes — on a big graph each step would yank `x_i` by something proportional to `M`, which is unstable and makes the per-step scale depend on graph size. So I divide by the count, `C = 1/(M - 1)`, turning the sum into a mean over the contributing differences. Then the displacement magnitude stays `O(1)` regardless of how many neighbours there are, and the same learning rate behaves consistently across graphs of different sizes. (Operationally this is just choosing "mean" rather than "sum" aggregation for the coordinate messages, while the feature messages `m_i` keep the plain GNN sum.)

There's a related stability worry in the displacement itself. The raw difference `(x_i - x_j)` can be large for distant pairs, so even with a modest weight the product can be big and the early-training dynamics can thrash as positions slosh around. A clean guard is to normalize the difference by its own length before scaling, either as `(x_i - x_j) / (||x_i - x_j|| + eps)` or, in a message-passing port, `(x_i - x_j) / (||x_i - x_j|| + 1)`, so what I scale is essentially a bounded direction and the magnitude is governed mostly by the learned weight rather than by how far apart the points happen to be. Dividing by a *scalar* function of the invariant distance doesn't touch equivariance — it's still `Q` times a direction times an invariant scalar — so I keep it as an optional stabilizer. And I'll initialize the last linear layer of `phi_x` very small (tiny Xavier gain), so the coordinate updates start near zero and the geometry barely moves at initialization; the network earns the right to move points as it trains.

On nonlinearities: notice every learnable nonlinearity in this design lives on the *invariant* channels — inside `phi_e`, `phi_x`, `phi_h`, all of which consume invariant arguments (`h`, the distance, the message) and produce invariant outputs. I never apply a pointwise nonlinearity to the coordinate `x` directly, and I'd better not: a pointwise nonlinearity does not commute with `Q` (it acts per-coordinate, so rotating the vector first versus applying the nonlinearity first give different answers), so it would silently break equivariance. The coordinate only ever gets touched by linear combination with invariant scalar weights, which is exactly the equivariance-safe operation. So I'm free to use a smooth activation such as Swish or SiLU on the feature channels without endangering the symmetry.

Now I want a feature channel that's genuinely flexible, unlike the Radial Field update, which only ever touches `x`. Köhler's radial-field layer is, in my notation, the coordinate update (2) with the weight a function of just the bare distance and *no* `h` channel propagated at all: `x_i += sum_j phi_rf(||x_i - x_j||)(x_i - x_j)`. It's equivariant and cheap, but it has no learned per-node memory, so its bias is high and it underfits once there's enough data to learn subtleties. My layer fixes exactly that: the coordinate weight `phi_x(m_ij)` depends on the full message `m_ij`, which depends on `h_i, h_j` — so the rich, learned feature channel feeds into the geometry, and the geometry (through the distance) feeds back into the features. The two channels exchange information in the edge operation. That's the missing flexibility, recovered without giving up equivariance.

Let me also handle the case where I'm given coordinates *and* an initial velocity, because for the N-body system I have `v^(0)` and I'd like to track momentum explicitly. A velocity is type-1, like a coordinate, so I can route it the same way. Break the coordinate update into two steps: first form a velocity, then move along it.

  v_i^{l+1} = phi_v(h_i^l) v_i^init + C sum_{j≠i} (x_i^l - x_j^l) phi_x(m_ij),
  x_i^{l+1} = x_i^l + v_i^{l+1}.

The new piece is `phi_v(h_i^l) v_i^init`: an invariant scalar `phi_v(h_i^l)` (an MLP of the invariant features, `R^nf -> R^1`) gating the *initial* velocity, which is itself a type-1 vector. Is it equivariant? Rotate and translate the inputs; the initial velocity transforms by `Q` only (velocities are unaffected by translation — a constant shift of all positions doesn't change relative motion). So `phi_v(h_i^l) Q v_i^init + C sum (Q x_i - Q x_j) phi_x(m_ij) = Q ( phi_v(h_i^l) v_i^init + C sum (x_i - x_j) phi_x(m_ij) ) = Q v_i^{l+1}` — `Q` factors out of both terms (`phi_v(h)` and `phi_x(m)` are invariant scalars, `g` cancels in the difference, and the velocity carries no `g`). Then `x_i^{l+1} = x_i^l + v_i^{l+1}` is equivariant because it's a sum of equivariant vectors, with the translation riding on `x_i^l`. And note the nice degeneracy: if `v^init = 0`, the `phi_v` term vanishes and this collapses back to the plain coordinate update (2). So the velocity variant is a strict generalization, used only when an initial velocity is actually provided.

There's still the question of edges. On a point cloud I may have no adjacency at all; the default is just to let every node talk to every other, `j ≠ i`. But fully connected doesn't scale to big point clouds, and sometimes I want the graph to *decide* its own edges. Rewrite the aggregation with explicit edge gates: `m_i = sum_{j≠i} e_ij m_ij`, where `e_ij ∈ {0,1}` says whether an edge exists. Now soften it — approximate the gate from the message, `e_ij ≈ phi_inf(m_ij)`, where `phi_inf` is a linear layer followed by a sigmoid, `R^nf -> [0,1]`. That's a soft, invariant attention over the (already invariant) messages, so it can't disturb equivariance, and it lets the network prune or weight its own connectivity. I'll use this for QM9, where molecules vary in size and I'm not handed an adjacency.

Now something that nags at me once the equivariant machinery is in place: for a purely *invariant* task — predict a scalar, positions are static, like QM9 energies — do I even need the equivariant coordinate update at all? If I never run equation (2), the coordinates just sit there and the model is plain invariant message passing on distances, essentially a GNN that gets `||x_i - x_j||^2` in every message. The worry is that by collapsing the geometry to scalar distances I've thrown away directional information — surely I'd do better keeping the type-1 difference vectors, or going to higher types like TFN does? Let me check whether that worry is real, because if it isn't, the simplest possible thing is also the best.

Claim: when I'm only given positions (no velocities, no higher-type inputs), the pairwise distance matrix for a fixed node indexing already determines the geometry completely, up to an `E(n)` transformation; if I relabel the nodes, the same matrix just relabels with them. If that's true, then distances lose no relevant information and there's nothing to gain from carrying the vectors. Two halves: invariance and uniqueness.

Invariance is the easy half and I already have it: for any orthogonal `Q` and translation `t`, `l2(Q x_i + t, Q x_j + t) = ||Q(x_i - x_j)||` (the `t`'s cancel) `= sqrt((x_i - x_j)^T Q^T Q (x_i - x_j)) = sqrt((x_i - x_j)^T(x_i - x_j)) = l2(x_i, x_j)`. Distances don't move under `E(n)`.

Uniqueness is the half that actually has content. Suppose two point sets `{x_i}` and `{y_i}` have identical pairwise distances, `l2(x_i, x_j) = l2(y_i, y_j)` for all `i, j`. I want to produce an orthogonal `A` and translation `t` with `x_i = A y_i + t`. First kill the translation: replace each `x_i` by `x_i - x_0` and each `y_i` by `y_i - y_0`. Distances are translation-invariant, so the hypothesis is unchanged, and after this centering I can write `x_0 = y_0 = 0`. Then `||x_i|| = l2(x_i, x_0) = l2(y_i, y_0) = ||y_i||`, so all the norms match. Now expand a squared distance: `||x_i - x_j||^2 = ||x_i||^2 - 2 <x_i, x_j> + ||x_j||^2`, and the same for `y`. The left sides are equal by hypothesis and the norm terms are equal, so the cross terms are equal too: `<x_i, x_j> = <y_i, y_j>` for all `i, j`. Every pairwise inner product agrees. The two centered configurations have the same Gram matrix.

That Gram equality is stronger than a list of matching lengths. For any two coefficient lists `c_i` and `d_i`,

  <sum_i c_i x_i, sum_j d_j x_j> = sum_{i,j} c_i d_j <x_i, x_j> = sum_{i,j} c_i d_j <y_i, y_j> = <sum_i c_i y_i, sum_j d_j y_j>.

Call this bilinear identity `(*)`; the squared-norm version is the special case `c = d`. Now I build the map in the orientation I actually need. Take a basis `{y_{i_1}, ..., y_{i_d}}` of the span of `{y_i}`. The corresponding `{x_{i_1}, ..., x_{i_d}}` is also independent, because if `sum_j a_j x_{i_j} = 0`, then by `(*)`, `||sum_j a_j y_{i_j}||^2 = ||sum_j a_j x_{i_j}||^2 = 0`, forcing all `a_j = 0`. Define a linear map `A` on `span{y_i}` by `A y_{i_j} = x_{i_j}`. I need it to send every `y_i` to the matching `x_i`, not just the basis points. Write `y_i = sum_j c_j y_{i_j}`; then `A y_i = sum_j c_j x_{i_j}`. The error is

  ||x_i - sum_j c_j x_{i_j}||^2
  = <x_i, x_i> - 2 <x_i, sum_j c_j x_{i_j}> + <sum_j c_j x_{i_j}, sum_j c_j x_{i_j}>.

By `(*)`, each term equals the corresponding expression with `y`'s:

  <y_i, y_i> - 2 <y_i, sum_j c_j y_{i_j}> + <sum_j c_j y_{i_j}, sum_j c_j y_{i_j}>
  = <y_i, y_i> - 2 <y_i, y_i> + <y_i, y_i> = 0,

using `y_i = sum_j c_j y_{i_j}`. The error is zero, so `A y_i = x_i` for all `i`. And `A` is an isometry on the span, not merely length-preserving on selected basis vectors: for arbitrary `u = sum_j a_j y_{i_j}` and `v = sum_k b_k y_{i_k}`, `<A u, A v> = sum_{j,k} a_j b_k <x_{i_j}, x_{i_k}> = sum_{j,k} a_j b_k <y_{i_j}, y_{i_k}> = <u, v>`. Extend this isometry by any orthogonal map on the orthogonal complement, and it becomes an orthogonal map on all of `R^n`. Returning to the uncentered points, `x_i - x_0 = A(y_i - y_0)`, so `x_i = A y_i + (x_0 - A y_0)`. Uniqueness holds with `t = x_0 - A y_0`.

So when only positions are given, the pairwise distances are a complete, lossless identifier of the indexed geometry up to `E(n)`, and the usual permutation equivariance handles relabelling of the nodes — there is genuinely no information in the difference *vectors* or in higher tensor types that isn't already in the *scalar* distances, once I quotient by the symmetries I'm forced to quotient by anyway. For an invariant target, the stripped-down model — just invariant message passing on `||x_i - x_j||^2`, skipping the coordinate update — is not throwing away geometric information. The coordinate update is for genuinely *equivariant* targets (predict a position, a velocity) where I do need to emit a vector.

Let me now write the layer as code I'd actually ship, filling that single open slot in the message-passing harness. I'll keep the four functions explicit: an edge MLP `phi_e` producing the message from `[h_i, h_j, squared-distance, edge-attr]`; a coordinate MLP `phi_x` mapping the message to a scalar weight, with its last layer initialized tiny; a node MLP `phi_h` updating `h` from `[h_i, aggregated message]` with a residual; and the coordinate update as a scalar-weighted, count-normalized sum of difference vectors. Aggregation of the feature messages is a sum; aggregation of the coordinate displacements is a mean (that's the `C = 1/(M-1)`). I'll allow the optional difference normalization and the optional soft edge gate.

```python
from torch import nn
import torch


class E_GCL(nn.Module):
    """One E(n)-equivariant graph-convolutional layer (EGCL).
    Carries invariant node features h and equivariant coordinates x; updates both.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention      # soft, invariant edge gate (phi_inf)
        self.normalize = normalize      # divide difference by its length for stability
        self.coords_agg = coords_agg    # 'mean' => C = 1/(M-1); 'sum' is the bare sum
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1              # the single invariant scalar: squared distance

        # phi_e: edge message from [h_i, h_j, ||x_i - x_j||^2, a_ij]
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # phi_h: node update from [h_i, aggregated message], with residual
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # phi_x: message -> scalar weight; last layer init tiny so x barely moves at start
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())   # optional bound on the weight (stability)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            # phi_inf: soft invariant edge gate, m_i = sum_j sigmoid(.) m_ij
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # radial = ||x_i - x_j||^2 (invariant). Build the invariant message m_ij.
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)                     # m_ij = phi_e(...)
        if self.attention:
            out = out * self.att_mlp(out)            # soft invariant gate on the message
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # x here is h (node features). Aggregate messages (sum), then phi_h with residual.
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # m_i = sum_j m_ij
        agg = torch.cat([x, agg], dim=1) if node_attr is None else torch.cat([x, agg, node_attr], dim=1)
        out = self.node_mlp(agg)                      # phi_h(h_i, m_i)
        if self.residual:
            out = x + out                             # h^{l+1} = h^l + phi_h(...)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        # The equivariant update: weighted sum of relative differences, weight = phi_x(m_ij).
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)   # (x_i - x_j) * phi_x(m_ij)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  # the C=1/(M-1)
        else:
            raise Exception('Wrong coords_agg parameter: %s' % self.coords_agg)
        return coord + agg                              # x^{l+1} = x^l + C sum (x_i-x_j) phi_x(m_ij)

    def coord2radial(self, edge_index, coord):
        # Squared distance (the invariant scalar) and the raw difference vector (the type-1 basis).
        row, col = edge_index
        coord_diff = coord[row] - coord[col]            # x_i - x_j  (type-1, equivariant)
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)   # ||x_i - x_j||^2  (invariant)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm              # unit direction; scaling by invariant scalar
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)   # (1) m_ij
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)  # (2) x update (equivariant)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)    # (3),(4) m_i and h update
        return h, coord, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    count = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)                  # mean = sum / count = the C normalization


class EGNN(nn.Module):
    """Stack of EGCLs: embed invariant features, run L equivariant layers, read out."""

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False,
                 normalize=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)     # h^0: invariant, no pose info
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                                  act_fn=act_fn, residual=residual, attention=attention,
                                  normalize=normalize, tanh=tanh))

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x                                              # h invariant, x equivariant
```
