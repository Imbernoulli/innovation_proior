I start at the cheapest place that still earns the word "geometric." The scaffold's default fill is a
basic invariant message-passing GNN: a kNN graph, an edge MLP that concatenates the two endpoints with a
distance-plus-direction edge feature, mean aggregation, a residual node MLP. It works, and it is
invariant — but the way it is invariant bothers me, and chasing that discomfort is what lands the first
real rung. The default edge feature carries the unit direction vector `(pos[dst] - pos[src]) / dist`.
A direction is *not* rotation-invariant; rotate the protein and it rotates. The default gets away with
it only because the downstream classifier reads the graph embedding and could in principle learn to
ignore the directional channels, but that is invariance by hope, not by construction — exactly the
"learn the symmetry from data" trap I want to refuse. So my first move is to ask what the smallest,
most honest invariant geometric encoder is, where invariance is structural and not left to the head to
clean up. I want every geometric quantity that touches a message to be invariant on its own, and I want
the filter that reads geometry to be *learned* rather than hand-built.

Let me write down what the rigid-motion constraint actually buys me, because the whole design hangs off
two facts. An orthogonal matrix `Q` preserves inner products and therefore norms: `(Qa)·(Qb) = a·b`. A
shared translation cancels in a difference: `(a+g) - (b+g) = a - b`. Put them together on a pair of
residues. The relative difference `pos_i - pos_j` is translation-invariant but rotates with `Q`; its
norm, the distance `d_ij = ‖pos_i - pos_j‖`, is invariant to translation *and* rotation *and*
reflection. So the distance is a genuine E(3)-invariant scalar built purely from geometry. If the only
geometric input a message ever sees is `d_ij`, the whole encoder is invariant by construction — no
equivariant coordinate channel, no spherical harmonics, nothing to prove per layer. That is the cheap
route and for the first rung I take it without hesitation: feed messages only distances, and the
embeddings cannot depend on the global pose.

Now the part that makes this more than a plain distance-GNN: how does the distance enter the message?
The blunt option is to bin it — a lookup table indexed by which distance bucket `d_ij` falls in, the
molecular cousin of a voxel grid or a one-hot bond type. But that reintroduces exactly the discreteness
I am trying to avoid: as a residue drifts and `d_ij` crosses a bin boundary, its contribution snaps from
one tap to the next, and the encoder's function of geometry has a jump. The root cause of every such
jump is that the filter is a table over a *discrete* index. So replace the table with a *function* of the
continuous distance: a small neural network `W(·)` that maps the scalar `d_ij` to a vector of filter
values, and convolve over neighbors by an elementwise (feature-wise) product,
`m_i = Σ_j (lin1·h_j) ∘ W(d_ij)`. There is no grid anymore; a residue at any distance contributes
through `W` evaluated at its exact `d_ij`, and as it moves, `W(d_ij)` moves continuously. This is the
continuous-filter convolution, and it is the conceptual core of the first rung. I make the filter
feature-wise rather than a full matrix so it costs `O(num_filters)` per edge instead of `O(F^2)`, and I
leave cross-channel mixing to ordinary per-node linear layers around it — geometry in the filter,
feature recombination in the dense layers.

There is a sharp practical failure I have to design around before this trains at all. The filter network
takes the single scalar `d_ij` and must emit many filter channels. At initialization a net is nearly
linear, so each output channel is approximately the *same* linear ramp in `d_ij`, just scaled — the
channels are almost identical, carrying one effective degree of freedom instead of many. That is a flat
plateau at the start of training with no diversity among filters to exploit. The fix is to lift the
single scalar into a representation where different channels naturally see different things: expand the
distance in a bank of Gaussians, `e_k(d) = exp(-γ (d - μ_k)^2)`, with centers `μ_k` on a uniform grid
from 0 to the cutoff and width matched to the spacing (`γ = 1/(2Δ^2)`). Now a given distance lights up
the few Gaussians whose centers are near it and leaves the rest near zero, so even a near-linear filter
net produces diverse filters — different channels latch onto different distance ranges. The number of
centers is the filter's resolution; the span of the centers is its size. Crucially the Gaussian
expansion is itself smooth, so I have not reintroduced any discontinuity.

The nonlinearity matters more than usual here. The cleanest activation for a geometric energy/feature
network is one that is smooth to all orders, because the function will be differentiated through and I do
not want kinks in the geometry response. The shifted softplus, `ssp(x) = softplus(x) - ln 2 =
ln(0.5 e^x + 0.5)`, is the C-infinity cousin of ReLU: it bends instead of cornering, and `ssp(0) = 0` so
zero pre-activations map to zero and the activations stay centered. I use it in the filter net and in the
per-node layers. And I fold a smooth cosine cutoff into the filter, `f_cut(d) = 0.5[1 + cos(π d /
cutoff)]`, whose value *and* slope are both zero at the cutoff, so a neighbor crossing the cutoff
boundary fades out without a jump rather than vanishing abruptly. The effective filter is
`W(d) = filter_net(e(d)) · f_cut(d)`.

The depth is what turns strictly pairwise radial filters into genuinely many-body structure, and this is
the payoff that makes a deep stack worth it. One continuous-filter convolution lets residue `i` feel each
neighbor `j` individually — pairwise. But after one block, `h_j` has already absorbed information about
`j`'s own neighbors `k`. So in the next block, when `i` pulls in `h_j`, it is implicitly pulling in
something that knows about `k` — `i` feels the `(i,j,k)` triple. A few blocks and a residue's
representation reflects its spatial environment several hops out, all while every individual filter only
ever looked at a single invariant distance. The depth manufactures the many-body character; the
invariance is never spent because each filter is radial. I wrap each block in a residual connection —
`h ← h + InteractionBlock(h)` — so a deep stack stays trainable and early local features survive to the
output, and I give each block its own weights (unshared), so earlier blocks build short-range structure
and later blocks build on it.

Now I make this concrete *inside this task's edit surface*, and here the implementation departs from the
generic continuous-filter encoder in ways I want to be explicit about, because the harness fixes choices
the generic version leaves open. First, I do not hand-roll the convolution; I use the components the
geometry library already ships — the interaction block (continuous-filter convolution plus shifted
softplus plus a linear), the Gaussian smearing, and the shifted-softplus activation — wired in the same
residual pattern. Second, there are no forces here and no energy: this is an *encoder*, so the landing
artifact is the node and graph embeddings, not a scalar with an autograd gradient. The whole
twice-differentiable, force-conserving story that motivates the smooth activation in the molecular
setting is *absent* from the harness; I keep the smooth activation anyway because it is the canonical
choice and costs nothing, but I am honest that the constraint that originally forced it is not present
here. Third, the graph: the encoder is handed raw `pos` and `batch` and must build its own edges, so I
build a **kNN graph** with `k = max_neighbors`, not a radius graph and not a provided spatial graph. The
edge weight is the Euclidean distance on those kNN edges; the edge attribute is the Gaussian expansion of
that distance. Fourth, the widths: I override the constructor defaults to the reference encoder
configuration — `hidden_channels = 512`, `num_filters = 128`, `num_gaussians = 50`, `cutoff = 10.0`,
`max_num_neighbors = 32`, six interaction layers, and an `add` (sum) readout for the graph embedding.
The output head is a linear from `hidden_channels` to `hidden_channels`, a shifted softplus, then a
linear to `out_dim` — the per-node embedding — and the graph embedding is the sum-scatter of the node
embeddings over the batch. The embedding lookup that in the molecular model is a per-element table
becomes here a plain `Linear(input_dim, hidden_channels)` over the 28-dim node features, because the
input is a feature vector rather than a single atom type. The whole scaffold module is in the answer.

So the first rung is the continuous-filter encoder: distances are the only geometric input (invariant by
construction), each is Gaussian-expanded to decorrelate the filter channels, a learned smooth filter with
a cosine cutoff carries the geometry into feature-wise messages, six unshared residual interaction blocks
build multi-scale structure out of radial pairwise filters, and a sum readout gives the per-protein
vector. It is the cheapest encoder that *learns* its geometry while staying honestly invariant — which is
exactly why it is the right floor to start the ladder from rather than the default fill, whose directional
edge feature was invariant only by the head's good graces.

Now I reason about what this floor must do, because that is the point of running it. The signal it can use
is entirely scalar-radial: residue `i` knows how far each neighbor is, and through depth, the distance
texture of its neighborhood several hops out. What it structurally *cannot* see is direction. Two
geometries that present the same multiset of neighbor distances are indistinguishable to it — the classic
illustration is that a single ring and two separate smaller rings with matching bond lengths look
identical from every node, because each node's distance-neighborhood is the same in both. In a folded
protein this is exactly the kind of degeneracy that matters: whether two contacts of a residue go off in
nearly the same direction or in opposite directions is a distance-blind quantity, and it is part of what
distinguishes one local fold from another. So I expect the continuous-filter encoder to get real traction
— it is a learned, invariant geometric encoder, far better than ignoring structure — but to be capped by
its radial blindness, and to be capped hardest on the task that most needs fine fold discrimination.

I expect the three benchmarks to split on how much directional and relational structure the metric
demands. EC reaction asks for the enzyme class, which is correlated with broad active-site geometry and
with sequence composition that the 28-dim node features already carry, so a radial encoder should do
*decently* there — this is where I expect the floor to look least embarrassing. GO biological process is
multilabel over a huge label set scored by threshold-swept f1_max, a coarse signal that even a modest
graph embedding can move, so I expect a middling-but-real number. Fold classification is the hardest for a
radial encoder by construction: distinguishing among ~1200 folds is precisely the task where two
different folds can share local distance statistics and differ in directional arrangement, so I expect
Fold to be the *lowest* of the three and the clearest evidence of the radial ceiling. Whatever the
precise split, the diagnosis is already pointed at the next rung: if the floor is capped by being unable
to read direction, the fix is to give the encoder a channel that carries and transforms directional —
type-1 — information through the layers, which is exactly the move the steerable ancestors made
expensively and which a cheaper equivariant coordinate update can make for almost nothing. That is the
second rung, and the Fold number here is the bar it has to clear.
