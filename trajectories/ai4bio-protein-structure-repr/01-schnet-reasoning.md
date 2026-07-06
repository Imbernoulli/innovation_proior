I start at the cheapest place that still earns the word "geometric." The scaffold's default fill is a
basic invariant message-passing GNN: a kNN graph, an edge MLP that concatenates the two endpoints with a
distance-plus-direction edge feature, mean aggregation, a residual node MLP. It works, and it is
invariant — but the way it is invariant bothers me, and chasing that discomfort is what lands the first
real rung. The default edge feature carries the unit direction vector `(pos[dst] - pos[src]) / dist`,
three numbers stacked next to the one distance number. A direction is *not* rotation-invariant; rotate the
protein by an orthogonal `Q` and those three numbers become `Q · direction` — three of the four edge
features move under the group. The graph embedding is a function of these edge features, so unless the
downstream classifier head learns to average over pose and zero the directional channels, two orientations
of the same protein land at two different embeddings and the head sees them as two different inputs.
Nothing in the fixed loss forces that averaging, and there is no rotation augmentation in the pipeline to
teach it. So the default is invariant only if the head cleans up after it — invariance by hope, not by
construction, exactly the "learn the symmetry from data" trap I want to refuse at the floor. My first move
is therefore to ask what the smallest, most honest invariant geometric encoder is: one where every
geometric quantity that touches a message is invariant on its own, and where the filter that reads geometry
is *learned* rather than hand-built.

Let me write down what the rigid-motion constraint actually buys me, because the whole design hangs off
two facts. An orthogonal matrix `Q` preserves inner products and therefore norms: `(Qa)·(Qb) = a·b`. A
shared translation cancels in a difference: `(a+g) - (b+g) = a - b`. Put them together on a pair of
residues. The relative difference `pos_i - pos_j` is translation-invariant but rotates with `Q`; its
norm, the distance `d_ij = ‖pos_i - pos_j‖`, is invariant to translation *and* rotation *and*
reflection. I want to see this survive a concrete transform rather than trust the algebra abstractly, so
I take three residues at `r1 = (0,0,0)`, `r2 = (3,0,0)`, `r3 = (0,4,0)`, giving `d12 = 3`, `d13 = 4`,
`d23 = 5`. Rotate ninety degrees about `z` so `(x,y,z) ↦ (-y,x,z)`: `r1 ↦ (0,0,0)`, `r2 ↦ (0,3,0)`,
`r3 ↦ (-4,0,0)`, and recomputing gives `d12 = 3`, `d13 = 4`, `d23 = √(16+9) = 5` — every distance intact.
Translate the rotated set by `(10,10,10)` and the differences are untouched. Reflect `x ↦ -x` and I get
`d12 = 3`, `d13 = 4`, `d23 = 5` again. All three distances survive every element of `E(3)`. So the
distance is a genuine `E(3)`-invariant scalar built purely from geometry, and if the only geometric input
any message ever sees is `d_ij`, the whole encoder is invariant by construction — no equivariant
coordinate channel, no spherical harmonics, nothing to prove per layer. That is the cheap route, and for
the first rung I take it without hesitation: feed messages only distances, and the embeddings cannot depend
on the global pose.

Before committing I want to be honest that this is a real choice among alternatives, so I lay out what the
floor could be. One option is a hand-crafted symmetry-function encoder in the Behler–Parrinello spirit:
give each residue a fixed descriptor vector — radial terms `Σ_j exp(-η(d_ij - R_s)²) f_cut(d_ij)` and
angular terms `Σ_{j,k} (1 + cos θ_ijk)^ζ …` — and feed it to a small per-residue net. It is honestly
invariant, but the descriptor bank is *hand-tuned*: I would be picking the `η`, `R_s`, `ζ` by hand, which
is a step backward from the very thing I am replacing, because the point of the floor is to *learn* the
geometry filter, not hand-design it; and the angular sum is `O(deg²)` per node, so at `k = 32` neighbors
that is on the order of a thousand triples per residue for a fixed, unlearned basis. A second option is
the steerable route the ancestors took — carry type-1 and higher features through every layer via
spherical harmonics and Clebsch–Gordan tensor products, so the layer commutes with `Q` by construction and
can even emit vectors. It is the most expressive of the three, but the harmonics must be recomputed per
geometry and the tensor products are the heavy part, and the whole apparatus is welded to three dimensions.
That is far more machinery than a *floor* deserves; if pure radial proves insufficient I will reach for
something with directional structure, but I refuse to pay for the full steerable apparatus before I have
measured what distance-only geometry buys. The third option is a distance-only message-passing net whose
filter is *learned* — cheapest, honestly invariant, and it learns its geometry instead of having it
hand-specified. I eliminate the first two on hand-tuning and cost and take the third.

Now the part that makes this more than a plain distance-GNN: how does the distance enter the message? The
blunt option is to bin it — a lookup table indexed by which distance bucket `d_ij` falls in, the molecular
cousin of a voxel grid or a one-hot bond type. Say I used half-Ångström bins across the ten-Ångström
window, twenty taps; then a residue drifting from `0.49` to `0.51` Å across a boundary flips its entire
filter tap discontinuously, and the encoder's function of geometry jumps precisely where a smooth physical
quantity crossed an arbitrary grid line. The root cause of every such jump is that the filter is a table
over a *discrete* index. So I replace the table with a *function* of the continuous distance: a small
neural network `W(·)` that maps the scalar `d_ij` to a vector of filter values, convolved over neighbors by
an elementwise (feature-wise) product, `m_i = Σ_j (lin1·h_j) ∘ W(d_ij)`. There is no grid anymore; a
residue at any distance contributes through `W` evaluated at its exact `d_ij`, and as it moves, `W(d_ij)`
moves continuously. This is the continuous-filter convolution, and it is the conceptual core of the first
rung. I make the filter feature-wise rather than a full matrix for a reason I can put a number on: a
feature-wise filter emits `num_filters = 128` values per edge and multiplies them in elementwise, whereas a
full-matrix filter `W(d) ∈ ℝ^{512×512}` would have the filter net emit `262,144` numbers on every edge, and
with roughly `32` neighbors across a couple hundred residues — call it `6,400` edges — that is on the order
of `1.7` billion filter outputs per protein per forward, which is absurd. The feature-wise filter is
`O(num_filters)` per edge, and I leave cross-channel mixing to ordinary per-node linear layers around it:
geometry in the filter, feature recombination in the dense layers.

There is a sharp practical failure I have to design around before this trains at all. The filter network
takes the single scalar `d_ij` and must emit many filter channels. At initialization a net is nearly
linear, so each output channel is approximately `a_c · d_ij + b_c` — the *same* linear ramp in `d_ij`, just
scaled — so the 128 channels are almost identical, carrying one effective degree of freedom instead of many.
That is a flat plateau at the start of training with no diversity among filters to exploit, and no strong
gradient toward diversifying them. The fix is to lift the single scalar into a representation where
different channels naturally see different things: expand the distance in a bank of Gaussians,
`e_k(d) = exp(-γ (d - μ_k)²)`, with centers `μ_k` on a uniform grid from 0 to the cutoff and width matched
to the spacing. Concretely with 50 centers over `[0, 10]` the spacing is `Δ = 10/49 = 0.204` Å, and the
smearing sets `γ = 0.5/Δ² = 12.0`, so each Gaussian's standard deviation is `σ = 1/√(2γ) = Δ = 0.204` —
neighboring bumps overlap at `exp(-0.125) = 0.88` at their midpoint and fall to `exp(-0.5) = 0.61` a full
spacing away, so the expansion is smooth with no bin edges anywhere. Tracing a specific distance makes the
decorrelation visible: at `d = 7.0` the centers `μ_33 = 6.74`, `μ_34 = 6.94`, `μ_35 = 7.14`, `μ_36 = 7.35`
light up to `0.43, 0.96, 0.78, 0.24` and essentially everything else is below `0.1` — a handful of active
coordinates whose *pattern* is unique to `d = 7.0`. Now even a near-linear map `W · e(d)` produces
genuinely different channels across distances, because the 50 inputs it reads already differ by distance;
different channels latch onto different distance ranges instead of all echoing one ramp. The number of
centers is the filter's resolution and the span of the centers its size, and crucially the Gaussian
expansion is itself smooth, so I have not reintroduced any discontinuity.

The nonlinearity matters more than usual here. The cleanest activation for a geometric feature network is
one that is smooth to all orders, because the function may be differentiated through and I do not want kinks
in the geometry response. The shifted softplus, `ssp(x) = softplus(x) - ln 2 = ln(0.5 e^x + 0.5)`, is the
`C∞` cousin of ReLU: it bends instead of cornering, and `ssp(0) = 0` so zero pre-activations map to zero
and the activations stay centered. I use it in the filter net and in the per-node layers. And I fold a
smooth cosine cutoff into the filter, `f_cut(d) = 0.5[1 + cos(π d / cutoff)]`, and I check by hand that
both its value *and* its slope vanish at the boundary rather than just asserting it: `f_cut(C) =
0.5[1 + cos π] = 0.5[1 - 1] = 0`, and differentiating, `f_cut'(d) = -(π/2C) sin(π d / C)`, so `f_cut'(C) =
-(π/2C) sin π = 0`. Value and derivative both zero at `d = C` means a neighbor crossing the cutoff fades
out with zero slope — no jump and no kink, unlike a hard truncation that would drop a contribution
abruptly. The effective filter is `W(d) = filter_net(e(d)) · f_cut(d)`.

The depth is what turns strictly pairwise radial filters into genuinely many-body structure, and this is
the payoff that makes a deep stack worth it. One continuous-filter convolution lets residue `i` feel each
neighbor `j` individually — pairwise, `h_i^{(1)}` a function of `{d_ij}` and the neighbors' `h_j^{(0)}`.
But after one block, `h_j^{(1)}` has already absorbed `{d_jk : k ∈ N(j)}`. So in the next block, when `i`
pulls in `h_j^{(1)}`, it is implicitly pulling in something that knows about `k` — `i` now depends on the
pair `(d_ij, d_jk)` jointly along every path `i–j–k`, and the triangle relation among `d_ij`, `d_jk`, and
(where the edge exists) `d_ik` is exactly what constrains the angle at `j`. A few blocks and a residue's
representation reflects its spatial environment several hops out, all while every individual filter only
ever looked at a single invariant distance. The depth manufactures the many-body character; the invariance
is never spent because each filter is radial. I wrap each block in a residual connection —
`h ← h + InteractionBlock(h)` — so a deep stack stays trainable and early local features survive to the
output, and I give each block its own weights (unshared), so earlier blocks build short-range structure and
later blocks build on it.

Now I make this concrete *inside this task's edit surface*, and here the implementation departs from the
generic continuous-filter encoder in ways I want to be explicit about, because the harness fixes choices the
generic version leaves open. First, I do not hand-roll the convolution; I use the components the geometry
library already ships — the interaction block (continuous-filter convolution plus shifted softplus plus a
linear), the Gaussian smearing, and the shifted-softplus activation — wired in the same residual pattern.
Second, there are no forces here and no energy: this is an *encoder*, so the landing artifact is the node
and graph embeddings, not a scalar with an autograd gradient. The whole twice-differentiable,
force-conserving story that motivates the smooth activation in the molecular setting is *absent* from the
harness; I keep the smooth activation anyway because it is the canonical choice and costs nothing, but I am
honest that the constraint that originally forced it is not present here. Third, the graph: the encoder is
handed raw `pos` and `batch` and must build its own edges, so I build a **kNN graph** with `k =
max_neighbors`, not a radius graph and not a provided spatial graph. The edge weight is the Euclidean
distance on those kNN edges; the edge attribute is the Gaussian expansion of that distance. Fourth, the
widths: I override the constructor defaults to the reference encoder configuration — `hidden_channels =
512`, `num_filters = 128`, `num_gaussians = 50`, `cutoff = 10.0`, `max_num_neighbors = 32`, six interaction
layers, and an `add` (sum) readout for the graph embedding. The output head is a linear from
`hidden_channels` to `hidden_channels`, a shifted softplus, then a linear to `out_dim` — the per-node
embedding — and the graph embedding is the sum-scatter of the node embeddings over the batch. The embedding
lookup that in the molecular model is a per-element table becomes here a plain `Linear(input_dim,
hidden_channels)` over the 28-dim node features, because the input is a feature vector rather than a single
atom type. It is worth a budget check to confirm this is genuinely a *floor* and not a heavy model in
disguise: each interaction block runs about `0.42M` parameters (filter MLP `23k` + `CFConv` `lin1` `66k` +
`lin2` `66k` + post-convolution linear `263k`), so six blocks are `~2.5M`, and with the `15k` embedding and
the `~0.33M` output head the whole encoder is `~2.8M` parameters — small, which is exactly right for the
cheapest honest starting point.

Two choices in that construction deserve their own reasoning rather than being inherited by default. The
first is the graph builder. I could connect residues by a radius rule — every pair within the cutoff — or
by kNN. A radius rule reports true local density: a residue in a tightly packed core gets many edges and a
residue on a loose loop gets few, which is honest but leaves loosely packed regions nearly edgeless, so
some residues would receive almost no messages and the encoder would have nothing to propagate through
them. For the floor I want a guaranteed degree everywhere so no residue is starved, so I take **kNN** with
`k = 32`: every residue gets exactly 32 neighbors regardless of how crowded its neighborhood is. The cost
is that kNN flattens away density variation in packed regions — a core residue is capped at 32 even if
fifty residues sit within the cutoff — but the continuous filter can still read the *distances* to those 32,
and with the cosine cutoff folded in, neighbors beyond the cutoff that sneak into the kNN list are faded
toward zero anyway, so the effective neighborhood is the near ones. It is the right trade for a floor whose
job is to never starve a node. The second choice is the readout. A mean pool would make the graph embedding
independent of protein length, which is tidy, but it also throws away size, and enzyme and fold classes are
not size-blind; a sum pool keeps a notion of the total accumulated signal and the protein's extent, which
the fixed classifier head can use, at the cost of the embedding scaling with length. I follow the reference
and take the `add` (sum) readout, accepting that the downstream `LayerNorm`-free head will have to cope with
length-dependent magnitudes — a thing I will keep an eye on if any task looks length-confounded.

It helps to trace the shapes through one interaction block once, so I know the wiring is consistent before I
trust the stack. A node feature `h_j` enters at width `512`; `CFConv.lin1` (no bias) projects it to the
filter width `128`; the filter `W(d_ij)`, itself a `128`-vector from `filter_net(e(d_ij)) · f_cut(d_ij)`,
multiplies it elementwise, still `128`; those per-edge products scatter-sum into each destination node,
`128`; `CFConv.lin2` lifts the aggregate back to `512`; a shifted softplus and the block's post-linear keep
it at `512`; and the residual adds it onto the incoming `512`. Everything closes at `512` from block to
block, the geometry only ever entered as the `50`-vector `e(d_ij)` feeding a filter net whose output is the
`128`-vector `W`, and the only place a distance touched the computation is inside that filter — precisely
the invariance guarantee I wanted, now checked at the level of shapes. The whole scaffold module is in the
answer.

So the first rung is the continuous-filter encoder: distances are the only geometric input (invariant by
construction), each is Gaussian-expanded to decorrelate the filter channels, a learned smooth filter with
a cosine cutoff carries the geometry into feature-wise messages, six unshared residual interaction blocks
build multi-scale structure out of radial pairwise filters, and a sum readout gives the per-protein vector.
It is the cheapest encoder that *learns* its geometry while staying honestly invariant — which is exactly
why it is the right floor to start the ladder from rather than the default fill, whose directional edge
feature was invariant only by the head's good graces.

Now I reason about what this floor must do, because that is the point of running it. The signal it can use
is entirely scalar-radial: residue `i` knows how far each neighbor is, and through depth, the distance
texture of its neighborhood several hops out. What it structurally *cannot* see at the layer that reads
`i`'s own edges is direction. Make the blindness concrete: suppose two neighbors `j` and `k` both sit at
`5` Å from `i`. On its edges `i` receives `d_ij = 5` and `d_ik = 5` whether `j` and `k` are `60°` apart —
`5` Å from each other — or `180°` apart — `10` Å from each other. The per-edge message carries only the
distance *to* `i`, so the angle at `i` is invisible where `i`'s edges are read; it leaks back only weakly
and lossily through multi-hop distance texture, if `j` and `k` happen to be neighbors so that `d_jk` rides
inside `h_j` and `h_k` and reaches `i` a layer later. Two geometries that present the same multiset of
neighbor distances *and* the same multi-hop texture are therefore indistinguishable to it — the classic
illustration is that a single ring and two separate smaller rings with matching bond lengths look identical
from every node, because each node's distance-neighborhood is the same in both. In a folded protein this is
exactly the kind of degeneracy that matters: whether two contacts of a residue go off in nearly the same
direction or in opposite directions is a distance-blind quantity, and it is part of what distinguishes one
local fold from another. So I expect the continuous-filter encoder to get real traction — it is a learned,
invariant geometric encoder, far better than ignoring structure — but to be capped by its radial blindness,
and to be capped hardest on the task that most needs fine fold discrimination.

I expect the three benchmarks to split on how much directional and relational structure the metric demands.
EC reaction asks for the enzyme class, which is correlated with broad active-site geometry and with sequence
composition that the 28-dim node features already carry, so a radial encoder should do *decently* there —
this is where I expect the floor to look least embarrassing. GO biological process is multilabel over a huge
label set scored by threshold-swept f1_max, a coarse signal that even a modest graph embedding can move, so
I expect a middling-but-real number. Fold classification is the hardest for a radial encoder by
construction: distinguishing among the roughly twelve hundred folds is precisely the task where two
different folds can share local distance statistics and differ in directional arrangement, so I expect Fold
to be the *lowest* of the three and the clearest evidence of the radial ceiling. Whatever the precise
split, the diagnosis is already sharp: if the floor is capped by being unable to read
direction, what it lacks is a channel that carries and transforms directional — type-1 —
information through the layers rather than collapsing every pair to a scalar the instant a message is built,
the degree of freedom the steerable ancestors carried expensively and that I refused to pay for at the
floor. The Fold number here is the bar a richer encoder has to clear.
