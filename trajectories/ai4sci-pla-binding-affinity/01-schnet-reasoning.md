The architecture is the whole contribution here, and before I reach for anything clever I want a
floor — the simplest fill of the `AffinityModel` slot that takes the binding problem seriously, so the
methods that follow have something to beat and something to diagnose against. The scaffold default is
not that floor: it runs two independent encoders over the ligand and the pocket, mean-pools each,
concatenates, and regresses, and it *ignores the inter-molecular edges entirely*. That is a non-starter
physically, and the why is the whole problem. Binding affinity is a free energy, and the free energy of
binding comes almost entirely from the non-covalent contacts *across the interface* — hydrogen bonds,
salt bridges, van der Waals packing, π-stacking — each between a particular ligand atom and a
particular pocket atom. The covalent skeletons inside each molecule exist whether or not the ligand is
in the pocket; they set the conformation but they are not the binding. A model that encodes the two
molecules separately and never lets a ligand atom see the pocket atom it is touching can only guess
`-logKd/Ki` from the *marginal* shapes of the two molecules — how big the ligand, how many rings, how
large the pocket — and those marginals are the dataset shortcut I distrust: they correlate with
affinity on the training distribution for reasons that are not binding and will not survive a held-out
split where a big ligand binds weakly or a small one tightly. My floor has to at least let the
interface edges into the message passing; anything less is measuring molecular size, not binding.

So what is the simplest principled fill? What physics will not let me get away with pins the design
harder than any architectural taste. The affinity is a scalar property of the *arrangement* of atoms,
and it cannot change if I pick up the whole complex and translate it by `t` or rotate it by `Q`; a
model that gives two affinities for the same complex viewed from two angles is predicting something
that is not physics. I refuse to learn that invariance from augmentation. The proper-rotation group is
three-dimensional, so covering it to within ~10° needs an ε-net of order `(180/10)³ ≈ 6000` rotated
copies of *every* complex — turning an order-10⁴ dataset into order-10⁷ — and even then the invariance
is only approximate, a residual the network can still exploit. Built into the representation it is exact
and free. What survives a rigid motion? Under `r → Q r + t` a raw coordinate scrambles and a difference
`(r_i − r_j) → Q(r_i − r_j)` still rotates, but its length `‖Q(r_i − r_j)‖ = ‖r_i − r_j‖` (since
`QᵀQ = I`) is fixed under translation, rotation, and reflection alike. The interatomic *distance* is the
free invariant the geometry hands me: build everything out of distances and I am invariant, full stop;
let a raw coordinate or bare difference vector leak in and I break it.

There is one honest cost buried in "reflection both." Distances are invariant under improper rotations
too, so a distance-only model is blind to chirality: a molecule and its mirror image have identical
pairwise-distance matrices and get the identical prediction, yet enantiomers can bind a chiral pocket
very differently. But the atom featurization I am handed — 35-dim one-hots of
element/degree/valence/hybridization/aromaticity/H-count — carries no R/S stereo tag, so the chirality
signal is absent from the inputs before I make any architectural choice. I am not discarding
information the harness gave me; I am accepting a limitation the featurization already imposes, in
exchange for exact rigid-motion invariance. A good trade at the floor.

And the edit surface is more restrictive than a from-scratch geometric net would assume, which shapes
the floor. The model is *not* handed raw 3D coordinates. The `PLABatch` edges already carry precomputed,
rigid-motion-invariant geometry: intra-molecular edges have 17 features (bond chemistry plus 11
geometric numbers — angle, triangle-area, and neighbour-distance statistics, plus pairwise L1 and L2),
inter-molecular edges the same 11. The invariance work is done for me in the features; there are no
positions to recompute distances from. The last geometric channel is the L2 distance scaled by 0.1, so
a layer wanting a scalar distance in angstroms reads `edge_attr[:, -1:] * 10`. That single scalar is
the only clean geometric handle, and it is enough to build a distance-driven message.

Which geometric-GNN idea do I instantiate first, and why not a richer one? At the floor the job is to
isolate one variable — what a purely distance-driven, invariant, interface-aware model can do — so that
when a later method adds the 11-dim geometry or a heavier readout the delta is attributable. That rules
out pouring the full edge geometry in now (I could never say what the angles bought) and rules out a
transformer over contacts (more machinery than a floor should carry). The real choice is narrower: feed
the bare scalar distance into an MLP, or expand it in a fixed Gaussian basis first. And that turns on a
concrete init pathology. Feed one scalar `d` into `Linear(1, H)`: at init every pre-activation channel
is `w_j·d + b_j`, an affine function of the *same* variable, so across a batch of edges the `H` columns
are all scalar multiples of one distance vector — rank one in the distance direction. A smooth monotone
nonlinearity keeps them comonotone, so the filter has effectively *one* degree of freedom in `d`: it can
scale a neighbour up or down as a single monotone function of distance and nothing more — it cannot
"boost contacts near 3 Å but suppress those near 4 Å," which needs two independent distance windows.
Training has to break that rank-one degeneracy from a plateau, and it crawls. A bank of Gaussians
centered on a grid instead feeds the filter near-decorrelated inputs — a short distance lights the near
centers, a long one the far — so the filter's first layer produces `H` distinct combinations of
near-orthogonal bumps and starts diverse. That decorrelation argument is decisive; I take the basis
expansion.

Fix the basis: centers 0 to 6 Å at 0.1 Å spacing — `torch.arange(0.0, 6.0, 0.1)` gives 60 — with width
σ equal to the 0.1 Å gap. That width is a smooth overlapping partition, not a near one-hot: a distance
halfway between two centers sits 0.05 Å from each and activates both at `exp(−0.5·0.5²) = 0.88`, no dead
gaps, and each distance meaningfully excites a band of about ±0.25 Å, roughly five centers. The two
regimes I care about land on disjoint parts of the basis — a covalent bond near 1.5 Å excites centers
around 1.4–1.6, a non-covalent contact near 4 Å excites 3.8–4.2 — so a downstream filter can respond to
bonds and contacts independently. The 6 Å ceiling gives headroom above the 5 Å contact cutoff so the
longest real contacts sit inside the basis rather than at its poorly-conditioned edge.

With the basis fixed, the message is the continuous-filter convolution: map the RBF expansion to a
per-edge filter `W = filter_net(RBF(d))` through a two-layer MLP with a Softplus middle, gate the
projected neighbour elementwise by `W`, sum over neighbours, add the destination's own feature as a
residual, pass through a Softplus output MLP: `h_dst ← h_dst + output(Σ_{s→dst} node_proj(h_s) ⊙
W_{s→dst})`. Softplus rather than ReLU because a response to a *continuous* physical distance ought to
be smooth — two contacts a hundredth of an angstrom apart should never see a kink. Only the invariant
distance enters, so every block is invariant and the whole predictor is invariant by construction, zero
augmentation.

But the continuous-filter convolution was built for a *single homogeneous molecule* where one shared
filter processes every edge. My complex is two molecules joined by an interface, with two physically
distinct edge kinds — stiff covalent bonds near a bond length, soft non-covalent contacts out to 5 Å —
living on disjoint parts of the RBF. One shared map would have to be correct at 1.5 Å and 4 Å
simultaneously, exactly the blur to avoid. The interaction-graph lineage this task lives in already
settled the fix, and I adopt it: keep one node set per molecule but four *separate* convolutions per
layer — covalent ligand, covalent pocket, non-covalent ligand→pocket, non-covalent pocket→ligand — all
computed in parallel from the same inputs, then summed per destination node type. A pocket atom gets its
covalent update plus the non-covalent update from ligand contacts pointing into it; a ligand atom gets
its covalent update plus pocket contacts. Sum, because covalent and non-covalent influences are
physically additive, and the shared input cleanly separates "what my own molecule tells me" from "what
my partner tells me." Each of the four is its own continuous-filter block with its own filter map, so
the two distance regimes get their own learned responses.

Three layers, width 256. The contact graph is a local 5 Å shell: one layer moves information across one
edge, so after three a ligand atom has heard from pocket atoms up to two covalent bonds beyond its
direct contacts, folding the pocket's local environment into the ligand and back. A fourth mostly starts
to homogenize node states across the shell — the oversmoothing to avoid. On budget this is not small: a
continuous-filter block at width 256 with a 60-dim RBF input is ≈ 279k params, twelve blocks ≈ 3.3M in
message passing alone, and the dual readout adds ≈ 1M more — a few million parameters against ~10⁴
training complexes, two orders of magnitude more parameters than examples. That is why the frozen harness
leans on early stopping, weight decay, gradient clipping, and dropout, and it is a standing risk: on
near-training benchmarks this capacity can memorize motifs and look strong, so the honest test of whether
it learned *binding* rather than *the training distribution* will be the temporally distant holdout, not
the core sets.

Now the readout, and here I deliberately do not fall back to "pool all atoms and regress," because the
interaction-graph skeleton offers the part that makes the floor meaningful: read affinity off the
interface. For each non-covalent contact I score a per-contact affinity from a low-rank triple product
of the projected source atom, projected destination atom, and projected edge geometry —
`e_proj ⊙ src_proj ⊙ dst_proj`, collapsed by a final linear — and sum over a complex's contacts in both
directions. The elementwise triple product is a rank-limited trilinear form that fires where source
atom, destination atom, and contact geometry agree in the same latent components: exactly "this kind of
atom meeting that kind of atom at this geometry is a favourable contact."

A raw sum over a *variable* number of contacts carries a size-dependent offset: a bigger complex simply
has more contacts in the 5 Å shell, many incidental, so the raw directional score drifts with atom count
independent of true binding. If each contact contributes about μ and a complex has `k` contacts, the raw
sum runs about `kμ`, extensive in `k`, while `-logKd/Ki` is not systematically larger for a bigger
complex. So per direction I subtract a bias correction whose attention is a softmax *over each complex's
contacts*. I should not overclaim what this does: the softmax weights sum to one, so the correction is an
*intensive*, scale-stable convex combination, and subtracting it from an extensive sum does not literally
cancel the `k`-growth — it hands the model a learned, count-aware per-complex reference to subtract off,
a de-biasing prior, not an exact normalization. Whether it is enough to keep size from dominating is
something only the numbers can tell me. I average the two corrected directional estimates, with the same
RBF expansion of the contact distance feeding the scoring. The dual heads live in the readout, but
`forward` emits a single averaged output, so I keep the harness's plain MSE and expose no `compute_loss`
hook.

One note on what this is *not*: it is the continuous-filter convolution's geometric core — RBF, Softplus
filter, elementwise gating — but not the canonical SchNet end-to-end. There are no forces to train (the
target is a scalar, not an energy whose gradient is a force), so the second-derivative machinery and
energy-plus-force loss are irrelevant, and I drop the cosine cutoff because the contacts are already
capped at 5 Å and the RBF bank only reaches 6 Å, so no neighbour drifts across a boundary mid-training.
What survives is the move that matters: distance → Gaussian expansion → learned filter → elementwise
gate, slotted into the heterogeneous interface skeleton.

What should this floor do? It sees the interface (unlike the default) and is invariant (unlike feeding
coordinates raw), but its only window onto geometry is the *scalar distance* through the RBF. The 11
geometric numbers include angles and triangle areas — a description of *orientation*, not just
separation — and the continuous filter discards all but the L2 distance. So two contacts at the same
distance but different orientation — an H-bond along its donor axis versus one splayed off at 40°, a
stacked aromatic pair versus a T-shaped one — are identical to a distance-only message, and binding
cares intensely about that difference. My a-priori expectation is that this floor is the weakest of the
geometric fills, and that it should suffer most on the temporally distant holdout — the largest, most
distributionally shifted test set, where the model must reason about contact geometry it has not seen
rather than recall a memorized motif, with only a scalar to reason with. If instead it matched the
edge-feature-rich methods, that would say the angle/area geometry is not pulling its weight and I would
stop adding it; I expect the opposite, and I expect the next method's gain to come from letting more
than the scalar distance into the message. (The full scaffold module is in the answer.)
