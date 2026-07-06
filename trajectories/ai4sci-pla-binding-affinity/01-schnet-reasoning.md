The architecture is the whole contribution here, and before I reach for anything clever I want a
floor — the simplest fill of the `AffinityModel` slot that actually takes the binding problem
seriously, so the methods that follow have something to beat and something to diagnose against. The
scaffold default is not that floor: it runs two independent encoders over the ligand and the pocket,
mean-pools each, concatenates, and regresses, and it *ignores the inter-molecular edges entirely*.
That is a non-starter physically, and it is worth saying exactly why, because the why is the whole
problem. Binding affinity is a free energy, and the free energy of binding comes almost entirely from
the non-covalent contacts *across the interface* — hydrogen bonds, salt bridges, van der Waals
packing, π-stacking — each between a particular ligand atom and a particular pocket atom. The covalent
skeletons inside each molecule exist whether or not the ligand is in the pocket; they set the
conformation but they are not the binding. A model that encodes the two molecules separately and never
lets a ligand atom see the pocket atom it is touching has thrown away the one thing that determines
the answer. So the default's `-logKd/Ki` can only be guessed from the *marginal* shapes of the two
molecules — how big the ligand is, how many aromatic rings it carries, how large the pocket — and
those marginals are exactly the dataset shortcut I distrust, because they correlate with affinity on
the training distribution for reasons that have nothing to do with the physics and will not survive a
held-out split where a big ligand binds weakly or a small one binds tightly. My floor has to at least
let the interface edges into the message passing; anything less is not measuring binding, it is
measuring molecular size.

So what is the simplest principled fill? Let me think about what physics will not let me get away
with, because that pins the design harder than any architectural taste. The affinity is a scalar
property of the *arrangement* of atoms, and it cannot change if I pick up the whole complex and
translate it or rotate it. If I slide ligand and protein together by a vector `t` or spin them by a
rotation `Q`, the answer must be byte-for-byte the same. A model that gives two different affinities
for the same complex viewed from two angles is predicting something that is not physics. I refuse to
learn that invariance from augmentation, and I want to be precise about why, because "just augment" is
the tempting shortcut. The proper-rotation group is three-dimensional; to approximate invariance by
covering it I would need an ε-net of orientations, on the order of `(π/ε)³` samples, so to pin
orientation to within ~10° I would need on the order of `(180/10)³ ≈ 6000` rotated copies of *every*
complex — turning an order-10⁴ dataset into order-10⁷ — and even then the invariance is only
approximate, a smoothed-over residual the network can still exploit. By construction it is exact and
free. So I want invariance built into the representation, not sprayed on by data. What survives a
rigid motion? Send every position to `Q r + t`. A raw coordinate changes completely. A difference of
two positions `(Q r_i + t) − (Q r_j + t) = Q(r_i − r_j)` loses the translation but still rotates. Its
length, though, `‖Q(r_i − r_j)‖ = √((r_i−r_j)ᵀ Qᵀ Q (r_i−r_j)) = ‖r_i − r_j‖` because `QᵀQ = I`, is
invariant to translation and to rotation and reflection both. The interatomic *distance* is the free
invariant the geometry hands me. If I build everything out of distances I am invariant, full stop; if
I let a raw coordinate or a bare difference vector leak in, I break it.

There is one honest cost buried in "reflection both" that I want to name now rather than discover
later. Distances are invariant under improper rotations too, so a distance-only model is blind to
chirality: a molecule and its mirror image have identical pairwise-distance matrices and will get the
identical prediction. Real binding is chirality-sensitive — enantiomers can bind a chiral pocket with
very different affinities. But the atom featurization I am handed is 35-dim one-hots of
element/degree/valence/hybridization/aromaticity/H-count, with no explicit R/S stereo tag, so the
chirality signal is already absent from the inputs before I make any architectural choice. I am not
throwing away information the harness gave me; I am accepting a limitation the featurization already
imposed, and gaining exact rigid-motion invariance in exchange. That is a good trade at the floor.

Now I look at what this task's harness actually exposes, because the edit surface is more restrictive
than a from-scratch geometric net would assume, and that restriction shapes the floor. The model is
*not* handed raw 3D coordinates. It is handed a `PLABatch` whose edges already carry precomputed,
rigid-motion-invariant geometry: the intra-molecular edges have 17 features (bond chemistry plus 11
geometric numbers — angle and triangle-area and neighbour-distance statistics, plus pairwise L1 and L2
distances), and the inter-molecular edges have those same 11 geometric numbers. The invariance work is
already done for me in the features; I do not get to recompute distances from positions because there
are no positions. The last geometric channel is the L2 distance (scaled by 0.1), so if a layer wants a
scalar distance per edge it reads `edge_attr[:, -1:] * 10` and gets back angstroms. That is the only
clean handle on a single geometric scalar the harness gives, and it is enough to build a
distance-driven message.

So which geometric-GNN idea do I instantiate first, and why that one and not a richer one? Let me
lay the real options on the table, because at the floor I am choosing deliberately for
*interpretability of the eventual failure*, not for the best possible number. Option one: feed the raw
scalar distance straight into a small MLP to produce the per-edge filter. Option two: expand the
distance in a fixed basis first — Gaussian radial functions on a grid — and map that expansion to the
filter. Option three: stop restricting myself to the distance and pour the full 11-dim edge geometry
(angles, triangle areas, neighbour statistics) into the message from the start. Option four: drop
message passing altogether and put a transformer/attention block over the contact edges. I reject
three and four for the same reason: this is the *floor*, and its whole job is to isolate one variable —
what a purely distance-driven, invariant, interface-aware model can do — so that when a later rung adds
the 11-dim geometry or a heavier readout, the delta is attributable. If I front-load the full geometry
now I cannot later say what the angles bought. And a transformer over contacts is more machinery than a
floor should carry and buries the message-passing skeleton I want to establish and reuse. So the real
choice is between options one and two, raw-distance-into-MLP versus a basis expansion, and that choice
turns on a concrete initialization pathology worth tracing.

Feed one scalar `d` into `Linear(1, H)`. At initialization the weights are small and roughly `N(0,
1/fan_in)`, so every pre-activation channel `j` is `w_j · d + b_j` — an affine function of the *same*
single variable. Across a batch of edges with varied distances, the `H` pre-activation columns are all
scalar multiples of the one distance vector plus a bias, i.e. the pre-activation matrix is rank-one in
the distance direction. A smooth monotone nonlinearity (Softplus, SiLU) warps each channel but keeps
them comonotone, so after the nonlinearity the `H` filter channels are still near-perfectly correlated
across edges: the whole filter has, in effect, *one* degree of freedom in `d` at init. It can scale a
neighbour up or down as a single monotone function of distance, and nothing more — it cannot carve out
"boost contacts near 3 Å but suppress those near 4 Å," because that needs two independent distance
windows and the filter has one. Training has to break this rank-one degeneracy channel by channel from
a plateau, and it crawls. Now do option two instead. A bank of Gaussians centered on a grid of
distances feeds the filter map `H`-ish decorrelated inputs: a short distance lights up the near
centers, a long distance the far ones, so the 60 basis channels are, across a batch of varied
distances, far from collinear — different bumps at different centers. The filter's first layer then
produces `H` *distinct* linear combinations of 60 near-orthogonal bumps, so the filter channels start
diverse and training does not have to manufacture that diversity from a rank-one start. That
decorrelation argument is decisive, so I take the basis expansion.

Let me fix the basis concretely and check it actually resolves the two distance regimes I care about.
I lay the centers from 0 to 6 Å at 0.1 Å spacing, which `torch.arange(0.0, 6.0, 0.1)` makes 60 of
(0.0 through 5.9), with the Gaussian width σ set equal to the 0.1 Å gap. Is that width sensible — not
so sharp the basis is a near one-hot with dead zones between centers, not so broad it blurs everything?
A distance falling exactly between two centers sits 0.05 Å from each, giving activation `exp(−0.5 ·
(0.05/0.1)²) = exp(−0.125) = 0.88` on both — so the basis is a smooth overlapping partition, no dead
gaps. One center away (0.1 Å) is `exp(−0.5) = 0.61`, at 0.15 Å it is `0.32`, at 0.2 Å `0.14`, at 0.25
Å `0.044`, at 0.3 Å `0.011`. So each distance meaningfully excites a band of about ±0.25 Å, roughly
five centers, smoothly — exactly the smooth, decorrelated, overlapping expansion I wanted. And the two
regimes land on disjoint parts of the basis: a covalent bond near 1.5 Å excites centers around
1.4–1.6, a non-covalent contact near 4 Å excites centers around 3.8–4.2, so the bond and contact
signals occupy different channels of the RBF and a downstream filter can respond to them
independently. The 6 Å ceiling gives headroom above the 5 Å contact cutoff so the longest real
contacts sit inside the basis rather than at its edge where the expansion is one-sided and poorly
conditioned. That is the floor's one geometric knob and it is set on purpose.

With the basis fixed, the message is the continuous-filter convolution: map the RBF expansion to a
per-edge filter `W = filter_net(RBF(d))` through a two-layer MLP with a Softplus in the middle, gate
the projected neighbour elementwise by that filter, sum over neighbours, add the destination's own
feature back as a residual, and pass the aggregate through a Softplus output MLP:

    h_dst ← h_dst + output( Σ_{s→dst} node_proj(h_s) ⊙ W_{s→dst} ).

Softplus rather than ReLU is the natural activation for this filter family — it is the smooth cousin of
ReLU, and a response to a *continuous* physical distance ought to be smooth, with no kinks, so that two
contacts a hundredth of an angstrom apart never see a discontinuity in the filter. Because only the
invariant distance enters, every block is invariant and the whole predictor is invariant by
construction, exactly as the physics demanded, with zero augmentation.

But the original continuous-filter convolution was built for a *single homogeneous molecule* where
every edge is the same kind of edge, processed by one shared filter. My complex is not homogeneous: it
is two molecules joined by an interface, with two physically distinct kinds of edge — stiff covalent
bonds around a bond length apart, and soft non-covalent contacts reaching out to 5 Å — and I just saw
that these live on disjoint parts of the RBF. If I throw both into one filter I am asking a single
distance-to-filter map to serve two regimes at once; even though the basis separates them, one shared
map has to learn a response that is correct at 1.5 Å and correct at 4 Å simultaneously, and the
homogeneous net's whole limitation was exactly that blur. The interaction-graph lineage this task lives
in already settled the skeleton that fixes this, and I adopt it rather than reinvent it: keep one node
set per molecule but four *separate* convolutions per layer — covalent on the ligand, covalent on the
pocket, non-covalent ligand→pocket, non-covalent pocket→ligand — all computed in parallel from the
same input features, then for each destination node type sum the contributions that land on it. A
pocket atom receives its own-molecule covalent update plus the non-covalent update from the ligand
contacts pointing into it; a ligand atom receives its covalent update plus the non-covalent update from
pocket contacts. Sum, because covalent and non-covalent influences on an atom are physically additive,
and using the same input features for both branches cleanly separates "what my own molecule tells me"
from "what my binding partner tells me." Each of the four convolutions is its own continuous-filter
block with its own filter map, so the covalent and non-covalent distance regimes get their own learned
responses instead of sharing one.

How many layers? The contact graph is a local 5 Å shell, and I want a protein atom's influence to
reach a few bonds into the ligand and back without oversmoothing the whole complex into an
indistinguishable mush. Trace the receptive field: one layer moves information across one edge, so
after one layer a ligand atom has heard from its bonded neighbours and its direct pocket contacts;
after three layers a ligand atom has heard from pocket atoms up to two covalent bonds beyond its direct
contacts, and the pocket's local environment has folded into the ligand's representation and back
again. Three rounds carry the joint covalent-plus-non-covalent context that far, and a fourth mostly
starts to homogenize node states across the whole shell, which is the oversmoothing I want to avoid.
So three layers, hidden width 256.

Before I go further, a budget check, because a floor that is quietly enormous is not a floor. One
`CFConv` at width 256 with a 60-dim RBF input is `filter_net` (`60→256`, `256→256`) = 81,408 params,
`node_proj` (`256→256`) = 65,792, and a two-layer `output` (`256→256` twice) = 131,584, so ≈ 279k
parameters per block. Four edge types times three layers is twelve blocks, ≈ 3.3M parameters in the
message passing alone; the node embeddings are negligible (`35→256` twice ≈ 18k), and the dual
interface readout with its two scoring heads and two attention-bias heads adds on the order of another
1M, so the whole floor is a few million parameters. Against a training set on the order of 10⁴
complexes that is roughly two orders of magnitude more parameters than examples — heavily
over-parameterized, which is exactly why the frozen harness leans on early stopping (patience 50 on
validation RMSE), weight decay `1e-6`, gradient clipping, and dropout in the head. I note this as a
standing risk: on the near-training benchmarks this capacity can memorize motifs and look strong, and
the honest test of whether it has learned *binding* rather than *the training distribution* will be the
temporally distant held-out split, not the core sets.

Now the readout, and here I deliberately do *not* fall back to "pool all atoms and regress," because
the interaction-graph skeleton offers something better and it is the part that makes the floor
meaningful: read the affinity off the interface. For each non-covalent contact I score a per-contact
affinity from a low-rank triple product of the projected source atom, the projected destination atom,
and the projected edge geometry — `e_proj ⊙ src_proj ⊙ dst_proj`, collapsed to a scalar by a final
linear — and sum the scalars over a complex's contacts, in both directions (ligand→pocket and
pocket→ligand). The elementwise triple product is a rank-limited trilinear form: it fires where the
source atom, the destination atom, and the contact geometry all agree in the same latent components,
which is exactly the statement "this kind of atom meeting that kind of atom at this geometry is a
favourable contact." Let me check the shapes so I know it composes: `prj_edge(rbf)` is `[E, H]`,
`prj_src(lig_h)[src]` is `[E, H]`, `prj_dst(poc_h)[dst]` is `[E, H]`, their elementwise product is
`[E, H]`, `fc_lp` collapses to `[E, 1]`, and `index_add_` over `inter_batch` scatters the per-contact
scalars into `[B, 1]` — the shapes line up and the sum is genuinely per-complex.

There is a wrinkle in a raw sum I have to confront, because it is the same wrinkle the additive form
always hits. A sum over a *variable* number of contacts carries a size-dependent offset: a complex with
more atoms simply has more contacts inside the 5 Å shell, many of them incidental rather than
favourable, so the raw directional score drifts with atom count independent of true binding strength.
If each contact contributes on the order of a mean μ and a complex has `k` contacts, the raw sum runs
about `kμ`, an *extensive* quantity that grows with `k`, while the label `-logKd/Ki` is not
systematically larger for a bigger complex. So I subtract a per-direction bias correction whose
attention is a softmax *over each complex's contacts*. I want to be honest about what this subtraction
does, because it is easy to overclaim. The softmax weights sum to one, so the attention-weighted
aggregate is an *intensive*, scale-stable quantity — a convex combination of per-contact terms — not an
extensive sum. Subtracting an intensive correction from an extensive sum does not literally cancel the
`k`-growth; what it does is hand the model a learned, count-aware, per-complex reference level to
subtract off, a soft baseline that the network can tune to counteract the systematic size drift. It is
a de-biasing prior, not an exact normalization, and whether it is enough to keep size from dominating
the score is something only the numbers can tell me. I average the two corrected directional estimates
for the prediction. For this rung this readout sits on top
of continuous-filter message passing, and the edge geometry fed into the scoring is the same RBF
expansion of the contact distance. I keep the harness's plain MSE loss here — the dual heads exist in
the readout, but the floor's `forward` emits a single averaged output, so the harness regresses that
against the label, and I do not expose a `compute_loss` hook.

One honest note about what this fill is *not*. It is the canonical continuous-filter convolution's
geometric core — RBF, Softplus filter, elementwise gating — but it is not the canonical SchNet
end-to-end. There are no forces to train (the target is a single scalar, not an energy whose gradient
is a force), so the second-derivative machinery and the energy-plus-force loss are irrelevant here, and
I drop the cosine cutoff because the contacts are already capped at 5 Å by the graph construction and
the RBF bank only reaches 6 Å, so there is no neighbour drifting across a boundary mid-training whose
message I would need to smooth continuously to zero. What survives is the move that matters: distance →
Gaussian expansion → learned filter → elementwise gate, slotted into the heterogeneous interface
skeleton.

Now reason about what this floor should do, because that is the entire point of running it. The model
sees the interface (unlike the default) and is invariant (unlike feeding coordinates raw), but its only
window onto geometry is the *scalar distance* through the RBF. The 11 geometric numbers per edge
include angles and triangle areas and neighbour-distance statistics — a rich local description of
*orientation*, not just separation — and the continuous filter ignores all of it except the single L2
distance channel. So I expect this to be a competent but *under-informed* predictor: it should beat the
interface-blind default comfortably, but it should sit at the bottom of any rung that actually uses the
full edge geometry or treats the covalent and non-covalent message passing with more chemistry-specific
structure. The distance-only filter is a real ceiling, and I can state the ceiling exactly: two
contacts at the same distance but different orientation — a hydrogen bond pointing along its donor axis
versus one splayed off at 40°, a stacked aromatic pair versus a T-shaped one — are *identical* to a
distance-only message, and binding cares intensely about that difference. So my falsifiable expectation
is that this floor lands as the weakest of the geometric fills — worst RMSE and worst Rp on the harder
benchmarks where the held-out chemistry rewards richer geometry, and where memorized distance
signatures from training cannot be reused. Concretely I expect the temporal holdout, the largest and
most distributionally distant test set, to be where the distance-only ceiling bites hardest: that is
the set where the model must *reason* about contact geometry it has not seen rather than recall a
motif, and it has only a scalar to reason with. If instead this floor matched the edge-feature-rich
methods, that would tell me the angle and area geometry is not pulling its weight on this data and I
would stop adding it; I expect the opposite, and I expect the next rung's gain to come precisely from
letting more than the scalar distance into the message. That is the diagnosis I want the first set of
numbers to either confirm or overturn. (The full scaffold module is in the answer.)
