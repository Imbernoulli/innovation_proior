GVP landed exactly where I feared and the numbers say why. On CATH 4.2 it recovered 0.4310 at
perplexity 5.8996, on CATH 4.3 0.4337 / 5.8209, on TS50 0.4602 / 5.4831 — the bottom of any ladder I'd
build. Before I diagnose, let me read the table properly rather than glancing at it. The two CATH sets
sit almost on top of each other, 0.4310 and 0.4337, a gap of 0.0027 that is within what one seed's
noise can explain, so whatever GVP is missing it misses equally on both structure splits. TS50 is
visibly easier — 0.4602 recovery, 5.4831 perplexity — which fits its nature: de novo designed proteins
are more idealized, more regular secondary structure, fewer awkward loops, so a weak encoder recovers
more of them. That ordering, TS50 easiest and the two CATH sets tied and hardest, is a property of the
benchmarks, not of the method, and I should carry it forward so I don't misread a later rung's gains as
mechanism when they are just the easy set being easy. And the failure itself is not invariance — GVP is
provably invariant and trained stably — but *information starvation on the edges*. I gave each edge one
`CA→CA` direction vector and one `CA`-only distance, and that under-determines the thing the prediction
most depends on: the relative pose of two residues. Two neighbors at the same `CA` separation can pack
against me from completely different angles and present completely different chemical environments, and
GVP's thin edge tuple barely tells them apart. The whole point of GVP — keep geometry as live vectors —
paid for invariance machinery that I then starved of input. And the encoder-only amputation hit GVP
harder than it should hit a method built to put its strength in the encoder, because GVP expected an
autoregressive decoder to carry the residue-residue couplings and here that decoder is a per-residue
MLP.

So the diagnosis is sharp, and I should convert the perplexity into something I can aim at. A perplexity
of 5.90 is `exp` of a mean per-residue NLL of 1.775 nats, about 2.56 bits of residual uncertainty; the
encoder has taken twenty options down to not quite six. If a richer encoder could reach the mid-5s, say
5.5, that is 1.705 nats, 2.46 bits — a drop of roughly 0.07 nats, a tenth of a bit per residue. It
sounds tiny written that way, but it is a tenth of a bit *averaged over every residue in every test
protein*, and it is the difference between a floor and a real baseline. The point is that the lever is
information, and I do not need *more* invariance machinery, I need *richer invariant features* poured
into the graph, and I should drop the equivariant-vector bookkeeping that bought me little.

Let me actually walk the options here, because "richer features" has more than one reading and I want to
reject the wrong ones by argument. One route keeps GVP and simply widens the vector path — more vector
channels, deeper GVP stacks. But the diagnosis was starvation of *input*, not shortage of *capacity*:
widening the vector channels gives the network more room to transform the same one `CA→CA` direction,
and no amount of channel width invents the neighbor-orientation degrees of freedom that were never fed
in. To feed them as vectors I would have to pick several atom-pair directions per edge and carry each as
an equivariant arrow, and now I am paying the full `W_h`/norm/scale-by-norm bookkeeping on every one of
them, every layer, for information I can get a cheaper way. And it *is* cheaper the other way: a distance
is a single number that any ordinary linear can consume with no bias-free constraint, no norm to take, no
scale-by-norm gate, and no separate vector width to tune — the invariance is free at the featurizer and
the rest of the network is a plain scalar MLP. So the equivariant path is strictly more machinery for a
strictly smaller slice of the pose than a generous bank of distances delivers. A second route stays scalar but hand-builds
a few angle features — the neighbor direction projected into a local frame, an orientation cosine or
two, the Ingraham recipe. That helps, but it is a thin, hand-curated slice of the pose, and it still
leans on me to guess which projections matter. The third route is the one the constraint actually points
at. Distances between atoms are invariant to rotation and translation *by construction* — the cleanest
invariance I have, with zero frame bookkeeping. The reason GVP needed vectors at all was the Ingraham
diagnostic: a single distance is not locally informative. But that diagnostic has an escape hatch I
didn't use. Orientation is recoverable from distances if I use *enough* of them.

I want to make that last claim quantitative, because it is the whole bet and I refuse to run on a slogan.
The relative pose of two rigid residues is an element of `SE(3)`: six degrees of freedom, three for the
displacement between them and three for the neighbor's orientation about that displacement. GVP handed
the edge one distance and one unit direction — a length and two angles, three numbers, and those three
fix only the displacement, leaving the neighbor's three orientation degrees of freedom to leak in
indirectly through node channels. Now suppose instead that for the edge `(i, j)` I take the pairwise
distances among a small set of backbone atoms on *both* residues. With five atoms on each side there are
`5 × 5 = 25` cross-distances between the two atom sets. Each atom set is a known rigid shape, so the 25
cross-distances are 25 constraints on the 6-DOF relative transform — massively over-determined, and
generically the map from pose to this distance vector is injective, so the pose is *pinned*. The cross-
distances between the two atom triads encode the relative rotation implicitly: the very orientation
degrees of freedom GVP's single direction missed are recovered, and recovered as invariant scalars with
no frame to track. That is the trade I want, and now I can see it is not a hopeful heuristic but a
distance-geometry fact: 25 constraints for 6 unknowns pins the pose the way 1 constraint never could.

It helps to see *which* of the 25 does what, because it tells me the set is not redundant padding. Split
it in two. The five same-atom distances — `Ca-Ca, N-N, C-C, O-O, Cb-Cb` — are five radial shells: how
far each of the neighbor's atoms sits from the corresponding atom of mine. Those shells are close cousins
of what GVP already had; they measure separation, and two neighbors related by a rotation about the
`Ca–Ca` axis can share nearly the same five shells while presenting opposite faces. The twenty cross
distances — `Ca_i` to `N_j`, `N_i` to `C_j`, and so on — are what break that degeneracy: each one is
sensitive to the *twist* of the neighbor's frame relative to mine, because it measures a diagonal across
the two atom sets rather than a shell. So the same-atom block localizes the neighbor and the cross block
orients it, and it is exactly the cross block that supplies the three orientation degrees of freedom GVP
never saw. That is why I want the full grid and not a tidy subset.

And it has to be the full *ordered* grid, 25 rather than the 15 an unordered atom-type accounting would
give, because the edge is directed. For the center residue `i` gathering a neighbor `j`, the distance
from `i`'s `Ca` to `j`'s `N` is a different physical quantity from `i`'s `N` to `j`'s `Ca` — they are two
different diagonals of the same contact, and each reports a different facet of how `j`'s frame is turned
relative to `i`'s. Collapsing them to one unordered `Ca–N` channel would throw away half the twist
information the cross block exists to carry. So both orientations stay, and the count is `5 × 5 = 25` by
construction, not by padding.

Which atoms? I have `N, CA, C, O`. The side chain is exactly what distinguishes many residues, and I
don't have it — but the *direction* a side chain would point is a strong cue (buried vs exposed, toward
a neighbor or away), and I can get a proxy for free. Place a virtual `Cβ` at the ideal tetrahedral
position from the backbone: with `b = N − Ca`, `c = C − Ca`, `a = b × c`,
`Cb = −0.58273431·a + 0.56802827·b − 0.54067466·c + Ca`, the constants being the tetrahedral geometry
that puts `Cβ` where it sits in a real residue. I should check this point behaves, because a virtual
atom that is not frame-consistent would silently break the invariance I am buying. But `b` and `c` are
differences of backbone positions, so they are equivariant vectors; `a = b × c` is their cross product,
which rotates with them; and `Cb` is a fixed linear combination of `a, b, c` added to `Ca`, so `Cb`
transforms exactly as the backbone does under any rotation or translation. Therefore every distance from
`Cb` to another atom is invariant, precisely like the real-atom distances. Good — the proxy is free and
it is clean. Now I have five atoms per residue, and for an edge I take the 25 ordered atom pairs — the
same-atom pairs `Ca-Ca, N-N, C-C, O-O, Cb-Cb` plus the cross pairs in *both* orientations, because
`Ca-N` and `N-Ca` are different facets of the local-frame pattern and keeping both directions helps the
network read relative pose. Twenty-five distances per edge, each lifted into the RBF basis. I move the
centers to 2–22 Å here rather than the node RBF's 0–20 Å, because the smallest of the 25 pairs (bonded
same-residue geometry aside, the closest inter-residue contacts) start near 2–3 Å and the tail of the
neighborhood reaches past 20 Å, so shifting the window keeps the informative distances inside the
resolved part of the basis. That is the concrete answer to GVP's starvation: where GVP saw one
direction, ProteinMPNN sees twenty-five invariant distances that together reconstruct the pose.

The RBF lift is not decoration, and it is worth saying why each distance goes through it rather than
entering as a raw scalar. A bare distance fed to a linear layer can only be read *monotonically* — the
layer can weight "farther" against "nearer," nothing more. But the signal in a contact is not monotone:
a hydrophobic packing distance near 5–6 Å means something specific that a 3 Å clash and a 12 Å glance do
not, and a linear read of the raw number cannot carve out that middle band. Lifting the distance into 16
Gaussians turns it into a soft one-hot over distance shells — a localized bump that fires when the
distance is *near* a particular value — so the downstream MLP can learn "I care about a contact
specifically around 6 Å" as a single weight on one basis channel. With centers 1.25 Å apart and a
matching width, adjacent bumps overlap by about one standard deviation, so the encoding is smooth rather
than a hard binning, and interpolates cleanly between shells. Every one of the 25 distances gets this
treatment, which is why the edge block is `25·16` and not `25`: the factor of 16 is buying non-monotone,
shell-localized reads of each pairwise distance.

Add sequence position to the edge, because two spatial neighbors that are also sequence-adjacent are a
different situation from two that are 200 residues apart. The chain has a direction but no canonical
origin, so encode the *relative* offset `i − j` with a sinusoidal positional encoding (`num_pos_emb`
dims). Concatenate onto the 25-pair RBF block, embed to `hidden_dim` with a bias-free linear and a
LayerNorm. The edge dimension is then `25·16 + 16 = 416` invariant channels compressed into 128 — a wide
geometric read funneled into the working width. For the nodes I keep only a thin geometric summary — a
forward orientation vector and a side-chain-direction proxy, six channels — because the heavy geometry
now lives on the edges and a redundant node channel would only add fold-specific quirks.

And here I have to make a decision that also settles an invariance worry, so let me take them together.
Those six node channels are orientation and side-chain *directions* — vectors, computed from atom
differences and fed straight into a linear embedding. On their own they are not frame-invariant: rotate
the input and those raw 3-vectors rotate, so a node embedding that consumed them would move, and the
output with it. The reference ProteinMPNN resolves this by going all the way — it zeros the node
features entirely at the encoder's input and forces every bit of geometry through the edges. I will
follow that: the featurizer computes the thin node summary, but the encoder starts each node from a
zero state and lets the invariant distance edges drive everything. That is not just faithfulness to the
reference; it is what makes the model *exactly* invariant, because now the only geometry entering the
network is inter-atomic distances (invariant) and the relative sequence offset (frame-independent), and
there is no world-frame vector anywhere in the forward pass to break it. The node-vector worry
evaporates because the node channel is not used. The edges are the carrier, and the carrier is clean.

Now the per-layer update, and this is the second concrete improvement over the default and over GVP.
The Structured Transformer (and the scaffold default, and GVP) refine only the *node* embeddings; the
edge features are computed once at featurization and frozen forever. That wastes the most informative
channel: a contact between two residues *means* different things depending on the rest of the local
environment, and a static edge feature can't express that. So let the edges update too. Each encoder
layer does two things in order. First the node update: gather for each edge the triple
`[h_i, h_j, e_ij]`, run a three-linear MLP with GELU, sum the per-neighbor messages, divide by a fixed
`scale = 30`, residual-add, LayerNorm, then a wide position-wise feed-forward with its own residual and
LayerNorm. The fixed divisor is worth a second look, because it is doing real work. A literal sum over
neighbors grows with how many valid neighbors a residue has, and that count varies — a padded edge of
the graph has fewer, a buried core residue has its full `k` — so a raw sum would hand LayerNorm a
different scale for every residue and make it fight the geometry for control of the norm. Dividing by a
constant near the typical degree, `k = 30`, turns the sum into an approximate mean that is stable across
residues without making the divisor itself data-dependent. Then the edge update the prior methods
lacked: recompute the edge's input triple from the *refreshed* node states, run a second three-linear
MLP, residual-add into the edge with a LayerNorm. So each round, nodes aggregate from neighbors and
edges, then edges are re-derived from the freshly updated nodes — "what this contact means" gets refined
through depth instead of frozen at featurization. Three encoder layers reach roughly three-hop
neighborhoods, enough for the local geometry that dominates and fast enough for the budget. Hidden 128,
GELU throughout, Xavier init.

One cost note so I know what I am spending. The 25 distance matrices are each computed over all atom
pairs before gathering to the `k` neighbors, so featurization is `O(L²)` per atom-pair — the one place
the model touches the full pair matrix. But it happens *once*, outside the layer loop, and the message
passing that repeats is `O(L·k)` per layer. So the quadratic term is a fixed setup cost, not a per-layer
tax, and for the protein lengths here it fits comfortably — even at `L ≈ 500` a full pair-distance matrix
is `500 × 500 = 2.5×10^5` entries, and twenty-five of them per batch element is a few tens of megabytes at
single precision, computed once and immediately gathered down to the `k`-neighbor slice the layers use.
This matters because it means I can be generous with the 25 distances without paying for them three times
over.

I have to be honest about *which ProteinMPNN this task wants*, because the famous version is much more
than an encoder. The full method's signature moves are an autoregressive decoder with order-agnostic
random decoding (so a designer can fix arbitrary positions and the training signal is BERT-like and
denser), implemented by conjugating a lower-triangular order-space mask through a permutation matrix;
sequence embeddings that flow in only from already-decoded neighbors; backbone-coordinate Gaussian noise
(`augment_eps`) as a regularizer against the crystal-vs-predicted distribution shift; a chain encoding
for multi-chain design; label smoothing; and the Noam warmup schedule. **None of that survives the edit
surface.** The harness gives me `forward(X, mask)` with no sequence input, a fixed MLP decoder head, one
chain, and a fixed `AdamW`/`OneCycleLR`/cross-entropy loop — so there is no decoder to make
autoregressive, no sequence to mask, no second optimizer to schedule, and `augment_eps` is left at 0. So
this rung is the *ProteinMPNN encoder*, transplanted whole: the 25-pair all-atom RBF featurizer with
virtual `Cβ` and positional encoding, the `EncLayer` with node *and* edge updates, zero node init,
three layers — feeding the scaffold's MLP decoder and one-shot log-softmax over 20 amino acids. I keep
the part of ProteinMPNN that is about *reading structure into rich invariant embeddings*, which is
exactly the component the task isolates, and I drop the part that is about *generating sequence
autoregressively*, which the task has fixed. Every layer's message is an MLP over node and edge states
summed symmetrically over neighbors, so permutation invariance is free.

Now the falsifiable expectations against GVP's measured numbers, and I want them specific enough to be
wrong. The whole bet is that 25 invariant inter-atomic distances plus a live edge update beat one
direction vector plus a frozen edge — that richer invariant features, not more invariance machinery, are
what move recovery. So I expect a clear jump over GVP's 0.4310 / 0.4337 / 0.4602: recovery into the
mid-0.46 range on CATH 4.2 and CATH 4.3, with perplexity dropping well below GVP's 5.8–5.9 toward the
mid-5s, and TS50 improving in step and staying the easiest of the three as the benchmark ordering
predicts. I want recovery and perplexity to move *together*, and that joint move is itself a check: if
recovery rose but perplexity barely budged, the gain would be argmax quantization — the distances tipping
a few near-ties across the decision boundary without actually sharpening the distribution — and I would
distrust it. A real information gain from richer edges should show up as both a higher argmax rate and a
lower `exp`-NLL, because it narrows the whole predictive distribution, not just its peak. I am
specifically predicting that the *edge-feature richness* is the lever: if ProteinMPNN does not clearly
beat GVP on both metrics at once, my whole diagnosis of GVP's failure — thin edges, not the wrong
invariance route — is wrong, and the 25-versus-1 degrees-of-freedom argument with it. There is a second, sharper
sub-prediction I can already make from the mechanism. Pouring in distances is a *local-geometry* fix; it
sharpens the reading of each residue's immediate neighborhood. So I expect it to help the in-distribution
CATH sets, where fine local discrimination directly converts to recovery, at least as much as the
easier-but-out-of-distribution TS50 — the gains should be broad and edge-driven, not concentrated on any
one benchmark's peculiarities. I also expect this rung to *not* reach the top. It pours rich distances in
but its aggregation is still a plain symmetric MLP sum, with no learned weighting of which neighbor
matters and no global context, and its side-chain proxy is a single fixed `Cβ`, identical for every
residue. If recovery lands in the mid-0.46s, the gap to a stronger rung tells me the next lever is the
*aggregation and the feature completeness*: attention over neighbors, virtual atoms that are *learned*
rather than a single fixed `Cβ`, dihedral and direction features alongside the distances, and a global
context gate. That is the move I will reach for next. The full encoder-only scaffold module is in the
answer.
