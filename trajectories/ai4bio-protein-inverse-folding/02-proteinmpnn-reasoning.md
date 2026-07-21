GVP landed exactly where I feared. On CATH 4.2 it recovered 0.4310 at perplexity 5.8996, on CATH 4.3
0.4337 / 5.8209, on TS50 0.4602 / 5.4831 — the bottom of any ladder I'd build. Reading the table: the two
CATH sets sit almost on top of each other, 0.4310 and 0.4337, a gap of 0.0027 that is within one seed's
noise, so whatever GVP is missing it misses equally on both structure splits. TS50 is visibly easier —
0.4602 / 5.4831 — which fits its nature: de novo designed proteins are more idealized, more regular
secondary structure, fewer awkward loops, so even a weak encoder recovers more of them. That ordering,
TS50 easiest and the two CATH sets tied and hardest, is a property of the benchmarks, not the method, and
I carry it forward so I don't misread a later rung's gains as mechanism when they are just the easy set
being easy. The failure itself is not invariance — GVP is provably invariant and trained stably — but
*information starvation on the edges*: one `CA→CA` direction vector and one `CA`-only distance
under-determine the relative pose of two residues, and the encoder-only amputation compounds it, since GVP
expected an autoregressive decoder to carry the residue-residue couplings and here that is a per-residue
MLP. The whole point of GVP — keep geometry as live vectors — paid for invariance machinery that I then
starved of input.

Perplexity 5.90 is `exp` of a mean per-residue NLL of 1.775 nats; reaching the mid-5s, say 5.5, is 1.705
nats, a drop of roughly 0.07 nats — a tenth of a bit averaged over every residue in every test protein,
the difference between a floor and a real baseline. The lever is information, not more invariance
machinery: I need *richer invariant features* poured into the graph, and I should drop the
equivariant-vector bookkeeping that bought me little.

There is more than one reading of "richer features," and the wrong ones fall to argument. One route keeps
GVP and widens the vector path — more channels, deeper stacks. But the diagnosis was starvation of
*input*, not shortage of *capacity*: more channel width gives the network more room to transform the same
one `CA→CA` direction and never invents the neighbor-orientation degrees of freedom that were never fed
in. To feed those as vectors I would carry several atom-pair directions per edge as equivariant arrows,
paying the full `W_h`/norm/scale-by-norm bookkeeping on each, every layer — and it is cheaper the other
way: a distance is a single number any ordinary linear can consume with no bias-free constraint, no norm,
no scale-by-norm gate, no separate vector width to tune, the invariance free at the featurizer. A second
route stays scalar but hand-builds a few angle features — the neighbor direction projected into a local
frame, an orientation cosine or two, the Ingraham recipe — but that is a thin, hand-curated slice of the
pose that still leans on me to guess which projections matter. The third route is the one the constraint
points at. Distances between atoms are invariant to rotation and translation *by construction* — the
cleanest invariance I have, zero frame bookkeeping. The reason GVP needed vectors at all was the Ingraham
diagnostic that a single distance is not locally informative. But that has an escape hatch: orientation is
recoverable from distances if I use *enough* of them.

That last claim is the whole bet, so make it quantitative. The relative pose of two rigid residues is an
element of `SE(3)`: six degrees of freedom, three for the displacement and three for the neighbor's
orientation about it. GVP handed the edge one distance and one direction — three numbers — fixing only the
displacement. Now suppose for the edge `(i, j)` I take the pairwise distances among a small set of
backbone atoms on *both* residues. With five atoms on each side there are `5 × 5 = 25` cross-distances
between the two atom sets. Each atom set is a known rigid shape, so the 25 distances are 25 constraints on
the 6-DOF relative transform — massively over-determined, and generically the map from pose to this
distance vector is injective, so the pose is *pinned*. The cross-distances between the two atom triads
encode the relative rotation implicitly: the very orientation degrees of freedom GVP's single direction
missed are recovered, as invariant scalars with no frame to track. It is not a hopeful heuristic but a
distance-geometry fact: 25 constraints for 6 unknowns pin the pose the way 1 constraint never could.

Which of the 25 does what tells me the set is not redundant padding. The five same-atom distances —
`Ca-Ca, N-N, C-C, O-O, Cb-Cb` — are radial shells: how far each of the neighbor's atoms sits from the
corresponding atom of mine. Those are close cousins of what GVP already had; two neighbors related by a
rotation about the `Ca–Ca` axis can share nearly the same five shells while presenting opposite faces. The
twenty cross distances — `Ca_i` to `N_j`, `N_i` to `C_j`, and so on — break that degeneracy: each measures
a diagonal across the two atom sets rather than a shell, so it is sensitive to the *twist* of the
neighbor's frame relative to mine. The same-atom block localizes the neighbor and the cross block orients
it, and it is exactly the cross block that supplies the three orientation degrees of freedom GVP never
saw. That is why I want the full grid.

And the full *ordered* grid, 25 rather than the 15 an unordered atom-type accounting gives, because the
edge is directed. For the center residue `i` gathering neighbor `j`, `i`'s `Ca` to `j`'s `N` is a
different physical quantity from `i`'s `N` to `j`'s `Ca` — two different diagonals of the same contact,
each reporting a different facet of how `j`'s frame is turned relative to `i`'s. Collapsing them to one
unordered `Ca–N` channel throws away half the twist information. So both orientations stay, and the count
is `5 × 5 = 25` by construction, not by padding.

Which atoms? I have `N, CA, C, O`. The side chain is what distinguishes many residues and I don't have it
— but the *direction* a side chain would point is a strong cue (buried vs exposed, toward a neighbor or
away), and I can get a proxy for free. Place a virtual `Cβ` at the ideal tetrahedral position from the
backbone: with `b = N − Ca`, `c = C − Ca`, `a = b × c`,
`Cb = −0.58273431·a + 0.56802827·b − 0.54067466·c + Ca`. A virtual atom that is not frame-consistent would
silently break the invariance I am buying, so check it: `b` and `c` are differences of backbone positions,
so equivariant vectors; `a = b × c` is their cross product, which rotates with them; and `Cb` is a fixed
linear combination of `a, b, c` added to `Ca`, so it transforms exactly as the backbone does under any
rotation or translation, and every distance from `Cb` is invariant like the real-atom distances. So the
proxy is free and clean. Now five atoms per residue, and for an edge the 25 ordered atom pairs — the
same-atom pairs plus the cross pairs in *both* orientations — each lifted into the RBF basis. I move the
centers to 2–22 Å here rather than the node RBF's 0–20 Å, because the closest inter-residue contacts start
near 2–3 Å and the tail reaches past 20 Å, so the shifted window keeps the informative distances inside
the resolved part of the basis. That is the concrete answer to GVP's starvation: where GVP saw one
direction, this encoder sees twenty-five invariant distances that reconstruct the pose.

Each distance goes through the RBF lift rather than in as a raw scalar, because a bare distance fed to a
linear layer can only be read *monotonically* — farther against nearer — while the signal in a contact is
not monotone: a hydrophobic packing distance near 5–6 Å means something a 3 Å clash and a 12 Å glance do
not. Lifting into 16 Gaussians turns each distance into a soft one-hot over distance shells, so the MLP
can key on a contact around 6 Å as a single channel weight; centers 1.25 Å apart with overlapping width
keep it smooth. That is why the edge block is `25·16`, not `25`.

Add sequence position to the edge, because two spatial neighbors that are also sequence-adjacent are a
different situation from two that are 200 residues apart. The chain has a direction but no canonical
origin, so encode the *relative* offset `i − j` with a sinusoidal positional encoding (`num_pos_emb`
dims), concatenate onto the 25-pair RBF block, embed to `hidden_dim` with a bias-free linear and a
LayerNorm — `25·16 + 16 = 416` invariant channels compressed into 128. For the nodes I keep only a thin
geometric summary — a forward orientation vector and a side-chain-direction proxy, six channels — because
the heavy geometry now lives on the edges.

Those six node channels are *directions* — vectors computed from atom differences — and on their own they
are not frame-invariant: rotate the input and they rotate, so a node embedding consuming them would move,
and the output with it. The clean resolution goes all the way: zero the node features entirely at the
encoder's input and force every bit of geometry through the edges. So the featurizer computes the thin
node summary, but the encoder starts each node from a zero state and lets the invariant distance edges
drive everything. That is what makes the model *exactly* invariant: the only geometry entering the network
is inter-atomic distances and the relative sequence offset, with no world-frame vector anywhere in the
forward pass. The edges are the carrier, and the carrier is clean.

Now the per-layer update, the second improvement over GVP and the default. The Structured Transformer and
GVP refine only the *node* embeddings; edge features are computed once at featurization and frozen. That
wastes the most informative channel: a contact between two residues *means* different things depending on
the rest of the local environment, and a static edge feature can't express that. So let the edges update
too. Each layer does two things in order. First the node update: gather for each edge the triple
`[h_i, h_j, e_ij]`, run a three-linear MLP with GELU, sum the per-neighbor messages, divide by a fixed
`scale = 30`, residual-add, LayerNorm, then a wide position-wise feed-forward with its own residual and
LayerNorm. The fixed divisor is doing real work: a literal sum over neighbors grows with the valid
neighbor count, which varies — a padded edge has fewer, a buried core residue its full `k` — so a raw sum
would hand LayerNorm a different scale for every residue and make it fight the geometry for control of the
norm. Dividing by a constant near the typical degree, `k = 30`, turns the sum into an approximate mean
that is stable across residues without making the divisor data-dependent. Then the edge update the prior
methods lacked: recompute the edge's input triple from the *refreshed* node states, run a second
three-linear MLP, residual-add into the edge with a LayerNorm. So each round, nodes aggregate from
neighbors and edges, then edges are re-derived from the freshly updated nodes — "what this contact means"
gets refined through depth instead of frozen. Three layers reach roughly three-hop neighborhoods, enough
for the local geometry that dominates and fast enough for the budget. Hidden 128, GELU throughout, Xavier
init.

One cost note. The 25 distance matrices are each computed over all atom pairs before gathering to the `k`
neighbors, so featurization is `O(L²)` — the one place the model touches the full pair matrix — but it
happens *once*, outside the layer loop, and the repeated message passing is `O(L·k)` per layer. Even at
`L ≈ 500` a full pair-distance matrix is `2.5×10^5` entries, twenty-five of them a few tens of megabytes at
single precision, computed once and immediately gathered down to the `k`-neighbor slice. So I can be
generous with the 25 distances without paying for them per layer.

The famous ProteinMPNN is much more than an encoder: an autoregressive decoder with order-agnostic random
decoding, sequence embeddings flowing in only from already-decoded neighbors, backbone-coordinate Gaussian
noise (`augment_eps`), a chain encoding, label smoothing, a Noam warmup. **None of that survives the edit
surface** — the harness gives `forward(X, mask)` with no sequence input, a fixed MLP decoder head, one
chain, and a fixed `AdamW`/`OneCycleLR`/cross-entropy loop, so `augment_eps` stays 0. What I keep is the
*encoder*: the 25-pair all-atom RBF featurizer with virtual `Cβ` and positional encoding, the encoder
layer with node *and* edge updates, zero node init, three layers — feeding the scaffold's MLP decoder and
one-shot log-softmax. Messages are MLPs over node and edge states summed symmetrically over neighbors, so
permutation invariance is free.

The bet against GVP's numbers: 25 invariant inter-atomic distances plus a live edge update beat one
direction vector plus a frozen edge. So I expect a clear jump over GVP's 0.4310 / 0.4337 / 0.4602 —
recovery into the mid-0.4s on both CATH sets, perplexity dropping from GVP's 5.8–5.9 toward the mid-5s,
TS50 improving in step and staying easiest. I want recovery and perplexity to move *together*: if recovery
rose but perplexity barely budged, the gain would be argmax quantization and I'd distrust it; a real
information gain narrows the whole distribution. If ProteinMPNN does not clearly beat GVP on both metrics
at once, my diagnosis — thin edges, not the wrong invariance route — is wrong, and the 25-versus-1 argument
with it. Since pouring in distances is a *local-geometry* fix, I expect it to help the in-distribution CATH
sets at least as much as the out-of-distribution TS50. And I expect this rung to *not* reach the top: its
aggregation is still a plain symmetric MLP sum with no learned weighting of which neighbor matters and no
global context, and its side-chain proxy is a single fixed `Cβ`. The gap to a stronger rung points the
next lever at *aggregation and feature completeness*: attention over neighbors, virtual atoms that are
*learned* rather than one fixed `Cβ`, and a global context gate.
