GVP landed exactly where I feared and the numbers say why. On CATH 4.2 it recovered 0.4310 at
perplexity 5.8996, on CATH 4.3 0.4337 / 5.8209, on TS50 0.4602 / 5.4831 — the bottom of any ladder I'd
build, and the failure is not invariance (GVP is provably invariant and trained stably) but
*information starvation on the edges*. I gave each edge one `CA→CA` direction vector and one `CA`-only
distance, and that under-determines the thing the prediction most depends on: the relative pose of two
residues. Two neighbors at the same `CA` separation can pack against me from completely different angles
and present completely different chemical environments, and GVP's thin edge tuple barely tells them
apart. The whole point of GVP — keep geometry as live vectors — paid for invariance machinery that I
then starved of input. And the encoder-only amputation hit GVP harder than it should hit a method built
to put its strength in the encoder, because GVP expected an autoregressive decoder to carry the
residue-residue couplings and here that decoder is a per-residue MLP. So the diagnosis is sharp: I do
not need *more* invariance machinery, I need *richer invariant features* poured into the graph, and I
should drop the equivariant-vector bookkeeping that bought me little. That points straight at distances.

Here is the principle I want to exploit. Distances between atoms are invariant to rotation and
translation *by construction* — the cleanest invariance I have, with zero frame bookkeeping. The reason
GVP needed vectors at all was the Ingraham diagnostic: a single distance is not locally informative. But
that diagnostic has an escape hatch I didn't use. Orientation is recoverable from distances if I use
*enough* of them. One `CA–CA` distance per edge is blind to which side a neighbor sits on; but if for
each edge I take the pairwise distances among a small set of backbone atoms on *both* residues, the
relative rigid-body pose of the two residues is essentially pinned down — it becomes a little
distance-geometry problem, and the cross-distances between the two atom triads encode the relative
rotation implicitly. So I can have the locality information GVP fought for, *and* keep everything as
invariant scalars, simply by being generous with how many inter-atomic distances I feed the edge. That
is the trade I want: trade GVP's equivariant vector path for a much richer bank of invariant distances.

Which atoms? I have `N, CA, C, O`. The side chain is exactly what distinguishes many residues, and I
don't have it — but the *direction* a side chain would point is a strong cue (buried vs exposed, toward
a neighbor or away), and I can get a proxy for free. Place a virtual `Cβ` at the ideal tetrahedral
position from the backbone: with `b = N − Ca`, `c = C − Ca`, `a = b × c`,
`Cb = −0.58273431·a + 0.56802827·b − 0.54067466·c + Ca`, the constants being the tetrahedral geometry
that puts `Cβ` where it sits in a real residue. Now I have five atoms per residue. For an edge `(i, j)`
take the distances between each atom of `i` and each atom of `j` — but be deliberate so the count is
fixed and the set is rich: 25 ordered atom pairs covers the full cross-product, the same-atom pairs
`Ca-Ca, N-N, C-C, O-O, Cb-Cb` plus the cross pairs in *both* orientations, because `Ca-N` and `N-Ca` are
different facets of the local-frame pattern and keeping both directions helps the network read relative
pose. Twenty-five distances per edge, each lifted into the RBF basis (here centers 2–22 Å, the contact-
to-neighborhood range), giving `25 × num_rbf` distance features. That is the concrete answer to GVP's
starvation: where GVP saw one direction, ProteinMPNN sees twenty-five invariant distances that together
reconstruct the pose.

Add sequence position to the edge, because two spatial neighbors that are also sequence-adjacent are a
different situation from two that are 200 residues apart. The chain has a direction but no canonical
origin, so encode the *relative* offset `i − j` with a sinusoidal positional encoding (`num_pos_emb`
dims). Concatenate onto the 25-pair RBF block, embed to `hidden_dim` with a bias-free linear and a
LayerNorm. For the nodes I keep a thin geometric summary — the forward orientation vector and a
side-chain-direction proxy, 6 channels embedded and normed — rather than the dihedrals, because the
heavy geometry now lives on the edges and a redundant node channel would only add fold-specific quirks.
The reference ProteinMPNN goes further and zeros the node features entirely, forcing all geometry through
the edges; this task's edit keeps a small node feature but the spirit is the same — the edges are the
carrier.

Now the per-layer update, and this is the second concrete improvement over the default and over GVP.
The Structured Transformer (and the scaffold default, and GVP) refine only the *node* embeddings; the
edge features are computed once at featurization and frozen forever. That wastes the most informative
channel: a contact between two residues *means* different things depending on the rest of the local
environment, and a static edge feature can't express that. So let the edges update too. Each encoder
layer does two things in order. First the node update: gather for each edge the triple
`[h_i, h_j, e_ij]`, run a three-linear MLP with GELU, sum the per-neighbor messages, divide by a fixed
`scale = 30` (a degree-stable approximate mean — a literal sum would grow with the variable neighbor
count and fight LayerNorm), residual-add, LayerNorm, then a wide position-wise feed-forward with its own
residual and LayerNorm. Then the edge update the prior methods lacked: recompute the edge's input triple
from the *refreshed* node states, run a second three-linear MLP, residual-add into the edge with a
LayerNorm. So each round, nodes aggregate from neighbors and edges, then edges are re-derived from the
freshly updated nodes — "what this contact means" gets refined through depth instead of frozen at
featurization. Three encoder layers reach roughly three-hop neighborhoods, enough for the local geometry
that dominates and fast enough for the budget. Hidden 128, GELU throughout, Xavier init.

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
virtual `Cβ` and positional encoding, the `EncLayer` with node *and* edge updates, zero/thin node init,
three layers — feeding the scaffold's MLP decoder and one-shot log-softmax over 20 amino acids. I keep
the part of ProteinMPNN that is about *reading structure into rich invariant embeddings*, which is
exactly the component the task isolates, and I drop the part that is about *generating sequence
autoregressively*, which the task has fixed.

Let me sanity-check invariance one more time, since I traded away the equivariant machinery for it. The
node features are orientation/side-chain *directions* — wait, those are vectors, so I must be careful:
they are computed from atom differences and fed as raw 3-vectors, but they are only ever consumed by a
linear embedding, so strictly the node channel is not frame-invariant on its own. The edges, which
carry the dominant signal, are pure inter-atomic *distances* (invariant) plus the relative sequence
offset (frame-independent), and zeroing-or-thinning the node channel is exactly why the reference leans
on edges: the invariance that matters and transfers to unseen folds rides on the distance edges, and the
node direction is a weak local hint. Every layer's message is an MLP over node and edge states summed
symmetrically over neighbors, so permutation invariance is free. Cost is `O(L·k)` per layer.

Now the falsifiable expectations against GVP's measured numbers. The whole bet is that 25 invariant
inter-atomic distances plus a live edge update beat one direction vector plus a frozen edge — that
richer invariant features, not more invariance machinery, are what move recovery. So I expect a clear
jump over GVP's 0.4310 / 0.4337 / 0.4602: recovery into the mid-0.46 range on CATH 4.2 and CATH 4.3,
with perplexity dropping well below GVP's 5.8–5.9 toward the mid-5s, and TS50 improving in step. I am
specifically predicting that the *edge-feature richness* is the lever: if ProteinMPNN does not clearly
beat GVP, my whole diagnosis of GVP's failure (thin edges, not the wrong invariance route) is wrong. I
also expect this rung to *not* reach the top — it pours rich distances in but its aggregation is still a
plain symmetric MLP sum, with no learned weighting of which neighbor matters and no global context, and
its node features are thinner than they could be. If recovery lands in the mid-0.46s, the gap to a
stronger rung tells me the next lever is the *aggregation and the feature completeness*: attention over
neighbors, virtual atoms that are *learned* rather than a single fixed `Cβ`, dihedral and direction
features alongside the distances, and a global context gate. That is the move I will reach for next. The
full encoder-only scaffold module is in the answer.
