The VGAE numbers told me exactly which of its two deliberate weaknesses actually cost me, and it was
both, in the proportions I half-expected. The AUC on Cora came in at 86.8 mean and on CiteSeer 86.9,
but look at the seed spread: seed 42 was strong (90.5 / 91.5) while seeds 123 and 456 collapsed to the
mid-84s on both graphs. That is not noise around a stable model — that is a model whose latent space is
being squeezed by the prior/inner-product fight I flagged, so that on the runs where the random init
and the sampling noise happen to start it badly, the KL pull and the reparameterization noise never let
it recover and spread the embeddings apart. The ranking metrics are where it really bled: Cora MRR
20.0, Hits@20 49.3; CiteSeer MRR 27.1, Hits@20 53.5. AUC of 87 says positives are *on average* above
negatives, but MRR of 20 says the *top* of the candidate list is a mess — the true edge is usually not
near the front. And the clearest verdict came from the large graph, exactly where I said the bare
inner product would struggle most: ogbl-collab Hits@50 of **31.77**. On a quarter-million-node
collaboration graph, where tens of thousands of non-edges have to be pushed below each true edge, a
single dot product of two independently-encoded points is simply too blunt, and the absence of any
learned pairwise interaction shows up as a Hits@50 in the low thirties.

So the diagnosis is sharp and it points at two specific pieces I left on the floor by design. First,
the *variational noise*: on these graphs the sampling noise plus KL pull cost me more in ranking
precision and seed stability than they bought in generalization — the small-graph seed collapses and
the soft MRR are both consistent with "the model is fighting its own regularizer." Second, and more
importantly, the *decoder*: a pure inner product gives the model exactly one way to compare two nodes —
the cosine-like alignment of their embeddings. It cannot represent "these two are similar in dimensions
1–50 but that's irrelevant; what matters is the interaction in dimensions 100–150," because the inner
product weights every dimension's product identically and then sums. The fix for both is the same move:
drop the variational machinery and make the embedding deterministic, and replace the bare dot product
with a *learned* decoder that can weight and mix the pairwise interaction. That is the second rung.

Let me derive the decoder, because the encoder choice is the easy part and the decoder is where the
gain has to come from. Keep the GCN encoder essentially as the scaffold default — a stack of `GCNConv`
layers with BatchNorm and ReLU on the intermediate layers, dropout, and crucially **no** sampling and
**no** KL: the embedding is just $z=\mathrm{GCN}(X,A)$, a point per node, trained end-to-end by the
loop's BCE. That alone removes the noise and the prior/inner-product fight that destabilized the VGAE
seeds. Now, given $z_i$ and $z_j$ for a candidate pair, what should the score be? The inner product
$\sum_d z_{id}z_{jd}$ is a single fixed reduction of the elementwise product $z_i\odot z_j$. The
obvious generalization is: don't sum the elementwise product with uniform weights — *feed it to an
MLP* and let the network learn which interaction dimensions matter and how to combine them. An MLP on
$z_i\odot z_j$ already strictly generalizes the dot product (a single linear layer with all-ones
weights and no bias recovers it), so it can only help if trained well.

But I can hand the decoder more than the Hadamard product. The two pieces of information a pair
carries are: *who each node is* (its own embedding) and *how they interact* (the elementwise product).
The inner product throws away the first entirely — it only ever sees the product. Yet "who each node
is" matters: a high-degree hub and a leaf interact differently than two leaves, and the raw embeddings
carry that. So I give the decoder the concatenation of all three:
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\,]$, a $3H$-dimensional
pair representation, and put a small MLP on top: $3H\to H\to H\to 1$, with ReLU and dropout between
layers, producing one logit per candidate edge. The first block ($z_{\text{src}},z_{\text{dst}}$) lets
the decoder condition the score on each node's identity and degree-like role; the third block
($z_{\text{src}}\odot z_{\text{dst}}$) is the learned-weight generalization of the dot product. This is
exactly the standard OGB GCN+MLP link-prediction predictor — the one that reliably reaches the
mid-fifties Hits@50 on collaboration graphs where the bare dot product sits in the low thirties — and
the reason it works is that the MLP can sharpen the *top* of the ranking, which is precisely where
VGAE's MRR told me the bare inner product was failing.

There is a subtlety in *symmetry* I should think about, because the graph is undirected and I am
concatenating an ordered pair $[z_{\text{src}}\,\|\,z_{\text{dst}}]$, which is not symmetric in
$i\leftrightarrow j$. The Hadamard block is symmetric, but the first two blocks are not — swapping
source and destination changes $h$. Is that a bug? In practice the harness samples each undirected
positive once with a fixed orientation and the negatives are sampled pairs, and the MLP learns to be
approximately symmetric because it sees both orientations across the training distribution (the graph
is stored undirected, so most node pairs that matter appear in both directions during message passing,
and the BCE target is orientation-independent). I could symmetrize explicitly by averaging the two
orderings, but that doubles decode cost on the large graph for a marginal gain, and the dominant signal
— the Hadamard interaction — is already symmetric. So I leave the concatenation ordered and let the MLP
absorb the asymmetry; this is the standard, cheaper choice and matches the predictor I want to
benchmark.

Now keep the encoder honest about the dot-product-versus-MLP distinction. In the VGAE rung I argued
*against* BatchNorm on the final embedding layer, because the inner product needs the embedding
magnitude — normalizing it away crushes the score spread. Does that argument still hold now that the
decoder is an MLP? Partly. The MLP has its own first linear layer that can rescale whatever magnitude
the embeddings have, so it is far less sensitive to the final-layer normalization than a bare dot
product is. But the scaffold's encoder convention — BatchNorm on intermediate layers only, none on the
last — is still the safe default: it stabilizes the depth of message passing without forcing a fixed
scale on the embedding the MLP consumes. So I keep the encoder exactly as the scaffold default
(intermediate BN, ReLU, dropout, no final BN) and put all the new capacity into the decoder. This also
keeps the parameter count modest — the MLP is $3H\cdot H+H\cdot H+H$ parameters, comfortably inside the
budget — and means the *only* change from the scaffold default is swapping the one-line dot product for
the three-block MLP, which is the cleanest possible test of "does a learned decoder fix VGAE's ranking
problem?"

So the step-2 edit lands directly in the editable region: the same GCN encoder as the scaffold default
(no sampling, no KL — deterministic embeddings), plus an MLP decoder on
$[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot z_{\text{dst}}]$ that returns one logit
per edge. `encode` is the plain GCN stack; `decode` builds the $3H$ concatenation and runs the MLP;
`forward` chains them. Nothing else about the loop changes. (The full scaffold module is in the
answer.)

Let me state the falsifiable expectations against the VGAE numbers, metric by metric, because that is
how I will know whether the diagnosis was right. The headline claim is on the ranking metrics: removing
the sampling noise should tighten the seed spread (no more mid-84 AUC collapses on seeds 123/456), and
the learned decoder should lift MRR and Hits@20 substantially above VGAE's Cora 20.0/49.3 and CiteSeer
27.1/53.5 — because the MLP can finally sharpen the front of the candidate list that the inner product
blurred. I expect Cora MRR into the thirties and Hits@20 into the high sixties / low seventies, and a
similar lift on CiteSeer. The second, sharper claim is on ogbl-collab: the GCN+MLP predictor is the
canonical baseline that reaches the mid-fifties Hits@50 on this exact graph, so I expect Hits@50 to
jump from VGAE's 31.77 into the low-to-mid fifties — that single number is the cleanest verdict on
whether a learned decoder beats a bare inner product at the large-pool ranking that is the real
problem. Where I am uncertain: AUC may *not* move much, or could even dip slightly, because AUC was
never the failing metric — VGAE already separated positives from negatives on average (87) — and a more
expressive decoder optimizes the loss in a way that trades a little average separation for much better
top-of-list precision. If AUC holds around the high eighties / low nineties while MRR and Hits jump,
that is the signal that the decoder, not the encoder, was the bottleneck, and the next rung should keep
this encoder/decoder skeleton and instead enrich what the decoder *sees about the pair* — explicit
structural features like shared-neighbor counts — rather than just giving it more parametric capacity.
If, instead, AUC and the ranking metrics both move only a little, the diagnosis flips: the GCN encoder
itself is the limit, and the next rung would need a structurally richer encoder, not a richer decoder.
