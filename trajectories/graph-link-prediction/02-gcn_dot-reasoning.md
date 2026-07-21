The VGAE numbers told me which of its two deliberate weaknesses actually cost me, and it was both, in
the proportions I half-expected. The AUC came in at 86.8 mean on Cora and 86.9 on CiteSeer, but the seed
spread is the real story: Cora seeds 90.51, 84.82, 85.10 (a 5.7-point range); CiteSeer 91.46, 84.78,
84.35 (a 7.1-point range). That is not gentle noise around a stable model — it is *bimodal*. Seed 42
lands well on both graphs (AUC $\sim91$, CiteSeer MRR 46.24), while seeds 123 and 456 collapse together
into the mid-84 AUC and the teens/low-twenties on MRR. One good basin, two bad ones — exactly the
failure mode I flagged closing the VGAE step: I predicted a squeezed, noisy latent space would "land
well or badly depending on the random init," and here it is made concrete. The KL squeeze plus
reparameterization noise carve a bad basin that the optimizer falls into on two of three seeds, and once
there the sampling noise never lets it climb out. The variational machinery is not paying for itself on
these small graphs; it is manufacturing variance. And a mean over a bimodal set is a fiction — the Cora
MRR mean of 20.0 is one run at 24, one at 21, one collapsed to 14.8; CiteSeer's 27.1 hides a 46/19/16
split — so I should fix the mechanism, not chase the mean.

The ranking metrics are where it really bled, and the numbers translate into a mechanism. Cora MRR 20.0,
Hits@20 49.3; CiteSeer MRR 27.1, Hits@20 53.5. An MRR of 20 means the mean reciprocal rank of the true
edge is about $0.20$, i.e. the true edge sits on average around rank 5 in its candidate list. Set that
beside an AUC of 87, which says positives are *on average* well above negatives: the two together say
the bulk is separated but the *top* of each list is a mess — the true edge is usually buried a few slots
down behind near-tied negatives. That is exactly the signature my sampling-noise arithmetic predicted: a
noise standard deviation near $28$ against a signal of order a couple hundred survives as bulk AUC but
randomizes the near-ties MRR and Hits@K read. And the clearest verdict came from the large graph, exactly
where I said the bare inner product would struggle most: ogbl-collab Hits@50 of **31.77**, MRR $7.70$. On
a quarter-million-node collaboration graph, where tens of thousands of non-edges have to be pushed below
each true edge, a single dot product of two independently-encoded points is simply too blunt.

So the diagnosis points at the two pieces I left on the floor by design. First, the *variational noise*:
on these graphs the sampling plus KL pull cost more in ranking precision and seed stability than they
bought in generalization. Second, and more importantly, the *decoder*: a pure inner product gives the
model exactly one way to compare two nodes — the cosine-like alignment of their embeddings. It cannot
represent "these two are similar in dimensions 1–50 but that's irrelevant; what matters is the
interaction in dimensions 100–150," because the inner product weights every dimension's product
identically and then sums. The fix for both is the same move: drop the variational machinery and make the
embedding deterministic, and replace the bare dot product with a *learned* decoder that can weight and
mix the pairwise interaction. Removing the sampling and the prior/inner-product fight alone should kill
the bimodal seed collapse; the learned decoder is where the ranking gain has to come from.

A two-dimensional toy makes the "weights every dimension identically" complaint undeniable and shows what
the MLP buys. Take $H=2$ and a source $z_i=(1,1)$. Let a true neighbor be $z_{j_1}=(2,0)$ and a non-edge
be $z_{j_2}=(0,2)$. The dot product scores them $z_i^\top z_{j_1}=2$ and $z_i^\top z_{j_2}=2$ —
*identical*: no inner product can separate this pair. But suppose the edge-relevant structure lives in
dimension 1. The Hadamard blocks are $z_i\odot z_{j_1}=(2,0)$ and $z_i\odot z_{j_2}=(0,2)$; a first
linear layer with weights $(1,0)$ maps them to $2$ and $0$ — cleanly separated. So the moment I let the
decoder learn a *non-uniform* weighting of the interaction dimensions, a pair the dot product declares a
tie becomes a decisive ranking. That is the whole mechanism behind the expected MRR gain, in miniature.

So keep the GCN encoder essentially as the scaffold default — `GCNConv` layers with BatchNorm and ReLU on
the intermediate layers, dropout, and crucially **no** sampling and **no** KL: the embedding is just
$z=\mathrm{GCN}(X,A)$, a point per node, trained end-to-end by the loop's BCE. Now, what should the score
be? It helps to write the dot product in its most revealing form: $z_i^\top z_j = z_i^\top I\, z_j$, a
*bilinear form with the identity matrix* — the most degenerate member of a whole family. That framing
lays out the design space, and I reject the tempting middle option on arithmetic. The learned bilinear
form $z_i^\top W z_j$ ($W\in\mathbb{R}^{H\times H}$, $65{,}536$ parameters) can up- and down-weight and
mix interaction dimensions, but it is still *bilinear*: a single scalar, linear in each embedding, monotone
in every product term, with no nonlinearity. It cannot carve a conditional region like "score this a true
edge only if the interaction in dims 100–150 is strong *and* both endpoints look like hubs," because that
needs a threshold. Since the failing metric was precisely the top of the ranking — the hardest, most
conditional pairs — a bilinear decoder would nudge average separation but leave MRR essentially where it
is. It adds capacity of the wrong shape.

The move that fits the diagnosis is to make the reduction *nonlinear and learned*: feed the elementwise
product $z_i\odot z_j$ to an MLP and let ReLU thresholds learn which interaction dimensions matter and how
to combine them. But I can hand the decoder more than the Hadamard product. A pair carries two pieces of
information: *who each node is* (its own embedding) and *how they interact* (the elementwise product). The
inner product throws away the first entirely. Yet "who each node is" matters — a high-degree hub and a
leaf interact differently than two leaves — so I give the decoder the concatenation of all three,
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\,]$, a $3H$-dimensional
pair representation, with a small MLP $3H\to H\to H\to 1$ (ReLU and dropout between layers). The identity
blocks let the decoder condition on each node's degree-like role; the Hadamard block is the learned-weight
generalization of the dot product. This is the standard OGB GCN+MLP link predictor, which reliably reaches
the mid-fifties Hits@50 on collaboration graphs where the bare dot product sits in the low thirties,
because the MLP can sharpen the *top* of the ranking — precisely where VGAE's MRR failed.

The identity blocks earn their place most clearly on ogbl-collab. A collaboration graph is degree-driven:
a candidate pair's plausibility depends heavily on how "popular" each endpoint is, and GCN aggregation
folds a node's degree and neighborhood scale into its embedding, so $z_{\text{src}}$ and $z_{\text{dst}}$
carry degree-correlated signal. The concatenated identity blocks let the MLP learn a degree prior
("high-degree endpoints are more likely to link") that a bare dot product literally cannot express,
because it only sees alignment, not magnitude-of-role. That is a second, independent reason this decoder
should beat the inner product on the large graph.

The move is strictly a generalization: a single linear row of all-ones over the Hadamard block emits
$\sum_d(z_{\text{src}}\odot z_{\text{dst}})_d$, the dot product itself, so every score the inner product
could produce is reachable (the middle ReLU is handled by the $x=\mathrm{relu}(x)-\mathrm{relu}(-x)$
split) and the extra decoder capacity can only add to it.

One subtlety in *symmetry*: the graph is undirected but I concatenate an ordered pair
$[z_{\text{src}}\,\|\,z_{\text{dst}}]$, which is not symmetric in $i\leftrightarrow j$. The Hadamard block
is symmetric; the identity blocks are not. In practice the graph is stored undirected so both orientations
appear during message passing, and the BCE target is orientation-independent, so the MLP learns to be
approximately symmetric from the training distribution. Explicit symmetrization by averaging the two
orderings doubles decode cost on the large graph for a marginal gain, and the dominant signal — the
Hadamard interaction — is already symmetric, so I leave the concatenation ordered.

Does the "no BatchNorm on the final layer" rule from the VGAE step still hold now that the decoder is an
MLP? Partly. The MLP's first linear layer can rescale whatever magnitude the embeddings have, so it is far
less sensitive to final-layer normalization than a bare dot product. But the scaffold convention —
BatchNorm on intermediate layers only, none on the last — is still the safe default: it stabilizes the
depth of message passing without forcing a fixed scale on the embedding the MLP consumes. So I keep the
encoder exactly as the scaffold default and put all the new capacity into the decoder.

Depth is a real choice, not just the default. A two-layer GCN gives each node an embedding pooling its
two-hop neighborhood, which is what a link decoder wants: a shared neighbor $w$ of a pair $(i,j)$ sits one
hop from each endpoint, so $w$'s features enter both $z_i$ and $z_j$'s receptive fields, and the Hadamard
product can pick up that co-appearance — the GCN senses neighborhood overlap, if only *indirectly*,
through the folded features. Going deeper would over-smooth these small graphs, pulling every embedding
toward the graph-average and collapsing the magnitude the ranking metrics depend on. So two layers is
right. But "senses overlap indirectly" is the load-bearing hedge: the decoder never *counts* shared
neighbors, it only reads whatever message passing chose to encode — sharp when a shared neighbor is a
distinctive feature-carrier, washed out when it is a generic hub whose contribution GCN normalization
averages away. If the ranking metrics come back still fragile after this step, that indirection is the
first suspect.

A parameter check confirms the redistribution is honest, not a capacity binge. The MLP is $3\cdot256^2 +
256^2 + 256\approx2.62\times10^5$ parameters. VGAE spent $\sim1.08\times10^6$ parameters on a
*parameter-free* dot product; this step deletes the variance head (which manufactured the seed noise),
keeping one input projection and one $\mathrm{GCNConv}(256,256)$ for a $\sim1.01\times10^6$ encoder, and
adds the $2.62\times10^5$ decoder, $\sim1.27\times10^6$ total. So I am spending roughly the same budget,
redirected from the variational variance head into the learned decoder where the diagnosis says the
bottleneck is — and the *only* change from the scaffold default is swapping the one-line dot product for
the three-block MLP, the cleanest possible test of "does a learned decoder fix VGAE's ranking problem?"

One number gives me pause: a $2.62\times10^5$-parameter decoder on Cora trains against $\sim9$k positives
plus an equal draw of negatives, $\sim1.8\times10^4$ pairs per epoch — more than ten parameters per
training pair, and `dropout=0.0` gives no explicit regularization. Do I raise the dropout? I keep the
default, for two reasons. The negatives are *re-sampled every epoch*, so the decoder never sees the same
batch twice — a form of augmentation that blunts memorization far more than dropout on a fixed set would.
And the loop early-stops on validation AUC with patience 20, halting a decoder that starts to overfit
before it can turn the ranking metrics soft. If the feedback showed a large train/val gap or ranking
metrics peaking early and decaying, dropout would be my first lever — but raising it now would just tune
against a problem I have not yet observed.

So the step-2 edit lands in the editable region: the same GCN encoder as the scaffold default (no
sampling, no KL) plus the MLP decoder on $[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot
z_{\text{dst}}]$ returning one logit per edge. `encode` is the plain GCN stack; `decode` builds the $3H$
concatenation and runs the MLP; nothing else about the loop changes. (The full module is in the answer.)

The falsifiable expectations against the VGAE numbers: removing the sampling noise should tighten the
seed spread (no more mid-84 AUC collapses on seeds 123/456), and the learned decoder should lift MRR and
Hits@20 substantially above VGAE's Cora 20.0/49.3 and CiteSeer 27.1/53.5. The best-anchored claim is on
ogbl-collab: the GCN+MLP predictor is the canonical baseline reaching the mid-fifties Hits@50 on this
exact graph, so I expect Hits@50 to jump from VGAE's 31.77 into the fifties — the cleanest verdict on
whether a learned decoder beats a bare inner product at large-pool ranking, and the loop already
early-stops on validation Hits@50 there. AUC I am less sure of: it was never the failing metric, and a
more expressive decoder can trade a little average separation for top-of-list precision, so it may hold
or dip slightly. If AUC holds while MRR and Hits jump, the decoder was the bottleneck and the next step
enriches what the decoder *sees about the pair* — explicit structural features like shared-neighbor
counts — rather than adding capacity; if both move only a little, the GCN encoder itself is the limit.
