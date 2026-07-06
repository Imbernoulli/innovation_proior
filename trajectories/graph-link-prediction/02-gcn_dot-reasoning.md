The VGAE numbers told me exactly which of its two deliberate weaknesses actually cost me, and it was
both, in the proportions I half-expected. Let me read the feedback carefully before I decide anything,
because the shape of the failure is what picks the next move. The AUC on Cora came in at 86.8 mean and on
CiteSeer 86.9, but the seed spread is the real story: on Cora the seeds were 90.51, 84.82, 85.10, a range
of 5.7 AUC points; on CiteSeer 91.46, 84.78, 84.35, a range of 7.1. That is not gentle noise around a
stable model — it is *bimodal*. Seed 42 lands well on both graphs (AUC ${\sim}91$, and on CiteSeer an MRR
of 46.24), while seeds 123 and 456 collapse together into the mid-84 AUC and the teens/low-twenties on
MRR. One good basin, two bad ones. That is precisely the failure mode I flagged closing the VGAE rung: I
predicted a squeezed, noisy latent space would "land well or badly depending on the random init," and the
feedback is that prediction made concrete — the KL squeeze plus reparameterization noise carve a bad
basin that the optimizer falls into on two of three seeds, and once there the sampling noise never lets it
climb out and spread the embeddings apart. So the variational machinery is not paying for itself on these
small graphs; it is manufacturing variance. It is worth noting which way the average is dragged: the Cora
MRR mean of 20.0 is not three runs near twenty, it is one run at 24, one at 21, one collapsed to 14.8 —
and on CiteSeer the mean of 27.1 hides a 46/19/16 split. A mean over a bimodal set is a fiction; what the
model actually does is either work or fall into the bad basin, and I should fix the mechanism, not chase
the mean.

The ranking metrics are where it really bled, and I want to translate the numbers into a mechanism rather
than just note they are low. Cora MRR 20.0, Hits@20 49.3; CiteSeer MRR 27.1, Hits@20 53.5. An MRR of 20
means the mean reciprocal rank of the true edge is about $0.20$, i.e. the true edge sits on average
around rank 5 in its candidate list. Set that beside an AUC of 87, which says positives are *on average*
well above negatives: the two together say the bulk is separated but the *top* of each candidate list is
a mess — the true edge is usually buried a few slots down behind near-tied negatives. That is exactly the
signature my sampling-noise arithmetic predicted last rung: a noise standard deviation near $28$ against a
well-aligned signal of order a couple hundred survives as bulk AUC but randomizes the near-ties that MRR
and Hits@K read. And the clearest verdict came from the large graph, exactly where I said the bare inner
product would struggle most: ogbl-collab Hits@50 of **31.77**, with MRR $7.70$. On a quarter-million-node
collaboration graph, where tens of thousands of non-edges have to be pushed below each true edge, a single
dot product of two independently-encoded points is simply too blunt, and the absence of any learned
pairwise interaction shows up as a Hits@50 stuck in the low thirties.

So the diagnosis is sharp and it points at two specific pieces I left on the floor by design. First, the
*variational noise*: on these graphs the sampling noise plus KL pull cost me more in ranking precision
and seed stability than they bought in generalization — the bimodal seed collapse and the soft MRR are
both consistent with "the model is fighting its own regularizer." Second, and more importantly, the
*decoder*: a pure inner product gives the model exactly one way to compare two nodes — the cosine-like
alignment of their embeddings. It cannot represent "these two are similar in dimensions 1–50 but that's
irrelevant; what matters is the interaction in dimensions 100–150," because the inner product weights
every dimension's product identically and then sums. The fix for both is the same move: drop the
variational machinery and make the embedding deterministic, and replace the bare dot product with a
*learned* decoder that can weight and mix the pairwise interaction. That is the second rung. Removing the
sampling and the prior/inner-product fight alone should kill the bimodal seed collapse; the learned
decoder is where the ranking gain has to come from.

Let me make that "weights every dimension identically" complaint concrete, because a two-dimensional toy
makes the failure undeniable and shows exactly what the MLP buys. Take $H=2$ and a source $z_i=(1,1)$.
Let a true neighbor be $z_{j_1}=(2,0)$ and a non-edge be $z_{j_2}=(0,2)$. The dot product scores them
$z_i^\top z_{j_1}=2$ and $z_i^\top z_{j_2}=2$ — *identical*. No inner product can separate this pair,
because it sums the elementwise products $(2,0)$ and $(0,2)$ with uniform weights and both sum to $2$.
But suppose the edge-relevant structure lives in dimension $1$ (dimension $2$ is nuisance). The Hadamard
blocks the MLP receives are $z_i\odot z_{j_1}=(2,0)$ and $z_i\odot z_{j_2}=(0,2)$; a first linear layer
with weights $(w_1,w_2)=(1,0)$ maps them to $2$ and $0$ — cleanly separated. So the moment I let the
decoder learn a *non-uniform* weighting of the interaction dimensions, a pair the dot product declares a
tie becomes a decisive ranking. That is the whole mechanism behind the expected MRR gain, in miniature.

So keep the GCN encoder essentially as the scaffold default — a stack of `GCNConv` layers with BatchNorm
and ReLU on the intermediate layers, dropout, and crucially **no** sampling and **no** KL: the embedding
is just $z=\mathrm{GCN}(X,A)$, a point per node, trained end-to-end by the loop's BCE. That alone removes
the noise and the prior/inner-product fight that destabilized the VGAE seeds. Now, given $z_i$ and $z_j$
for a candidate pair, what should the score be? Here it helps to write the dot product in its most
revealing form. $z_i^\top z_j = z_i^\top I\, z_j$ is a *bilinear form with the identity matrix* — the most
degenerate member of a whole family. That framing lays out the design space cleanly, so let me walk it and
reject the tempting middle options on arithmetic rather than taste.

The first upgrade is the learned bilinear form, $z_i^\top W z_j$ with a trainable $W\in\mathbb{R}^{H\times
H}$. This costs $H^2=256^2=65{,}536$ parameters and it does buy something real: it can up- and
down-weight, and even mix, interaction dimensions instead of summing them uniformly. But it is still
*bilinear* — a single scalar that is linear in each embedding separately, monotone in every product term,
with no nonlinearity anywhere. It cannot carve a conditional decision region of the form "score this a
true edge only if the interaction in dims 100–150 is strong *and* both endpoints look like hubs," because
that requires a threshold, and a bilinear map has none. And it still reduces the pair to one interaction
scalar; it never sees "who each node is" apart from how they interact. Since the failing metric was
precisely the *top of the ranking* — the hardest, most conditional pairs — a bilinear decoder would nudge
average separation but leave the MRR problem essentially where it is. So I reject it: it adds capacity of
the wrong shape.

The move that fits the diagnosis is to make the reduction *nonlinear and learned*. Do not sum the
elementwise product $z_i\odot z_j$ with fixed (or even learned-linear) weights — feed it to an MLP and let
the network learn which interaction dimensions matter and how to combine them through ReLU thresholds. But
I can hand the decoder more than the Hadamard product. The two pieces of information a pair carries are:
*who each node is* (its own embedding) and *how they interact* (the elementwise product). The inner
product throws away the first entirely — it only ever sees the product. Yet "who each node is" matters: a
high-degree hub and a leaf interact differently than two leaves, and the raw embeddings carry that. So I
give the decoder the concatenation of all three:
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\,]$, a $3H$-dimensional
pair representation, and put a small MLP on top: $3H\to H\to H\to 1$, with ReLU and dropout between
layers, producing one logit per candidate edge. The first block ($z_{\text{src}},z_{\text{dst}}$) lets
the decoder condition the score on each node's identity and degree-like role; the third block
($z_{\text{src}}\odot z_{\text{dst}}$) is the learned-weight generalization of the dot product. This is
exactly the standard OGB GCN+MLP link-prediction predictor — the one that reliably reaches the mid-fifties
Hits@50 on collaboration graphs where the bare dot product sits in the low thirties — and the reason it
works is that the MLP can sharpen the *top* of the ranking, which is precisely where VGAE's MRR told me
the bare inner product was failing.

The identity blocks earn their place most clearly on ogbl-collab, and it is worth spelling out why,
because that is where the biggest jump should come. A collaboration graph is degree-driven: a prolific
author collaborates with many, and a candidate pair's plausibility depends heavily on how "popular" each
endpoint is. GCN aggregation folds a node's degree and neighborhood scale into its embedding, so
$z_{\text{src}}$ and $z_{\text{dst}}$ carry degree-correlated signal — and the concatenated identity
blocks let the MLP learn a degree prior ("high-degree endpoints are more likely to link"), a preference a
bare dot product literally cannot express because it only sees alignment, not magnitude-of-role. That is a
second, independent reason the concat-plus-Hadamard decoder should beat the inner product on the large
graph, on top of the learned interaction weighting.

I want to verify that this MLP does not merely resemble the dot product but genuinely *contains* it, so
that the move is strictly a generalization — worst case it matches, and its extra capacity can only help
if trained well. The linear function $\sum_d(z_{\text{src}}\odot z_{\text{dst}})_d$ is the dot product;
can the three-layer ReLU MLP reproduce it exactly? A single linear layer reading the Hadamard block with
an all-ones weight row and no bias emits exactly that sum, so the function is in the linear span the MLP's
first layer can hit. The only subtlety is the ReLU in the middle: a linear target has to survive a
rectifier. The standard gadget handles it — $x=\mathrm{relu}(x)-\mathrm{relu}(-x)$ — so the first layer
maps the summed Hadamard signal into two channels, $\mathrm{relu}(s)$ and $\mathrm{relu}(-s)$, and the
second layer subtracts them, recovering $s$ exactly before the final projection. So the dot product is
*representable*, not merely approximable, by this decoder; every VGAE score is reachable, and everything
the MLP learns beyond that is pure upside. That settles the "can only help" claim on a construction rather
than a hope.

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
(intermediate BN, ReLU, dropout, no final BN) and put all the new capacity into the decoder.

While I am fixing the encoder I should be deliberate about its depth, because `num_layers=2` is a choice
and not just the default. A two-layer GCN gives each node an embedding that pools its two-hop
neighborhood. That depth is exactly what a link decoder wants: a shared neighbor $w$ of a candidate pair
$(i,j)$ sits one hop from each endpoint, so $w$'s features enter both $z_i$ and $z_j$'s receptive fields
at two layers, and the Hadamard product can pick up that co-appearance — the GCN senses neighborhood
overlap, if only *indirectly*, through the folded features. Going deeper would be a mistake here: a
three- or four-layer GCN over these small graphs starts to over-smooth, pulling every node's embedding
toward the graph-average and collapsing the very magnitude and spread the ranking metrics depend on — I
would be trading the top-of-list precision I am trying to win for a wider but blurrier receptive field.
So two layers is the right setting: deep enough that the embeddings can register the one- and two-hop
structure that decides a link, shallow enough not to wash it out. I flag, though, that "senses overlap
indirectly" is the load-bearing hedge — the decoder never *counts* shared neighbors, it only reads
whatever the message passing chose to encode about them. Concretely, the Hadamard block can register that
a candidate pair shares a neighbor $w$ only when $w$'s features survived into *both* endpoint embeddings
with correlated sign — sharp when $w$ is a distinctive feature-carrier, but washed out when $w$ is a
generic high-degree hub whose contribution is averaged away by GCN normalization. So the overlap signal
the decoder can actually see is real but lossy, and if the ranking metrics come back still fragile after
this rung, that indirection is the first suspect.

That decision is worth a parameter check, because I want to confirm the redistribution is honest rather
than a capacity binge. The MLP is $3H\!\cdot\!H + H\!\cdot\!H + H = 3\cdot256^2 + 256^2 + 256 \approx
2.62\times10^5$ parameters (about four times the bilinear form I rejected, but nonlinear where it
matters). Compare the two rungs end to end. VGAE spent about $1.08\times10^6$ parameters — a
$948\text{k}$ input projection plus two $65.8\text{k}$ GCN heads — on a *parameter-free* dot-product
decoder. This rung deletes the variance head, keeps one input projection and one $\mathrm{GCNConv}(256,
256)\approx6.58\times10^4$, giving a ${\sim}1.01\times10^6$ encoder, and adds the $2.62\times10^5$
decoder, for ${\sim}1.27\times10^6$ total. So I am not inflating the model: I am spending roughly the same
budget, redirected from the variational variance head (which manufactured the seed noise) into a learned
decoder (where the diagnosis says the bottleneck is). And the *only* change from the scaffold default is
swapping the one-line dot product for the three-block MLP, which is the cleanest possible test of "does a
learned decoder fix VGAE's ranking problem?"

The one number that gives me pause in that count is what it means on Cora. A $2.62\times10^5$-parameter
decoder is trained, on Cora, against roughly $85\%$ of $10{,}556$ edges — about $9\text{k}$ positives —
plus an equal draw of negatives, so on the order of $1.8\times10^4$ pair examples per epoch. That is a
decoder with more than ten parameters per training pair: overparameterized enough that it could, in
principle, memorize the training pairs and generalize poorly, and the default `dropout=0.0` gives it no
explicit regularization. So do I raise the dropout? I decide to keep the default, for two concrete
reasons. First, the negatives are *re-sampled every epoch*, so the decoder never sees the same
positive-plus-negative batch twice — that fresh-negative resampling is a form of data augmentation that
blunts memorization far more effectively than dropout would on a fixed set. Second, the loop early-stops
on validation AUC with patience 20, so a decoder that starts to overfit the training pairs is halted
before it can turn the ranking metrics soft. If the feedback were to show a large train/validation gap or
ranking metrics that peak early and decay, dropout would be my first lever — but on this evidence the
default is the right starting point, and raising it now would just be tuning against a problem I have not
yet observed.

So the step-2 edit lands directly in the editable region: the same GCN encoder as the scaffold default
(no sampling, no KL — deterministic embeddings), plus an MLP decoder on
$[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot z_{\text{dst}}]$ that returns one logit
per edge. `encode` is the plain GCN stack; `decode` builds the $3H$ concatenation and runs the MLP;
`forward` chains them. A shape pass: $z\in\mathbb{R}^{N\times H}$, the gathered $z_{\text{src}},
z_{\text{dst}}\in\mathbb{R}^{M\times H}$, their concatenation with the Hadamard block is
$\mathbb{R}^{M\times 3H}$, and the MLP maps it to $\mathbb{R}^{M\times1}$, squeezed to the $M$-vector the
loop's BCE expects. Nothing else about the loop changes. (The full scaffold module is in the answer.)

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
problem. One structural reason to trust that ogbl-collab prediction: the loop early-stops on validation
Hits@50 there (not AUC, as on the citation graphs), so the checkpoint I actually evaluate is the one the
model reached with the headline ranking metric in the objective's crosshairs — the selection pressure is
already aligned to Hits@50, which is exactly the metric a learned decoder is built to sharpen, so if the
decoder helps at all the improvement should read cleanly rather than being washed out by a mismatched
early-stop criterion. Where I am uncertain: AUC may *not* move much, or could even dip slightly, because
AUC was never the failing metric — VGAE already separated positives from negatives on average (87) — and a more
expressive decoder optimizes the loss in a way that trades a little average separation for much better
top-of-list precision. If AUC holds around the high eighties / low nineties while MRR and Hits jump,
that is the signal that the decoder, not the encoder, was the bottleneck, and the next rung should keep
this encoder/decoder skeleton and instead enrich what the decoder *sees about the pair* — explicit
structural features like shared-neighbor counts — rather than just giving it more parametric capacity.
If, instead, AUC and the ranking metrics both move only a little, the diagnosis flips: the GCN encoder
itself is the limit, and the next rung would need a structurally richer encoder, not a richer decoder.
