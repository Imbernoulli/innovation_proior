The scaffold default is a GCN encoder feeding a dot product, trained by binary cross-entropy on
sampled positive and negative pairs. Before I reach for anything heavier I want to start from the
principled version of exactly that object, because the dot-product-of-GCN-embeddings idea is not an
arbitrary baseline — it is the shape of a real generative model of the graph, and starting there tells
me whether the *probabilistic* framing buys anything before I start bolting on structural features and
MLP decoders. So the first rung is the variational graph auto-encoder: the same GCN-encoder /
inner-product-decoder skeleton, but derived as a latent-variable model of the adjacency matrix and
trained with the regularizer that framing demands.

Let me write down what I am actually modeling, because the whole method falls out of taking the
generative story seriously. I have an undirected graph on $N$ nodes with adjacency $A$ and node
features $X$. I posit a latent embedding $z_i\in\mathbb{R}^F$ per node, collected in $Z\in\mathbb{R}^{N
\times F}$, and a generative model in which an edge between $i$ and $j$ is a Bernoulli draw whose
probability is a function of the two latents: $p(A_{ij}=1\mid z_i,z_j)=\sigma(z_i^\top z_j)$. That is
the *inner-product decoder*, and it is not a modeling afterthought — it is the natural likelihood for a
symmetric binary relation. The score $z_i^\top z_j$ is large when two embeddings point the same way,
the sigmoid turns it into an edge probability, and reconstructing the whole adjacency means
$\hat A=\sigma(ZZ^\top)$. The prior on the latents is the usual isotropic Gaussian
$p(Z)=\prod_i\mathcal N(z_i\mid 0,I)$. So the object I want to fit is a model that, given features and
the observed (training) edges, places each node in a latent space such that connected nodes land near
each other in inner-product geometry, and I want to do this by maximizing the probability the model
assigns to the observed edges.

The inference side — the encoder — is where the graph structure and features enter. I cannot do exact
posterior inference over $Z$, so I take the variational route: an approximate posterior
$q(Z\mid X,A)=\prod_i\mathcal N(z_i\mid \mu_i,\operatorname{diag}(\sigma_i^2))$ whose per-node mean and
log-standard-deviation are produced by a GCN read off the features and the connectivity. Concretely a
shared GCN stack computes a hidden representation, and then two separate GCN heads on top of it produce
$\mu=\mathrm{GCN}_\mu(X,A)$ and $\log\sigma=\mathrm{GCN}_\sigma(X,A)$. The reason both heads are GCNs
and not plain linear layers is that I want the *uncertainty* of a node's embedding to also depend on
its neighborhood — a node with few, ambiguous connections should be allowed a wider posterior than a
hub with a clear structural role. This is the GCN-as-encoder idea from the prior art, but now it is
producing the parameters of a distribution rather than a point.

The objective is the variational lower bound (ELBO):
$\mathcal L=\mathbb E_{q(Z\mid X,A)}[\log p(A\mid Z)]-\mathrm{KL}[q(Z\mid X,A)\,\|\,p(Z)]$. The first
term is the expected reconstruction log-likelihood — under the Bernoulli-inner-product decoder this is
exactly a (re-weighted) cross-entropy between the true adjacency entries and $\sigma(z_i^\top z_j)$,
which is the same edge-classification loss the harness already runs. The second term is the
KL divergence from the approximate posterior to the prior, and for two Gaussians it has the closed
form $\mathrm{KL}=-\tfrac12\sum_i\sum_f\big(1+\log\sigma_{if}^2-\mu_{if}^2-\sigma_{if}^2\big)$. To make
the expectation differentiable I use the reparameterization trick: instead of sampling $z_i$ directly,
I sample $\epsilon\sim\mathcal N(0,I)$ and set $z_i=\mu_i+\epsilon\odot\sigma_i$ during training, which
moves the stochasticity off the parameters so gradients flow through $\mu$ and $\sigma$. At evaluation
I drop the noise and use $z_i=\mu_i$, the posterior mean, as the deterministic embedding.

Now the part where the principled story has to bend to fit this task's harness, and getting it right is
the whole reason to think carefully rather than copy a generic VGAE. The fixed training loop does
*not* know about an ELBO. It calls my `forward`, gets back a vector of edge scores, and computes
binary cross-entropy on those scores against {1 for positives, 0 for negatives}. That BCE is precisely
the reconstruction term — so the reconstruction half of the ELBO is handled for me, for free, by the
loop. But the loop will never add a KL term, because it only ever sees my scalar scores; there is no
hook for an auxiliary loss. If I do nothing, the KL gradient never flows, the encoder is unregularized,
$\sigma$ can collapse and $\mu$ can drift to whatever the reconstruction term wants, and I have a
plain (non-variational) auto-encoder with extra sampling noise — strictly worse than just fitting
point embeddings. So the KL has to be smuggled into the computation graph *of the scores themselves*.

The mechanism I use is to add the KL, as a single scalar, onto every returned score:
$\text{score}\leftarrow\text{score}+w\cdot\mathrm{KL}_{\text{per-node}}$, where
$\mathrm{KL}_{\text{per-node}}=-\tfrac12\,\overline{\sum_f(1+\log\sigma^2-\mu^2-\sigma^2)}$ is the
mean-over-nodes KL (the standard $1/N$ VGAE normalization, here realized as a mean). This is a uniform
additive shift on all scores, so it barely perturbs the *ranking* the BCE cares about, but because the
KL scalar is built from $\mu$ and $\log\sigma$ — which are differentiable functions of the GCN
parameters — adding it to a quantity the loop then differentiates makes the KL's gradient flow into
the encoder during backprop. The coefficient $w$ controls how hard the regularizer pulls. I keep it
small, $w=0.005$: the KL is already normalized per node, and the dominant gradient on any single score
must remain the reconstruction signal — if $w$ were large, the shared shift would start dragging every
logit toward the same value and the BCE would degrade. This injection trick is specific to *this*
harness's score-only interface; it is not how a standalone VGAE is trained (there one simply sums
ELBO terms in the loss), and the small coefficient is a direct consequence of the KL entering through
the scores rather than as a separate loss term. One more harness-driven detail: I put BatchNorm only
on the shared intermediate layers and *not* on the $\mu$/$\log\sigma$ heads, because the decoder is a
raw inner product and normalizing the final embeddings would crush the magnitude that the dot product
needs to spread positive and negative pairs apart.

I should be honest with myself about the known tension in this model before I run it, because it
predicts where the rung will be weak. The Gaussian prior pulls every embedding toward the origin, but
the inner-product decoder wants to push embeddings *outward* and apart so that $z_i^\top z_j$ is large
for edges and small for non-edges. These two forces fight: the prior is, in a real sense, a poor match
for an inner-product likelihood, and the consequence is that the latent space the encoder is allowed to
use is squeezed. On top of that, the sampling noise injected at train time is pure variance from the
ranking metrics' point of view — it helps calibration and prevents overfitting on a small graph, but
it blurs the exact top-of-list ordering that MRR and Hits@K reward. And the decoder is the weakest
possible one: a single inner product, no learned interaction between the two embeddings, no structural
features about the *pair* (how many neighbors do $i$ and $j$ share?). The model only ever knows about a
pair through the geometry of two independently-encoded points. That is a deliberate floor: I want to
see how far the clean probabilistic version of "GCN embeddings, dot product" gets before I start adding
machinery, so that every later rung is measured against a principled, not a sloppy, starting point.

So the step-1 edit is concrete and lands directly in the editable region. I replace the default class
with: a shared GCN stack (BatchNorm + ReLU + dropout on its layers) when there is more than one layer;
two GCN heads `conv_mu` and `conv_logstd` producing $\mu$ and $\log\sigma$ with no BatchNorm;
`encode` that runs the shared stack, computes $\mu$ and $\log\sigma$, and returns the reparameterized
sample during training or $\mu$ at eval; a `decode` that is the plain inner product
$(z_{\text{src}}\cdot z_{\text{dst}}).\text{sum}$; and a `forward` that, when training, computes the
mean-per-node KL from the cached $\mu,\log\sigma$ and adds $w\cdot\mathrm{KL}$ to the scores before
returning them, so the KL gradient reaches the encoder. The encoder caches $\mu$ and $\log\sigma$ as
instance attributes during `encode` precisely so `forward` can reach them for the KL injection without
re-running the GCN. (The full scaffold module is in the answer.)

What do I expect, and against what would I judge it? This is the floor, so I am not expecting it to
win — I am expecting it to *separate*, i.e. produce respectable AUC (positives clearly above
negatives), while being noticeably weaker on the ranking-sensitive MRR and Hits@K, for the three
structural reasons above: the prior/inner-product mismatch squeezing the space, the sampling noise
blurring the top of the list, and the structureless decoder having no way to sharpen the hardest pairs.
On the small citation graphs the variational regularization might even hurt relative to a deterministic
embedding, because the graphs are small and the KL pull plus sampling noise cost more in ranking
precision than they buy in generalization. On the large `ogbl-collab` graph, where the metric is
Hits@50 over a huge candidate pool, I expect the inner-product decoder to struggle the most — a single
dot product is a blunt instrument when tens of thousands of non-edges have to be pushed below each true
edge, and the absence of any explicit common-neighbor signal (the heuristic that classically dominates
on dense collaboration graphs) should bite hardest there. That is the falsifiable shape of the result:
decent AUC, soft ranking metrics, and a collaboration-graph Hits@50 that I will read as the clearest
verdict that the next rung must (a) drop the variational noise in favor of a deterministic embedding
and (b) replace the bare inner product with a *learned* decoder that can mix the two embeddings, since
those are exactly the two pieces this model deliberately leaves on the floor. If instead the KL
injection destabilizes training and even AUC comes in soft, the diagnosis flips to "the score-shift
trick is too crude," and the next move would lean even harder on a learned decoder that does not depend
on the inner-product geometry the Gaussian prior fights.
