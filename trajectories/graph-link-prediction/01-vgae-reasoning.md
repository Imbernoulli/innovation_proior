The scaffold default is a GCN encoder feeding a dot product, trained by binary cross-entropy on
sampled positive and negative pairs. The prior art gives me four shapes of model, and two I can
eliminate immediately on the data I have been handed. The fixed heuristics — common neighbors,
Adamic–Adar, resource allocation — score a pair by neighborhood overlap and learn nothing; the
random-walk embeddings in the DeepWalk / node2vec family read structure but never touch $X$. Both
throw the node features away, and these are *feature-rich* citation graphs — 1,433 dimensions per node
on Cora, 3,703 on CiteSeer — so discarding the largest signal at the very start is a move I want to
*arrive at* by proving the learned encoder is not enough, not one I open with. So the real contest is
between the deterministic GCN-plus-dot scaffold and its probabilistic sibling, and the question that
decides the first step is narrow: does framing "GCN embeddings, dot product" as a genuine *generative
model of the adjacency* — with the regularizer that framing forces on me — buy anything before I bolt
on learned decoders and structural features? Starting from the principled version of exactly the
scaffold's own object is the honest floor, and it gives everything I build later a non-sloppy baseline
to be measured against. So I start with the variational graph auto-encoder: the same GCN-encoder /
inner-product-decoder skeleton, but derived as a latent-variable model and trained with the KL the
derivation demands.

Let me write down what I am modeling, because the whole method falls out of taking the generative
story seriously. I have an undirected graph on $N$ nodes with adjacency $A$ and features $X$. I posit
a latent embedding $z_i\in\mathbb{R}^F$ per node and a generative model in which an edge between $i$
and $j$ is a Bernoulli draw $p(A_{ij}=1\mid z_i,z_j)=\sigma(z_i^\top z_j)$. That is the *inner-product
decoder*, and it is the natural likelihood for a symmetric binary relation: the score $z_i^\top z_j$ is
large when two embeddings point the same way, the sigmoid turns it into an edge probability, and
reconstructing the whole adjacency means $\hat A=\sigma(ZZ^\top)$. The prior on the latents is the
isotropic Gaussian $p(Z)=\prod_i\mathcal N(z_i\mid 0,I)$. So I want to fit a model that, given features
and the observed edges, places connected nodes near each other in inner-product geometry, by
maximizing the probability it assigns the observed edges.

The inference side — the encoder — is where structure and features enter. I cannot do exact posterior
inference over $Z$, so I take the variational route: an approximate posterior $q(Z\mid X,A)=\prod_i
\mathcal N(z_i\mid \mu_i,\operatorname{diag}(\sigma_i^2))$ whose per-node mean and log-variance come
from a GCN. A shared GCN stack maps $X$ to a hidden $\mathbb{R}^{N\times H}$, and two GCN heads on top
produce $\mu=\mathrm{GCN}_\mu(X,A)$ and $\log\sigma^2=\mathrm{GCN}_\sigma(X,A)$. Both heads are GCNs and
not plain linear layers because I want a node's *uncertainty* to depend on its neighborhood — a leaf
with one ambiguous citation should be allowed a wider posterior than a hub with a clear structural
role. The cheaper alternative — fix $\sigma^2\equiv1$ and learn only $\mu$, which collapses the KL to a
plain $\ell_2$ shrinkage on the means and saves a whole GCN head — throws away the one thing the
variational framing is *for*: with $\sigma$ fixed every node carries the same posterior width
regardless of the evidence its neighborhood provides, an $\ell_2$-regularized deterministic embedding
wearing the reparameterization costume, not the principled floor I set out to build. So I keep both
heads. The whole encoder is one input-width GCN plus two narrow heads feeding a parameter-free
decoder; on CiteSeer that is around $1.08\times10^6$ weights, dominated entirely by the first-layer
feature projection ($3703\times256$ alone is $\sim9.5\times10^5$), about half that on Cora, and far
less on ogbl-collab whose input is only 128-wide. So almost every weight sits in the input projection:
a 256-wide two-layer net is not where the startup budget bites, and anything wanting capacity later
will have to buy it in the decoder, not by widening an encoder already saturated by the input
dimension.

The objective is the ELBO, $\mathcal L=\mathbb E_{q}[\log p(A\mid Z)]-\mathrm{KL}[q\,\|\,p]$. The first
term is the expected reconstruction log-likelihood, which under the Bernoulli-inner-product decoder is
exactly a cross-entropy between the true adjacency entries and $\sigma(z_i^\top z_j)$. The one wrinkle
is class balance: a graph has $O(|E|)$ edges among $O(N^2)$ pairs, so the naive sum is swamped by
non-edges — but the harness samples as many negatives as positives each epoch, exactly the standard
VGAE re-weighting to a balanced cross-entropy. So the loop's per-batch BCE *is* the reconstruction half
of my ELBO, already balanced, and I do not re-derive the weighting. The KL, for two Gaussians, has the
closed form $\mathrm{KL}=-\tfrac12\sum_i\sum_f(1+\log\sigma_{if}^2-\mu_{if}^2-\sigma_{if}^2)$; I keep
the heads in log-variance so the reparameterization exponential is numerically tame and the KL reads
directly off the head outputs. Early in training, with per-dimension $\mu^2\approx0.5$ and
$\log\sigma^2\approx-0.5$ (so $\sigma^2\approx0.607$), each dimension contributes
$-\tfrac12(1-0.5-0.5-0.607)=0.303$, and over $F=256$ dimensions the per-node KL is about $77.6$ — a
number I need below when I size the score injection. To make the
expectation differentiable I use the reparameterization trick, $z_i=\mu_i+\epsilon\odot\sigma_i$ with
$\epsilon\sim\mathcal N(0,I)$ at train time; at eval I drop the noise and use $z_i=\mu_i$.

Now the part where the principled story has to bend to this harness, and getting it right is the whole
reason to think carefully rather than copy a generic VGAE. The fixed loop does *not* know about an
ELBO: it calls my `forward`, gets back a vector of edge scores, and computes BCE against {1 for
positives, 0 for negatives}. That BCE is the reconstruction term, handled for free — but the loop will
never add a KL term, because it only ever sees my scalar scores and offers no hook for an auxiliary
loss. If I do nothing, the KL gradient never flows, $\sigma$ collapses, $\mu$ drifts to whatever
reconstruction wants, and I have a plain (non-variational) auto-encoder with extra sampling noise —
strictly worse than fitting point embeddings. So the KL has to be smuggled into the computation graph
*of the scores themselves*. I add it, as a single scalar, onto every returned score:
$\text{score}\leftarrow\text{score}+w\cdot\mathrm{KL}_{\text{per-node}}$, where
$\mathrm{KL}_{\text{per-node}}=-\tfrac12\,\overline{\sum_f(1+\log\sigma^2-\mu^2-\sigma^2)}$ is the
mean-over-nodes KL.

That normalization is a real choice among three, and only one is scale-stable. The raw total
$\mathrm{KL}_{\text{total}}=\sum_i\mathrm{KL}_i$ scales with $N$ — roughly $2.7\times10^3$ times a
per-node value on Cora but $2.36\times10^5$ times it on ogbl-collab — so a single $w$ against the total
would pull almost a hundred times harder on the large graph, entangling regularization strength with
graph size, and no one $w$ could transfer across the three datasets. Dividing by $N$ makes the quantity
*intensive* — the average divergence a node's posterior carries, comparable across graphs — so one $w$
is meaningful everywhere. Dividing by the batch's score count $M$ is wrong for a subtler reason: the KL
is a property of the encoder's latent distribution over *nodes*, not of the sampled *pairs*, so tying
it to $M$ would weaken the regularizer whenever more negatives are drawn and strengthen it whenever
fewer are, coupling it to a nuisance parameter of the sampler. So the per-node mean it is.

I want to know exactly what the injection does to the two things I care about — the ranking the metrics
read and the gradient the encoder feels. The shift $c=w\cdot\mathrm{KL}_{\text{per-node}}$ is one number
added identically to all $M$ scores. With $\mathrm{KL}_{\text{per-node}}\approx77.6$ and $w=0.005$ that
is about $0.39$ logits, added to positive scores that run into the hundreds — and AUC, MRR, Hits@K are
all pure functions of the *order* of the scores, which a common additive shift preserves exactly, so
the injection cannot move a single ranked position (and at eval the shift is not even applied). So it
cannot corrupt the ranking. The gradient is the point. Writing each logit as $\ell_p=s_p+c$, the loop's
mean BCE has $\partial\mathcal L/\partial c=\tfrac1M\sum_p(\sigma(\ell_p)-y_p)$, and since $c=w\,
\mathrm{KL}(\theta)$ depends on the encoder through $\mu$ and $\log\sigma^2$, the KL's contribution to
the encoder gradient is $\big[\tfrac1M\sum_p(\sigma(\ell_p)-y_p)\big]\cdot w\cdot\partial\mathrm{KL}/
\partial\theta$. The gradient is real — it flows into exactly the parameters I want to regularize — but
its magnitude is gated by the batch's *mean residual*, which near calibration sits near zero (positives
contribute $\sigma-1<0$, negatives $\sigma-0>0$, and they cancel). So the regularizer self-throttles,
pulling hardest precisely when the model is most miscalibrated and backing off as it calibrates — a
second layer of gentleness on top of the small coefficient.

The coefficient $w=0.005$ I keep small: as $w\to\infty$ the KL dominates, dragging $\mu\to0$ and
$\sigma^2\to1$, collapsing every embedding onto the prior so $z_i^\top z_j$ carries no signal, so the
useful model sits near the small-$w$ end, close to a deterministic auto-encoder but with just enough KL
pressure to keep the posterior honest. One harness-driven detail: BatchNorm only on the shared
intermediate layers, not on the $\mu$/$\log\sigma^2$ heads, because the decoder is a raw inner product
and $z_i^\top z_j$ is quadratic in embedding magnitude — a BatchNorm forcing $\|z\|$ to a fixed scale
would flatten exactly the dynamic range that separates positives from negatives.

There is a cost to the sampling that predicts which metrics this step will disappoint. At train time a
pair's score is $z_i^\top z_j$ with $z=\mu+\epsilon\odot\sigma$; its mean over the noise is the clean
$\mu_i^\top\mu_j$, but its variance is $\sum_f(\mu_{if}^2\sigma_{jf}^2+\mu_{jf}^2\sigma_{if}^2+
\sigma_{if}^2\sigma_{jf}^2)$. Early in training, with $\sigma^2\approx1$ and per-dimension $\mu^2$ of
order one across $F=256$ dimensions, that variance is of order $256\times3\approx7.7\times10^2$, a
standard deviation near $28$. A well-aligned positive pair with $|\mu_{if}|\sim1$ and matching signs has
$\mu_i^\top\mu_j\sim F=256$, so its signal-to-noise ratio is roughly $256/28\approx9$: the sampling
perturbs the best pairs by only about $11\%$ of their score, nowhere near enough to flip them below a
random negative, which is why the *average* separation AUC measures survives. But AUC is a bulk
statistic dominated by the well-separated majority; MRR and Hits@K read the *marginal* pairs clustered
near the top of the candidate list, where a true edge sits only tens of logits above the strongest
negatives — there a noise standard deviation of $28$ is comparable to the gap and randomizes their
order run to run. The KL/reconstruction tug drives $\sigma^2$ down as training proceeds, but by then the
coarse geometry is largely set. So the arithmetic says: expect the sampling to cost me most on the
ranking-sensitive metrics and least on AUC, and most where the candidate pool is largest and the
top-of-list ties densest.

The second structural tension is baked into the model class, not the harness. The Gaussian prior pulls
every embedding toward the origin through the $-\mu^2$ KL term, while the inner-product likelihood wants
to push embeddings *outward* and apart. These fight over $\|\mu\|$; with $w=0.005$ reconstruction wins
comfortably, but the pull never disappears — the prior is a poor match for an inner-product likelihood,
so the latent space is gently squeezed no matter how well I train. And underneath both tensions sits the
decoder, the weakest possible one: a single inner product, no learned interaction between the two
embeddings, no structural feature about the *pair* — the model only ever knows a pair through the
geometry of two independently-encoded points. That is a deliberate floor: I want to see how far the
clean probabilistic version of "GCN embeddings, dot product" gets before I add machinery.

So the step-1 edit replaces the default class with: a shared GCN stack (BatchNorm + ReLU + dropout on
its layers) when there is more than one layer; two GCN heads `conv_mu` and `conv_logstd` with no
BatchNorm; an `encode` that returns the reparameterized sample during training or $\mu$ at eval; a plain
inner-product `decode`; and a `forward` that, when training, computes the mean-per-node KL from the
cached $\mu,\log\sigma^2$ and adds $w\cdot\mathrm{KL}$ to the scores so the gradient reaches the encoder.
The encoder caches $\mu$ and $\log\sigma^2$ as attributes during `encode` precisely so `forward` can
reach them for the injection without re-running the GCN, and every tensor the loop touches keeps exactly
the shape the default scaffold produced.

I am not expecting this floor to win — I expect it to *separate*: respectable AUC, noticeably softer MRR
and Hits@K, for the three reasons the arithmetic pins down — the prior/inner-product squeeze on
$\|\mu\|$, the sampling noise of standard deviation $\sim28$ blurring the top of the list, and the
structureless decoder having no way to sharpen the hardest pairs. On the small citation graphs the
variational regularization might even hurt relative to a deterministic embedding, and I would expect the
seed-to-seed spread to widen, since a squeezed, noisy latent space lands well or badly depending on the
random init. On ogbl-collab, where Hits@50 reads a huge candidate pool and common-neighbor overlap is
the classically dominant signal, the bare inner product should struggle most. That shape would point the
next step at dropping the variational noise and replacing the bare inner product with a *learned* decoder
that can mix the two embeddings, the two pieces this model deliberately leaves on the floor.
