The scaffold default is a GCN encoder feeding a dot product, trained by binary cross-entropy on
sampled positive and negative pairs. Before I reach for anything heavier I want to be deliberate about
where on the ladder I start, so let me lay the real options side by side. The prior art gives me four
shapes of model. There are the fixed heuristic scores — common neighbors, Adamic–Adar, resource
allocation — which score a pair by a hand-designed function of its shared neighborhood and learn
nothing; there are random-walk embeddings in the DeepWalk / node2vec family, which learn a node
embedding from walk co-occurrence and score a pair by an embedding dot product; there is the GCN as a
learned encoder folding features and structure into every node embedding; and there is the scaffold
default itself, a GCN encoder with a bare inner-product decoder trained end-to-end by the loop's BCE.
Two of these I can eliminate immediately on the data I have been handed. The pure heuristics ignore the
node features entirely, and these are *feature-rich* citation graphs — Cora carries a 1,433-dimensional
feature per node, CiteSeer 3,703 — so throwing the features away to score by neighborhood overlap alone
is leaving the largest signal on the table at the very first rung; that is a move I want to *arrive at*
by proving the learned encoder is not enough, not one I want to open with. Random-walk embeddings share
that defect: skip-gram over walks reads structure but never touches $X$, and again the features are too
rich to discard first. So the real contest is between the deterministic GCN-plus-dot scaffold and its
probabilistic sibling, and the question that actually decides the first rung is narrow and worth
answering cleanly: does framing "GCN embeddings, dot product" as a genuine *generative model of the
adjacency* — with the regularizer that framing forces on me — buy anything before I start bolting on
learned decoders and structural features? Starting from the principled version of exactly the scaffold's
own object is the honest floor: it tells me whether the probabilistic story pays for itself, and it
gives every later rung a non-sloppy baseline to be measured against. So the first rung is the
variational graph auto-encoder: the same GCN-encoder / inner-product-decoder skeleton, but derived as a
latent-variable model and trained with the KL that the derivation demands.

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
log-variance are produced by a GCN read off the features and the connectivity. Concretely a shared GCN
stack maps $X\in\mathbb{R}^{N\times \text{in}}$ to a hidden $\mathbb{R}^{N\times H}$, and then two
separate GCN heads on top of it produce $\mu=\mathrm{GCN}_\mu(X,A)\in\mathbb{R}^{N\times H}$ and
$\log\sigma^2=\mathrm{GCN}_\sigma(X,A)\in\mathbb{R}^{N\times H}$. The reason both heads are GCNs and not
plain linear layers is that I want the *uncertainty* of a node's embedding to also depend on its
neighborhood — a node with few, ambiguous connections should be allowed a wider posterior than a hub
with a clear structural role. It is worth checking the parameter count this implies, because a budget is
verified at startup and I do not want the very first rung to trip it. With $H=256$ and a two-layer
encoder, the shared stack is a single $\mathrm{GCNConv}(\text{in},256)$: on CiteSeer that is
$3703\times256+256\approx 9.48\times10^5$ weights, on Cora $1433\times256+256\approx 3.67\times10^5$,
on `ogbl-collab` only $128\times256+256\approx3.3\times10^4$. Each head is a
$\mathrm{GCNConv}(256,256)\approx 6.58\times10^4$, so two heads add ${\sim}1.3\times10^5$. The whole
encoder is thus around $1.08\times10^6$ parameters on CiteSeer and about half that on Cora, dominated
entirely by the input projection — and the decoder side is *parameter-free*, since a raw inner product
has no weights at all. So the entire model is one input-width GCN plus two narrow heads; a two-layer,
256-wide net is not where a parameter budget bites, and I can spend the design attention on the
objective rather than on shrinking the net. That count also tells me where the money goes: almost every
weight sits in the first-layer feature projection, so if a later rung wants capacity it will have to buy
it in the decoder, not by widening the encoder, because the encoder is already saturated by the input
dimension.

Before I commit to two learned heads, I should ask whether the variance head earns its keep, because the
cheaper alternative is real and I want to reject it on a reason rather than by default. I could fix the
posterior variance to a constant — $\log\sigma^2\equiv 0$, i.e. $\sigma^2\equiv1$ — and learn only
$\mu$. That collapses the KL to $-\tfrac12\sum_{i,f}(1-\mu_{if}^2)=\tfrac12\|\mu\|_F^2-\text{const}$, a
plain $\ell_2$ shrinkage on the means, and it saves one whole GCN head. But it throws away the one thing
the variational framing is *for*: with $\sigma$ fixed, every node is forced to carry the same posterior
width regardless of how much evidence its neighborhood provides, so a leaf with one ambiguous citation
is treated as exactly as certain as a densely-connected hub. The point of a *variational* auto-encoder
over a deterministic one is precisely that the model can widen the posterior where the graph is
uninformative and narrow it where it is not, and only a learned $\sigma$ head expresses that. Fixing
$\sigma$ would leave me with an $\ell_2$-regularized deterministic embedding wearing the
reparameterization costume — which is not the principled floor I set out to build. So I keep both heads;
the extra $6.58\times10^4$ parameters buy the data-dependent uncertainty that is the whole reason to be
variational here.

The objective is the variational lower bound (ELBO):
$\mathcal L=\mathbb E_{q(Z\mid X,A)}[\log p(A\mid Z)]-\mathrm{KL}[q(Z\mid X,A)\,\|\,p(Z)]$. The first
term is the expected reconstruction log-likelihood — under the Bernoulli-inner-product decoder this is
$\sum_{i,j}\big[A_{ij}\log\sigma(z_i^\top z_j)+(1-A_{ij})\log(1-\sigma(z_i^\top z_j))\big]$, exactly a
cross-entropy between the true adjacency entries and $\sigma(z_i^\top z_j)$. The one wrinkle is class
balance: a real graph has $O(|E|)$ edges among $O(N^2)$ possible pairs, so the naive sum over all
entries is swamped by the negative (non-edge) class. The harness resolves this for me by sampling as
many negatives as positives each epoch, which is exactly the standard VGAE re-weighting of the
reconstruction term to a balanced cross-entropy — so the loop's per-batch BCE *is* the reconstruction
half of my ELBO, already correctly balanced, and I do not have to re-derive the weighting. The second
term is the KL divergence from the approximate posterior to the prior, and for two Gaussians it has the
closed form
$\mathrm{KL}=-\tfrac12\sum_i\sum_f\big(1+\log\sigma_{if}^2-\mu_{if}^2-\sigma_{if}^2\big)$. I keep
the heads parametrized in log-variance so that the exponential in the reparameterization is numerically
tame and the KL reads directly off the head outputs. It is worth pinning down numerically where this KL
lives, because that tells me the regime training starts in and how large a shift the regularizer will
ever inject. At initialization the GCN biases are zero, so $\mu\approx0$ and $\log\sigma^2\approx0$
(hence $\sigma^2\approx1$), and every dimension contributes $-\tfrac12(1+0-0-1)=0$: the KL is exactly
zero when the posterior *is* the prior. Now walk one step into training: suppose reconstruction has
pushed a node to average per-dimension $\mu^2\approx0.5$ and pulled its log-variance to
$\log\sigma^2\approx-0.5$ (so $\sigma^2=e^{-0.5}\approx0.607$). Each dimension now contributes
$-\tfrac12\big(1-0.5-0.5-0.607\big)=-\tfrac12(-0.607)=0.303$, and over $F=256$ dimensions that is a
per-node KL of about $77.6$. So the KL swells from $0$ at init toward tens-per-node as the encoder
pushes the posterior away from the prior — a small quantity early, growing over training, which is
precisely the regime a small fixed coefficient is tuned for. To make the expectation differentiable I
use the reparameterization trick: instead of sampling $z_i$ directly, I sample
$\epsilon\sim\mathcal N(0,I)$ and set $z_i=\mu_i+\epsilon\odot\sigma_i$ during training, which moves the
stochasticity off the parameters so gradients flow through $\mu$ and $\sigma$. At evaluation I drop the
noise and use $z_i=\mu_i$, the posterior mean, as the deterministic embedding.

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
mean-over-nodes KL (the standard $1/N$ VGAE normalization, here realized as a mean). The normalization
constant in that expression is not cosmetic and I want to derive it rather than inherit it, because
there are three plausible choices and only one is scale-stable. The raw quantity is
$\mathrm{KL}_{\text{total}}=\sum_i\mathrm{KL}_i$, which scales with $N$: it is roughly $2.7\times10^3$
times a per-node value on Cora but $2.36\times10^5$ times it on `ogbl-collab`. If I injected
$w\cdot\mathrm{KL}_{\text{total}}$ the *same* coefficient $w$ would pull almost a hundred times harder on
the large graph than on Cora — the regularization strength would be entangled with graph size, and no
single $w$ could transfer across the three datasets. Dividing by $N$ fixes that: the per-node mean is an
*intensive* quantity, the average divergence a node's posterior carries, comparable across graphs, so
one $w$ is meaningful everywhere. The tempting third option — dividing by the batch's score count $M$ —
is wrong for a subtler reason: the KL is a property of the encoder's latent *distribution over nodes*,
not of the sampled *pairs*, so tying it to $M$ would make the regularizer weaker whenever more negatives
are sampled and stronger whenever fewer are, coupling it to a nuisance parameter of the sampler. That is
exactly why the per-node mean is the right normalization and the num-scores division is not. And it
composes cleanly with what the loop does downstream: the loop averages BCE over the $M$ scores, so the
gradient the injected scalar $c=w\cdot\mathrm{KL}_{\text{per-node}}$ receives is
$\partial\mathcal L/\partial c=\tfrac1M\sum_p(\sigma(\ell_p)-y_p)$, already a per-pair average, and
$\partial c/\partial\theta=w\cdot\tfrac1N\,\partial\mathrm{KL}_{\text{total}}/\partial\theta$ is a
per-node average — a product of two intensive quantities, both properly normalized, which is why the
trick behaves the same across a 2.7k-node and a 236k-node graph.

Before I trust the injection I want to trace exactly what it does to the two things I care about — the
ranking the metrics read and the gradient the encoder feels — because a trick that quietly wrecks either
is worse than no regularizer at all. Take the ranking first. The shift
$c=w\cdot\mathrm{KL}_{\text{per-node}}$ is one number, added identically to all $M$ scores in the batch.
Concretely, with the $\mathrm{KL}_{\text{per-node}}\approx77.6$ I computed above and $w=0.005$, that
shift is about $0.39$ logits — added to positive scores that, as I will show below, run into the
hundreds. AUC, MRR and Hits@K are all pure functions of the *order* of the scores, and adding a constant
to every score preserves order exactly, so a common $+0.39$ cannot move a single ranked position. The
injection has literally zero effect on any reported metric — and in any case at evaluation I run in eval
mode where the shift is not even applied. That is reassuring: whatever the KL does, it cannot corrupt the
ranking directly. Now the gradient, which is the point of the whole exercise. Writing each logit as
$\ell_p=s_p+c$ with $s_p=z_{i}^\top z_{j}$, the loop's mean BCE has
$\partial\mathcal L/\partial c=\tfrac1M\sum_p(\sigma(\ell_p)-y_p)$, the mean residual over the batch,
and since $c=w\,\mathrm{KL}(\theta)$ depends on the encoder parameters $\theta$ through $\mu$ and
$\log\sigma^2$, the KL's contribution to the encoder gradient is
$\big[\tfrac1M\sum_p(\sigma(\ell_p)-y_p)\big]\cdot w\cdot\partial\mathrm{KL}/\partial\theta$. Two things
fall out of that expression. The gradient is real — it flows into exactly the parameters I wanted to
regularize — which is the whole reason the injection works at all. But its magnitude is gated by the
batch's *mean residual*: with a balanced positive/negative sample and a roughly calibrated model that
mean sits near zero (positives contribute $\sigma-1<0$, negatives $\sigma-0>0$, and near calibration
they cancel), so the regularizer self-throttles, pulling hardest precisely when the model is most
miscalibrated and backing off as it calibrates. Put a number on it: a model outputting
$\sigma\approx0.7$ on positives and $\sigma\approx0.3$ on negatives has residuals $-0.3$ and $+0.3$ that
cancel to a mean near zero, so the KL barely pulls; the throttle keeps the KL quiet except transiently,
when a batch happens to be lopsidedly wrong in one direction. That is a second layer of gentleness on top
of the small coefficient, and it is why I am comfortable that this trick regularizes without hijacking
training.

The coefficient $w$ then controls how hard the regularizer pulls even in the miscalibrated regime, and I
keep it small, $w=0.005$. I can bracket the choice by its two limits. As $w\to0$ the KL gradient
vanishes and I am back to a deterministic auto-encoder wearing useless sampling noise — the failure mode
I flagged. As $w\to\infty$ the KL term dominates, dragging $\mu\to0$ and $\sigma^2\to1$; the posterior
collapses onto the prior, every embedding becomes standard Gaussian noise, $z_i^\top z_j$ carries no
signal, and AUC decays to chance. The useful model lives near the small-$w$ end of that interval,
deliberately close to the deterministic auto-encoder but with just enough KL pressure to keep the
posterior honest; $0.005$ sits there. One more harness-driven detail belongs with this: I put BatchNorm
only on the shared intermediate layers and *not* on the $\mu$/$\log\sigma^2$ heads, because the decoder
is a raw inner product and normalizing the final embeddings to unit variance would crush the very
magnitude the dot product needs to spread positive and negative pairs apart — the score $z_i^\top z_j$
is quadratic in that magnitude, so a BatchNorm forcing $\|z\|$ to a fixed scale would flatten exactly
the dynamic range that separates positives from negatives.

There is a cost to the sampling that I should quantify rather than wave at, because it predicts exactly
which metrics this rung will disappoint. At train time the score of a pair is
$z_i^\top z_j=(\mu_i+\epsilon_i\odot\sigma_i)^\top(\mu_j+\epsilon_j\odot\sigma_j)$. Its mean over the
noise is the clean $\mu_i^\top\mu_j$, but its variance is
$\sum_f(\mu_{if}^2\sigma_{jf}^2+\mu_{jf}^2\sigma_{if}^2+\sigma_{if}^2\sigma_{jf}^2)$. Early in training,
with $\sigma^2\approx1$ and per-dimension $\mu^2$ of order one across $F=256$ dimensions, that variance
is of order $256\times(1+1+1)\approx 7.7\times10^2$, a standard deviation near $28$. Now set that against
the signal. A well-aligned positive pair with $|\mu_{if}|\sim1$ and matching signs has
$\mu_i^\top\mu_j\sim\sum_f\mu_{if}\mu_{jf}\sim F=256$ — of order a couple hundred — so its
signal-to-noise ratio is roughly $256/28\approx9$: the sampling perturbs the best pairs by only about
$11\%$ of their score, nowhere near enough to flip them below a random negative, which is why the
*average* separation AUC measures survives. But AUC is a bulk statistic dominated by the well-separated
majority; MRR and Hits@K read the *marginal* pairs, the ones clustered near the top of the candidate
list where the score gaps between adjacent candidates are small. For those pairs — a true edge whose
$\mu_i^\top\mu_j$ sits only tens of logits above the strongest negatives — a noise standard deviation of
$28$ is comparable to or larger than the gap, so their relative order gets randomized run to run. The
KL/reconstruction tug will drive $\sigma^2$ down as training proceeds, shrinking this noise, but by then
the coarse geometry is largely set. So the arithmetic says: expect the sampling to cost me most on the
ranking-sensitive metrics and least on AUC, and to cost me most where the candidate pool is largest and
the top-of-list ties densest.

I should be equally explicit about the second structural tension, because it is baked into the model
class, not the harness. The Gaussian prior pulls every embedding toward the origin through the $-\mu^2$
term in the KL, while the inner-product likelihood wants to push embeddings *outward* and apart so that
$z_i^\top z_j$ is large and positive for edges and large and negative for non-edges. These two forces
fight over $\|\mu\|$: the reconstruction gradient inflates it, the KL gradient (scaled by $w$) deflates
it, and the equilibrium norm the encoder is allowed to use is set by their ratio. With $w=0.005$ the
reconstruction side wins that tug comfortably, but the pull never disappears — the prior is, in a real
sense, a poor match for an inner-product likelihood, and the consequence is that the latent space is
gently squeezed no matter how well I train. And underneath both tensions sits the decoder, which is the
weakest possible one: a single inner product, no learned interaction between the two embeddings, no
structural feature about the *pair* — how many neighbors do $i$ and $j$ actually share? The model only
ever knows a pair through the geometry of two independently-encoded points. That is a deliberate floor:
I want to see how far the clean probabilistic version of "GCN embeddings, dot product" gets before I add
machinery, so that every later rung is measured against a principled, not a sloppy, starting point.

So the step-1 edit is concrete and lands directly in the editable region. I replace the default class
with: a shared GCN stack (BatchNorm + ReLU + dropout on its layers) when there is more than one layer;
two GCN heads `conv_mu` and `conv_logstd` producing $\mu$ and $\log\sigma^2$ with no BatchNorm;
`encode` that runs the shared stack, computes $\mu$ and $\log\sigma^2$, and returns the reparameterized
sample during training or $\mu$ at eval; a `decode` that is the plain inner product
$(z_{\text{src}}\cdot z_{\text{dst}}).\text{sum}$; and a `forward` that, when training, computes the
mean-per-node KL from the cached $\mu,\log\sigma^2$ and adds $w\cdot\mathrm{KL}$ to the scores before
returning them, so the KL gradient reaches the encoder. The encoder caches $\mu$ and $\log\sigma^2$ as
instance attributes during `encode` precisely so `forward` can reach them for the KL injection without
re-running the GCN. A quick shape pass confirms it all fits the contract: $X\in\mathbb{R}^{N\times
\text{in}}$ through the stack and heads gives $\mu,\log\sigma^2\in\mathbb{R}^{N\times H}$, the training
sample $z=\mu+\epsilon\odot\sigma$ is also $\mathbb{R}^{N\times H}$, `decode` gathers
$z[\text{src}],z[\text{dst}]\in\mathbb{R}^{M\times H}$, their elementwise product summed over $H$
returns the $\mathbb{R}^{M}$ scores the loop expects, and the KL scalar broadcasts cleanly onto that
$M$-vector without changing its shape. Every tensor the loop touches is exactly the shape the default
scaffold produced, so the drop-in is contract-clean.

What do I expect, and against what would I judge it? This is the floor, so I am not expecting it to
win — I am expecting it to *separate*, i.e. produce respectable AUC (positives clearly above
negatives), while being noticeably weaker on the ranking-sensitive MRR and Hits@K, for the three
structural reasons the arithmetic above pins down: the prior/inner-product mismatch squeezing $\|\mu\|$,
the sampling noise of standard deviation ${\sim}28$ blurring the top of the list, and the structureless
decoder having no way to sharpen the hardest pairs. On the small citation graphs the variational
regularization might even hurt relative to a deterministic embedding, because the graphs are small and
the KL pull plus sampling noise cost more in ranking precision than they buy in generalization; I would
also not be surprised to see the seed-to-seed spread widen, since a squeezed, noisy latent space is
exactly the kind that lands well or badly depending on the random init — the same run-to-run variance
the noise analysis predicts should show up as unstable AUC across seeds, not just soft ranking within a
seed. On the large `ogbl-collab` graph, where the metric is Hits@50 over a huge candidate pool, I expect
the inner-product decoder to struggle the most — a single dot product is a blunt instrument when tens of
thousands of non-edges have to be pushed below each true edge, and the absence of any explicit
common-neighbor signal (the heuristic that classically dominates on dense collaboration graphs) should
bite hardest there. That is the falsifiable shape of the result: decent AUC, soft ranking metrics, and a
collaboration-graph Hits@50 that I will read as the clearest verdict that the next rung must (a) drop the
variational noise in favor of a deterministic embedding and (b) replace the bare inner product with a
*learned* decoder that can mix the two embeddings, since those are exactly the two pieces this model
deliberately leaves on the floor. If instead the KL injection destabilizes training and even AUC comes in
soft, the diagnosis flips to "the score-shift trick is too crude," and the next move would lean even
harder on a learned decoder that does not depend on the inner-product geometry the Gaussian prior fights.
