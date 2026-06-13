# Research question

The dominant supervised-learning toolkit — linear/logistic regression, kernel machines, feed-forward and convolutional networks — assumes each instance is a **fixed-dimensional vector with a fixed coordinate ordering**. A growing class of problems violates both assumptions at once: the natural input is an **unordered collection of variable size**, i.e. a *set* $X=\{x_1,\dots,x_M\}$, $x_m\in\mathfrak X$, so the input domain is the power set $2^{\mathfrak X}$.

Two regimes arise. In the **set → label** regime, the target depends only on the set, so the model must be **permutation invariant**: for every permutation $\pi$,
$$f(\{x_1,\dots,x_M\}) = f(\{x_{\pi(1)},\dots,x_{\pi(M)}\}).$$
In the **set → per-element** (transductive) regime, each element carries its own output and reordering the input must reorder the output identically — **permutation equivariance**:
$$\mathbf f(\pi\mathbf x)=\pi\,\mathbf f(\mathbf x).$$

The goal is a single architecture that (i) accepts variable-size sets, (ii) is guaranteed invariant or equivariant *by construction* rather than by data augmentation, and (iii) scales to large set counts and large sets. The hard part is not engineering one such network but knowing the *space* of admissible functions: what is the most general invariant function, and the most general equivariant linear layer? Without that characterization, any pooling heuristic is a guess that might be missing expressive power or might secretly break the symmetry.

# Background

**Symmetry and invariant functions.** A set has no canonical ordering, so a function on it must not exploit one. The relevant symmetry group is the symmetric group $\mathcal S_M$ acting by permutation matrices on the index set. A function is invariant if it is constant on each orbit and equivariant if it intertwines the action. Classical mathematics already describes invariant functions of $M$ scalars: the **symmetric polynomials**. The **Fundamental Theorem of Symmetric Functions** states every symmetric polynomial is a polynomial in the power sums $p_q=\sum_m x_m^q$ (equivalently the elementary symmetric polynomials), and **Newton–Girard formulae** convert between power sums and elementary symmetric polynomials. The **Kolmogorov–Arnold representation theorem** gives a representation for *any* continuous multivariate function of $M$ variables.

**Exchangeability and de Finetti.** In Bayesian statistics an exchangeable model factors as $p(X\mid\alpha)=\int\big[\prod_m p(x_m\mid\theta)\big]\,p(\theta\mid\alpha)\,d\theta$. For an exponential family $p(x\mid\theta)=\exp(\langle\phi(x),\theta\rangle-g(\theta))$ with a conjugate prior, marginalizing $\theta$ yields a likelihood that depends on the data only through the sufficient statistic $\sum_m\phi(x_m)$.

**Pooling as a recurring heuristic.** Several systems already pool a per-element feature across members to obtain order-independence: pooling across a panoramic projection or across multiple rendered views of a 3D object for classification (Shi 2015; Su 2015), pooling over a sample set for a causality decision (Lopez-Paz 2016), and exploiting the row/column permutation symmetry of a normal-form payoff matrix (Hartford 2016). In multi-agent / sensor-network models (Sukhbaatar 2016, CommNet), each agent combines its own state with a pooled summary of the others. These are effective but *ad hoc*: pooling is adopted because it happens to be invariant, with no statement of which functions it can or cannot express.

**Group-equivariant networks.** A parallel line builds networks equivariant to general transformation groups by weight-sharing tied to the group structure (Gens & Domingos 2014; Cohen & Welling 2016, group-equivariant CNNs; Ravanbakhsh 2017). Permutation symmetry is one instance of this program, but the precise minimal parameter-sharing for the permutation group in a plain dense layer was not pinned down.

**Motivating diagnostic.** When a set is force-fed to a sequence model (LSTM/GRU) — the natural fallback when no set-native model exists — performance is order-dependent and degrades sharply as the test set grows beyond training sizes; a sum-of-digits task trained on length $\le 10$ and tested up to length 100 makes this failure visible. Kernel-on-distributions methods, the other established route, require an $N\times N$ kernel matrix and become infeasible past tens of thousands of sets.

# Baselines

**Distribution / set kernels (support distribution machines).** To predict from a set treated as an i.i.d. sample from a distribution $p$, one builds $f(p)=\sum_i\alpha_i y_i K(p_i,p)+b$ with a kernel between distributions estimated from samples, $\hat K(p,q)=\frac{1}{MM'}\sum_{i,j}k(x_i,y_j)$ (Poczos 2012/2013; Muandet 2012/2013; Szabo 2016; Oliva 2013). Core idea: lift each set to a distribution embedding and run kernel regression/classification. Gap: the $N\times N$ Gram matrix and its inversion cost $O(N^2)$–$O(N^3)$, prohibitive beyond $\sim\!16{,}000$ sets; the kernel is fixed in advance rather than learned end-to-end with the task.

**Sequence models on an imposed order (LSTM/GRU; "order matters").** Feed the set elements one at a time to a recurrent network and read the final state. Vinyals (2015) explicitly searches for a "good" ordering to feed the set, conceding that recurrent readers are order-sensitive. Gap: not invariant by construction (different orders give different outputs), and the learned dependence on position generalizes poorly to set sizes unseen in training.

**Voxel / multi-view pipelines for 3D shapes.** Convert an unordered point cloud into a voxel grid or a stack of rendered 2D views, then apply a 3D or 2D CNN (3DShapeNets, VoxNet, MVCNN, VRN). Core idea: re-impose a regular grid so standard convolution applies. Gap: voxelization is lossy and memory-heavy ($O(\text{res}^3)$), discards the native point representation, and bakes in a particular discretization rather than operating on the set of points directly.

**Bayesian sets / exchangeable generative models.** Score a candidate set by an exchangeable likelihood (de Finetti, exponential-family conjugate form). Core idea: high score for "coherent" sets, used for set expansion. Gap: tied to a hand-chosen likelihood family and its sufficient statistics; the per-element feature $\phi$ is fixed by the family, not learned.

# Evaluation settings

- **Population-statistic estimation.** Sets are i.i.d. samples from Gaussians; targets are entropy or mutual information under several generators (random 2-D rotation; correlation between two 16-D blocks; rank-1 perturbation in 32-D; random covariance in 32-D), set sizes $M\in[300,500]$. Metric: mean-squared error vs. the number of training sets $N$. Natural comparison: support distribution machines with an RBF kernel.
- **Sum of digits.** Build sets of digits (text tokens, or MNIST8m image stamps) with set-label equal to the sum; train on sets of length $\le 10$, test on lengths up to 100 (text) / 50 (image). Metric: accuracy = exact match after rounding. Comparison: LSTM and GRU with matched layers/parameters.
- **Point-cloud classification.** ModelNet40: 9{,}843 train / 2{,}468 test 3-D objects in 40 classes, sampled into point clouds of 100 / 1000 / 5000 points ($x,y,z$). Metric: classification accuracy. Comparison: voxel/multi-view CNNs.
- **Red-shift regression.** redMaPPer galaxy-cluster catalog, 17 photometric features per galaxy, clusters of $\sim$20–300 galaxies treated as sets. Metric: scatter $\frac{|z_{\text{spec}}-z|}{1+z_{\text{spec}}}$. Comparison: MLP, redMaPPer.
- **Set expansion / retrieval.** LDA top-word concept sets; concept-set retrieval and image tagging. Metric: recall@$k$ / retrieval rank.
- **Set anomaly detection.** Faces (CelebA): per-element outlier scoring within a set (equivariant regime).

# Code framework

The primitives that already exist: an autodiff tensor library with a dense layer, pointwise nonlinearities, an Adam/SGD optimizer, dropout, and a training loop with a regression (MAE/MSE) or classification (cross-entropy) loss. A set is stored as a `(batch, M, in_dim)` tensor (with a mask for variable $M$). What does **not** yet exist is the set-native module: the piece that turns the per-element tensor into one output while respecting the permutation symmetry. The scaffold leaves exactly that slot empty.

```python
import torch
import torch.nn as nn

class InvariantSetModel(nn.Module):
    """A model whose input is an unordered, variable-size set (batch, M, in_dim).
    The body must be designed so the set-to-label output ignores element order."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # TODO: build the invariant set-to-label body
        pass

    def forward(self, x):
        # x: (batch, M, in_dim)
        # TODO: produce an invariant output
        raise NotImplementedError


class SetLayer(nn.Module):
    """A single layer mapping (batch, M, in_dim) -> (batch, M, out_dim) that
    should commute with reordering the M elements. What linear maps are allowed
    here is itself unknown and must be characterized."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # TODO: the constrained linear map(s) permitted under permutation symmetry
        pass

    def forward(self, x):
        # TODO: apply the constrained map; stay order-respecting
        raise NotImplementedError


class SetClassifier(nn.Module):
    """A set-to-label classifier respecting permutation symmetry."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # TODO: build the invariant classifier body
        pass

    def forward(self, x):
        # x: (batch, M, in_dim)
        raise NotImplementedError


def train(model, loader, opt, loss_fn):
    for x, y in loader:                 # x: (batch, M, in_dim)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
```
