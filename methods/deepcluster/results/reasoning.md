OK, let me think about what I actually need. I want general-purpose visual features — the kind of convnet representation that transfers to detection, segmentation, retrieval — but learned with *no labels at all*, and learned at a scale that could eventually mean billions of images. The reason this matters now: supervised ImageNet pre-training is running out of room. The error left on ImageNet is small, performance is saturating across new architectures, and ImageNet itself is only a million images curated for object classification. If I want better features, the obvious move is a much bigger, more diverse dataset — but labeling billions of images by hand is hopeless, and if I substitute raw metadata like hashtags for labels I import all the biases of that metadata into the representation. So I need to learn from the *images alone*.

Let me survey the unsupervised toolbox and find where each piece breaks, because the cracks will tell me what to build.

Classical unsupervised learning — clustering, dimensionality reduction, density estimation — is appealing because it needs no labels and works on any domain (satellite, medical, depth). The bag-of-features model, for instance, clusters handcrafted descriptors into image-level features. But every one of these was designed for a *linear* model sitting on top of *fixed* features. The moment I want the features themselves to be learned by the convnet *at the same time* as I cluster, the whole thing falls apart. Concretely: suppose I try to train a convnet jointly with k-means — push images through `f_θ`, cluster the outputs, and use "be close to your centroid" as the training signal. What does the optimizer discover? That it can drive the features to *zero* and collapse every cluster into a single point. The loss is trivially minimized and the features are useless. So naive convnet-plus-clustering is degenerate. That degeneracy is the wall I have to get around; everything below is in service of it.

What about self-supervised learning? It's the popular alternative: invent a *pretext* task whose labels can be computed from the raw data — predict the relative position of two image patches, solve a jigsaw of shuffled patches, inpaint a missing region, predict color from grayscale, exploit temporal coherence in video. These give a genuine supervisory signal and good transfer features. But notice what they all share: each is a *hand-designed* task, domain-specific, requiring expert intuition about which artificial task will happen to force the network to learn transferable structure. I'd rather not bet the representation on cleverly guessing a pretext. I want something that needs almost no domain knowledge.

Generative models are another route — a VAE or GAN learns a mapping from noise to images, and the GAN discriminator (or an added encoder) yields features. But the features are a side effect, not the objective, and plain discriminator features have been disappointing. So that's indirect.

And the few prior attempts to marry clustering with convnets each stop short: one pre-trains with k-means but layer by layer, bottom-up, not end-to-end; another learns features and clusters jointly in a recurrent scheme that's promising on small data but hard to scale to the image counts convnets need; another uses an information-preserving loss that discriminates between individual images. None had been pushed to the scale and the modern architectures where I could actually study whether clustering works.

Now let me look for the thread to pull. Here is the observation that reframes everything: a convnet with *random* weights — sampled from a Gaussian, never trained — already produces features well above chance. Put an MLP on the last convolutional layer of a random AlexNet and it reaches around 12% on ImageNet, against 0.1% chance. That's not nothing; that's a hundredfold over chance. Where does it come from? From the convolutional architecture itself — weight sharing, locality, hierarchy — which is a strong prior on natural images even before any learning. So there is *already* a weak discriminative signal sitting in the untrained network. What if I could bootstrap it — use that weak signal to produce a rough grouping of images, then train the network to reproduce that grouping, which sharpens the features, which gives a better grouping, and iterate?

That's the whole idea, and it lets me reuse the machinery I trust. The supervised convnet pipeline is

  min_{θ,W} (1/N) Σ_n ℓ( g_W(f_θ(x_n)), y_n ),

with `ℓ` the multinomial logistic loss, trained by mini-batch SGD and backprop, dressed up with batch norm, dropout, momentum, augmentation. The only thing it's missing in my setting is the labels `y_n`. So *manufacture* the labels by clustering, and then run exactly this pipeline. Alternate two steps: cluster the current features to get pseudo-labels, then train the network to predict those pseudo-labels. Each round, better features yield a better clustering, and a better clustering trains better features.

For the clustering step I'll use k-means — it's standard, it scales (there are billion-scale implementations), and preliminary checks suggest the exact algorithm isn't crucial (power-iteration clustering works too). k-means jointly learns a `d×k` centroid matrix `C` and a one-hot assignment `y_n` for each image:

  min_{C ∈ ℝ^{d×k}} (1/N) Σ_n min_{y_n ∈ {0,1}^k} ‖ f_θ(x_n) − C y_n ‖₂²,  subject to  y_n^⊤ 1_k = 1.

It returns optimal assignments `y_n*` and centroids `C*`. I'll keep *only the assignments* as pseudo-labels and throw the centroids away — my goal is the features, and the classifier `g_W` will relearn a decision boundary from the pseudo-labels each round; the centroids are scaffolding I don't need to carry.

But I haven't solved the degeneracy yet — I've only named it. Alternating between learning a classifier and learning the labels it predicts is *exactly* the setup that's prone to trivial solutions; discriminative clustering hits this even with linear models. The classical fixes constrain or penalize the minimum number of points per cluster, but those terms are computed over the whole dataset, which I can't do inside mini-batch SGD on a million-plus images. I need scalable, local workarounds. Let me trace the two distinct ways the thing can collapse and patch each.

The first collapse is on the *k-means* side: empty clusters. A discriminative objective is happiest assigning everything to one cluster — an optimal decision boundary can simply lump all inputs together, leaving the other clusters empty, and nothing in plain k-means prevents that. If clusters keep emptying out, the pseudo-labels carry no information. The fix is local and cheap, borrowed from large-scale feature quantization: during the k-means optimization, whenever a cluster goes empty, pick a non-empty cluster at random, copy its centroid with a small random perturbation to serve as the empty cluster's new centroid, and split the points of that non-empty cluster between the two. This keeps all `k` clusters alive without any global penalty term.

The second collapse is on the *convnet* side: trivial parametrization. Even with all clusters populated, suppose the vast majority of images land in a handful of clusters. Then when I train the network to predict pseudo-labels, `θ` only ever needs to discriminate those few dominant clusters; in the worst case, with all but one cluster reduced to singletons, minimizing the classification loss drives the convnet to predict the same output regardless of the input — useless features again. This is the same pathology supervised classification has under severe class imbalance, and imbalance is the default for unsupervised pseudo-labels (cluster sizes are wildly uneven). The fix: don't sample training images uniformly over *images*; sample uniformly over *clusters*. Equivalently, weight each image's contribution to the loss by the inverse of the size of its assigned cluster. Now a giant cluster doesn't get to dominate the gradient, and the network is forced to discriminate the small clusters too, so it can't collapse to a constant.

There's a subtlety the alternation forces that I should pin down before writing code. Each round I re-run k-means from scratch on the new features. The cluster *indices* are arbitrary — cluster "5" this round has no relationship to cluster "5" last round; the labeling is a fresh, permuted partition. So the final classification layer `g_W` — the linear head that maps features to one logit per pseudo-class — learned a mapping to *this round's* arbitrary indices, which are meaningless next round. If I kept that head, I'd be fighting a relabeling every round. So at the start of each round I *reinitialize the top classification layer* (random Gaussian weights, zero bias) to match the new, arbitrary cluster identities, while keeping the convnet backbone, which is the thing I actually want to improve and which carries over meaningfully. The backbone persists; the head is reborn each round.

A few more choices the setup demands. Unsupervised feature learning is notorious for latching onto *color* as a cheap shortcut — color is low-level and too easy, and it doesn't force the shape and texture structure I want. So I strip color out up front with a fixed Sobel filtering, which removes color and boosts local contrast, leaving the network to learn from edges and structure. Before clustering, I PCA-reduce the features to 256 dimensions, whiten them, and L2-normalize them — whitening decorrelates and equalizes the feature scales so that the Euclidean geometry k-means assumes is meaningful, and the reduction makes the clustering fast. I cluster the features of *central-cropped* images (a stable view) but train the network with data augmentation — random flips and random-size, random-aspect crops — which makes the features invariant to those augmentations, a property I want. And I re-cluster every epoch: the features drift as training proceeds, so the pseudo-labels need to be refreshed; on ImageNet, re-clustering once per epoch turns out to be nearly optimal, and the clustering itself costs about a third of the time mostly because it needs a forward pass over the whole dataset.

The last knob is the number of clusters `k`. I'm training on ImageNet, which has 1000 classes, so the reflexive choice is `k = 1000`. But there's no reason the pseudo-classes should match the true classes — and intuitively, *over-segmenting* (more clusters than true classes) gives the network finer distinctions to learn, which should produce richer features. So I'd expect a larger `k` to help, and I'll use `k = 10000`: ten times the number of true classes, betting that the extra granularity in the pseudo-labels forces a more discriminative representation.

Let me write the pieces. The feature preprocessing before k-means — PCA-whiten to 256 dims, then L2-normalize:

```python
import numpy as np
import faiss
import torch
from torch import nn

def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)   # eigen_power=-0.5 => whitening
    mat.train(npdata); npdata = mat.apply_py(npdata)
    npdata = npdata / np.linalg.norm(npdata, axis=1)[:, None]   # L2 normalize each row
    return npdata
```

The clustering returns one pseudo-label per image; empty clusters are reassigned during the k-means solve so the partition never degenerates:

```python
def run_kmeans(x, k, n_iter=20):
    n, d = x.shape
    clus = faiss.Clustering(d, k)
    clus.niter = n_iter
    clus.max_points_per_centroid = 10_000_000           # don't subsample; empty clusters handled by faiss
    index = faiss.IndexFlatL2(d)
    clus.train(x, index)                                # solves the k-means objective; revives empty clusters
    _, assignments = index.search(x, 1)                 # nearest centroid = pseudo-label
    return assignments.reshape(-1)
```

The sampler that prevents convnet collapse — draw images so that clusters are represented uniformly, i.e. inverse-size weighting:

```python
class UnifLabelSampler(torch.utils.data.Sampler):
    def __init__(self, N, images_lists):                # images_lists[c] = indices in cluster c
        self.N = N
        self.images_lists = images_lists
    def __iter__(self):
        nonempty = [c for c in self.images_lists if len(c) > 0]
        per = int(self.N / len(nonempty)) + 1           # equal budget per cluster -> uniform over clusters
        res = np.concatenate([
            np.random.choice(c, per, replace=(len(c) <= per)) for c in nonempty])
        np.random.shuffle(res)
        return iter(res[: self.N].astype(int))
    def __len__(self):
        return self.N
```

And the round: compute features over the whole dataset, cluster to pseudo-labels, *reinitialize the top layer* because cluster identities are arbitrary this round, then train the supervised objective with the uniform sampler:

```python
def deepcluster_round(model, dataset, k, optimizer, optimizer_tl, fd):
    feats = compute_features(model, dataset)                 # forward pass over all images
    feats = preprocess_features(feats)
    assignments = run_kmeans(feats, k)                       # pseudo-labels
    images_lists = [[] for _ in range(k)]
    for i, a in enumerate(assignments):
        images_lists[a].append(i)

    # cluster indices are arbitrary every round -> rebuild the classification head
    model.top_layer = nn.Linear(fd, k)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()

    train_dataset = cluster_assign(images_lists, dataset.imgs)   # attach pseudo-labels (with augmentation)
    sampler = UnifLabelSampler(len(train_dataset), images_lists) # uniform over clusters
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=sampler)

    crit = nn.CrossEntropyLoss()                             # multinomial logistic loss
    for x, pseudo_y in loader:
        out = model(x)
        loss = crit(out, pseudo_y)
        optimizer.zero_grad(); optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step(); optimizer_tl.step()                # update backbone and new head
```

The convnet itself is a standard AlexNet (five conv layers with 96, 256, 384, 384, 256 filters and three fully-connected layers), with the local response normalization removed and batch norm added; the input is the Sobel-filtered image. Outer loop: run `deepcluster_round` each epoch for a few hundred epochs, with momentum 0.9, L2 weight decay, dropout, and a constant step size.

So the causal chain, end to end: I want label-free, scalable visual features, but classical clustering assumes fixed features and naive convnet-plus-k-means collapses to zero features in one cluster. The opening is that a random convnet already encodes a weak signal (≈12% on ImageNet vs 0.1% chance) from its convolutional prior — so I bootstrap it by alternating between clustering the features into pseudo-labels (k-means, keeping only the assignments) and training the convnet to predict those pseudo-labels with the ordinary supervised pipeline. Because jointly learning a classifier and its labels is degenerate, I patch the two collapse modes locally and scalably: revive empty clusters by splitting a perturbed non-empty centroid, and sample uniformly over clusters (inverse-size weighting) so the network can't collapse onto a few dominant clusters. Since each re-clustering produces an arbitrary permutation of cluster indices, I reinitialize the classification head every round while carrying the backbone forward, strip color with Sobel so the features don't shortcut on it, PCA-whiten and L2-normalize before k-means so its Euclidean geometry is meaningful, re-cluster every epoch as the features drift, and over-segment with `k = 10000` so the finer pseudo-classes force a richer representation.
