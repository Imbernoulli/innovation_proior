I want a convnet feature extractor without labels, but I do not want to throw away the supervised training machinery that already works. The supervised pipeline is clean: an image goes through `f_theta`, a classifier `g_W` maps the feature to logits, and I minimize a negative log-softmax against a one-hot label. Batch norm, dropout, augmentation, momentum SGD, weight decay, all of that already exists. The missing object is only the label.

The first temptation is to make the representation itself satisfy a clustering objective: take the current features, learn centroids `C`, and minimize the joint cost over both the features and the centroids,

`min_{theta, C} (1/N) sum_n min_{y_n} ||f_theta(x_n) - C y_n||_2^2`.

Before I commit to this I want to know what its minimizer actually is, because the worry is that with `theta` free there is a cheap way to drive the cost to zero. Let me just check the value at the suspicious point. Pick any vector `v` and suppose `theta` maps every image to `v`, with a single centroid sitting at `v`. Then every assignment distance is `||v - v||^2 = 0`, so the total cost is exactly 0. I ran this against a representation that genuinely has structure — two well-separated Gaussian blobs in 8 dimensions, clustered with their true means:

```
collapsed joint cost: 0.0
structured joint cost (2 true clusters): 364.45
```

So the collapsed constant map is not just *a* low-cost solution, it is strictly better under this objective than a representation that actually separates the data. A free `theta` is pulled toward it. That settles it: I should not backpropagate through a k-means reconstruction objective as if k-means were the final loss. Whatever I build, the gradient that updates `theta` must not be the clustering cost itself.

There is still a useful clue in the other direction. A random convnet is bad, but it is not random in the statistical sense. A classifier over a random AlexNet's last convolutional layer gets about 12% ImageNet accuracy. Against the 0.1% chance of a 1000-way problem that is 120x above chance, so the convolutional structure alone already carries a real, if weak, signal about the images. If I can read a rough partition out of that structure, then train the network to predict the partition, the feature space may sharpen; a sharper feature space gives a better partition; then I repeat. The initial partition does not have to be semantically correct for this to start moving.

That suggests splitting the process into two phases and only backpropagating through the supervised phase. First, freeze the current network long enough to compute features for all images and cluster those features. Second, treat the resulting cluster index for each image as a pseudo-label and train the convnet with the ordinary classification loss. The supervised objective is

`min_{theta,W} (1/N) sum_n ell(g_W(f_theta(x_n)), y_n)`,

where `ell` is the multinomial logistic loss. The only change from a fully supervised run is that `y_n` now comes from a clustering step rather than a human annotation. Crucially, the clustering cost never enters the backward pass — it only manufactures `y_n` between phases — which is exactly what the collapse check above told me to avoid.

For the clustering step I can use k-means because it is simple and scalable. With `k` clusters and a `d x k` centroid matrix `C`, the assignment problem is

`min_{C in R^{d x k}} (1/N) sum_n min_{y_n in {0,1}^k} ||f_theta(x_n) - C y_n||_2^2`

subject to `y_n^T 1_k = 1`. The shape matters: each column of `C` is a centroid, `y_n` is one-hot, and `C y_n` selects the assigned centroid. Once k-means returns `C*` and the assignments `y_n*`, I keep the assignments and discard the centroids. The centroids are useful only to name the current groups; the object I am trying to improve is the convnet feature map, not a stored codebook.

This gives me the main loop: cluster features into pseudo-labels, train the convnet to classify those pseudo-labels, then cluster again. But notice I have not actually escaped collapse — I have only moved the freedom of `theta` out of the clustering gradient. `theta` is still updated, now by the classification loss, and there are still degenerate configurations the loop could drift into. Let me find them concretely instead of trusting that the two-phase split is enough.

The first one lives in the clustering step. If a cluster becomes empty, the pseudo-label set loses capacity, and nothing stops the process from emptying more clusters on the next pass; in the extreme everything falls into one cluster, the classification loss can be driven to zero by a constant prediction, and `theta` is free to collapse again. Discriminative clustering has exactly this pathology: if the method chooses labels and a boundary together, a one-cluster assignment can be optimal unless a balance constraint intervenes. Whole-dataset convex balance constraints are not something I want inside minibatch convnet training. A local k-means fix is cheaper: when a cluster is empty, choose a non-empty cluster, create the empty centroid as a small perturbation of that non-empty centroid, and split the old cluster's points between the two. That keeps all `k` slots alive during the k-means solve without adding any global penalty to the neural training loss.

The second degeneracy lives in the convnet training step, and it survives even when every cluster is non-empty. Suppose k-means returns `k` non-empty clusters but the sizes are wildly uneven — most images in a few huge clusters, the rest in tiny ones. If I sample images uniformly, the gradient is dominated by the huge clusters, the classifier learns to discriminate only those, and in the worst case `theta` drifts toward an almost input-independent output again. This is ordinary class imbalance, except the classes are pseudo-classes. The natural fix is to sample uniformly over pseudo-labels rather than over images: draw roughly the same number of training examples from each non-empty cluster per epoch, with replacement for the small clusters.

I want to be careful that "uniform over labels" really does what I think to the per-image gradient weight, because the clusters are so unequal that intuition is shaky here. So let me trace the sampler logic on a deliberately lopsided example: clusters of size 100, 10, and 2, one empty cluster, and a target of `N = 300` draws for the epoch. The sampler computes `per_label = floor(N / non_empty) + 1 = floor(300/3)+1 = 101`, draws `per_label` indices from each non-empty cluster (with replacement when the cluster is smaller than `per_label`), concatenates, shuffles, and truncates to `N`. Running it:

```
non-empty clusters: 3
per_label (target draws per cluster): 101
draws per cluster over the epoch: {0: 99, 1: 100, 2: 101, 3: 0}
cluster 0: size 100, expected draws/image = per_label/size = 1.0100
cluster 1: size 10, expected draws/image = per_label/size = 10.1000
cluster 2: size 2, expected draws/image = per_label/size = 50.5000
```

Two things to read off. The draws per cluster come out 99 / 100 / 101 even though the clusters differ by 50x in size, so the gradient now sees the three pseudo-classes about equally — that is the balance I wanted. And the per-image draw rate is `per_label / size`: roughly 1, 10, 50 for the three clusters. Each image therefore enters the loss in inverse proportion to the size of its assigned cluster. So "sample uniformly over pseudo-labels" is, at the level of the gradient, the same as weighting each image by the inverse of its cluster size — which is exactly the correction class imbalance asks for. The empty cluster contributes nothing, as it should.

There is another subtlety, and this one is about identity rather than collapse. Cluster indices have no meaning across reclustering. Cluster 17 this epoch is not "the same class" as cluster 17 next epoch — the k-means seed changes and the labels are an arbitrary permutation. If I keep the final classifier head across reassignments, its weights are tied to last epoch's permutation and become noise the moment the labels are reshuffled. The backbone should persist, because it carries visual features that are permutation-independent; the top classification layer should not. So each time I make a fresh set of pseudo-labels, I throw away and reinitialize the final linear layer. Concretely: before extracting features I remove `top_layer` and drop the last classifier ReLU, then after clustering I append the ReLU back, create a new `Linear(fd, k)`, initialize weights from `Normal(0, 0.01)`, zero the bias, and train that layer with its own optimizer.

Now I need k-means to see a feature geometry that is not dominated by arbitrary per-axis scales, since the raw convnet features have no reason to be isotropic and Euclidean k-means is scale-sensitive. The plan is to reduce to 256 dimensions with PCA and whiten. The `faiss.PCAMatrix` exposes an `eigen_power` argument, and I want to be sure `eigen_power = -0.5` is the whitening I think it is rather than some other power. Whitening should divide each PCA coordinate by `sqrt(lambda_i)`, i.e. scale axis `i` by `lambda_i^{-1/2}`, leaving unit variance on every axis. I checked this on anisotropic data (variances spanning 10:0.3 across five axes) by projecting onto the PCA basis and scaling by `lambda^{-1/2}`:

```
variance per axis after eigen_power=-0.5 whitening: [0.999 0.999 1. 1. 0.999]
```

so `-0.5` does flatten the spectrum to unit variance per axis. Good — after this I L2-normalize each row, and then Euclidean k-means is operating on a roughly angular geometry where no single high-variance direction dominates the partition, and the clustering pass is also much cheaper in 256 dimensions.

I also need to decide which image view defines the target and which views train the invariance. For the clustering pass I want a stable, deterministic descriptor, so I use a resized center crop. For the classification pass I want invariance, so I train with random resized crops and horizontal flips. That means the target is computed from one canonical view, while the convnet learns to predict it from augmented views — which is what should push the features to be augmentation-invariant. Cropping is not decoration here; it is part of what makes the representation useful.

Color is a cheap shortcut. If unsupervised clusters can be balanced by color alone, the model may learn attractive but low-level partitions instead of structure that transfers. I will remove much of that shortcut with a fixed Sobel transform: grayscale the input with a `1/3` channel average, then apply horizontal and vertical derivative filters. Those filters are fixed, not learned. The network then has to build on edge and local-contrast structure rather than raw RGB color.

For the number of clusters, there is no reason to force the pseudo-labels to mimic the 1000 ImageNet classes — the labels are internal scaffolding, not a target taxonomy. Finer groups ask the convnet to discriminate more visual detail, so I over-segment, `k = 10000`. The training recipe uses 256-image minibatches, momentum 0.9, weight decay `10^-5`, a constant step size, and repeated full-dataset reclustering; I run AlexNet training for 500 epochs, though something closer to 200 epochs already produces useful features, and the algorithmic loop is identical either way.

Putting the code skeleton together, the core pieces look like this:

```python
import numpy as np
import faiss
import torch
from torch import nn
import torchvision.transforms as transforms


def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype("float32")
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    npdata = mat.apply_py(npdata)
    row_sums = np.linalg.norm(npdata, axis=1)
    return npdata / row_sums[:, np.newaxis]


def run_kmeans(x, k, verbose=False):
    n_data, d = x.shape
    clus = faiss.Clustering(d, k)
    clus.seed = np.random.randint(1234)
    clus.niter = 20
    clus.max_points_per_centroid = 10_000_000
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16 = False
    config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, config)
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print("k-means loss evolution: {0}".format(losses))
    return [int(n[0]) for n in I], losses[-1]


class Kmeans:
    def __init__(self, k):
        self.k = k
        self.images_lists = None

    def cluster(self, features, verbose=False):
        xb = preprocess_features(features)
        assignments, loss = run_kmeans(xb, self.k, verbose=verbose)
        self.images_lists = [[] for _ in range(self.k)]
        for i, a in enumerate(assignments):
            self.images_lists[a].append(i)
        return loss


class UnifLabelSampler(torch.utils.data.Sampler):
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        non_empty = [c for c in self.images_lists if len(c) != 0]
        size_per = int(self.N / len(non_empty)) + 1
        res = np.array([])
        for images in non_empty:
            chosen = np.random.choice(images, size_per, replace=(len(images) <= size_per))
            res = np.concatenate((res, chosen))
        np.random.shuffle(res)
        res = list(res.astype("int"))
        if len(res) >= self.N:
            return res[:self.N]
        return res + res[: self.N - len(res)]

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
```

And the epoch choreography is the part I cannot skip, because it is where the top-layer reset and the feature-view bookkeeping actually happen:

```python
def one_epoch(model, dataset, feature_loader, deepcluster, optimizer, fd, args):
    # Feature extraction uses the backbone/classifier feature, not the old pseudo-label head.
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    features = compute_features(feature_loader, model, len(dataset))

    deepcluster.cluster(features, verbose=args.verbose)

    train_dataset = cluster_assign(deepcluster.images_lists, dataset.imgs)
    sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                               deepcluster.images_lists)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )

    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(fd, len(deepcluster.images_lists)).cuda()
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()

    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(), lr=args.lr, weight_decay=10 ** args.wd
    )
    train(train_loader, model, nn.CrossEntropyLoss().cuda(), optimizer, optimizer_tl)
```
