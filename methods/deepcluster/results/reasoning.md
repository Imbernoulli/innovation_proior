I want a convnet feature extractor without labels, but I do not want to throw away the supervised training machinery that already works. The supervised pipeline is clean: an image goes through `f_theta`, a classifier `g_W` maps the feature to logits, and I minimize a negative log-softmax against a one-hot label. Batch norm, dropout, augmentation, momentum SGD, weight decay, all of that already exists. The missing object is only the label.

The first temptation is to make the representation itself satisfy a clustering objective. Take the current features, learn centroids, and move features toward their centroid. But if `theta` is free, that objective has a stupid solution: set every feature to the same point, even zero, and the geometry collapses. That is not a training signal; it is an invitation to destroy the representation. So I should not backpropagate through a k-means reconstruction objective as if k-means were the final loss.

There is still a useful clue. A random convnet is bad, but it is not random in the statistical sense. Its convolutional structure already gives a weak visual prior; a classifier over a random AlexNet's last convolutional layer gets about 12% ImageNet accuracy, far above 0.1% chance. That means the starting feature space has some structure. If I can read a rough partition out of that structure, then train the network to predict the partition, the feature space may sharpen; a sharper feature space gives a better partition; then I repeat. I do not need the initial partition to be semantically correct. I need it to be consistent enough to bootstrap.

So I will split the process into two phases and only backpropagate through the supervised phase. First, freeze the current network long enough to compute features for all images and cluster those features. Second, treat the resulting cluster index for each image as a pseudo-label and train the convnet with the ordinary classification loss. The supervised objective is

`min_{theta,W} (1/N) sum_n ell(g_W(f_theta(x_n)), y_n)`,

where `ell` is the multinomial logistic loss. The only change is that `y_n` now comes from a clustering step rather than a human annotation.

For the clustering step I can use k-means because it is simple and scalable. With `k` clusters and a `d x k` centroid matrix `C`, the assignment problem is

`min_{C in R^{d x k}} (1/N) sum_n min_{y_n in {0,1}^k} ||f_theta(x_n) - C y_n||_2^2`

subject to `y_n^T 1_k = 1`. The shape matters: each column of `C` is a centroid, `y_n` is one-hot, and `C y_n` selects the assigned centroid. Once k-means returns `C*` and the assignments `y_n*`, I keep the assignments and discard the centroids. The centroids are useful only to name the current groups; the object I am trying to improve is the convnet feature map, not a stored codebook.

This gives me the main loop: cluster features into pseudo-labels, train the convnet to classify those pseudo-labels, then cluster again. But I have not solved collapse yet. I have merely moved it into two concrete places where I can patch it.

The first collapse is in the clustering step. If a cluster becomes empty, then the pseudo-label set loses capacity, and in the extreme everything can fall into one cluster. Discriminative clustering has exactly this pathology: if the method is allowed to choose labels and a boundary together, a one-cluster assignment can be optimal unless balance constraints intervene. Whole-dataset convex balance constraints are not something I want inside minibatch convnet training. A practical k-means fix is enough here: when a cluster is empty, choose a non-empty cluster, create the empty centroid as a small perturbation of that non-empty centroid, and split the old cluster's points between the two. That keeps all clusters alive during the k-means solve without adding a global penalty to the neural training loss.

The second collapse is in the convnet training step. Suppose k-means technically returns many non-empty clusters, but most images are in a few huge ones. If I sample images uniformly, the gradient mostly sees those huge clusters. The classifier can learn to discriminate only the dominant groups, and in the worst case the feature extractor can drift toward an almost input-independent output. This is just severe class imbalance, except the classes are pseudo-classes. The fix is to sample uniformly over pseudo-labels: give each non-empty cluster roughly the same number of training examples per epoch, sampling with replacement for small clusters. In loss terms, this is equivalent to weighting an image by the inverse of the size of its assigned cluster.

There is another subtlety. Cluster indices have no identity across reclustering. Cluster 17 this epoch is not "the same class" as cluster 17 next epoch. If I keep the final classifier head across reassignments, its weights are tied to an arbitrary old permutation of labels. The backbone should persist, because it carries visual features; the top classification layer should not. So each time I make a fresh set of pseudo-labels, I throw away and reinitialize the final linear layer. The official code is even more precise: before extracting features it removes `top_layer` and drops the last classifier ReLU, then after clustering it appends the ReLU back, creates a new `Linear(fd, k)`, initializes weights from `Normal(0, 0.01)`, zeroes the bias, and trains that layer with its own optimizer.

Now I need to make k-means see a feature geometry that is not dominated by arbitrary feature scales. Before clustering I reduce the features to 256 dimensions with PCA, whiten them by using eigenvalue power `-0.5`, and L2-normalize each row. Then Euclidean k-means is closer to a meaningful angular/geometric partition, and the clustering pass is much cheaper.

I also need to decide which image view defines the target and which views train the invariance. For the clustering pass, I want a stable descriptor, so I use a resized center crop. For the classification pass, I want invariance, so I train with random resized crops and horizontal flips. That means the target is computed from one canonical view, while the convnet learns to predict it from augmented views. Cropping is not decoration here; it is part of what makes the representation useful.

Color is a cheap shortcut. If unsupervised clusters can be balanced by color alone, the model may learn attractive but low-level partitions. I will remove much of that shortcut with a fixed Sobel transform: grayscale the input with a `1/3` channel average, then apply horizontal and vertical derivative filters. Those filters are fixed, not learned. The network now has to build on edge and local-contrast structure rather than raw RGB color.

For the number of clusters, I should not force the pseudo-labels to mimic the 1000 ImageNet classes. Finer groups can ask the convnet to discriminate more visual detail, so I choose over-segmentation, `k = 10000`. The paper-level training recipe uses 256-image minibatches, momentum 0.9, weight decay `10^-5`, a constant step size, and repeated full-dataset reclustering; the reported AlexNet training runs for 500 epochs. The released code defaults to 200 epochs, but the algorithmic loop is the same.

Putting the faithful code skeleton together, the core pieces look like this:

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

And the epoch choreography is the part I cannot skip:

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

The whole construction is therefore a bootstrapping loop. A random convnet gives weak structure; k-means turns the current structure into pseudo-labels; the supervised loss makes the convnet predict those labels under augmentation; empty-cluster reassignment and uniform pseudo-label sampling keep the two collapse modes from taking over; and the top layer is reset each round because the labels are only temporary names for the current partition. The centroids disappear after assignment, but the backbone survives and gradually becomes the useful representation.
