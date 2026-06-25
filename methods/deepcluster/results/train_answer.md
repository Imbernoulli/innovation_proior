Training a convolutional network that generalizes to classification, detection, and retrieval usually starts from a supervised pretraining dataset such as ImageNet, but those labels are expensive, curated for object classes, and do not scale to the billions of uncurated images available on the internet. Self-supervised pretext tasks such as patch prediction, jigsaw puzzles, or colorization remove the annotation cost, yet each task is hand-designed for a specific visual domain and may not transfer to the broad visual cues that downstream tasks actually need. Generative models such as autoencoders or GANs learn an image distribution, but their internal representations are a side effect of reconstruction or generation, not the direct target, and they tend to underperform the best discriminative features. The real goal is therefore a label-free procedure that still uses the standard discriminative convnet training machinery and can run at ImageNet scale.

A natural next idea is to train the convnet and a clustering model jointly, letting k-means discover groups while the network learns to represent them. That idea is unstable because backpropagating through a clustering reconstruction loss invites collapse: the network can push every image to the same point and still minimize the distance to its centroid. Whole-dataset balance constraints can prevent this in principle, but they are awkward to enforce inside a minibatch SGD loop over millions of images. What is needed is a way to reuse the reliable supervised pipeline while stopping the two failure modes of degenerate clusters and degenerate representations.

The method is DeepCluster. It treats clustering as a pseudo-label generator and the convnet as a pseudo-label classifier, then alternates the two steps. In each round the current network is frozen, features are extracted for the whole dataset, and k-means partitions those features into a fixed number of clusters. The resulting cluster index for each image is used as a one-hot pseudo-label, and the network is trained for a few epochs with ordinary multinomial logistic regression, just as if the cluster indices were human class labels. The convnet objective is therefore exactly the standard supervised loss, with the only difference that the targets come from the data itself rather than manual annotation. Because the classifier is trained on the current pseudo-labels, the representation sharpens; because the next clustering pass sees the sharper representation, the pseudo-labels improve. The loop bootstraps from the weak prior already present in a random convnet.

Several design choices make this loop stable. Features are preprocessed before clustering by reducing them to 256 dimensions with PCA, whitening them with eigenvalue power negative one half, and L2-normalizing each vector. This removes arbitrary scaling and makes Euclidean k-means act more like a meaningful angular partition. For the clustering target each image is represented by a resized center crop, while the classification training uses random resized crops and horizontal flips so the network learns to predict the stable target from augmented views. To remove the easy shortcut of grouping by raw color, inputs are converted to grayscale and passed through fixed Sobel filters before being fed to the network, so the representation must build on edges and local contrast rather than global color.

The most important safeguards are against collapse. Empty clusters are revived inside the k-means routine by perturbing a non-empty centroid and splitting its points, which keeps the full set of pseudo-labels alive. Since k-means assignments can be extremely imbalanced, training is performed with a sampler that draws roughly the same number of examples from every non-empty cluster each epoch, which is equivalent to weighting each image by the inverse size of its cluster. This prevents the gradient from being dominated by a few massive clusters and stops the feature extractor from drifting toward an input-independent output. Finally, cluster indices have no stable identity across reclustering, so the final linear classification layer is discarded and reinitialized every round while the backbone and its lower classifier are retained. The centroids themselves are also discarded after each clustering step; they are only a temporary naming device for the current partition.

The implementation uses FAISS for fast GPU k-means with ten thousand clusters and twenty iterations, batch size 256, momentum SGD with weight decay one times ten to the negative five, and a constant learning rate. The backbone optimizer trains all convnet parameters, while the freshly created top layer receives its own SGD optimizer. I run five hundred epochs on AlexNet, though something closer to two hundred epochs already works; either way the reassignment loop repeats every epoch.

```python
import numpy as np
import faiss
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        gray = torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("gray", gray)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        x = nn.functional.conv2d(x, self.gray, padding=0)
        dx = nn.functional.conv2d(x, self.sobel_x, padding=1)
        dy = nn.functional.conv2d(x, self.sobel_y, padding=1)
        return torch.cat([dx, dy], dim=1)


def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype("float32")
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    npdata = mat.apply_py(npdata)
    norm = np.linalg.norm(npdata, axis=1)
    return npdata / norm[:, np.newaxis]


def run_kmeans(x, k, niter=20, verbose=False):
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.seed = np.random.randint(1234)
    clus.niter = niter
    clus.max_points_per_centroid = 10_000_000
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print("k-means losses:", losses)
    return [int(n[0]) for n in I], losses[-1]


class DeepClusterKmeans:
    def __init__(self, k):
        self.k = k
        self.images_lists = []

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
        res = np.array([], dtype=np.int64)
        for images in non_empty:
            chosen = np.random.choice(images, size_per, replace=(len(images) <= size_per))
            res = np.concatenate((res, chosen))
        np.random.shuffle(res)
        res = list(res.astype("int"))
        if len(res) >= self.N:
            return res[: self.N]
        return res + res[: self.N - len(res)]

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


def compute_features(loader, model, N):
    model.eval()
    features = np.zeros((N, model.feature_dim), dtype="float32")
    for i, (x, _) in enumerate(loader):
        x = x.cuda(non_blocking=True)
        with torch.no_grad():
            z = model.features(x)
            z = z.view(z.size(0), -1)
        start = i * loader.batch_size
        end = min(start + x.size(0), N)
        features[start:end] = z.cpu().numpy()[: end - start]
    return features


def cluster_assign(images_lists, dataset_imgs):
    # dataset_imgs is assumed to be a list of (path, class_index) pairs.
    new_images = []
    for cluster_idx, images in enumerate(images_lists):
        for i in images:
            new_images.append((dataset_imgs[i][0], cluster_idx))
    return new_images


def deepcluster_epoch(model, train_dataset, feature_loader, k, fd, args):
    # Extract features using the backbone only.
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    features = compute_features(feature_loader, model, len(train_dataset))

    deepcluster = DeepClusterKmeans(k)
    deepcluster.cluster(features, verbose=args.verbose)

    train_dataset.imgs = cluster_assign(deepcluster.images_lists, train_dataset.imgs)
    sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)), deepcluster.images_lists)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True
    )

    # Reattach ReLU and a fresh top layer for the new pseudo-labels.
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(fd, k).cuda()
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()

    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(), lr=args.lr, weight_decay=10 ** args.wd
    )

    criterion = nn.CrossEntropyLoss().cuda()
    model.train()
    for x, y in train_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        output = model(x)
        loss = criterion(output, y)
        args.optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        args.optimizer.step()
        optimizer_tl.step()
```
