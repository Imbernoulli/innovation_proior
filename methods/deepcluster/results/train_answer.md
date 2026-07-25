Training a convolutional network that generalizes to classification, detection, and retrieval usually starts from a supervised pretraining dataset such as ImageNet, but those labels are expensive, curated for object classes, and do not scale to the billions of uncurated images available on the internet. Self-supervised pretext tasks such as patch prediction, jigsaw puzzles, or colorization remove the annotation cost, yet each task is hand-designed for a specific visual domain and may not transfer to the broad visual cues that downstream tasks actually need. Generative models such as autoencoders or GANs learn an image distribution, but their internal representations are a side effect of reconstruction or generation, not the direct target, and they tend to underperform the best discriminative features. The real goal is therefore a label-free procedure that still uses the standard discriminative convnet training machinery and can run at ImageNet scale.

A natural next idea is to train the convnet and a clustering model jointly, letting k-means discover groups while the network learns to represent them. That idea is unstable because backpropagating through a clustering reconstruction loss invites collapse: the network can push every image to the same point and still minimize the distance to its centroid. Whole-dataset balance constraints can prevent this in principle, but they are awkward to enforce inside a minibatch SGD loop over millions of images. What is needed is a way to reuse the reliable supervised pipeline while stopping the two failure modes of degenerate clusters and degenerate representations.

The method is DeepCluster. It treats clustering as a pseudo-label generator and the convnet as a pseudo-label classifier, then alternates the two steps. In each round the current network is frozen, features are extracted for the whole dataset, and k-means partitions those features into a fixed number of clusters. The resulting cluster index for each image is used as a one-hot pseudo-label, and the network is trained for a few epochs with ordinary multinomial logistic regression, just as if the cluster indices were human class labels. The convnet objective is therefore exactly the standard supervised loss, with the only difference that the targets come from the data itself rather than manual annotation. Because the classifier is trained on the current pseudo-labels, the representation sharpens; because the next clustering pass sees the sharper representation, the pseudo-labels improve. The loop bootstraps from the weak prior already present in a random convnet.

Several design choices make this loop stable. Features are preprocessed before clustering by reducing them to 256 dimensions with PCA, whitening them with eigenvalue power negative one half, and L2-normalizing each vector. This removes arbitrary scaling and makes Euclidean k-means act more like a meaningful angular partition. For the clustering target each image is represented by a resized center crop, while the classification training uses random resized crops and horizontal flips so the network learns to predict the stable target from augmented views. To remove the easy shortcut of grouping by raw color, inputs are converted to grayscale and passed through fixed Sobel filters before being fed to the network, so the representation must build on edges and local contrast rather than global color.

The most important safeguards are against collapse. Empty clusters are revived inside the k-means routine by perturbing a non-empty centroid and splitting its points, which keeps the full set of pseudo-labels alive. Since k-means assignments can be extremely imbalanced, training is performed with a sampler that draws roughly the same number of examples from every non-empty cluster each epoch, which is equivalent to weighting each image by the inverse size of its cluster. This prevents the gradient from being dominated by a few massive clusters and stops the feature extractor from drifting toward an input-independent output. Finally, cluster indices have no stable identity across reclustering, so the final linear classification layer is discarded and reinitialized every round while the backbone and its lower classifier are retained. The centroids themselves are also discarded after each clustering step; they are only a temporary naming device for the current partition.

The implementation uses FAISS for fast GPU k-means with ten thousand clusters and twenty iterations, batch size 256, momentum SGD with weight decay one times ten to the negative five, and a constant learning rate. The backbone optimizer trains all convnet parameters, while the freshly created top layer receives its own SGD optimizer. I run five hundred epochs on AlexNet, though something closer to two hundred epochs already works; either way the reassignment loop repeats every epoch.

```python
def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype("float32")
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    npdata = mat.apply_py(npdata)
    return npdata / np.linalg.norm(npdata, axis=1)[:, None]


def run_kmeans(x, k, verbose=False):
    n_data, d = x.shape
    clus = faiss.Clustering(d, k)
    clus.seed = np.random.randint(1234)
    clus.niter = 20
    clus.max_points_per_centroid = 10_000_000
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    return [int(n[0]) for n in I], losses[-1]


class UnifLabelSampler(torch.utils.data.Sampler):
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        non_empty = [xs for xs in self.images_lists if len(xs) != 0]
        per_label = int(self.N / len(non_empty)) + 1
        res = np.array([])
        for xs in non_empty:
            sample = np.random.choice(xs, per_label, replace=(len(xs) <= per_label))
            res = np.concatenate((res, sample))
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
