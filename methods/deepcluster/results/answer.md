# DeepCluster

## Problem

Train a convnet end-to-end on unlabeled images so its features transfer to downstream visual tasks. A direct joint convnet+k-means objective is degenerate: the representation can collapse and the clusters can become uninformative.

## Method

Alternate between clustering the current features and training the network to predict the resulting cluster assignments.

Supervised template:

`min_{theta,W} (1/N) sum_n ell(g_W(f_theta(x_n)), y_n)`

where `ell` is multinomial logistic loss.

k-means assignment step:

`min_{C in R^{d x k}} (1/N) sum_n min_{y_n in {0,1}^k} ||f_theta(x_n) - C y_n||_2^2`

subject to `y_n^T 1_k = 1`.

Use the optimal assignments `y_n*` as pseudo-labels and discard the centroids `C*`.

## Anti-Collapse Details

- Empty clusters: during k-means, revive an empty cluster by perturbing a non-empty centroid and splitting that non-empty cluster's points.
- Trivial parametrization: train with a sampler that is uniform over non-empty pseudo-labels, equivalent to inverse-cluster-size weighting.
- Cluster-index permutation: reset the final classification layer at every reassignment; keep the backbone.

## Canonical Implementation Shape

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

Reference training loop:

1. Remove `top_layer` and the last classifier ReLU; compute full-dataset features from resized center crops.
2. PCA-whiten to 256 dimensions and L2-normalize.
3. Run FAISS k-means with `k=10000`, 20 iterations, and changing random seed.
4. Build a reassigned dataset with random resized crop and horizontal flip.
5. Use `UnifLabelSampler` for uniform sampling over pseudo-labels.
6. Re-append the classifier ReLU, create a fresh `Linear(fd, k)` top layer, initialize weights with `Normal(0, 0.01)` and zero bias.
7. Train with cross-entropy; backbone optimizer uses momentum 0.9 and weight decay `10^-5`, while the fresh top layer has its own SGD optimizer.

The ECCV paper reports AlexNet training for 500 epochs with batch size 256, Sobel-filtered input, central-crop clustering, augmented training crops, and reclustering every epoch. The released repository defaults to 200 epochs but implements the same reassignment loop.
