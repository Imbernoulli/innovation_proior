I propose applying Principal Component Analysis to the Olivetti faces dataset as a concrete, canonical instance of linear dimensionality reduction for high-correlation image data. The Olivetti collection gives 400 grayscale face images of 40 distinct subjects, each rasterized into a vector of pixel intensities, typically 4096 dimensions for 64-by-64 images. Every image is a point in this high-dimensional pixel space, but the coordinates are far from independent: eyebrows, eyes, noses, and chins move together in constrained ways because all the images depict human faces under similar lighting and pose. That strong correlation is exactly the situation PCA is designed to exploit. Rather than treating each pixel as its own isolated measurement, I want to find a small set of orthogonal directions along which the face cloud varies most, and then represent each face as a compact linear combination of those dominant directions.

The regression approach fails here for the same reason it fails in any correlated system. If I pick one pixel as a dependent variable and regress it on neighboring pixels, I get a prediction line that minimizes error along only that pixel's axis. Swap the roles and I get a different line. The truth is that no pixel is measured exactly while the others are noisy; every pixel intensity is part of one underlying face. I need a symmetric criterion that treats all pixel coordinates on equal footing and returns one unique lower-dimensional flat for the cloud. Factor analysis, which posits a fixed number of latent traits, is also unsatisfying because it imposes a hypothesis about the number of factors and suffers from rotational indeterminacy: the latent axes can be rotated without changing the model's fit, so they do not give a determinate, ordered decomposition of variance.

The right criterion is the sum of squared perpendicular distances from each face point to the fitting flat. Perpendicular distance is geometrically symmetric, so it does not privilege any pixel axis. This objective is the second moment of the point system about the flat, and a standard fact about second moments forces the best-fitting flat to pass through the centroid of the data. After subtracting the mean face from every image, the problem reduces to finding directions rather than locations. Introducing a Lagrange multiplier for the unit-length constraint on a direction vector and setting the derivative to zero produces an eigenvalue equation involving the covariance matrix of the centered face data. The eigenvectors are the principal directions, and the corresponding eigenvalues are the variances captured along those directions. Minimizing perpendicular residual is therefore identical to maximizing projected variance, because total scatter splits Pythagoreanly into the part along the flat plus the part perpendicular to it, and the total is fixed.

For Olivetti faces, the first principal component is typically a broad lighting or left-right intensity gradient that accounts for the largest share of pixel variance across the whole dataset. The second component captures the next largest orthogonal mode of variation, perhaps a lighting direction orthogonal to the first or a chin-to-forehead contrast. Each successive component explains as much residual variance as possible while remaining uncorrelated with all previous ones. The fraction of total variance retained by the first k components gives a natural stopping rule: if the top 50 components explain ninety percent of the variance, then a 50-dimensional PCA code is a faithful compressed representation of the original 4096-dimensional image. The components themselves can be visualized as eigenfaces: reshaping each principal direction back into the original image geometry reveals ghostly face-like patterns that, when added with appropriate weights, reconstruct any individual face.

Whether to use the covariance or the correlation matrix is a modeling choice about the metric. For Olivetti faces, all pixels share the same intensity units, so centering and diagonalizing the covariance matrix is the natural default. Standardizing each pixel to unit variance would treat a low-variance background pixel as equally important as a high-variance eye pixel, which is usually not what we want for images. I therefore center the data matrix by subtracting the mean face and then compute the principal directions from the centered covariance structure.

In practice I compute the directions via the singular value decomposition of the centered data matrix rather than by explicitly forming the covariance matrix and diagonalizing it. If the centered data matrix has shape n_samples by n_features, then forming its Gram matrix squares the condition number and can destroy accuracy on the small-variance directions. The SVD avoids that problem. Writing the centered matrix as X_c equals U S V transpose, the right singular vectors V are exactly the eigenvectors of X_c transpose X_c, so they are the principal axes, and S_i squared divided by n_samples minus one is the variance explained by component i. The embedding of the data onto the top k components is X_c V[:, :k], which equals the first k columns of U S. Singular vector signs are arbitrary, so I deterministically fix each right singular vector by forcing its largest-magnitude entry to be positive, making the result reproducible across runs.

```python
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def svd_flip(U: NDArray[np.float64], Vt: NDArray[np.float64]):
    """Deterministic singular-vector signs from the largest entry in each row of Vt."""
    max_abs = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs])
    return U * signs[np.newaxis, :], Vt * signs[:, np.newaxis]


class OlivettiFacePCA:
    """Principal Component Analysis for the Olivetti faces dataset.

    Center the image vectors, take the SVD of the centered matrix, and project
    onto the top-k right singular vectors (eigenfaces). The implementation uses
    the SVD of the centered data rather than eig(X_c.T @ X_c) for numerical
    stability.
    """

    def __init__(self, n_components: int = 50, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def _fit_full_svd(self, X: NDArray[np.float64]):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.mean_face_ = X.mean(axis=0)
        Xc = X - self.mean_face_
        U, S, Vt = linalg.svd(Xc, full_matrices=False)
        U, Vt = svd_flip(U, Vt)
        k = self.n_components
        self.components_ = Vt[:k]
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)
        total_var = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.singular_values_ = S[:k]
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        return U, S, Vt, Xc

    def fit(self, X: NDArray[np.float64]) -> "OlivettiFacePCA":
        self._fit_full_svd(X)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_face_
        return Xc @ self.components_.T

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        U, S, Vt, Xc = self._fit_full_svd(X)
        return U[:, : self.n_components] * S[: self.n_components]

    def reconstruct(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        codes = self.transform(X)
        return codes @ self.components_ + self.mean_face_


def demo_olivetti_face_pca():
    """Synthetic verification: generate correlated face-like patches and check PCA."""
    rng = np.random.default_rng(0)
    n_samples = 200
    img_h, img_w = 32, 32
    n_features = img_h * img_w

    # Build a synthetic dataset with a shared global intensity and horizontal gradient.
    base_faces = rng.uniform(0.3, 0.7, size=(n_samples, 1))
    global_mode = np.ones((1, n_features))
    gradient_mode = np.tile(np.linspace(-1.0, 1.0, img_w), (img_h, 1)).reshape(1, -1)
    noise = rng.normal(0, 0.05, size=(n_samples, n_features))
    X = (
        base_faces @ global_mode
        + rng.normal(0, 0.2, size=(n_samples, 1)) @ gradient_mode
        + noise
    )

    pca = OlivettiFacePCA(n_components=10).fit(X)
    X_recon = pca.reconstruct(X)
    mse = np.mean((X - X_recon) ** 2)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    print(f"Features: {n_features}, retained components: {pca.n_components}")
    print(f"Explained variance ratio (first 5): {pca.explained_variance_ratio_[:5].round(4)}")
    print(f"Cumulative variance ratio (top 10): {cum_var[-1]:.4f}")
    print(f"Reconstruction MSE: {mse:.6f}")

    # The first two synthetic components should capture most variance.
    assert cum_var[1] > 0.6, "First two components should explain most synthetic variance"
    assert pca.components_.shape == (10, n_features)
    return pca, X, X_recon


if __name__ == "__main__":
    demo_olivetti_face_pca()
```

The reconstruction step completes the story. Given a new face, I subtract the mean face, project onto the retained eigenfaces to get a short code, and then rebuild the image by adding back the weighted eigenfaces plus the mean. The quality of the reconstruction improves as more components are kept, and the cumulative explained variance ratio tells me precisely the trade-off between compression and fidelity. The canonical method name is Olivetti Face PCA, and it is the standard principal component analysis procedure specialized to the Olivetti faces image corpus.
