When two point sets are supposed to represent the same shape but have been measured in different coordinate systems, a naive coordinate-wise comparison is meaningless. The same triangle can look like completely different matrices of numbers if one copy has been shifted to another origin, rotated in the plane, scaled by a different unit, or reflected across an axis. The question Procrustes analysis answers is how to strip away these exactly-the-shape nuisance transformations before comparing configurations, so that the remaining disagreement is genuine geometric difference rather than arbitrary pose. I propose the classical Procrustes analysis as the canonical solution: translate, rotate, scale, and optionally reflect one configuration onto another in the least-squares sense, and report the residual as a shape distance.

The setup is simple but rich. A configuration is an m times p matrix whose rows are landmarks observed in p-dimensional space. Let X and Y be two such matrices. We want to find a scalar s greater than zero, an orthogonal p times p matrix Q satisfying Q^T Q = I, and a translation vector t in R^p that minimize the Frobenius norm ||s X Q + 1 t^T - Y||_F^2. Translation and scaling are nuisance parameters, so the first step is to center both matrices at the origin by subtracting their row means and then rescale them to unit Frobenius norm. After this standardization the comparison is reduced to the orthogonal Procrustes problem: minimize ||X Q - Y||_F^2 over orthogonal Q. The solution comes from the singular value decomposition of Y^T X = U Sigma V^T, and the minimizer is Q = U V^T. If a proper rotation is required, that is det Q must be plus one rather than plus or minus one, one flips the sign of the smallest singular value before forming the product, giving Q = U diag(1, ..., 1, det(U V^T)) V^T. Once the optimal rotation is found, the minimal residual sum of squares divided by the squared norm of the centered data gives the Procrustes disparity, a number between zero and one; its square root is the usual Procrustes distance. A distance of zero means the two configurations are identical up to the allowed transformations, while a large distance indicates real shape dissimilarity.

The same idea scales to more than two configurations through generalized Procrustes analysis. Suppose we have K configurations and want a single consensus shape. We initialize the consensus as one of the configurations, then repeatedly align every configuration to the current consensus using ordinary Procrustes analysis and replace the consensus by the mean of the aligned configurations. Because the mean is itself only defined up to rotation, scale, and translation, the iterations are anchored by keeping the consensus centered and normalized at each step. The algorithm converges to a least-squares group average that removes the pose variability common to all observations and exposes the remaining shape variation. This is the workhorse of geometric morphometrics, where biologists compare anatomical landmarks across specimens, and it is equally useful in machine learning when one needs to align embeddings produced by different runs or different algorithms before comparing them.

A subtle but important point is what the method does not do. It aligns pointwise correspondences; the first row of X must already correspond to the first row of Y, and so on. It does not solve correspondence itself. It also treats all landmarks as equally weighted and uses Euclidean distance, so outliers can dominate the fit. The standard version removes only rigid similarity transforms, though the framework extends to affine or projective transformations when the application demands it. Within its stated domain, however, the method is remarkably clean: it has a closed form for the two-configuration case, a straightforward iterative form for many configurations, and a clear geometric interpretation as the closest point on the orbit of one configuration under the similarity group to another configuration.

The reason Procrustes analysis keeps appearing across disciplines is that shape comparison is almost never coordinate-free in practice. In dimensionality reduction, one often wants to know whether two embeddings show the same global arrangement of clusters up to rotation; a Procrustes alignment followed by the residual distance is the standard answer. In structural biology, one uses it to superimpose protein backbones after translation and rotation. In computer vision, it underlies the estimation of rigid-body pose from corresponding points. Even the word itself captures the spirit of the method: like the mythical bed, we force configurations into a common frame and measure how much they have to be stretched or trimmed to fit.

```python
import numpy as np

def center_and_scale(X):
    """Remove translation and scale so the configuration has centroid zero and unit norm."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc / np.linalg.norm(Xc, "fro")

def orthogonal_procrustes(X, Y, allow_reflection=True):
    """Return the orthogonal matrix Q minimizing ||X Q - Y||_F."""
    M = Y.T @ X
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    if allow_reflection:
        Q = U @ Vt
    else:
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0] * (U.shape[1] - 1) + [d])
        Q = U @ D @ Vt
    return Q

def procrustes_distance(X, Y, allow_reflection=True):
    """Procrustes distance after centering and unit-scaling both configurations."""
    Xs = center_and_scale(X)
    Ys = center_and_scale(Y)
    Q = orthogonal_procrustes(Xs, Ys, allow_reflection)
    residual = np.linalg.norm(Xs @ Q - Ys, "fro")
    return residual / np.sqrt(2.0)  # normalized to [0, 1]

# --- demo: recover a rigid transformation and verify the residual is small ---
np.random.seed(0)
# original 2-D "L" shape
X = np.array([[0, 0], [1, 0], [2, 0], [2, 1]], dtype=float)
# apply random similarity transform
true_angle = np.radians(37)
true_scale = 2.5
true_t = np.array([3.0, -1.5])
R = np.array([[np.cos(true_angle), -np.sin(true_angle)],
              [np.sin(true_angle),  np.cos(true_angle)]])
Y = true_scale * (X @ R) + true_t + 0.02 * np.random.randn(*X.shape)

d = procrustes_distance(X, Y, allow_reflection=False)
print(f"Procrustes distance after alignment: {d:.6f}")
assert d < 0.05, "alignment should remove the similarity transform"

# --- generalized Procrustes analysis for three noisy copies of a shape ---
def generalized_procrustes(configs, n_iter=20):
    configs = [center_and_scale(C) for C in configs]
    consensus = configs[0].copy()
    for _ in range(n_iter):
        aligned = []
        for C in configs:
            Q = orthogonal_procrustes(C, consensus)
            aligned.append(C @ Q)
        consensus = np.mean(aligned, axis=0)
        consensus = center_and_scale(consensus)
    return consensus, aligned

base = np.array([[0, 0], [1, 0.1], [2, -0.1], [2.5, 1.0], [1.2, 1.3]], dtype=float)
copies = [
    base @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) + np.random.randn(*base.shape) * 0.05
    for a in [0.2, 0.8, -0.4]
]
consensus, aligned = generalized_procrustes(copies)
residuals = [np.linalg.norm(A - consensus, "fro") for A in aligned]
print("Mean consensus residual:", np.mean(residuals))
```
