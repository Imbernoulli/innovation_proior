I propose the canonical name "Kernel Trick and Representer Theorem" for this method. The starting problem is that ordinary linear learners can only separate or regress along straight directions in the original input space, while many useful decision boundaries and response surfaces are curved. One natural fix is to map each input into a much richer feature space where the problem becomes linear again, but writing out that feature map explicitly is often impractical or impossible: the useful expansion may have thousands, millions, or even infinitely many coordinates. The kernel trick solves this by replacing explicit feature coordinates with a single similarity function, and the representer theorem guarantees that even though the search is over an infinite-dimensional function space, the optimal solution can always be written as a finite weighted sum over the training examples.

The first insight is that many learning algorithms do not actually need the coordinates of the lifted points; they only need the inner products between lifted points. A perceptron update, a support-vector margin, or a least-squares fit in feature space can be rewritten so that every occurrence of the lifted inner product <phi(x), phi(z)> is isolated. If we can find a real-valued function k(x, z) that equals that inner product, the algorithm can run without ever computing phi(x). A valid such function is called a positive definite kernel. Concretely, for any finite collection of points x_1, ..., x_m and any real coefficients c_1, ..., c_m, the matrix K with entries K_ij = k(x_i, x_j) must be symmetric and satisfy sum_i sum_j c_i c_j K_ij >= 0. This condition is not merely a technical nicety: it is exactly what guarantees that the kernel evaluations behave like legitimate squared lengths and angles in some Hilbert space. If it failed, we could find examples whose "feature-space norm" would be negative, which is impossible for a genuine inner product.

When the positive-definiteness condition holds for every finite sample, the Moore-Aronszajn theorem constructs a unique reproducing kernel Hilbert space, or RKHS, in which the feature map can be taken to be Phi(x) = k(., x), the function that reports the similarity of any input to x. The reproducing property is the identity <f, k(., x)>_H = f(x), which means that evaluating a function at x is the same as taking its inner product with the representer of point evaluation. In particular, <k(., x), k(., z)>_H = k(x, z), so the kernel value is exactly the inner product in this space. For a Mercer expansion k(x, z) = sum_r lambda_r phi_r(x) phi_r(z) with lambda_r >= 0, the coordinates are sqrt(lambda_r) phi_r(x); the eigenvalues appear linearly in the inner product, not squared. Zero eigenvalues contribute no coordinate and simply drop out.

The second insight is that regularized learning over this entire Hilbert space still collapses to a finite-dimensional optimization. Consider an objective of the form c((x_1, y_1, f(x_1)), ..., (x_m, y_m, f(x_m))) + g(||f||_H), where c is any empirical cost that depends only on the fitted values at the training inputs and g is a strictly increasing function of the RKHS norm, such as a multiple of the squared norm. Take any candidate function f and split it as f = f_parallel + f_perp, where f_parallel lies in the span of the representers k(., x_i) and f_perp is orthogonal to every one of them. By the reproducing property, f_perp contributes nothing to any training value f(x_j) = <f, k(., x_j)>_H, because <f_perp, k(., x_j)>_H = 0. Therefore the empirical cost c is unchanged when f_perp is removed. By Pythagoras, ||f||_H^2 = ||f_parallel||_H^2 + ||f_perp||_H^2, and because g is strictly increasing, the penalty strictly improves when f_perp is removed. So every minimizer must have f_perp = 0, which means the optimal function can be written as f(.) = sum_i alpha_i k(., x_i) for some real coefficients alpha_i. This is the representer theorem. It turns an optimization over functions into an optimization over m numbers.

For the concrete case of kernel ridge regression, the objective is sum_i (y_i - f(x_i))^2 + lambda ||f||_H^2 with lambda > 0. Writing f in its representer form gives f_train = K alpha for the vector of training predictions, where K is the m-by-m Gram matrix with entries K_ij = k(x_i, x_j), and ||f||_H^2 = alpha^T K alpha. The stationarity condition is K ((K + lambda I) alpha - y) = 0. For lambda > 0 the matrix K + lambda I is positive definite even when K is only positive semidefinite, so the canonical stable solution is alpha = (K + lambda I)^{-1} y. A prediction on a new input x is then f(x) = sum_i alpha_i k(x_i, x), which only requires storing the training inputs and the kernel function. This is the same pattern used in libraries such as scikit-learn's KernelRidge: fit by solving the regularized linear system in the dual, then predict by forming kernel products with the stored training set.

A useful extension appears when the regularizer ignores a finite-dimensional null space, as in smoothing splines and related semiparametric models. If the penalty only controls a roughness component and leaves low-degree polynomials unchanged, the representer argument removes only the part of f that is invisible to the observations and penalized. The remaining unpenalized basis functions must be retained, giving a solution of the form f(x) = sum_i alpha_i k(x_i, x) + sum_p beta_p psi_p(x). The coefficients are then identified by additional constraints appropriate to the spline system, such as orthogonality between the penalized coefficients and the null-space design matrix. The core message remains the same: the penalized component lives in the finite span generated by the training data, while any explicitly unpenalized structure is handled separately.

Together, the kernel trick and the representer theorem explain why a nonlinear, potentially infinite-dimensional learning problem can be solved with finite linear algebra. The kernel trick hides the feature coordinates behind a similarity function, and the representer theorem confines the optimizer to the span of the training representers. The resulting algorithm needs only the m-by-m Gram matrix and a single kernel routine.

```python
import numpy as np

def polynomial_kernel(x, z, c=1.0, d=2):
    """Homogeneous polynomial kernel k(x,z) = (c + x^T z)^d."""
    return (c + np.dot(x, z)) ** d

def rbf_kernel(x, z, gamma=1.0):
    """RBF kernel k(x,z) = exp(-gamma ||x - z||^2)."""
    return np.exp(-gamma * np.sum((x - z) ** 2))

# 1. Verify that the polynomial kernel equals an explicit feature inner product.
np.random.seed(0)
x = np.random.randn(3)
z = np.random.randn(3)
c, d = 1.0, 2

# Explicit quadratic feature map for (c + x^T z)^2 in 3D.
# Expanding gives c^2 + 2c x^T z + sum_i x_i^2 z_i^2 + 2 sum_{i<j} x_i x_j z_i z_j.
def phi_poly(u, c, d):
    assert d == 2
    linear = np.sqrt(2 * c) * u
    pure = u ** 2
    cross = np.array([np.sqrt(2) * u[i] * u[j] for i in range(len(u)) for j in range(i + 1, len(u))])
    return np.concatenate([[c], linear, pure, cross])

phi_x = phi_poly(x, c, d)
phi_z = phi_poly(z, c, d)
explicit = np.dot(phi_x, phi_z)
kernel_val = polynomial_kernel(x, z, c, d)
print("explicit inner product:", explicit)
print("kernel evaluation:   ", kernel_val)
print("difference:          ", abs(explicit - kernel_val))

# 2. Demonstrate kernel ridge regression on a small 1D problem.
X_train = np.linspace(0, 1, 8).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train.ravel()) + 0.05 * np.random.randn(8)

lam = 0.01
gamma = 10.0
m = len(X_train)
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        K[i, j] = rbf_kernel(X_train[i].ravel(), X_train[j].ravel(), gamma)

alpha = np.linalg.solve(K + lam * np.eye(m), y_train)

def predict(x_new):
    k_new = np.array([rbf_kernel(x_new.ravel(), X_train[i].ravel(), gamma) for i in range(m)])
    return k_new @ alpha

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = np.array([predict(x) for x in X_test])

print("\nTrain predictions:", np.array([predict(x) for x in X_train]))
print("Train targets:    ", y_train)
print("Residual norm:    ", np.linalg.norm(y_train - np.array([predict(x) for x in X_train])))

# 3. Check positive semidefiniteness of the RBF Gram matrix.
eigvals = np.linalg.eigvalsh(K)
print("\nSmallest eigenvalue of K:", eigvals[0])
print("Largest eigenvalue of K: ", eigvals[-1])
print("All nonnegative?         ", np.all(eigvals >= -1e-10))
```
