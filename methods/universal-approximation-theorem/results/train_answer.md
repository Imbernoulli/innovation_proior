I will explain the universal approximation theorem for single-hidden-layer neural networks, the result that established shallow feedforward networks as a legitimate function class for nonlinear approximation, and I will propose that the canonical name for this method be the Universal Approximation Theorem. The theorem addresses the following representational question. A one-hidden-layer real-valued network computes a finite ridge expansion of the form G(x) = sum_{j=1}^N alpha_j sigma(w_j . x + theta_j), where the same scalar activation sigma is reused in every hidden unit and the output weights alpha_j, direction vectors w_j, and thresholds theta_j are free parameters. Given a compact input domain such as the n-dimensional unit cube [0,1]^n, a continuous target function f, and a tolerance epsilon greater than zero, does there exist some finite choice of hidden units that makes the uniform error smaller than epsilon? The theorem answers yes, provided sigma is a non-degenerate activation such as a continuous sigmoid. No training algorithm is promised; the statement is purely about the expressive power of the architecture.

The classic sigmoidal version, due to Cybenko, can be stated as follows. Let I_n be [0,1]^n and let C(I_n) carry the supremum norm. Let sigma be a continuous sigmoidal function with distinct finite limits a_- at negative infinity and a_+ at positive infinity. Then the finite sums G(x) = sum_{j=1}^N alpha_j sigma(w_j . x + theta_j), with alpha_j and theta_j real and w_j in R^n, are dense in C(I_n). In other words, for every continuous target f on the cube and every positive epsilon, some finite single-hidden-layer network satisfies sup_{x in I_n} |f(x) - G(x)| < epsilon. It is important to emphasize that this is an existence result. It gives no construction rule for the hidden units, no practical width bound in terms of epsilon or the complexity of f, and no guarantee that gradient descent or any optimizer will find the approximating parameters. The theorem removes representability as a fundamental obstruction, but it leaves width, conditioning, optimization, data, and inductive bias entirely untouched.

The proof is elegant because it converts an approximation problem into a separation problem using duality. One first normalizes the activation so that its limits are 0 and 1; this is harmless because constant functions can be obtained by setting the direction vector w to zero, so spans with the original and normalized activations have the same closure. An activation rho is called discriminatory if the only finite signed regular Borel measure mu on I_n satisfying int rho(w . x + theta) dmu(x) = 0 for all directions w and thresholds theta is the zero measure. If rho is discriminatory, then its ridge span must be dense. The argument is by contradiction: if the span were not dense, the Hahn-Banach theorem would supply a nonzero bounded linear functional that vanishes on the closed span, and the Riesz representation theorem would represent that functional as integration against a nonzero signed measure mu. That measure would annihilate every ridge function, contradicting the assumption that rho is discriminatory. Thus the entire approximation question reduces to showing that common activations are discriminatory.

For a continuous normalized sigmoid, discriminatory follows from the scaling behavior of the activation. Suppose mu annihilates every ridge function built from tau. Then for every w, theta, phi, and large lambda, the function x -> tau(lambda(w . x + theta) + phi) also has zero integral against mu. Letting lambda tend to infinity and using bounded convergence, one obtains mu({w . x + theta > 0}) + tau(phi) mu({w . x + theta = 0}) = 0. Because tau is continuous and not constant, one can choose two values of phi with different tau(phi) values, which forces both the open-half-space mass and the hyperplane mass to be zero. Fixing a direction w and pushing mu forward by the map T_w(x) = w . x gives a signed measure nu_w on the real line. Half-lines and points have zero nu_w-mass, so intervals have zero mass, and the monotone-class theorem implies nu_w = 0. Therefore the one-dimensional Fourier integral int exp(i s w . x) dmu(x) equals zero for every real s. Since s w ranges over all frequencies in R^n as w and s vary, the Fourier transform of mu vanishes identically. Uniqueness of Fourier transforms for finite compactly supported measures then gives mu = 0, completing the proof.

The theorem also extends to classification-style decision regions in a limited but useful way. A continuous network cannot uniformly reproduce an arbitrary discontinuous indicator function, because continuous functions on a compact set are uniformly continuous and cannot jump. Instead, one can approximate a measurable decision function outside a set of arbitrarily small Lebesgue measure. The argument uses Lusin's theorem to replace the measurable decision function by a continuous function on a large-measure set, then applies the uniform approximation result. For a closed decision region D, one builds a continuous ramp that equals 1 on D and 0 outside an epsilon-neighborhood of D using the distance function. If the network approximates that ramp within less than one half, thresholding its output correctly classifies every point inside D and every point outside the epsilon-neighborhood; only the boundary band remains ambiguous. The bad set is therefore not a failure of the theorem but the price of enforcing continuity.

The condition that sigma be sigmoidal is sufficient but not the final word. A sharper frontier characterizes which activations work. For locally bounded piecewise continuous activations whose discontinuities have negligible closure, the thresholded span span{ sigma(w . x + theta) : w in R^n, theta in R } is dense in C(K) for every compact K contained in R^n if and only if sigma is not an algebraic polynomial almost everywhere. Polynomial activations fail for a simple reason: affine substitution of x into a degree-d polynomial yields a multivariate polynomial of degree at most d, and finite linear combinations stay inside that finite-dimensional space, which cannot approximate arbitrary continuous functions. For non-polynomial activations, the multivariate problem reduces to ridge functions of the form f_i(a_i . x), and the remaining burden is one-dimensional density for shifted and scaled copies of the activation. In the smooth non-polynomial case, difference quotients and derivatives bring monomials into the closed span, and the Weierstrass approximation theorem brings in every continuous univariate target. For nonsmooth activations, convolution with smooth test functions moves the problem to the smooth case; if every convolution were polynomial with a uniform degree bound, distribution theory would force the original activation to be polynomial almost everywhere, contradicting the hypothesis.

The threshold parameter theta is essential, not merely a cosmetic bias term. Without shifts, even a non-polynomial activation such as sin(w x) spans only odd functions on a symmetric interval and cannot approximate an even function such as cos(x). Adding a threshold gives translations of the univariate activation and escapes this symmetry trap. The threshold therefore unlocks the representation power that the direction weights alone cannot provide.

The broader lesson is that shallow networks are universal approximators because their affine copies of a fixed activation can separate finite signed measures, and the activation frontier says the thresholded span fails exactly when the activation is polynomial almost everywhere. The theorem explains why a single hidden layer is, in principle, enough for continuous function approximation, but it should not be read as a justification for ignoring depth, regularization, or architecture search in practice. Depth can offer more efficient representations, better conditioning, and richer feature hierarchies, while the universal approximation theorem only guarantees that some sufficiently wide shallow network exists.

To make these ideas concrete, I will run a small numerical illustration. The code below samples a continuous target function on the unit square, fits a one-hidden-layer network with a sigmoidal activation using ordinary least squares on the output weights, and compares the approximation error to the theorem's prediction. This is not a proof; it is merely an empirical demonstration that a small number of randomly oriented ridge functions can approximate a simple continuous target.

```python
import numpy as np

np.random.seed(0)

# Target function on [0,1]^2: f(x,y) = sin(2*pi*x) * cos(2*pi*y)
def f(x):
    return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])

# Build a grid of training points
n_grid = 20
x1 = np.linspace(0, 1, n_grid)
x2 = np.linspace(0, 1, n_grid)
X1, X2 = np.meshgrid(x1, x2)
X = np.column_stack([X1.ravel(), X2.ravel()])
y = f(X)

# Number of hidden units (random ridge functions with sigmoid activation)
n_hidden = 200
W = np.random.randn(n_hidden, 2)
theta = np.random.uniform(-3, 3, size=n_hidden)

# Hidden feature matrix: each column is sigma(w_j . x + theta_j)
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))

H = sigmoid(X @ W.T + theta)

# Fit output weights by least squares (ridge-free for this toy example)
alpha = np.linalg.lstsq(H, y, rcond=None)[0]

# Predict and compute uniform error on the grid
y_pred = H @ alpha
max_error = np.max(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

print(f"Hidden units: {n_hidden}")
print(f"Grid points: {len(y)}")
print(f"RMSE: {rmse:.4f}")
print(f"Max uniform error: {max_error:.4f}")
```
