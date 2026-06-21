I will explain the Nadaraya-Watson kernel regression estimator, which is the canonical nonparametric method I am presenting. The goal of regression is to estimate the conditional expectation of a response variable given a predictor variable from observed data. When we do not want to commit to a specific parametric form such as a line or a polynomial, kernel regression offers a flexible local averaging approach, and the Nadaraya-Watson estimator is the most classical form of this approach.

Suppose we observe a dataset of pairs, where each pair consists of a real-valued predictor and a real-valued response. We believe there is an unknown function relating the predictor to the response, but we do not know the shape of that function. The Nadaraya-Watson estimator constructs the predicted response at a query point by taking a weighted average of all observed responses. The weights are determined by a kernel function that assigns larger weights to observations whose predictors are close to the query point and smaller weights to observations that are far away. The kernel is usually symmetric and peaks at zero, and the width of the kernel is controlled by a bandwidth parameter.

The estimator can be written as a ratio. For a query point, the numerator is the sum of each observed response multiplied by the kernel evaluated at the scaled distance between the query point and the corresponding predictor. The denominator is the sum of those kernel weights. This ratio form ensures that the estimate is a proper weighted average. When the query point coincides with one of the observed predictors, that observation receives the largest weight, and the estimate is pulled toward its response. When the bandwidth is small, only very nearby observations influence the prediction, which can lead to a highly variable but very flexible fit. When the bandwidth is large, many observations contribute significantly, which smooths the estimate but may oversimplify the underlying relationship.

The choice of kernel is often less critical than the choice of bandwidth. Common kernels include the Gaussian kernel, which gives positive weight to every observation but decays exponentially with squared distance, and the Epanechnikov kernel, which gives zero weight beyond a certain distance and is theoretically optimal in a mean squared error sense for certain smoothness classes. The uniform kernel simply counts observations within a fixed window around the query point. In practice, the Gaussian kernel is widely used because it is smooth and convenient.

The bandwidth selection problem is central to using the Nadaraya-Watson estimator well. If the bandwidth is too small, the estimator interpolates noise and produces a wiggly estimate. If the bandwidth is too large, the estimator may miss important local structure and oversmooth the data. A common way to choose the bandwidth is cross-validation, where we try several candidate bandwidths and select the one that gives the best out-of-sample prediction performance. Leave-one-out cross-validation is particularly natural because the estimator can be evaluated at each observation using all other observations, and the bandwidth with the lowest average squared error is chosen.

The Nadaraya-Watson estimator is closely related to locally weighted regression and can be derived from kernel density estimation. If we estimate the joint density of the predictor and response using a product kernel, and separately estimate the marginal density of the predictor, then the conditional expectation is the ratio of these density estimates. This density estimation perspective explains the ratio form of the estimator and connects it to the broader literature on kernel smoothing.

The estimator has several appealing theoretical properties. Under regularity conditions on the unknown regression function, the kernel, and the bandwidth, the Nadaraya-Watson estimator is consistent, meaning that as the sample size grows and the bandwidth shrinks at an appropriate rate, the estimate converges to the true regression function. The rate of convergence depends on the smoothness of the function and the dimension of the predictor space. In one dimension, the typical rate is comparable to other nonparametric methods, but the curse of dimensionality means that performance degrades as the predictor dimension increases because data become sparse in high-dimensional spaces.

One subtle issue with the Nadaraya-Watson estimator is bias near the boundaries of the predictor range. Near a boundary, observations are only available on one side of the query point, so the kernel weights become asymmetric and the estimate is pulled toward the interior. This boundary bias can be reduced by using local polynomial regression, of which the local linear estimator is the most common alternative. Local linear regression fits a weighted linear model locally and has better boundary behavior, though the Nadaraya-Watson estimator remains simpler and is essentially a local constant fit.

Despite its simplicity, the Nadaraya-Watson estimator remains widely used and taught because it provides an intuitive introduction to nonparametric regression. It is also a building block for more sophisticated methods, such as kernel smoothing in conditional density estimation, conditional mode estimation, and various semi-parametric models. In machine learning, it connects to nearest-neighbor methods and Gaussian process regression, where the kernel plays a similar role in determining similarity between points.

I will now provide a compact, runnable Python illustration that implements the Nadaraya-Watson kernel regression estimator, generates synthetic data from a nonlinear relationship, fits the estimator with a Gaussian kernel and a fixed bandwidth, and visualizes the result. The script also includes a small leave-one-out cross-validation routine to show how the bandwidth can be selected from a small grid.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(u):
    """Standard Gaussian kernel."""
    return np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)

def nadaraya_watson(x_query, x_train, y_train, bandwidth):
    """
    Nadaraya-Watson kernel regression estimate at points x_query.
    """
    x_query = np.atleast_1d(x_query)
    estimates = np.zeros(len(x_query))
    for i, xq in enumerate(x_query):
        weights = gaussian_kernel((x_train - xq) / bandwidth)
        denom = np.sum(weights)
        if denom > 1e-12:
            estimates[i] = np.sum(weights * y_train) / denom
        else:
            estimates[i] = np.mean(y_train)
    return estimates

def leave_one_out_cv(x_train, y_train, bandwidths):
    """Select bandwidth by leave-one-out cross-validation."""
    best_bw = bandwidths[0]
    best_mse = np.inf
    for bw in bandwidths:
        preds = np.zeros(len(y_train))
        for i in range(len(y_train)):
            mask = np.ones(len(y_train), dtype=bool)
            mask[i] = False
            preds[i] = nadaraya_watson(
                x_train[i], x_train[mask], y_train[mask], bw
            )[0]
        mse = np.mean((y_train - preds) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_bw = bw
    return best_bw, best_mse

#Synthetic data: a sinusoidal relationship with noise.
np.random.seed(0)
x_train = np.sort(np.random.uniform(0, 2 * np.pi, 80))
y_train = np.sin(x_train) + 0.2 * np.random.randn(len(x_train))

#Fit and predict.
bandwidth_grid = np.linspace(0.1, 1.0, 20)
best_bw, best_mse = leave_one_out_cv(x_train, y_train, bandwidth_grid)
x_plot = np.linspace(0, 2 * np.pi, 500)
y_plot = nadaraya_watson(x_plot, x_train, y_train, best_bw)

print(f"Selected bandwidth: {best_bw:.3f}, LOOCV MSE: {best_mse:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="black", alpha=0.5, label="Observations")
plt.plot(x_plot, y_plot, color="red", linewidth=2, label="Nadaraya-Watson fit")
plt.plot(x_plot, np.sin(x_plot), color="blue", linestyle="--", label="True function")
plt.title("Nadaraya-Watson Kernel Regression")
plt.xlabel("Predictor")
plt.ylabel("Response")
plt.legend()
plt.tight_layout()
plt.savefig("nadaraya_watson_demo.png", dpi=150)
plt.show()
```

This code demonstrates the core idea of the Nadaraya-Watson estimator in a self-contained way. The function `nadaraya_watson` computes a weighted average of the training responses at each query point, with weights given by the Gaussian kernel. The bandwidth selection routine searches over a grid and picks the value with the lowest leave-one-out cross-validation error. The resulting fit adapts to the local structure of the data while avoiding the need to specify a global parametric model.

In summary, the method I have described is Nadaraya-Watson kernel regression, a foundational nonparametric technique for estimating conditional expectations. It combines a kernel similarity function with local weighted averaging, and its performance is governed primarily by the bandwidth. With proper bandwidth selection, it provides a flexible and interpretable approach to regression that remains relevant across statistics and machine learning.
