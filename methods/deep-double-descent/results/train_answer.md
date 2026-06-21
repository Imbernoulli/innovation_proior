The classical theory of generalization says that increasing model capacity first helps and then hurts. A model with too few parameters underfits; a model with too many overfits. This gives a U-shaped risk curve, and the point where training error hits zero is treated as the start of overfitting. That picture works for shallow models and small hypothesis classes, but it fails for modern deep networks. In practice, large networks can interpolate the training set and still generalize well. They can even fit random labels, so capacity alone cannot distinguish good generalization from bad. Earlier work by Belkin, Hsu, Ma, and Mandal had already shown that test risk can rise near interpolation and then fall again, but that evidence was mostly framed around model size or parameter count. That leaves out training time, regularization, data augmentation, label noise, optimizer choice, and sample size. We need a way to put all of these on the same axis.

The missing step is to stop measuring nominal capacity and start measuring effective fitting capacity. The right quantity is not the number of parameters; it is how many samples the whole training procedure can fit to approximately zero error. Once that quantity is close to the actual sample size, the learner is critically parameterized and fragile. Far below or far above that point, it is safer. This coordinate change turns the U-curve into a richer picture with three distinct crossings: model-wise, epoch-wise, and sample-wise.

The method is called Deep Double Descent. It treats a training procedure T as the complete map from a labeled training set to a trained predictor, including architecture, optimizer, number of steps, data augmentation, regularization, and anything else that affects fitting. For a distribution D and a small threshold epsilon, define Effective Model Complexity as EMC_{D,epsilon}(T) = max { n : E_{S ~ D^n}[ Error_S(T(S)) ] <= epsilon }. The paper uses epsilon = 0.1 as a heuristic for "approximately zero" training error. EMC rises when width increases, training time increases, regularization weakens, augmentation is reduced, or labels are cleaner. It is an empirical quantity that tracks the interpolation threshold, not a classical complexity bound.

The generalized hypothesis is that test error is governed by the relationship between EMC and the actual sample size n. If EMC is much smaller than n, increasing effective complexity decreases test error because the procedure is underfitting. If EMC is much larger than n, increasing effective complexity also decreases test error because the procedure has slack and can select a better interpolant. The dangerous region is when EMC is approximately equal to n. There the procedure is barely able to fit the training set, and test error can peak or plateau. So the interpolation threshold is not the end of generalization; it is a critical boundary between the classical under-parameterized regime and the modern interpolating regime.

Model-wise double descent fixes a long training budget and varies model width. As width grows, EMC grows, and the test curve follows the classical descent and ascent up to the point where train error first reaches zero, then descends again. Epoch-wise double descent fixes a sufficiently large model and varies training time. Training longer increases EMC even though parameter count is fixed, so the same model can travel through underfitting, critical fitting, and overfitting in a single run. Test error can fall, rise near the epoch where train error hits zero, and fall again. Sample-wise double descent fixes the model and procedure and varies n. More data usually lowers error, but it also raises the interpolation threshold. Near the critical region these effects can fight, so more data can locally hurt test performance. The mechanism in linear and random-feature settings is that near EMC = n the data matrix is poorly conditioned and the interpolating solution is highly sensitive to noise; past the threshold, many interpolants exist and the implicit bias of the optimizer can select a better one.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def fit_procedure(X_train, y_train, X_val, y_val, width, epochs, l2=1e-4):
    """A simple training procedure whose EMC can be tuned by width or epochs."""
    np.random.seed(0)
    n_samples, n_features = X_train.shape
    W1 = np.random.randn(n_features, width) * 0.01
    b1 = np.zeros(width)
    W2 = np.random.randn(width, 1) * 0.01
    b2 = 0.0

    for _ in range(epochs):
        # Forward pass
        H = np.maximum(X_train @ W1 + b1, 0)  # ReLU
        y_pred = H @ W2 + b2
        err = y_pred.ravel() - y_train

        # Backward pass with gradient descent
        dW2 = (H.T @ err) / n_samples + l2 * W2.ravel()
        db2 = err.mean()
        dH = np.outer(err, W2.ravel()) * (H > 0)
        dW1 = (X_train.T @ dH) / n_samples + l2 * W1
        db1 = dH.mean(axis=0)

        lr = 0.05
        W2 -= lr * dW2.reshape(-1, 1)
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    H_val = np.maximum(X_val @ W1 + b1, 0)
    y_val_pred = H_val @ W2 + b2
    train_mse = mean_squared_error(y_train, y_pred.ravel())
    test_mse = mean_squared_error(y_val, y_val_pred.ravel())
    return train_mse, test_mse

def measure_emc(X, y, width, max_n, eps=0.05, epochs=2000):
    """Estimate EMC as the largest sample size n the procedure fits below eps."""
    for n in range(50, max_n + 1, 50):
        idx = np.random.permutation(len(y))[:n]
        tr, te = fit_procedure(X[idx], y[idx], X, y, width=width, epochs=epochs)
        if tr > eps:
            return n - 50
    return max_n

# Generate a noisy 1-D regression task
np.random.seed(42)
X_full = np.linspace(-3, 3, 500).reshape(-1, 1)
X_poly = PolynomialFeatures(degree=5).fit_transform(X_full)
y_clean = np.sin(X_full).ravel()
y_noisy = y_clean + 0.2 * np.random.randn(len(y_clean))

# Model-wise double descent: sweep hidden width
widths = np.arange(5, 200, 10)
train_errs, test_errs = [], []
for w in widths:
    tr, te = fit_procedure(X_poly, y_noisy, X_poly, y_clean, width=w, epochs=2000)
    train_errs.append(tr)
    test_errs.append(te)

# Epoch-wise double descent: fix width, sweep epochs
epochs_list = np.arange(200, 5001, 200)
train_e, test_e = [], []
for e in epochs_list:
    tr, te = fit_procedure(X_poly, y_noisy, X_poly, y_clean, width=80, epochs=e)
    train_e.append(tr)
    test_e.append(te)

# Sample-wise double descent: fix procedure, vary sample size
sizes = np.arange(50, 451, 25)
train_s, test_s = [], []
for n in sizes:
    idx = np.random.permutation(len(y_noisy))[:n]
    tr, te = fit_procedure(X_poly[idx], y_noisy[idx], X_poly, y_clean, width=80, epochs=2000)
    train_s.append(tr)
    test_s.append(te)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(widths, test_errs, label='test')
axes[0].plot(widths, train_errs, label='train')
axes[0].set_xlabel('model width')
axes[1].plot(epochs_list, test_e, label='test')
axes[1].plot(epochs_list, train_e, label='train')
axes[1].set_xlabel('epochs')
axes[2].plot(sizes, test_s, label='test')
axes[2].plot(sizes, train_s, label='train')
axes[2].set_xlabel('sample size')
for ax in axes:
    ax.set_ylabel('MSE')
    ax.legend()
plt.tight_layout()
plt.savefig('deep_double_descent_demo.png')
plt.show()
```
