A feed-forward network trained with ordinary squared error converges to the conditional mean of the targets, which is the right point predictor but nothing more. The squared-error loss also collapses all residual information into a single global variance, so it produces one error bar for the entire input space. That is fine when the target noise is homoscedastic, but many regression problems are not. In the climate-emulation setting, for example, boundary-layer convection is genuinely stochastic while the free troposphere is almost deterministic, so a global variance is wrong everywhere the noise level changes with the atmospheric state. Mixture-density networks can represent arbitrary conditional densities, yet they are overkill for unimodal but input-dependent spread and are hard to fit when data are limited. Non-parametric density estimators face the same data problem. What is needed is a way to learn an input-dependent variance from the same feed-forward, gradient-descent machinery, without requiring variance labels or an enormous training set.

The method is Heteroskedastic Regression, abbreviated HSR. The core move is to stop treating squared error as fundamental and remember that it is the negative log-likelihood of a Gaussian with constant variance. HSR replaces that constant with a variance that depends on the input. The network therefore has two output heads: one predicts the conditional mean mu of each target dimension, and the other predicts the log-precision ell, defined as the natural logarithm of one over the variance. Because ell is an unconstrained real number, the variance is automatically positive when passed through an exponential, and zero variance is unreachable. The training objective is the Gaussian negative log-likelihood. Dropping constants and writing the loss per output element in the log-precision convention gives (d - mu)^2 times exp(ell) minus ell. The first term is a squared error weighted by the predicted precision; the second term penalizes the model for claiming large variance. If the precision is held fixed, the loss reduces to plain squared error, so HSR contains ordinary regression as a special case.

The two terms balance in a useful way. Holding the mean fixed, the per-example minimizer sets the predicted variance equal to the squared residual, so the variance head learns to predict the squared error without ever being given a variance target. The likelihood invents that target. Differentiating the loss shows that the mean head receives the ordinary delta-rule update scaled by one over the predicted variance, which means low-noise examples get a larger effective learning rate and noisy outliers stop dragging the fit around. The variance head update is driven by the difference between the current squared residual and the predicted variance, pushing the variance up when the residual is larger and down when it is smaller. There is a practical danger, however: at the start of training the mean is poor, so large residuals appear everywhere. If the inverse-variance weighting is active too early, the model can mistake its own underfitting for high noise and freeze out the very examples it most needs to learn. The standard remedy is a staged schedule. First, train only the mean with plain mean-squared error until it is a reasonable conditional-mean estimate. Then switch on the full Gaussian negative log-likelihood and let the mean and precision co-adapt from a sensible initialization, such as setting the variance bias to the logarithm of the global mean squared error. This warmup keeps the weighted regression from harming learning while the mean is still wrong.

For the architecture, the mean and the variance are generally different functions of the input, so each is given its own multilayer perceptron rather than forcing the variance to use features built only for the mean. When the target is a vector, as in the climate emulator, HSR uses a diagonal Gaussian: each output dimension gets its own mean and its own variance, and the joint negative log-likelihood is the sum of the per-element terms. At inference time the evaluation metrics score only the predicted value, so the mean head is returned as the point prediction; the log-precision head is converted to a standard deviation for calibration, error bars, or sampling. A coarse numerical clamp on the scalar loss is also useful because a very confident correct prediction can send the negative log-likelihood toward very large negative values.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(p=dropout),
            ))
        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for block in self.blocks:
            x = F.relu(block(x))
        return self.final_linear(x)


class HeteroskedasticRegression(nn.Module):
    """Two MLPs: one predicts the per-dimension mean, the other the log-precision."""

    def __init__(self, input_dim, output_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.mean = MLP(input_dim, output_dim, hidden_dim, layers, dropout)
        self.logprec = MLP(input_dim, output_dim, hidden_dim, layers, dropout)

    def forward(self, x):
        return self.mean(x), self.logprec(x)

    def predict_mean(self, x):
        mu, _ = self(x)
        return mu

    def predict_distribution(self, x):
        mu, logprec = self(x)
        std = torch.exp(-0.5 * logprec)
        return mu, std


def gaussian_nll(mu, logprec, target):
    # Twice the per-element negative log-likelihood, dropping the constant ln(2pi):
    # (target - mu)^2 / sigma^2 + ln(sigma^2) = (target - mu)^2 * exp(logprec) - logprec
    return (torch.exp(logprec) * (target - mu) ** 2 - logprec).mean()


def training_loss(model, x, y, epoch, total_epochs):
    mu, logprec = model(x)
    if epoch < total_epochs / 3:
        # Warm up the mean with plain squared error before the precision can reweight learning.
        return ((y - mu) ** 2).mean()
    loss = gaussian_nll(mu, logprec, y)
    return torch.clamp(loss, min=-1e5, max=1e5)
```
