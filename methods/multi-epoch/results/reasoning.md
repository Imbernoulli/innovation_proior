Let me start from the resource mismatch, because that is the thing I have to exploit without fooling myself. The target is a hidden `k`-sparse parity: `y = chi_S(x)`, with `|S| = k`, and statistically I only need enough labels to distinguish one subset among `C(n,k)` possibilities, so the information floor is `log C(n,k) = Theta(k log n)`. But a gradient learner is not doing Gaussian elimination; it is reading correlations through noisy gradients. Parity gives no partial credit to a wrong subset, and the SQ lower bound says that `T` tolerance-`tau` correlational queries must satisfy `T / tau^2 >= Omega(n^k)`. Online SGD accepts that computational price and pays it in samples too, because every step draws a fresh batch. If I need `T` steps and the batch size is `B`, I consume `T * B` independent examples. In the window `k log n << m << n^k`, that coupling is the blocker: the dataset can contain enough information about `S`, but the online recipe runs out of fresh samples before it has spent enough compute.

So I want to separate the two resources. I can draw a finite dataset once and keep optimizing on it. Then the number of gradient steps is no longer the number of fresh examples; I can spend compute by revisiting the same `m` labels. But that move immediately breaks the clean guarantee I had. An online minibatch gradient is an unbiased estimate of the population gradient. A minibatch drawn from a fixed dataset is only an estimate of the empirical gradient of that dataset. Reuse does not create new independent samples, and it cannot make the empirical distribution closer to the population distribution than the original draw already made it. The only way this can work is if the fixed sample already preserves enough trace of the hidden parity, and the repeated steps give the optimizer time to integrate that trace.

What exactly is the trace I am hoping survives? I should look at the population gradient, because that is where the sparse feature first becomes visible. Take a single ReLU neuron and the correlation loss. At initialization, the coordinate gradient has the form `E[-y x_i 1[w*x + b > 0]]`. For a sign or all-ones threshold, the indicator is `1/2 + 1/2 Majority` after flipping coordinates by the signs of `w`. Multiplying by `y x_i` decides which Fourier coefficient I am reading. If `i` is in `S`, the `x_i` cancels one parity bit and I read an order-`k-1` coefficient. If `i` is not in `S`, the extra `x_i` adds one bit and I read an order-`k+1` coefficient. So the feature signal is a simple ordering question: are the order-`k-1` majority coefficients separated from the order-`k+1` coefficients?

The exact majority formula says yes. For odd order `q`, all size-`q` coefficients are equal to
`xi_q = (-1)^((q-1)/2) * C((n-1)/2, (q-1)/2) / C(n-1, q-1) * 2^{-(n-1)} C(n-1, (n-1)/2)`, and the even coefficients vanish. The two orders I need satisfy
`|xi_{k-1}| = ((n-k)/(k-1)) |xi_{k+1}|`. Therefore
`|xi_{k-1}| - |xi_{k+1}| = ((n-2k+1)/(k-1)) |xi_{k+1}|`, and the binomial estimates give the concrete lower bound
`gamma_Maj >= 0.03 (n-1)^(-(k-1)/2)` for `n >= 4k`. The ReLU indicator carries half of the nonempty majority coefficient, so the gradient-coordinate gap changes only by a constant factor. The exponent is the important part: the signal is real, but its scale is `Theta(n^(-(k-1)/2))`.

Now I can see the one-shot recovery condition. If the gradient function has gap `gamma`, and my estimated gradient is within `gamma/2` of the population gradient in `infinity` norm, then no irrelevant coordinate can overtake a relevant one. The top `k` absolute coordinates are exactly `S`. For bounded stochastic gradients, a union bound over the `n` coordinates gives `O(log n / gamma^2)` samples to reach that accuracy. With `gamma = Theta(n^(-(k-1)/2))`, this is `O(log n / gamma^2) = tilde O(n^(k-1))` samples just to resolve the gap in this read-off sense, and the clean two-layer MLP theorem uses an `n^{Omega(k)}`-scale batch for the full construction. So the population mechanism explains why gradient training can find the subset, but it also explains why online training spends so many independent samples.

The finite-sample route has to live between two thresholds, not pretend they are the same. The `Theta(k log n)` floor says how many labels are needed in principle to specify `S`. The `O(log n / gamma^2)` bound says how many fresh samples are enough for a direct, uniformly accurate gradient estimate. A fixed dataset with `m` between those scales does not satisfy the one-shot certificate by magic. What it can do is contain enough information about `S` for an optimizer with many passes to extract it gradually. Repeated epochs are compute, not new evidence; they substitute for fresh samples only because the empirical objective is informative enough to keep nudging the same sparse coordinates over and over.

This immediately creates the wall on the other side. A finite dataset is also easy to memorize. The MLP can fit a sample-specific labeling with a dense interpolating function, and that route can be much faster than resolving the tiny parity gap. If I optimize the empirical loss and stop when train error is zero, I may have learned only the sample. The population loss is no longer what each step sees; the empirical gradient rewards whatever fits those `m` points. So the training trajectory I should expect in the informative-but-small window is not smooth population learning. It is fit-the-sample first, then, if the sparse trace keeps accumulating, recover the rule later.

This is where the delayed-generalization picture becomes more than a curiosity. During the early phase, the dense interpolator can explain the finite table and make the training metric look solved. But if the dataset is large enough that the empirical correlations still favor the true sparse coordinates, continued passes can keep reinforcing the low-complexity parity solution after the training set is already fit. Held-out accuracy can therefore remain near chance through a long plateau and then change abruptly when the sparse circuit overtakes the memorizer. That is the grokking-shaped trajectory: memorization first, delayed generalization later. I should not describe that as extra samples being created; it is a competition between two solutions under a fixed empirical objective.

Now I need a bias that makes the right competitor win. A dense memorizer uses many degrees of freedom to fit idiosyncratic labels; the sparse parity rule concentrates dependence on `k` coordinates. A norm penalty is the natural bias toward the lower-complexity explanation. With weight decay, the update has the shrinkage term `theta <- (1 - eta lambda) theta - eta grad empirical_loss`. The decay does not know which weights are relevant, so it is dangerous, but it changes the competition: weights that are only useful for a brittle sample fit are continually eroded, while weights that keep receiving coherent reinforcement from the parity trace can survive and grow. This is the role of weight decay in the finite-sample run: it makes "fit the sample and stay there" less stable and gives the sparse solution time to become the preferred explanation.

The danger is the same sentence read backward. The relevant signal starts tiny. If `lambda` is too large, the shrinkage term overwhelms the early Fourier-gap drift before it has separated the relevant coordinates. Then regularization does not help generalization; it kills optimization. So the parameter has to sit in a window: large enough to erode the dense memorizer, small enough not to crush the nascent sparse circuit. That is the computational-statistical trade in one scalar. More decay expands the small-data region where the sparse solution can win, until it crosses the point where the optimizer cannot amplify the feature signal at all.

The sample size `m` has its own window. If `m` is extremely large, the empirical objective is close to the population objective and there is little reason for a long memorization-generalization split. If `m` is near or below the statistical floor, the sample may not identify `S` reliably, so no amount of repeated optimization can reconstruct missing information. The useful regime is in between: enough examples that the fixed sample carries the sparse trace, far fewer than the online fresh-sample cost, and enough passes that compute can be spent extracting what is already present.

I also have to make the repeated passes behave like stochastic optimization on the empirical distribution, not like a deterministic artifact of one fixed order. If the batches arrive in the same cycle every time, the optimizer sees the same temporal pattern at every pass boundary. Reshuffling each pass keeps each step closer to a fresh minibatch from the empirical distribution and avoids making the batch order another object to memorize. It does not change the sample count; it just makes the reused-gradient stream better mixed.

Now the fixed harness matters. The trainer is already written as `while steps < max_steps`, and inside each pass it samples a fresh permutation of the returned training set before stepping through minibatches. Therefore the epoch count is not an independent knob. Up to last-batch rounding,

```text
epochs = max_steps * B / m.
```

If I return the maximal allowed dataset, each example is seen about once and the run behaves like single-pass online training under the fixed compute budget. If I return a small fixed dataset, the same `max_steps` forces many reshuffled passes over those examples. The method is exactly that choice, paired with weight decay kept in the working window: make `m` small enough that the harness becomes a multi-pass finite-sample trainer, and keep the regularization bias that lets the sparse rule beat the dense memorizer.

The two slots I need to fill are the dataset construction and the optimizer configuration:

```python
import torch


def parity_labels(x, secret):
    idx = torch.as_tensor(secret, dtype=torch.long, device=x.device)
    return x.index_select(1, idx).sum(dim=1).remainder(2).to(torch.float32)


def make_dataset(secret, n_features, seed, max_train_examples):
    # A small fixed sample makes the fixed step budget revisit examples many times.
    # It must stay well above the statistical floor while remaining far below the
    # maximal single-pass budget.
    train_examples = 10_000
    m = min(train_examples, max_train_examples)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (m, n_features), generator=generator, dtype=torch.float32)
    y = parity_labels(x, secret)
    return x, y


def get_optimizer_config():
    # Keep weight decay in the finite-sample window: enough regularization to
    # penalize dense memorization, not so much that it erases the early sparse signal.
    return {"lr": 1e-3, "wd": 1e-2, "beta1": 0.9, "beta2": 0.999}
```

And this is the fixed loop that turns that small `m` into many epochs:

```python
def train(model, train_x, train_y, optimizer, loss_fn, batch_size, max_steps):
    steps = 0
    while steps < max_steps:
        perm = torch.randperm(train_x.shape[0], device=train_x.device)
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[idx]).view(-1), train_y[idx])
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                return
```

So the causal chain is tight. Sparse parity is statistically cheap but correlationally expensive. Online SGD pays the `n^{Omega(k)}` computation in fresh samples because every step consumes new data. The Fourier gap explains the sparse feature signal and the `O(log n / gamma^2)` one-shot concentration cost; finite reuse does not evade that concentration theorem, it changes the resource accounting by optimizing repeatedly on one informative empirical distribution. The fixed sample creates a memorization competitor, so weight decay is needed as a low-norm bias, but only inside the window where it does not erase the tiny early signal. In the harness, all of this reduces to returning a small dataset and keeping weight decay on: the loop itself supplies `max_steps * B / m` reshuffled passes over the same examples.
