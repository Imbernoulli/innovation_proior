The task is to recover a hidden size-k subset S of n binary features from labels y = chi_S(x), the parity of the bits in S.  Although only Theta(k log n) samples are needed to identify S information-theoretically, the parity functions are mutually orthogonal: any guess that overlaps S in k-1 coordinates has exactly zero correlation with the label, just as a guess that overlaps none.  That orthogonality underwrites an n^{Omega(k)} statistical-query lower bound, and the classic gradient-concentration argument over all 2^n parities suggests that gradients carry no signal about S.  So the puzzle is not sample complexity but optimization: ordinary gradient training has to find the needle in a polynomially large haystack, and the observed loss curves sit at chance for a long time before snapping to solved.

The reason the pessimistic arguments miss the mark is that they measure against the wrong family.  The bound over all parities uses a family of size 2^n, which drives the gradient variance to zero exponentially; the actual sparse problem fixes |S| = k, so the relevant family has only C(n,k) ~ n^k members.  The same variance bound then leaves a polynomial residual signal of scale about n^{-k/2}, exactly the scale the SQ floor predicts.  A fixed-feature or NTK analysis cannot explain success either: representing all C(n,k) parities with fixed features requires Omega(n^k) dimensions, yet the network learns at width 512 or even width 1, so the weights must leave their initialization.  Memoryless search or Langevin diffusion also fails to match the black-box dynamics, which show no early successes, runtimes concentrated on the initialization, and sub-linear returns to width.

The method is sparse parity learning with SGD.  It trains a generic two-layer ReLU MLP with completely standard initialization and optimizer, no sparsity prior, on fresh online batches of uniform binary vectors.  The mechanism is a Fourier gap in the population gradient.  For a ReLU neuron f(x;w) = (w^T x)_+ at a random-sign or all-ones initialization, the j-th coordinate of the expected gradient under the correlation loss is a Fourier coefficient of the threshold derivative 1[w^T x >= 0].  When j is in S, y x_j is the parity of S \ {j}, a degree-(k-1) function; when j is not in S, y x_j is the parity of S union {j}, a degree-(k+1) function.  Majority's spectrum falls off with degree, and adjacent coefficients satisfy |xi_{k-1}| = ((n-k)/(k-1)) |xi_{k+1}|, so the lower-degree relevant-coordinate gradient dominates the higher-degree irrelevant one by a factor of order n/k.  The resulting Fourier gap is gamma = Theta(n^{-(k-1)/2}).

That gap has two consequences.  If the batch is large enough to estimate the population gradient to within gamma/2 in sup norm, the k largest-magnitude coordinates are exactly S, recoverable in one step from roughly n^{k-1} samples.  With small batches the per-step noise swamps gamma, but each relevant weight follows a biased random walk: drift proportional to gamma times the number of steps, noise proportional to the square root of the number of steps.  Drift eventually outruns noise, so the relevant weights grow steadily throughout the flat phase.  The classifier threshold does not change labels until those weights overtake the irrelevant ones, which is why the loss stays flat and then jumps discontinuously.  The plateau is amplification, not search.  A simple progress probe, rho = ||W_t - W_0||_inf, makes the hidden growth visible while accuracy is still at chance.

In practice the protocol is intentionally off-the-shelf: a two-layer MLP with 512 hidden units, Xavier uniform initialization, zero biases, AdamW with learning rate 1e-3 and weight decay 1e-2, binary cross-entropy loss, and fresh i.i.d. batches each step.  Online sampling is important because it makes every gradient an unbiased estimate of the same population gradient, removing overfitting as a confound and giving the drift a clean signal.  The implementation below follows the standard training harness and adds only the weight-displacement probe.

```python
import torch
from torch import nn


def build_model(n_features: int, width: int = 512) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_features, width),
        nn.ReLU(),
        nn.Linear(width, 1),
        nn.Sigmoid(),
    )


def init_model(model: nn.Sequential) -> None:
    # Standard, data-independent init (must not see the secret).
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def parity_labels(x: torch.Tensor, secret) -> torch.Tensor:
    # y = parity (sum-mod-2 / XOR) of the bits at the hidden indices.
    idx = torch.tensor(secret, dtype=torch.long, device=x.device)
    return x.index_select(1, idx).sum(dim=1).remainder(2).to(torch.float32)


def make_online_batch(secret, n_features: int, batch_size: int, generator):
    # Fresh i.i.d. uniform binary batch -> unbiased sample of the same population gradient.
    x = torch.randint(0, 2, (batch_size, n_features), generator=generator).float()
    return x, parity_labels(x, secret)


def make_test_set(secret, n_features: int, test_size: int, generator):
    x = torch.randint(0, 2, (test_size, n_features), generator=generator).float()
    return x, parity_labels(x, secret)


def get_optimizer_config() -> dict[str, float]:
    return {"lr": 1e-3, "wd": 1e-2, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


def state_probe(model: nn.Sequential, init_state) -> float:
    # ||W_t - W_0||_inf on the first layer: rises with the drift while loss stays flat.
    W0 = init_state["0.weight"]
    W = model[0].weight.detach()
    return (W - W0).abs().max().item()


def train_parity(secret, n_features, width=512, batch_size=128, steps=100_000,
                 test_size=16_384, seed=0, device="cpu"):
    model = build_model(n_features, width).to(device)
    init_model(model)
    init_state = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt_cfg = get_optimizer_config()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["wd"],
    )
    criterion = nn.BCELoss()
    train_gen = torch.Generator().manual_seed(seed)
    test_gen = torch.Generator().manual_seed(seed + 1)
    test_x, test_y = make_test_set(secret, n_features, test_size, test_gen)

    for t in range(1, steps + 1):
        xb, yb = make_online_batch(secret, n_features, batch_size, train_gen)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xb).view(-1), yb)
        loss.backward()
        opt.step()
        if t % 250 == 0:
            print(f"step {t}  loss {loss.item():.4f}  rho {state_probe(model, init_state):.4f}")

    with torch.no_grad():
        preds = model(test_x.to(device)).view(-1)
        acc = ((preds >= 0.5) == (test_y.to(device) >= 0.5)).float().mean().item()
    return model, acc
```
