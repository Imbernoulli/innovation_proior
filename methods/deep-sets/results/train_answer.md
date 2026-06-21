The problem is learning from sets: inputs are unordered, variable-size collections such as point clouds, galaxy clusters, or bags of digits. A model must produce the same answer no matter how the elements are ordered, and it must generalize across different set sizes. The usual neural-network tools break this symmetry by construction. A feed-forward net expects a fixed-dimensional vector with fixed coordinates, and an RNN processes the elements in a chosen order, so different permutations or unseen lengths change the result. Pooling heuristics give invariance, but without a characterization of all admissible functions it is unclear whether they are expressive enough or whether they silently reintroduce ordering bias. Kernel methods on distributions can be invariant, yet they scale quadratically or cubically with the number of sets and keep the kernel fixed rather than learning it end to end.

The right starting point is to ask what form every permutation-invariant function must take. If each element is mapped through a shared function phi and the results are summed, the sum is automatically order-independent, so rho(sum phi(x)) is invariant. The deeper fact is that this decomposition is also necessary in the countable case: one can encode a set as a sum of positional codes, recover the set from the sum, and let rho apply the target function to the recovered set. For fixed-size continuous inputs, taking phi(x) = [1, x, x^2, ..., x^M] makes the sum-of-powers embedding injective by the Newton-Girard identities, and the inverse from power sums back to the set is continuous. Universal approximation then follows from the fact that symmetric polynomials are dense among continuous invariant functions. Thus the architecture is not a convenient trick; it is the canonical form forced by permutation invariance.

The method is Deep Sets. The invariant Deep Sets model has three pieces: a shared per-element encoder phi, a commutative pooling operation over the set, and a readout network rho. Every element is processed with the same weights, because using element-specific weights would let the model distinguish positions and violate invariance. The encoder turns each element into a feature vector, the pool aggregates these vectors with a commutative reduction such as sum, mean, or max, and the readout maps the pooled representation to the desired output. Because the pool ignores order, the whole model is invariant by construction and accepts any set size. For tasks where each element needs its own prediction, the model should instead be permutation equivariant, meaning that permuting the input permutes the output in the same way. The only permutation-equivariant linear layer is the two-parameter tied form, which can be written as transforming each element and subtracting a broadcast pooled summary of the whole set. Stacking such layers keeps equivariance, and appending a final commutative pool turns the stack into an invariant set-to-label model.

Deep Sets therefore unifies the two regimes: set-to-label tasks use an invariant encoder-pool-readout stack, while set-to-per-element tasks use an equivariant stack followed by a pool when a single output is needed. The code below implements both forms in PyTorch. The invariant variant sums the per-element embeddings, with optional masking for variable-size batches. The equivariant variant uses the unique tied layer and max-pools between elements, then max-pools over the set at the end for classification.

```python
import torch
import torch.nn as nn

class InvariantDeepSet(nn.Module):
    """f(X) = rho(sum_x phi(x)): invariant by construction."""
    def __init__(self, in_dim, phi_dim=64, hidden=128, out_dim=1, pool='sum'):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, phi_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.pool = pool

    def forward(self, x, mask=None):
        # x: (batch, M, in_dim)
        h = self.phi(x)                       # (batch, M, phi_dim)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        if self.pool == 'sum':
            s = h.sum(dim=1)
        elif self.pool == 'mean':
            if mask is not None:
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                s = h.sum(dim=1) / counts
            else:
                s = h.mean(dim=1)
        else:
            s, _ = h.max(dim=1)
        return self.rho(s)


class PermEquivariant(nn.Module):
    """The only permutation-equivariant linear layer: self term minus broadcast pool."""
    def __init__(self, in_dim, out_dim, pool='max'):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.pool = pool

    def forward(self, x):
        # x: (batch, M, in_dim)
        if self.pool == 'max':
            p, _ = x.max(dim=1, keepdim=True)
        else:
            p = x.mean(dim=1, keepdim=True)
        return self.Gamma(x) - self.Lambda(p)


class EquivariantDeepSet(nn.Module):
    """Stacked equivariant layers with a final pool for set-level classification."""
    def __init__(self, in_dim, d=256, num_classes=40, pool='max'):
        super().__init__()
        self.phi = nn.Sequential(
            PermEquivariant(in_dim, d, pool), nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
        )
        self.rho = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d, d),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d, num_classes),
        )

    def forward(self, x):
        h = self.phi(x)
        s, _ = h.max(dim=1)
        return self.rho(s)
```
