Let me start from the failure mode. I have two inputs that live in different spaces, say a question and an image, or a class label and a noised image, and I want the first one to change how the network processes the second. The usual move is to make an embedding of the conditioning signal, make features from the data, fuse the two near the top, and ask the final classifier or decoder to sort everything out. That feels too late. If the image pipeline has already built the same intermediate representation for "red cubes" and "leftmost spheres", then the question is only selecting from a representation that was not built for the question. I want the conditioning signal to reach into the computation while the features are being formed, and I want that without paying for a whole separate network or a hand-built reasoning module.

The simplest thing I can move earlier is concatenation. Broadcast a conditioning vector `z` into constant feature maps and concatenate it to the feature vector `F` before a convolution or dense layer. But a convolution is still linear in its local input, so if a linear map `W` follows the concatenation I should be able to split it across the two blocks of the stacked vector:

```text
W [F; z] = W_F F + W_z z.
```

I want to be sure this split is exact and not just suggestive, so let me check it on small random tensors. Take `F` of width 4, `z` of width 3, and `W` of shape `5 x 7` acting on the concatenation. Forming `W [F; z]` directly and forming `W_F F + W_z z` from the two column blocks `W_F = W[:, :4]`, `W_z = W[:, 4:]` agree to floating-point error (max absolute difference about `1e-7`). And if I hold `z` fixed and resample `F`, the `W_z z` term does not move at all. So the decomposition is an identity, and the `W_z z` term genuinely depends on `z` but not on `F`; from the feature pathway's point of view it is just a conditioning-dependent offset added to each output feature. So concatenation is not a mysterious fusion mechanism. Once a linear layer follows it, it is a conditional bias, `beta(z) = W_z z`. That is useful, because a question or class can push features up or down, but it cannot rescale a feature map, shut it off by multiplication, amplify it, or reverse its sign. If this is all I do, I have chosen the additive half of the story and thrown away the multiplicative half.

The other familiar half is gating. LSTMs and channel gates multiply features by a learned factor, usually a sigmoid output in `(0, 1)`. That gives conditional scale, but only attenuation: a gate can pass or damp a feature, not amplify it above its current magnitude, not negate it, and not add a threshold shift. Concatenation gives me `W_F F + beta(z)`, an unscaled feature computation plus a conditional offset. Gating gives me `gamma(z) * F`, with `gamma` bounded and no shift. The natural object sitting between those two partial mechanisms is a per-feature affine transform, with the conditioning signal producing both knobs:

```text
conditioned feature = gamma(z) * F + beta(z).
```

Set `gamma = 1` and I get the additive-bias case that concatenation reduces to. Set `beta = 0` and force `gamma` through a sigmoid and I get ordinary gating. Let both be free real values and I keep the useful parts of both without their artificial restrictions.

This affine shape is already hiding in plain sight. Batch norm computes a normalized activation and then applies a learned per-channel scale and shift:

```text
x_hat = (F - E[F]) / sqrt(Var[F] + eps)
BN(F | gamma_c, beta_c) = gamma_c * x_hat + beta_c.
```

Instance norm changes which statistics are used, but the tail is the same. Normalization strips and stabilizes statistics, then the affine gives each channel its scale and offset back. If I want a small conditioning signal to control feature maps, that two-scalar channel control is exactly the right kind of handle.

The style-transfer evidence makes the handle hard to ignore. A single style network can share all convolutional weights across many paintings and swap only `(gamma_s, beta_s)` after instance normalization for each style id. That means a tiny per-channel affine table can carry a large fraction of the style-specific behavior. But a table indexed by `s` is too rigid for the setting I care about. A table works for a fixed finite menu; it does not naturally accept a sentence, a class embedding that should share structure with other classes, or a continuous conditioning vector. So the table has to become a learned function. For input `x_i`, I want

```text
gamma_{i,c} = f_c(x_i)
beta_{i,c}  = h_c(x_i).
```

The two functions consume the same conditioning embedding and emit vectors of the same size, so in practice I should share the computation and have one generator produce the whole pair. The target network then receives per-channel scale and shift values for each modulated layer.

Conditional batch normalization already takes one step in that direction: it maps a language embedding to `Delta_gamma` and `Delta_beta`, adds those deltas to frozen pretrained BatchNorm scalars, and applies the result inside a pretrained ResNet. That shows how a language vector can reach early visual layers with very few parameters. But I should separate the pieces that are essential from the pieces that belong only to that retrofit. The frozen ResNet is not essential if I am training the target network directly. The BatchNorm wrapper is not essential to the affine formula either; `gamma * F + beta` is defined on any feature tensor. Normalization may still help training, but the conditioning operation itself does not require the preceding statistics computation. So the cleaner primitive is the affine itself, placed wherever the architecture has a useful feature map to condition.

There is one detail in that delta parameterization I want to understand on its own terms, not just keep out of deference to a pretrained BatchNorm layer. Suppose I let the generator emit `gamma` directly and its outputs are zero-centered at initialization, so `gamma` starts near zero. What actually happens on the forward and backward passes? Let me trace it. The forward path is `gamma * F + beta`, so with `gamma` near zero the output collapses to roughly `beta`: I checked this with `gamma = 0.01`, and the output-to-input ratio came out around `0.01`, i.e. the feature map is almost entirely erased before it is even used. The backward pass is worse than I first assumed: the local derivative of the output with respect to the incoming feature is exactly `gamma`, so the gradient that flows back into the feature pathway is also scaled by `gamma`. Computing it autograd-style with `gamma = 0.01` gives `dout/dF` of about `0.01` as well — the conditioning layer chokes off both the signal and its gradient at the same time. The fix is to notice that the identity operation is not `gamma = 0`; it is `gamma = 1, beta = 0`. So I should make the generator emit a residual scale:

```text
gamma_{i,c} = 1 + Delta_gamma_{i,c}.
```

I re-ran the same trace with `gamma = 1 + Delta` and `Delta = 0.01`: now the forward ratio is about `1.01` and `dout/dF` is about `1.01`, so a near-zero generator output starts as the identity affine rather than a feature-killing multiplier, and the gradient passes through essentially untouched. `beta` needs no matching offset because `beta = 0` is already the identity shift.

I also need to keep both knobs unrestricted, and I want to be precise about what "negative scale" actually buys, because it is tempting to dismiss it as just a small gate. A free `beta` is more than a bias when a ReLU follows it: it moves the threshold, so the condition can decide which values survive. A free `gamma` can attenuate, amplify, shut a channel off at `gamma = 0`, or negate it. With a following ReLU, the unit is active when `gamma * F + beta > 0`. If `gamma > 0`, that means `F > -beta / gamma`; if `gamma < 0`, dividing by a negative flips the inequality and the unit is active when `F < -beta / gamma`. Let me make that concrete with `beta = 0.5`. For `gamma = 2` the threshold is `-0.25` and the surviving inputs are exactly the `F` values above `-0.25`; I enumerated `F` on a grid and the analytic predicate `F > -0.25` matched the actual `gamma*F+beta > 0` test on every point. For `gamma = -2` the threshold is `+0.25` and the surviving inputs are exactly the `F` values below `+0.25` — again an exact match on the grid, and the surviving set is the opposite tail of the feature distribution. So a negative scale is not just a smaller gate. It changes which side of the feature distribution can pass onward. If I clamp `gamma` into `(0, 1)`, I only get attenuation. If I clamp it to `(-1, 1)`, I lose amplification. If I force it positive, I lose negation. The unrestricted affine keeps the full menu: shift, threshold, amplify, suppress, zero, and sign flip.

The cost is exactly what makes the operation usable throughout a network. For a feature map `F_{i,c}` with spatial dimensions `H x W`, the condition supplies one `gamma_{i,c}` and one `beta_{i,c}` for the whole channel, and those scalars are broadcast over all spatial locations:

```text
FiLM(F_{i,c} | gamma_{i,c}, beta_{i,c}) = gamma_{i,c} * F_{i,c} + beta_{i,c}.
```

That is two numbers per modulated feature map. The number of emitted modulation parameters depends on channel count and number of modulated layers, not on image resolution. A pairwise relational module pays more as the spatial grid grows; this channel-wise affine does not. That is why I can afford to apply it early and repeatedly.

The generator can be an RNN over a question, a linear projection from a class embedding, or any other network that emits the two vectors. The modulation layer itself should not care. Its whole job is to receive `x`, `gammas`, and `betas`, broadcast the two `[B, C]` tensors over `H, W`, and return the affine result:

```python
import torch.nn as nn


class FiLM(nn.Module):
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas
```

Before I trust this layer I want to run it once on a `[2, 3, 4, 4]` tensor and confirm it does what the formula says. Picking out one example and one channel, the layer's output over all 16 spatial positions equals `gammas[b,c] * x[b,c] + betas[b,c]` exactly — a single scale and a single shift reused at every position, which is the whole point. The modulation it consumed was `2 * C = 6` numbers regardless of the `4 x 4` grid, confirming the resolution-independence. And if I feed `gammas = 1`, the output reduces exactly to `x + betas`, the pure additive-bias case I started from.

That separation also keeps the concatenation baseline honest. If I want true concat conditioning in the visual-reasoning code, I should not pretend it is calling the affine layer. I should take the conditioning parameters that would have become `(gamma, beta)`, expand them as constant spatial maps, and concatenate them before the convolution. If I want the affine path, I split the same generator output into `gammas` and `betas` and pass them to the layer:

```python
if self.condition_method == 'concat':
    cond_params = film[:, :, :2 * self.module_dim]
    cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(
        cond_params.size() + x.size()[-2:]
    )
else:
    gammas, betas = torch.split(
        film[:, :, :2 * self.module_dim], self.module_dim, dim=-1
    )
```

The additive-bias corner I started from is still important because it is the cheapest baseline. If I fix `gamma = 1`, the modulated layer output becomes its ordinary feature computation plus `beta(cond)`, exactly the conditional-bias form that follows from concatenation after the linear map is split. In the diffusion UNet scaffold, the residual blocks already consume a side-conditioning vector derived from the timestep embedding. A projected class embedding can ride that same route by being added to the timestep embedding, and then no extra hidden-state conditioner is needed:

```python
import torch.nn as nn


def prepare_conditioning(time_emb, class_emb):
    return time_emb + class_emb


class ClassConditioner(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()

    def forward(self, h, class_emb):
        return h
```
