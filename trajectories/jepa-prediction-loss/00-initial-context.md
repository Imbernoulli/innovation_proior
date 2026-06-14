## Research question

Temporal JEPA on Moving MNIST learns an encoder and an autoregressive predictor jointly: the encoder
turns each frame into a spatial latent feature map, and the predictor rolls those latents forward in
time. Training drives one scalar — the **prediction loss** — that measures how far the predicted latent
feature map is from the encoder's target latent feature map, summed with a Variance–Covariance
anti-collapse regularizer the trainer adds separately. The single thing being designed is that
prediction term: a function `forward(state, predicted)` over two same-shaped `[B, C, T, H, W]` latent
tensors, returning a scalar that is small when the prediction matches the target. Everything else —
encoder, predictor, regularizer, optimizer, the 50-epoch schedule, the detection-AP probe — is fixed.

## Prior art before the first rung (latent-prediction lineage)

The substrate the first rung fills is itself the resolution of a line of self-supervised
representation methods; these are the ancestors the ladder reacts to.

- **Predict-the-pixels autoencoding / next-frame prediction.** Reconstruct the raw future frame and
  minimize pixel error. It forces the model to spend capacity on every pixel — backgrounds, texture,
  uncontrollable flicker — most of which is irrelevant to dynamics. Gap: the loss is dominated by
  high-frequency pixel detail, not by the structure that matters for prediction.
- **Contrastive instance discrimination (SimCLR, Chen et al. 2020; MoCo, He et al. 2020).** Pull
  augmented views of one instance together and push other instances apart in latent space. It learns
  good features but needs large batches / a memory bank of explicit negatives, and the InfoNCE
  objective is a classification surrogate, not a direct match between a *predicted* and a *target*
  representation. Gap: negatives and batch-size dependence; no notion of predicting a future state.
- **Joint-Embedding Predictive Architecture (I-JEPA, Assran et al. 2023).** Drop pixels and negatives
  both: encode context and target, and train a predictor to match the *target's latent
  representation* from the context, with a stop-gradient / momentum target to avoid collapse. The
  prediction happens entirely in feature space. The temporal extension used here makes the predictor
  autoregressive over time on a frame sequence. Gap left open for this task: I-JEPA fixes the
  *architecture* (encoder, predictor, target) but leaves the *prediction cost* itself a plain
  squared error over latent coordinates — the component this ladder redesigns.
- **VICReg (Bardes, Ponce, LeCun 2022).** Replace negatives and stop-gradient with explicit
  regularizers: a variance hinge that keeps each feature dimension's std above a floor, a covariance
  penalty that decorrelates dimensions, and an invariance term that pulls the two embeddings together.
  Here the variance + covariance pair is the anti-collapse regularizer the trainer adds *separately*;
  the invariance term — the part that actually matches predicted to target — is exactly the prediction
  loss being redesigned, and VICReg's default for it is plain MSE.

So the architecture, the anti-collapse regularizer, and the autoregressive roll-out are all settled.
The one undetermined choice the ladder climbs is the per-coordinate function that turns the residual
between a predicted and a target latent feature map into the scalar the predictor descends.

## The fixed substrate

A self-contained Video-JEPA training script is frozen and must not be touched. It builds: a `ResNet5`
encoder shared between context and target (producing a `[B, C, T, H, W]` latent map per frame), a
`ResUNet`-based `StateOnlyPredictor` with context length 2 that unrolls autoregressively, a `Projector`
feeding a `VCLoss(std_coeff=10, cov_coeff=100)` Variance–Covariance regularizer, and image-decoder /
detection-head probes used only for evaluation. The `JEPA.unroll(...)` method calls the prediction
loss as `predcost(state, predicted_states)` on two tensors of identical `[B, C, T, H, W]` shape; the
returned scalar is added to the regularization loss and the probe losses and backpropagated with Adam
(`lr=1e-3`) for 50 epochs. Three model widths are instantiated from `MODEL_SIZE` — small
(`henc=16,dstc=8,hpre=16`), base (`henc=32,dstc=16,hpre=32`), large (`henc=64,dstc=32,hpre=64`) — and
the same prediction-loss code must serve all three.

## The editable interface

Exactly one region is editable — the `CustomPredictionLoss` class in `custom_prediction_loss.py`
(the `__init__` and `forward` methods, plus any helper methods or imports). The contract:

```
forward(state, predicted):
    state:     [B, C, T, H, W]  target encoded representations (from the encoder)
    predicted: [B, C, T, H, W]  predicted representations (from the predictor)
    returns:   scalar loss tensor (lower = predicted closer to state)
```

Both tensors share the same shape; the scalar is added to the VC regularizer by the trainer. The
encoder emits spatial feature *maps* (not vectors), so the channel dimension `C` carries the feature
content at each spatial location, and the time dimension `T` is the autoregressive roll-out — temporal
weighting or ordering can in principle be exploited. The starting point is the scaffold default: a
no-op that returns a constant zero (a placeholder with no learning signal). Each rung replaces exactly
this `forward` and nothing else.

```python
# EDITABLE region of custom_prediction_loss.py -- default fill (placeholder, no signal)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Prediction cost function for temporal JEPA.

    Measures discrepancy between predicted and target representations in the
    latent space. Used to train the predictor network.

    Args:
        state:     [B, C, T, H, W] target encoded representations
        predicted: [B, C, T, H, W] predicted representations

    Returns:
        Scalar loss tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return torch.tensor(0.0, device=state.device, requires_grad=True)
```

## Evaluation settings

Mean detection Average Precision (AP) across prediction timesteps on Moving MNIST, higher is better.
The model trains for 50 epochs (Adam, `lr=1e-3`) and the final mean detection AP is reported. The
prediction loss is graded across the three model sizes — **small**, **base**, **large** — to test that
the loss generalizes across width, with one seed ({42}). The detection head and image decoder are
probes trained alongside but read out only at evaluation; the prediction loss is judged solely by the
downstream detection AP its learned representations support, never by its own training value.
