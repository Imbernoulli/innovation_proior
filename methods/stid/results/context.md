# Context: graph-heavy multivariate time-series forecasting

## Research question

Given a multivariate time series with `N` correlated variables, predict the next `F`
steps from the last `P` observed steps. In the traffic setting, the variables are
loop detectors or road segments; in the electricity setting, they are clients or
meters. The practical question is whether the forecasting architecture really has
to model a spatial graph and a temporal sequence at every layer, or whether a much
smaller direct forecaster can match the same benchmark protocol.

The target output is still the standard direct multi-step panel forecast:
`Y_hat in R^{F x N}` for every sample. The model may use the observed values and
calendar/time covariates available at the forecast origin, but it must not use
future target values.

## Data and task

The benchmark data are arranged as sliding windows. A processed sample has
historical input `history_data` with shape `[B, P, N, C]` and future target
`future_data` with shape `[B, F, N, C]`. Channel `0` is the value to predict.
For traffic datasets, channels `1` and `2` are normalized time-of-day and
day-of-week features, tiled across nodes. In the canonical preprocessing,
time-of-day is generated as `i % steps_per_day / steps_per_day`, so it lies in
`{0, ..., steps_per_day - 1} / steps_per_day`; day-of-week lies in `{0, ..., 6}/7`.

The evaluation protocol considered here uses PEMS04, PEMS07, PEMS08, PEMS-BAY, and Electricity.
PEMS04/07/08 and PEMS-BAY are five-minute traffic datasets; Electricity is hourly.
The forecasting horizon is `F = 12` for all of them. The historical length is
`P = 12` for the traffic datasets and `P = 168` for Electricity. PEMS-BAY uses a
`7:1:2` train/validation/test split, while the other listed datasets use `6:2:2`.

## Existing model family

The dominant models are Spatial-Temporal Graph Neural Networks. They combine a
graph module for cross-variable structure with a temporal module for sequence
evolution. DCRNN uses bidirectional diffusion random walks inside a recurrent
encoder-decoder with scheduled sampling. STGCN stacks temporal convolutions around
graph convolutions. Graph WaveNet learns an adaptive dependency matrix from node
embeddings and combines it with dilated temporal convolutions. AGCRN learns
node-specific graph/recurrent parameters and a data-adaptive graph.

These designs are strong, but they repeatedly pay for graph construction or graph
learning, message passing, and recurrent or convolutional temporal processing.
Some need a pre-defined road graph; others avoid that graph by learning another
spatial relation. The baseline question is therefore not whether spatial and
temporal information matter; it is whether this much machinery is the only way to
make those distinctions available to a regressor.

## Failure mode to explain

A channel-shared direct regressor sees each variable's recent values through the
same weights. If two variables have nearly the same recent value trace, the
regressor receives nearly the same value input for both. Yet those two variables
can have different future traces because they occupy different locations in the
system. The same issue appears along the time axis: similar recent traces can
continue differently at different times of day or on different days of week.

Any simpler model has to confront this collision directly. It cannot rely on
universal approximation alone if the representation collapses samples whose
future behavior is different. The open design slot is therefore a direct
forecaster whose input representation does not force these cases to share the
same continuation.

## Evaluation and scaffold

Use the standard metrics MAE, RMSE, and MAPE on inverse-transformed values. The
BasicTS-style runner passes selected forward features to the model and compares
the model prediction against `TARGET_FEATURES = [0]`. The architecture slot is the
module from a normalized history window to a normalized future value window.

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """Direct multivariate forecaster.

    Input:
        history_data: [batch, P, num_nodes, num_features]
        future_data:  [batch, F, num_nodes, num_features]  # decoder/time features only

    Output:
        prediction:   [batch, F, num_nodes, 1]
    """

    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        # TODO: choose the direct architecture.

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: produce [batch, output_len, num_nodes, 1].
        raise NotImplementedError
```
