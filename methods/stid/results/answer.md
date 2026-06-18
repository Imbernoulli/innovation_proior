# STID, Distilled

STID, Spatial-Temporal IDentity, is a graph-free multivariate time-series forecaster. It embeds each node's own history, concatenates learned node and periodic-time identity embeddings, encodes the result with residual MLP blocks, and regresses the forecast horizon. It uses no graph convolution, attention, or recurrence.

## Core Formulas

For node `i` at forecast time `t`, embed its historical value trace:

```text
H^i_t = FC_embedding(X^i_{t-P:t}),        H^i_t in R^D
```

Attach the spatial and temporal identities:

```text
Z^i_t = H^i_t || E_i || T^TiD_t || T^DiW_t
```

where `E in R^{N x D}`, `T^TiD in R^{N_d x D}`, and `T^DiW in R^{7 x D}` are trainable tables. In the paper's all-identities-enabled notation, `Z^i_t in R^{4D}`.

Encode with residual MLP layers:

```text
(Z^i_t)^{l+1} = FC^l_2(ReLU(FC^l_1((Z^i_t)^l))) + (Z^i_t)^l
```

Regress the horizon:

```text
Y_hat^i_{t:t+F} = FC_regression((Z^i_t)^L),       Y_hat^i_{t:t+F} in R^F
```

The paper loss is:

```text
L(Y_hat, Y) = (1 / (N F)) sum_i sum_j |Y_hat^i_j - Y^i_j|
```

The BasicTS implementation uses masked MAE and masked MAE/RMSE/MAPE metrics for datasets with missing-value markers.

## Constants and Cases

For the short-paper traffic experiments, `P = 12`, `F = 12`, `D = 32`, `num_layer = 3`, `N_d = 288`, and `N_w = 7`. For Electricity, the paper uses `P = 168` and `F = 12`.

The canonical implementation is more general than the clean `4D` equation:

```text
hidden_dim =
    embed_dim
  + node_dim     * int(if_node)
  + temp_dim_tid * int(if_T_i_D)
  + temp_dim_diw * int(if_D_i_W)
```

The time table size is dataset-specific: `288` for five-minute data, `96` for fifteen-minute data, `24` for hourly data, and `1` when there is no meaningful within-day slot. The generated normalized time features are in `[0, 1)`, so multiplying by the table size and casting to integer gives valid indices.

## Canonical Implementation

This is the faithful core of the reference implementation at `GestaltCogTeam/STID`, commit `e8b313bc591bdd0101a1619962c9b503e75127c0`.

```python
import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data


class STID(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        # history_data: [B, input_len, N, C]
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)
        return prediction
```

## Faithfulness Notes

The temporal indices are read from the final history step, not the future window. The reference implementation indexes them per node as `[B, N]`; the preprocessing tiles the same time value across nodes for traffic, but the shape remains per-node. The `future_data` argument is accepted for the runner interface and is not used by this architecture.

The paper's compact formula assumes all three identity tables are enabled and have the same width. The code supports ablations by turning any identity branch off and by choosing separate widths.
