# Context: relation-aware stock trend forecasting (circa 2020-2021)

## Research question

Stocks do not move independently. Companies in the same sector, industry, or line of business
co-move; they react jointly to macro shocks; information flows between them through institutional
holdings and news. The concrete goal is a model that, given each stock's recent price/volume
history at date `t` and a set of human-curated concepts that group stocks (sector, industry, main
business), produces a next-day return signal that ranks the cross-section of stocks. The open
question is *how* the cross-stock structure carried by those concepts should enter a temporal
predictor. The model must fit a fixed benchmarking pipeline — fixed features, labels, splits, and
backtest — changing only the network that maps a day's stock features and concept membership to one
prediction per stock.

## Background

**Single-stock technical analysis.** The oldest line predicts a stock's trend from its own
historical market data — trading price and volume. Linear, stationary models — autoregression (AR)
and ARIMA — were standard. Deep sequence models followed: recurrent networks, and in particular the
Long Short-Term Memory network (Hochreiter & Schmidhuber 1997) and the Gated Recurrent Unit
(Chung et al. 2014), capture long-range dependencies in the price/volume series. Variants add
structure: State Frequency Memory (Zhang et al. 2017) decomposes the hidden state into frequency
components to model multi-frequency trading patterns; an attentive LSTM (Feng et al. 2019)
aggregates over all past hidden states; adversarial training (the same line) simulates the
stochasticity of price data. Each stock is encoded in isolation.

**Observed cross-stock behavior.** Two properties of the market are visible before any model is
built. A stock's relevance to a given theme is time-varying: a company that a membership table
assigns to several themes can have its trend driven by different ones at different times, so the
same fixed membership carries different amounts of signal on different dates. And the set of
concepts an analyst has written down is never complete — themes emerge faster than they are
catalogued, stocks that share no curated tag can begin to co-move when an unlabeled theme becomes
salient, and newly listed names sit outside the curated graph until someone fills in their tags.

**Graph neural networks for relational structure.** The general machinery for letting entities
exchange information along a relation graph had matured: the graph convolutional network
(Kipf & Welling 2017) averages each node's neighbors through a normalized adjacency, and the graph
attention network (Veličković et al. 2018) replaces that fixed averaging with learned per-edge
attention weights — for node `i` it computes `e_ij = a(W h_i, W h_j)` for each neighbor `j`,
normalizes `α_ij = softmax_{j∈N_i} e_ij`, and outputs `h'_i = σ(Σ_j α_ij W h_j)`. The attention is
masked to the graph: coefficients are computed only for `j` in `i`'s predefined neighborhood `N_i`.
These are the off-the-shelf relational primitives for a relation-aware stock model.

**Residual decomposition in time-series forecasting.** A recent architectural idea: in a deep
forecasting stack (Oreshkin et al. 2020), each block emits *two* outputs — a forward forecast `ŷ_ℓ`
and a "backcast" `x̂_ℓ`, the block's reconstruction of its own input. Blocks are chained by a
residual on the input, `x_ℓ = x_{ℓ-1} − x̂_{ℓ-1}`, so a block sees only what the previous blocks did
not reconstruct, and the partial forecasts are summed, `ŷ = Σ_ℓ ŷ_ℓ`. It was built for
trend/seasonality decomposition of a single series.

## Baselines

**LSTM / GRU single-stock encoders (Hochreiter & Schmidhuber 1997; Chung et al. 2014).** Encode a
stock's 60-day, 6-channel price/volume window with a recurrent net and read out the last hidden
state to predict the trend.

**State Frequency Memory; attentive LSTM; Transformer encoders (Zhang et al. 2017; Feng et al.
2019; Ding et al. 2020).** More expressive single-stock encoders — frequency decomposition,
temporal attention over hidden states, self-attention over the time axis.

**GCN / GAT over a predefined stock-relation graph (Kipf & Welling 2017; Veličković et al. 2018,
applied to stocks).** Build a stock graph whose edges connect two stocks when they share a curated
relation (e.g. the same industry), encode each stock with a GRU, then pass messages along that graph
— GCN with fixed normalized weights, GAT with learned attention masked to the curated neighborhood.

**Hierarchical relation attention (Kim et al. 2019).** Stack two attention levels — one selecting
among multiple curated *relation types*, one aggregating neighbors within a type — over a
multi-relation predefined graph.

**Temporal relational ranking (Feng et al. 2019, TOIS).** Couple a temporal encoder with a
relation-graph convolution over sector and knowledge-base relations, trained with a ranking
objective for the top-of-book.

## Evaluation settings

- **Universes:** the CSI 100 and CSI 300 — the 100 and 300 largest stocks by market capitalization
  in the China A-share market. Membership rebalances over time, so the set of stocks and their
  concept tags change across dates.
- **Features:** Alpha360 from the qlib platform — for each stock on date `t`, 6 raw daily series
  (opening, closing, highest, lowest, VWAP, volume) looked back over 60 days, giving a
  360-dimensional vector that reshapes to `[60, 6]` for a sequence encoder.
- **Auxiliary structure:** a stock-concept membership (industry and main-business tags, collected
  from Tushare), exposed to the model as a membership matrix. The number of concepts per day is
  large and varies (hundreds to over a thousand).
- **Label:** the next-day return, `d_i^t = (Price_i^{t+1} − Price_i^t)/Price_i^t`, with closing
  price; labels are cross-sectionally normalized per date.
- **Splits:** by time — train 2007–2014, validation 2015–2016, test 2017–2020 — fixed.
- **Batching:** one date's full cross-section of stocks is a batch, because the relational
  computation is over all stocks present on that day.
- **Metrics:** Information Coefficient (Pearson correlation of prediction vs. label) and Rank IC
  (Spearman), averaged over days; Precision@N for N in {3,5,10,30}; and a top-`k`/drop portfolio
  backtest reporting cumulative/annualized return, information ratio, and max drawdown.

## Code framework

The model plugs into the qlib benchmarking harness: the dataset handler yields, per date, the
matrix of Alpha360 stock features and the stock-concept membership matrix; the optimizer (Adam) and
the MSE loss already exist; the training loop iterates over daily batches; the backtest is fixed.
What is not settled is the network that turns a day's stock features and the membership matrix into
one prediction per stock. A per-stock temporal encoder is uncontroversial, so the scaffold fixes a
recurrent encoder and leaves a single empty slot for everything that consumes the per-stock
embeddings and the membership matrix.

```python
import torch
import torch.nn as nn


class StockFeatureEncoder(nn.Module):
    """Per-stock temporal encoder. Maps a [N, 6*60] day of Alpha360 features to a
    [N, hidden] embedding by reshaping to a [N, 60, 6] sequence and reading out the
    last recurrent hidden state. This part is uncontroversial."""

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)   # [N, 6, 60]
        x = x.permute(0, 2, 1)                    # [N, 60, 6]
        out, _ = self.rnn(x)
        return out[:, -1, :]                      # [N, hidden]: per-stock embedding


class StockTrendModel(nn.Module):
    """Maps a day's stock embeddings (and the stock-concept membership matrix) to one
    scalar prediction per stock."""

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = StockFeatureEncoder(d_feat, hidden_size, num_layers, dropout)
        # TODO: the architecture we will design.

    def forward(self, x, concept_matrix):
        x_hidden = self.encoder(x)               # [N, hidden]
        return self.day_model(x_hidden, concept_matrix)

    def day_model(self, x_hidden, concept_matrix):
        pass

def train_epoch(model, optimizer, daily_batches):
    model.train()
    for feature, concept_matrix, label in daily_batches:   # one date = one batch
        pred = model(feature, concept_matrix)
        mask = ~torch.isnan(label)
        loss = ((pred[mask] - label[mask]) ** 2).mean()     # per-day mean MSE
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
```

The encoder and the training loop are fixed; the single empty slot maps a day's stock embeddings
plus the membership matrix to one prediction per stock.
