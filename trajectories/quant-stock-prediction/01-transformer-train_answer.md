The scaffold I am opening from is a Ridge regression on the flat 360-dimensional Alpha360 vector — one global linear map, one score per stock-day. Its single disqualifying weakness is that the 360 numbers are not 360 unrelated features: they are six base ratios (open, close, high, low, volume, vwap, each normalized to the latest close) unrolled across sixty trading days. A flat regressor sees `[N, 360]` and has no idea that column 5 and column 65 are the *same* ratio one day apart. The temporal structure — the entire reason a 60-day window is the input — is invisible to it. So the first thing worth trying is a model whose inductive bias is exactly "this is a length-60 sequence of 6-dimensional daily observations," whose job is to summarize that window into one forward-return score per stock-day. That is a sequence-to-one regression, and I want to start from an architecture that, in principle, lets any day in the window talk to any other day in a single hop.

I propose an **encoder-only, sequence-to-one Transformer**. The reason I reach for attention rather than a recurrence for this *first* attempt is precisely that one-hop reach. In a recurrent encoder the state at day $t$ is $h_t = f(h_{t-1}, x_t)$, so information from day 1 must survive sixty sequential transformations before it can influence the read-out at day 60, and each transformation can attenuate it. A self-attention layer instead forms, for every position, a weighted average over *all* positions at once: every day reaches every other day through a single content-based lookup, with no distance-dependent attenuation in the routing. For a window where a move sixty days ago might matter as much as yesterday's, that any-to-any reach is exactly the right bias. So the model lifts each day's 6 features to a working width, runs a couple of self-attention layers over the 60 positions, and reads out a single score.

Because the harness fixes this as an *encoder-only*, *sequence-to-one* regressor, most of the machinery of a from-scratch transduction model is simply absent here and I drop it deliberately. There is no decoder, no autoregressive generation, no target sequence — so no causal mask (every day may legitimately attend to every other day; nothing is "in the future" to hide, since the label lies strictly outside the window), no cross-attention, no tied input/output embeddings, no label smoothing, and no token vocabulary at all. The "tokens" are continuous 6-vectors, one per day. What I keep is the part that *scores*: a linear that lifts each day's 6 features to the model width, positional information so the stack is not order-blind, scaled multi-head self-attention, a position-wise feed-forward, residual-plus-LayerNorm glue, and a final linear that reads one position's representation down to one number.

Each surviving choice is forced. **Lifting the input**: a day arrives as a 6-vector, far too thin to host multi-head attention — I want to slice the representation into heads, each in its own subspace, and 6 dimensions cannot be meaningfully split. So `feature_layer = Linear(6, d_model)` lifts each day to $d_{\text{model}} = 64$. That width is the qlib Alpha360 benchmark setting and it is deliberate: 64 is wide enough to host `nhead = 2` heads of 32 dimensions each, yet small enough that the $n^2 \cdot d$ attention over only 60 positions is cheap and, more importantly, has too little capacity to memorize the noise in a notoriously low signal-to-noise regression. Financial cross-sectional prediction has an IC ceiling in the low single-digit percents; the danger is never underfitting, it is fitting noise. A narrow model is a regularizer.

**Position**: pure self-attention is permutation-equivariant — $\text{softmax}(QK^\top/\sqrt{d_k})V$ is all dot products and weighted sums over the *set* of positions, with no term that knows which day is which. Shuffle the sixty days and the output shuffles identically; the layer literally cannot tell a recent observation from an old one. For a window where recency matters, that is fatal, so I inject order with a sinusoidal positional code added to each lifted day-vector before the stack:
$$PE(pos, 2i) = \sin\!\big(pos/10000^{2i/d}\big), \qquad PE(pos, 2i+1) = \cos\!\big(pos/10000^{2i/d}\big).$$
I *add* rather than concatenate because a learned linear over the sum $e + p$ is $We + Wp$, so the downstream projections can already separate content from position into different directions of the 64-dim space; concatenation would only widen every matrix in the stack for marginal benefit. I use sinusoids rather than a learned table because each dimension pair is a sine/cosine at its own frequency, so a shift by a fixed offset $k$ is a fixed, position-independent rotation of that pair — letting a head learn a relative rule like "look a few days back" that applies uniformly along the window. At a fixed length of 60 a learned table would also work, but the sinusoid is the canonical parameter-free choice and there is no extrapolation concern either way.

**The attention primitive and its one wart.** I score query against key by a dot product, softmax it, and mix the values: $\text{softmax}(QK^\top/\sqrt{d_k})V$. The $1/\sqrt{d_k}$ is not decorative. If $q$ and $k$ components are roughly unit-variance and independent, then $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$, so its typical magnitude grows like $\sqrt{d_k}$. Feed logits of that size into a softmax and it saturates toward one-hot; its Jacobian $p_i(\delta_{ij} - p_j)$ then collapses toward zero, the attention weights stop receiving gradient, and the model freezes in a near-argmax it cannot learn its way out of. Dividing the scores by $\sqrt{d_k}$ restores unit logit variance and keeps the softmax responsive. With $d_k = 32$ here the effect is real, so the scaling stays.

**Multiple heads.** A single softmax distribution per day is a single averaged summary, and an average blurs: if a day needs to track a volume spike at one lag *and* a price reversal at another, one head smears them. So I run `nhead = 2` attentions in parallel over 32-dim projections, each free to attend to a different pattern, and concatenate. Two heads is modest, but the regime rewards restraint — more heads is more capacity to overfit a faint signal — and the qlib benchmark settled on two.

**The per-position feed-forward, the residual-plus-LayerNorm wrapping, and the depth.** Attention mixes information *across* days but does almost no nonlinear processing *within* a day — its only nonlinearity is the softmax on the weights. So between attention sublayers I apply a position-wise feed-forward identically to every day, giving the model somewhere to transform each day's mixed representation nonlinearly. I wrap each sublayer as $x + \text{Sublayer}(x)$ so the identity path keeps gradients flowing through depth, and LayerNorm — normalizing each day-vector across its own 64 features, batch-size-invariant and immune to the variable, padded batches — keeps the residual scale from drifting. I stack `num_layers = 2` such encoder layers; two is shallow on purpose, because depth multiplies capacity and on this data capacity is liability.

The read-out is the last position. After the stack I take the day-60 representation — `output.transpose(1,0)[:, -1, :]` in the loop's `[T, N, F]` convention — and a `Linear(d_model, 1)` maps it to the score. The last day rather than a pooled summary because day 60 is the most recent observation and, having attended over all sixty days, its representation is the encoder's summary of the whole window conditioned on "now." Training is masked MSE over the finite labels (NaN targets dropped), gradient value-clipping at 3.0 to cap the exploding side, and early stopping on the validation score with the best parameters restored. The flat vector is reshaped into the sequence with `src.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`, recovering the time axis the attention needs.

The one hyperparameter I am most wary of is the optimization, because the harness gives this most-init-sensitive architecture the least protective loop. The benchmark sets `dropout = 0`, Adam at a very small `lr = 1e-4` (an order of magnitude below the LSTM's $10^{-3}$), weight decay $10^{-3}$, batch size 2048, MSE loss, early-stop patience 5. Transformers are brittle at initialization: random projections can make the dot-product logits large, the softmax saturates early (the same Jacobian collapse, now at the *start* of training), the first gradients are large and noisy, and Adam — dividing by a second-moment estimate built from only a few early samples — records those noisy gradients and distorts the step scale for many steps after. A from-scratch transduction transformer defuses this with a warmup-then-inverse-sqrt schedule; the harness runs *plain* constant-rate Adam with no warmup. So the tiny constant `lr` and patience-5 early stop are the only guards, and they may not suffice — a constant small rate that never warms up can crawl too slowly to escape a bad initial basin, or, on a few bad early batches, still take the destabilizing step warmup was meant to prevent. The honest expectation is that this is the *risky* opening: if the constant $10^{-4}$ Adam threads the needle, the attention bias could land IC in the low-single-digit range a working temporal model reaches; but the failure I brace for is that it does not catch and early-stops on a barely-trained model whose ranking is near noise — IC near 0.01 rather than 0.04–0.05, and, the sharper tell, a *negative* portfolio information ratio, since a noise ranking still gets forced through TopkDropout to hold 50 names and churn 5 a day, paying transaction costs on noise. If that is what I see, the next rung needs not more architecture but a learner robust to this data without delicate optimization.

```python
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
import copy
import math
import os
import torch.optim as optim


class PositionalEncoding(nn.Module):
    """Positional encoding — verbatim from qlib/contrib/model/pytorch_transformer.py."""

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    """Transformer network — verbatim from qlib/contrib/model/pytorch_transformer.py.

    Reshapes flat Alpha360 features internally:
    [N, F*T] -> [N, d_feat, T] -> [N, T, d_feat]
    """

    def __init__(
        self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])

        return output.squeeze()


class CustomModel(Model):
    """Transformer model — faithful to qlib's official TransformerModel
    (pytorch_transformer.py).

    Uses DatasetH with Alpha360 features. The Transformer reshapes the flat
    360-dim feature vector internally: [N, 360] -> [N, 6, 60] -> [N, 60, 6].

    Hyperparameters from official benchmark:
    examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml
    """

    def __init__(self):
        super().__init__()
        # Official Alpha360 benchmark hyperparameters
        self.d_feat = 6
        self.d_model = 64
        self.nhead = 2
        self.num_layers = 2
        self.dropout = 0
        self.n_epochs = 100
        self.lr = 0.0001
        self.metric = ""
        self.batch_size = 2048
        self.early_stop = 5
        self.loss = "mse"
        self.reg = 1e-3
        self.seed = int(os.environ.get("SEED", "42"))
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(
            self.d_feat,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dropout,
            self.device,
        )
        self.train_optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.reg
        )
        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print(
                "Epoch%d: train %.6f, valid %.6f"
                % (step, train_score, val_score)
            )

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = (
                torch.from_numpy(x_values[begin:end]).float().to(self.device)
            )

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
```
