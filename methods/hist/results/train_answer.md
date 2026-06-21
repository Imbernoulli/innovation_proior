Stock trend forecasting is fundamentally cross-sectional: a company does not trade in isolation, and its next-day return is shaped by the sector, industry, and latent themes that make groups of stocks co-move. The concrete task is to predict next-day returns from a 60-day window of six daily price and volume channels per stock, together with a binary stock-concept membership matrix that records which curated concepts each stock belongs to. Treating every stock with its own LSTM or GRU ignores this cross-sectional structure. The obvious patch is a graph neural network over a predefined stock graph, such as GCN or GAT, where an edge exists whenever two stocks share a curated concept. But that freezes the edge set: a stock assigned to both e-commerce and cloud computing is forced to propagate information through both links with the same neighborhood every day, even when only one theme is currently relevant. It also cannot use themes the analysts never wrote down, and newly listed or untagged stocks sit outside the graph entirely. We need a representation in which concepts are first-class objects whose relevance is recomputed from the day's embeddings, plus a way to discover hidden themes from the data itself.

The method is HIST, a graph-based framework for stock trend forecasting via mining concept-oriented shared information. HIST replaces the fixed stock-stock graph with a stock-concept bipartite graph whose edge strengths are derived, not frozen. Each date starts by encoding every stock's feature window with a recurrent net, producing an embedding x_i. The predefined concept module initializes each concept vector as a smoothed aggregate of the stocks tagged with it; singleton and empty concepts are handled safely so nothing explodes. Rather than keeping those weights fixed, each stock then computes cosine similarity to every concept vector and normalizes across concepts, so a stock can borrow strongly from a theme it currently resembles even if its membership row attaches it weakly. The resulting shared information is passed through a learned linear layer and split into two heads: a forecast that contributes to the final return prediction, and a backcast that represents the part of x_i the module has explained.

The hidden concept module runs on the residual left after subtracting the predefined backcast from the original embedding. That residual contains cross-stock commonality the curated concepts could not explain. It discovers unlabeled concepts by seeding one concept per stock, computing all pairwise cosines among the residual embeddings, zeroing the diagonal so a stock does not trivially choose itself, and assigning each stock to its single most similar seed. Seeds that attract no other stock are discarded, and the surviving seeds become hidden concepts. Their representations are aggregated from the stocks that chose them, then sent back to all stocks through the same cosine-weighted, softmax-normalized attention over concepts. This yields a second forecast and a second backcast.

Finally, an individual information module takes the residual after both backcasts are subtracted, representing the part of the stock that neither shared module could account for, and applies a nonlinear head to produce the third forecast. The three forecasts are summed and passed through one linear layer to produce the predicted next-day return. The backcast-subtract, forecast-sum structure prevents the three channels from triple-counting the same sector move and preserves the idiosyncratic spread that ranking tasks need. Training uses daily cross-section batches, because the relational computation is over all stocks present on one date. The loss is mean squared error against cross-sectionally normalized next-day returns, optimized with Adam, gradient value clipping at 3.0, and early stopping on validation information coefficient. Typical settings use a 2-layer GRU or LSTM with hidden size 64, learning rate 1e-4, and the Alpha360 features from the qlib platform.

```python
import torch
import torch.nn as nn


class HISTModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        if base_model == "GRU":
            self.rnn = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc_es = nn.Linear(hidden_size, hidden_size)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        for layer in [self.fc_es, self.fc_is, self.fc_es_fore, self.fc_is_fore,
                      self.fc_es_back, self.fc_is_back, self.fc_indi]:
            nn.init.xavier_uniform_(layer.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_t2s = nn.Softmax(dim=1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def cal_cos_similarity(self, x, y):
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        return xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)

    def forward(self, x, concept_matrix):
        device = x.device

        x_hidden = x.reshape(len(x), self.d_feat, -1)
        x_hidden = x_hidden.permute(0, 2, 1)
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Predefined concept module
        stock_to_concept = concept_matrix
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(
            stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        stock_to_concept_sum = stock_to_concept_sum + torch.ones(
            stock_to_concept.shape[0], stock_to_concept.shape[1], device=device)
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        hidden = hidden[hidden.sum(1) != 0]

        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)
        e_shared_info = self.fc_es(concept_to_stock.mm(hidden))
        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.leaky_relu(self.fc_es_fore(e_shared_info))

        # Hidden concept module on the residual
        i_shared_info = x_hidden - e_shared_back
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, i_shared_info)
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        i_stock_to_concept = i_stock_to_concept * (
            torch.ones(dim, dim, device=device) - torch.eye(dim, device=device))
        row = torch.arange(dim, device=device).long()
        column = i_stock_to_concept.max(1)[1].long()
        value = i_stock_to_concept.max(1)[0]
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed(
            (i_stock_to_concept.sum(0) != 0).float() * diag)
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
        hidden = hidden[hidden.sum(1) != 0]

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = self.fc_is(i_concept_to_stock.mm(hidden))
        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.leaky_relu(self.fc_is_fore(i_shared_info))

        # Individual information module
        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = self.leaky_relu(self.fc_indi(individual_info))

        all_info = output_es + output_is + output_indi
        return self.fc_out(all_info).squeeze()
```
