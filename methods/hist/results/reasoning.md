Let me start from what actually goes wrong when I predict stock trends one stock at a time. I have, for each stock `i` on date `t`, a window of recent price and volume — six raw series over sixty days — and I want next-day return. A recurrent encoder handles the temporal part fine: feed the `[60, 6]` sequence through a GRU, take the last hidden state, call it `x_i` for stock `i`. GRU over LSTM because the window is long and I want the cheaper gating that still captures long-range dependence; the choice barely matters here, the encoder is the easy part. The problem is that this `x_i` only ever saw stock `i`. And stocks are not independent — companies in the same sector co-move, they react jointly to a macro shock, a theme that lifts one lifts its peers. Throwing that cross-sectional structure away has to be leaving signal on the table. So the real question is not "how do I encode a stock" but "how do I let a stock's prediction borrow from the right other stocks."

The off-the-shelf answer is a graph neural network over a stock-relation graph. I have curated concepts — sector, industry, main business — that tell me which stocks share a theme. So I could build a graph: nodes are stocks, edge between two stocks when they share a curated concept, then run graph attention. For each stock `i`, attend over its neighbors `N_i`: score `e_ij = a(W x_i, W x_j)`, normalize `α_ij = softmax_{j∈N_i} e_ij`, output `Σ_j α_ij W x_j`. The attention even learns how much each neighbor matters instead of averaging them blindly. This is the natural thing to reach for, and it's what the cross-stock baselines do.

But let me poke at it, because something about it bothers me. The attention is *masked* to the graph — I only compute `e_ij` for `j` already in `i`'s neighborhood. So the model can re-weight the edges I gave it, but the edge *set* is frozen, fixed once from the curated relations and held constant on every date. Now think about what a concept membership actually means in the market. Take a company tagged with both "cloud computing" and "e-commerce." During one regime its price is driven almost entirely by the e-commerce side; during another, by the cloud side. The membership table says it belongs to both, equally, forever. The graph says the same edges with the same potential weight every single day. Even with attention, the model is stuck choosing among a fixed neighborhood through a stock-to-stock adjacency — it can dial an existing edge up or down a little, but it can't say "today this stock's relevance to the e-commerce theme is high and to cloud is near zero," because there is no theme object in the graph at all, only stock-to-stock links, and the links it can touch are exactly the ones the curator drew. The relevance of a stock to a theme *drifts*, and a frozen, masked graph can't express drift. That's the first wall.

There's a second wall, and it's worse because it's about edges that *aren't there*. The curated concepts are incomplete. Themes emerge faster than analysts catalogue them — when some unlabeled theme suddenly becomes salient, stocks that share no curated tag start co-moving anyway. A protective-equipment maker tracking an e-commerce firm during a pandemic: there's a real shared driver, "pandemic-exposed companies," but nobody wrote that concept down, so there is no edge, and a graph that only knows curated edges can never propagate that signal. And a freshly listed stock might have *no* tags yet, sitting outside the graph entirely. So the curated graph is wrong in two ways at once: the edges it has are static when they should drift, and it's missing edges it should have.

So I can't just take the graph as given and attend over it. I need the relevance between a stock and a theme to be *computed from the current embeddings*, not read from a frozen table, so that it can drift; and I need a way to find structure the curated relations don't name. Let me think about how to get drift first, since the curated concepts at least give me a starting point there.

Maybe the object I am passing messages along is wrong. If a concept stays only an *edge* between two stocks, I never get a current representation of the concept itself. If a concept has a vector — a dynamic embedding recomputed every date from whatever stocks currently look like it — then "relevance of stock `i` to concept `k`" is just the similarity between `x_i` and that concept vector, computed fresh each day. Nothing is frozen: the same curated membership can yield wildly different relevance on different dates because the embeddings move. A bipartite graph then, stocks on one side, concepts on the other, and the edges' strengths are *derived*, not given. The question becomes: where does a concept's representation come from? It has to come from the stocks under it — a concept is, after all, defined by its member stocks. So initialize concept `k`'s representation as an aggregate of the embeddings of stocks tagged with it: `e_k = Σ_{i∈N_k} α_ki x_i`, where `N_k` is the set of stocks the curated table puts under concept `k`.

What should the weights `α_ki` be? My first instinct is a plain average, `1/|N_k|`. If market capitalization is available, the same idea can become an index-style weighted average, because large constituents dominate how a theme moves. But the benchmarking harness I need to fill only hands the model the stock features and the binary stock-concept matrix. So the code has to get a stable concept initialization from that matrix alone. Let `M_{ik}` be 1 when stock `i` carries concept `k` and 0 otherwise, and let `d_k = Σ_r M_{rk}` be the number of current stocks with that concept. For a member stock I use a smoothed degree denominator, not a raw average:

  `α_{ik}^{init} = M_{ik} / (M_{ik} d_k + 1)`.

For a non-member the numerator is zero, so the extra `+1` does nothing except keep the tensor finite. For a member it contributes `1/(d_k+1)`. That means a singleton concept does not explode into a full-strength copy of one stock, and an empty concept produces the zero row that I can drop. The initial concept vector is

  `e_k^{0} = Σ_i α_{ik}^{init} x_i`.

But this initial `e_k^0` still leans entirely on the curated membership, which I just argued is wrong in two ways. So the next operation has to stop treating membership as the final edge weight. The flaws were: (1) a stock genuinely belongs to a theme but the curator forgot the tag — a missing edge; (2) a stock is tagged to a theme but the theme barely drives it — an edge that should not carry much weight. Both are statements about similarity: a missing edge is a stock that looks like the concept yet is not connected; an over-weighted edge is a connected stock that does not look like the concept. The clean mathematical correction compares every stock to every initial concept by cosine,

  `v_ki = cos(x_i, e_k) = (x_i · e_k) / (‖x_i‖ ‖e_k‖)`,

cosine rather than the GAT-style learned scorer `a([Wx_i ‖ We_k])` because cosine is scale-free and adds no parameters per bipartite edge. It measures "does this stock currently point in the same direction as this concept," which is exactly the relevance question. If I rewrite the concept nodes themselves, the softmax axis is stocks for each fixed concept,

  `α'_ki = softmax_i v_ki = exp(v_ki) / Σ_{j∈S^t} exp(v_kj)`,

and the corrected concept is

  `e_k^{corr} = LeakyReLU(W_e Σ_{i∈S^t} α'_ki x_i + b_e)`.

Look at what this softmax-over-all-stocks does. A stock that is very similar to the concept gets a large `α'_ki` even if the curator never tagged it; a tagged but dissimilar stock gets little influence. That is the dynamic edge editing I need: missing stock-concept edges can appear through similarity, weak existing edges can fade, and the weights are recomputed every date. In the qlib path I have to land in, market values are not part of the forward signature and the separate stock-softmax correction is not present; the implementation keeps the smoothed membership initialization as the concept matrix and performs the current-embedding comparison when sending concept information back to stocks. For stock `i`, normalize the cosine scores across concepts, not across stocks,

  `β_ik = softmax_k cos(x_i, e_k^0)`,

and aggregate the concept vectors into a per-stock shared vector,

  `ŝ_i = W_s Σ_k β_ik e_k^0 + b_s`.

`ŝ_i` is "what stock `i` shares with the themes it currently resembles." The important part is the axis: `softmax_k` makes every stock choose a date-specific mixture over concepts, so a stock can borrow from a known concept even when its original membership row did not attach it strongly. First wall is handled for predefined concepts.

Second wall remains: a completely unlabeled theme has no concept node to begin with. There is nothing in the predefined concept set for a protective-equipment-and-e-commerce pandemic theme to attach to. I need to *discover* concepts from the data, with no labels. And I want to do it on the part of the signal the predefined concepts couldn't explain, otherwise I'll just rediscover the sectors I already have.

How do I get "the part the predefined module couldn't explain"? This is exactly the residual-decomposition discipline from deep forecasting: a block emits not just a forecast but a *backcast* — its reconstruction of its own input — and the next block runs on the residual `x − x̂`, so it only sees what's left. Borrow it. Let the predefined module also emit a backcast `x̂^0_i` of the stock embedding, the part of `x_i` it has accounted for via shared concept information, and feed the *residual* into the hidden-concept stage:

  `x^1_i = x_i − x̂^0_i`.

Now the hidden module operates on the leftover, so it won't just re-find the sectors — it's looking at the cross-stock commonality that survives after the known themes are stripped out. In code the backcast is a learned linear head off `ŝ_i`, `x̂^0_i = W_b ŝ_i + b_b`, while the forecast head gets the nonlinearity, `ŷ^0_i = LeakyReLU(W_f ŝ_i + b_f)`. I'll come back to what the forecasts are for.

So how do I find concepts with no labels, given the residual embeddings `x^1_i`? I don't know how many hidden concepts there are or which stocks belong to which. The most parameter-free construction is to posit exactly `n` hidden concepts, one *seeded by each stock* — initialize hidden concept `k`'s representation as `u_k = x^1_k`. That sounds wasteful (n concepts for n stocks), but the deletion step below collapses it. Now measure every stock's similarity to every seed, `γ_ki = cos(x^1_i, u_k)`, and connect each stock to its single most similar seed. If many stocks all point at the same seed, that seed *is* an emergent group — a hidden concept — and the seeds nobody points to are spurious and get deleted. So the algorithm is: connect each stock to its argmax seed, delete seeds with no incoming stock, and the survivors are the discovered concepts.

Wait — there's an immediate bug. Every stock is its own seed, so `cos(x^1_i, u_i) = cos(x^1_i, x^1_i) = 1`, the maximum possible. The argmax for stock `i` is always *itself*, every seed survives as a singleton, and no grouping happens at all. So I must remove that guaranteed self-score before the row-max: zero out the diagonal of the similarity matrix. Then stock `i` connects to the most similar remaining seed, genuinely similar stocks pile onto a shared seed, and the structure emerges. Concretely: take the `[n × n]` cosine matrix `γ`, multiply by `(1 − I)` to kill the diagonal, take each row's argmax column as that stock's chosen concept, keep only those `(row, column)` connections, and drop any column (concept) that received no connection. If I wanted a hard mathematical exclusion even when all off-diagonal cosines are negative, I would mask the diagonal with `-∞`; the qlib tensor path uses the zero diagonal mask.

One more refinement: after I've decided which seeds survive, a surviving seed should include *its own originating stock* in its membership — the stock that defined it obviously belongs to it. So if seed `k` survived (someone pointed at it), re-add the `(k, k)` self-connection with its diagonal similarity value. Now each surviving hidden concept `k` has a member set `M_k` (the stocks that chose it, plus its originator if it survived), and I aggregate their residual embeddings, weighted by similarity, into the hidden concept's representation. In the formula-level version this representation can pass through a learned LeakyReLU transform,

  `u_k = LeakyReLU(W_u Σ_{i∈M_k} γ_ki x^1_i + b_u)`.

The qlib forward keeps the surviving hidden-concept rows as the weighted residual sums, then applies the learned `fc_is` transform after concept→stock aggregation. The same concept→stock aggregation as before sends the hidden shared information back to the stocks: cosine similarity from each stock to each surviving hidden concept, softmax over concepts, weighted sum, and a learned linear transform before the forecast/backcast heads. This whole hidden-concept stage has essentially no concept-specific parameters — it's a similarity-driven clustering that runs fresh every date — which is what I want, because the concepts it finds should be free to change as the market changes.

Now repeat the residual discipline once more. The hidden module emits its own backcast `x̂^1_i` of its input, and the *individual* information is whatever survives after both shared parts are removed:

  `x^2_i = x^1_i − x̂^1_i = x_i − x̂^0_i − x̂^1_i`.

That's the part of stock `i` that is neither explained by the predefined themes nor by any discovered hidden theme — its idiosyncratic signal. I want to keep it, because a stock's own peculiarities matter for its return too; if I only used shared information I'd predict every member of a theme to move identically, which is wrong. So a third, individual module is just a nonlinear head on the residual: `ŷ^2_i = LeakyReLU(W_f^2 x^2_i + b_f^2)`.

Why subtract the backcasts at all, instead of just running three parallel modules on the same `x_i`? Because without the subtraction the three modules would all see the full embedding and redundantly re-encode the same shared trend three times — the predefined-shared, hidden-shared, and "individual" outputs would overlap, double-counting the sector move and starving the genuinely idiosyncratic part. The backcast residual *forces a decomposition*: each module operates on what the previous ones could not explain, so the three pieces are (softly) disjoint — known-theme-shared, then leftover-theme-shared, then truly individual. It also eases each downstream module's job (smaller residual to model) and gives cleaner gradient paths, which is the reason the residual trick was invented in the first place.

And that decomposition is also why the *forecasts* combine the way they do. Each module produced a forecast head — `ŷ^0_i`, `ŷ^1_i`, `ŷ^2_i` — from its slice of the signal. Since the slices add up to the original — `x = x̂^0 + x^1 = x̂^0 + x̂^1 + x^2` exactly, by the two residual subtractions — the natural recombination of their forecasts is a *sum*. So pool the three forecast vectors and read out the prediction with one final linear layer:

  `p_i = W_p (ŷ^0_i + ŷ^1_i + ŷ^2_i) + b_p`.

This is the forecast-residual side of the same decomposition: backcasts subtract on the way down, forecasts sum on the way out. The prediction is "predefined-shared trend + hidden-shared trend + individual trend," exactly the additive structure the residual stack gives.

For training, the label is the next-day return `d_i^t = (Price_i^{t+1} − Price_i^t)/Price_i^t`, and I regress to it with mean squared error, averaged per day so each date's cross-section contributes equally regardless of how many stocks were present:

  `L = Σ_t (1/|S^t|) Σ_{i∈S^t} (p_i^t − d_i^t)^2`,

optimized with Adam. The batch is naturally one date's full set of stocks, because the entire relational computation — concept representations, similarities, clustering — is over the cross-section present on that day; a "batch" that mixed dates would mix unrelated graphs.

Let me write the whole forward as the code I'd actually run, filling the empty slot in the harness. I'll keep the implementation honest to how the relational steps are actually realized in tensors — the predefined initialization as a membership-normalized aggregation, the hidden clustering as a masked-argmax on the cosine matrix — and I'll name the three modules predefined / hidden / individual:

```python
import torch
import torch.nn as nn


class StockTrendModel(nn.Module):
    """Encode each stock with a GRU, then mine three kinds of signal in a doubly
    residual chain: shared info over predefined concepts, shared info over
    discovered hidden concepts, and each stock's individual residual."""

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        # predefined-concept (es) and hidden-concept (is) shared-info transforms
        self.fc_es = nn.Linear(hidden_size, hidden_size)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es.weight)
        torch.nn.init.xavier_uniform_(self.fc_is.weight)

        # forecast heads
        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
        torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)

        # backcast heads (what each module "explains", subtracted from the residual)
        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
        torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
        # individual module
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_t2s = nn.Softmax(dim=1)     # normalize concept->stock weights over concepts
        self.fc_out = nn.Linear(hidden_size, 1)  # final read-out of the summed forecasts

    def cal_cos_similarity(self, x, y):          # cosine of every row of x vs every row of y
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        return xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)

    def forward(self, x, concept_matrix):
        device = x.device

        # ---- stock feature encoder: [N, 6*60] -> [N, hidden] ----
        x_hidden = x.reshape(len(x), self.d_feat, -1)   # [N, 6, 60]
        x_hidden = x_hidden.permute(0, 2, 1)            # [N, 60, 6]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]                   # last hidden state per stock

        # ---- Predefined Concept Module ----
        # initialize each concept from its member stocks with membership-degree
        # normalization and +1 smoothing.
        stock_to_concept = concept_matrix
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(
            stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        stock_to_concept_sum = stock_to_concept_sum + torch.ones(
            stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device)
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)  # concept reps = aggregate of members
        hidden = hidden[hidden.sum(1) != 0]              # drop concepts with no member

        # send back: similarity-weighted concept->stock aggregation (dynamic edges)
        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)
        e_shared_info = concept_to_stock.mm(hidden)
        e_shared_info = self.fc_es(e_shared_info)

        e_shared_back = self.fc_es_back(e_shared_info)   # backcast: what this module explains
        output_es = self.leaky_relu(self.fc_es_fore(e_shared_info))   # forecast head

        # ---- Hidden Concept Module: runs on the residual x_hidden - e_shared_back ----
        i_shared_info = x_hidden - e_shared_back
        hidden = i_shared_info
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, hidden)  # [N, N], seed=each stock
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        # remove the guaranteed diagonal self-score (cos=1) before the row-wise max
        i_stock_to_concept = i_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        row = torch.linspace(0, dim - 1, dim).to(device).long()
        column = i_stock_to_concept.max(1)[1].long()     # each stock's largest remaining seed score
        value = i_stock_to_concept.max(1)[0]
        # keep only the argmax entry in each row (single most-similar connection)
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        # re-add a surviving seed's own originating stock
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed(
            (i_stock_to_concept.sum(0) != 0).float() * diag)
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()  # hidden concept reps
        hidden = hidden[hidden.sum(1) != 0]              # delete seeds nobody connected to

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = i_concept_to_stock.mm(hidden)
        i_shared_info = self.fc_is(i_shared_info)

        i_shared_back = self.fc_is_back(i_shared_info)   # hidden-module backcast
        output_is = self.leaky_relu(self.fc_is_fore(i_shared_info))   # forecast head

        # ---- Individual Information Module: what neither shared part explained ----
        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = self.leaky_relu(self.fc_indi(individual_info))

        # ---- Stock Trend Prediction: sum the three forecasts, read out ----
        all_info = output_es + output_is + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        return pred_all
```

The tensor details matter because the axes are the method. The predefined initialization divides
each member's contribution by the membership-masked concept degree plus one, so empty concepts stay
zero and singleton concepts are smoothed before zero rows are deleted. The concept-to-stock
attention uses `Softmax(dim=1)` because each row is a stock choosing among concept columns. The
hidden-module assignment is a single masked row-argmax: remove the diagonal self-score, keep exactly
one remaining maximum per stock, restore the chosen values, re-add the diagonal only for surviving
columns, and delete empty columns. The `+ 1e-6` in the cosine guards the zero-norm case.

So the causal chain. I was stuck with a per-stock encoder that can't see cross-stock structure, and
the obvious fix — graph attention over a curated stock graph — fails twice: its edges are frozen so
it can't track relevance that drifts day to day, and it's blind to themes the curators never
labeled. Treating a concept as a node with a date-specific representation makes stock-concept
relevance recomputed, not frozen: initialize each concept as a smoothed aggregate of member stocks,
then let each stock choose a current mixture of concept vectors by cosine similarity and a softmax
over concepts. The full concept-correction equation uses the other axis — softmax over stocks for a
fixed concept — when I want to rewrite the concept nodes themselves. To reach the themes that are not
labeled at all, I run a parameter-free clustering on the residual embedding after subtracting what
the predefined module explained: seed one concept per stock, connect each stock to its most similar
other seed, delete empty seeds, and the survivors are discovered hidden concepts. A final residual
leaves each stock's individual signal. The backcast-subtract / forecast-sum discipline keeps the
three slices disjoint and recombines them additively, and an MSE objective over daily cross-sections
trains the whole thing end to end.
