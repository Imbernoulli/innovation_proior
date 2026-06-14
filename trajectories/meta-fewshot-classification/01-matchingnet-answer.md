**Problem.** N-way K-shot classification of unseen classes from a tiny support set, with no test-time
fine-tuning. Parametric nets cannot absorb a class in a few gradient steps; the comparison must be
non-parametric so a fresh support set instantly *is* the classifier, and the metric must be learned
under the exact episodic task.

**Key idea.** Classify a query as an attention-weighted vote over the support labels,
ŷ = Σ_i a(x̂, x_i) y_i, with a a softmax over the **cosine similarity** between deep embeddings. This
is simultaneously a KDE, subsumes kNN, and reads as an associative memory that grows with the support
set. Cosine (not raw dot or Euclidean) compares *direction* so an embedding's magnitude cannot dominate
the vote. The embeddings are made **fully conditional** on the whole support set — a bidirectional LSTM
over support (with a skip connection), an attention LSTM that re-reads the support over K = |S| steps
for the query — and the model is trained **episodically** so train matches test.

**Why these choices.** Non-parametric vote → a new support set needs no weight update. Cosine + softmax
→ bounded, smooth, convex weights forming a probability distribution per query. Episodic training → the
only way features learn to serve comparison against a support set of unseen classes. FCE LSTMs → let
near-colliding support examples adjust given each other and condition the query embedding on the
candidate classes; the skips keep them no worse than the myopic embedding. Output is
**log-probabilities**, so the loss is **NLL**, not cross-entropy (which would softmax twice).

**Hyperparameters.** `LR_OVERRIDE = 1e-3` — the bidirectional support LSTM and the multi-step query
attention loop destabilize under the scaffold's plain SGD@1e-2 and collapse the softmax toward uniform;
the smaller step keeps them trainable (the loop's grad-norm clip at 5.0 also helps). The cosine vote
normalizes the **support side only** so the softmax stays peaked; `+1e-6` floors `log(0)`.
`is_transductive = False`.

**What to watch.** The fine-grained benchmark (CUB) should be the weakest: a per-point cosine vote has
the least margin when all classes look alike, and FCE context cannot create a margin the embedding lacks.
That points the next rung at a cleaner per-class summary with a metric consistent with it.

```python
class CustomFewShotMethod(FewShotClassifier):
    """Matching Networks (Vinyals et al., 2016).

    Contextualizes support and query features using LSTMs, then classifies
    queries via cosine-similarity-weighted voting over support labels.
    Uses NLLLoss since output is log-probabilities.
    """

    # Vinyals et al. 2016 trains MatchingNet with Adam@1e-3 and gradient
    # clipping at 5; SGD@1e-2 (the global default for ProtoNet/RelationNet)
    # destabilizes the bidirectional LSTM and the 25-step query encoder loop,
    # collapsing softmax to uniform output. The framework's training loop
    # honours LR_OVERRIDE to keep this baseline trainable.
    LR_OVERRIDE = 1e-3

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone, use_softmax=False)
        self.feature_dimension = FEATURE_DIMENSION

        # Bidirectional LSTM to contextualize support features
        self.support_features_encoder = nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # LSTM cell for attention-based query encoding
        self.query_features_encoding_cell = nn.LSTMCell(
            self.feature_dimension * 2, self.feature_dimension
        )
        self.softmax = nn.Softmax(dim=1)

        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        self.contextualized_support_features = self._encode_support(support_features)
        self.one_hot_support_labels = F.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        contextualized_query_features = self._encode_query(query_features)

        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                F.normalize(self.contextualized_support_features, dim=1).T
            )
        )
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def _encode_support(self, support_features: Tensor) -> Tensor:
        hidden_state = self.support_features_encoder(
            support_features.unsqueeze(0)
        )[0].squeeze(0)
        contextualized = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )
        return contextualized

    def _encode_query(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.nll_loss(scores, labels)
```
