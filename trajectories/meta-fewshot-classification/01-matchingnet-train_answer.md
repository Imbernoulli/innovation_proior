The scaffold hands me a default classifier — per-class mean prototypes scored by L2 distance — but I want to begin from the weakest *principled* fill of the contract and understand exactly where it stalls, because that failure is what the later rungs react to. And the thing that hurts here is structural, not a tuning problem. The backbone would be wonderful if I could pour thousands of labeled images per class into it, and it is useless the instant a brand-new class arrives with five examples: the only way a parametric net absorbs a class is to take gradient steps, and five images are a few tiny steps. Worse, the task forbids the move outright — at test time I am handed a fresh support set of unseen classes and may not update weights on it, and if I tried, hammering the backbone on twenty-five images would overfit catastrophically and overwrite what the base classes taught it. So whatever I build has to absorb a never-before-seen class *instantly*, with no weight update, and still classify a query into it. The knowledge that adapts to the task cannot live in the slowly-moving weights; it has to live in the support set itself. The piece that already assimilates new examples instantly is *non-parametric* — nearest neighbours, kernel density estimation — where the model *is* the data; but a nearest-neighbour classifier is only as good as the metric that decides what "near" means, and a raw-feature metric tuned to the base-class softmax was never optimized to make instance-to-instance comparison meaningful for an unseen support set. That mismatch is the thing to fix, and fixing it means making the metric *learnable* and training it under the exact episodic task I will be judged on.

I propose **Matching Networks**: classify a query as an attention-weighted vote over the support labels, with the attention a softmax over the cosine similarity between deep embeddings, and the whole map trained episodically. Concretely, given a support set $S = \{(x_i, y_i)\}$ and a query $\hat{x}$, the prediction is

$$\hat{y} = \sum_i a(\hat{x}, x_i)\, y_i,$$

where the $y_i$ are one-hot and the weights $a(\hat{x}, x_i)$ are non-negative and sum to one — so $\hat{y}$ is *literally* a probability distribution over the classes present in $S$, the mass on each class being the total attention the query pays to support examples of that class. This single expression is doing several jobs at once, and that is exactly why I trust it: if $a$ is a kernel on the embedding space it is kernel density estimation (a Nadaraya–Watson smoothing of the labels by similarity); if $a$ is zero except on the nearest support points it is nearest neighbours; so the vote *subsumes* both KDE and kNN — I have not thrown away the non-parametric behaviour, I have parameterized the kernel. And there is a third reading: $a$ is an *attention* mechanism, the $y_i$ are memories bound to the $x_i$, and given a query I point into the support set and read out the label of whatever I am pointing at — an associative memory whose slots carry labels and whose size grows with the support set. That is the instant-assimilation property recovered: a freshly sampled support set of novel classes simply *becomes* a new classifier, with no weight update — exactly what the test protocol demands.

For the attention I use a softmax,

$$a(\hat{x}, x_i) = \frac{\exp\!\big(c(f(\hat{x}), g(x_i))\big)}{\sum_j \exp\!\big(c(f(\hat{x}), g(x_j))\big)},$$

with $f$ embedding the query and $g$ embedding the support examples (both the deep ResNet-12 backbone), and $c$ a similarity. The softmax gives exactly the non-negative, sum-to-one weights I assumed; it is differentiable; and it is the same smoothing that, in Neighbourhood Components Analysis (Goldberger et al., 2004), replaced the discontinuous hard nearest neighbour with a soft, stochastic one to make kNN trainable by gradient descent — only here it is applied over a tiny support set that changes every episode rather than point-against-all-of-training, and the metric is learned through deep $f, g$ rather than NCA's single linear map. What should $c$ be? The backbone's outputs have no constraint on magnitude — one support embedding can come out with a much larger norm than another just because of where it landed. If $c$ is a raw dot product, a long support vector scores high for almost any query regardless of *direction* and dominates the softmax; if $c$ is negative Euclidean, the absolute scale of the embeddings sets the softmax temperature. I do not want an embedding's *magnitude* deciding the vote — I want its *direction*. So I normalize magnitude out and use cosine similarity, $c(u, v) = u^\top v / (\lVert u\rVert\,\lVert v\rVert)$, bounded in $[-1, 1]$, so the softmax sees scores on a fixed scale and the comparison is about alignment, not length. This is a genuine divergence from the scaffold default, which scores by L2 distance to a *mean*; I vote over individual support points by cosine instead.

How do I train a map $S \mapsto c_S(\hat{x})$ so it generalizes to support sets of classes it has never seen? The trap is to train an ordinary flat softmax over all base classes and *hope* the features transfer — precisely the train/test shape mismatch I diagnosed. So I state the principle baldly: the conditions I train under must match the conditions I test under. At test time I am handed a small support set of a few unseen classes and asked to label queries using only it, so I train exactly that way —

$$\theta = \arg\max_\theta\; \mathbb{E}_{L\sim T}\,\mathbb{E}_{S,B\sim L}\sum_{(x,y)\in B}\log P_\theta(y \mid x, S),$$

sampling a handful of classes per step, drawing a support set and a query batch, and resampling which classes are in play every step so the network never binds a fixed label set to output units. Because the support classes are never bound to fixed outputs, a genuinely novel support set at test time simply *is* the new classifier — no fine-tuning, by construction. The honest caveat is that this holds only while the test task distribution resembles training: if novel classes are drawn very differently — all fine-grained and mutually similar, when training sampled broadly — the learned comparison transfers worse.

I could stop at a cosine-softmax vote, but there is a myopia I want to remove: $g$ embeds each support example *independently*, so if two support examples of different classes land close together, $g$ has no way to push them apart, and symmetrically the query embedding ought to depend on which candidate classes it is being compared against. The classification is already conditioned on the whole support set through $P(\cdot\mid\hat{x},S)$; I want the *features* conditioned too — **fully conditional embeddings**. On the support side, $g(x_i, S)$ must be a function of one example and the whole set. The set has no natural order, so a one-way RNN would impose a spurious directionality; a *bidirectional* LSTM lets each element's encoding see both sides and washes that out. The piece that keeps it from breaking is a *skip connection* back to the raw feature, $g(x_i, S) = \vec{h}_i + \overleftarrow{h}_i + g'(x_i)$: at initialization $g(x_i,S)\approx g'(x_i)$, so the network only has to learn how the set should *modify* the embedding, and it can never do worse than the myopic version because it can drive the LSTM contributions to zero. On the query side, $f(\hat{x}, S)$ is an attention LSTM that takes no changing external input and runs a fixed number of steps, each step reading the support by content attention and folding the read-out back into its state. Each step computes $a(h_{k-1}, g(x_i)) = \mathrm{softmax}_i\big(h_{k-1}^\top g(x_i)\big)$, reads $r_{k-1} = \sum_i a\, g(x_i)$, feeds $[f'(\hat{x}); r_{k-1}]$ to the LSTM cell, and adds the skip $h_k = \hat{h}_k + f'(\hat{x})$; after $K$ steps $f(\hat{x}, S) = h_K$. I take $K$ equal to the number of support examples so the read budget scales with the set being read. The attention *inside* this refinement is the unnormalized dot product, not cosine, because here I am shaping an internal representation, not casting the final vote; the final vote stays the cosine-softmax over the contextualized $f(\hat{x}, S)$ and $g(x_i, S)$.

Two implementation choices bite and I pin them down. First, the cosine vote: strictly, cosine normalizes both sides, but if I L2-normalize the query too, every dot product is squeezed into a narrow band, the support softmax comes out nearly uniform, and the vote loses its bite. So I normalize the *support* side only and leave the query unnormalized — keeping the query vectors sharp so similarities spread and the softmax can concentrate, a deliberate trade of strict query-side scale invariance for a softmax that discriminates. The output $\hat{y} = \sum_i a(\hat{x}, x_i)\,y_i$ is already a probability distribution per query, so I take its log and apply *negative log-likelihood* — not cross-entropy, which would softmax an already-normalized distribution a second time — and I floor with a tiny additive $\varepsilon = 10^{-6}$ before the log so a class with zero total mass does not give $\log(0) = -\infty$. Second, the learning rate: the scaffold's plain SGD at $10^{-2}$ is fine for a fixed-metric mean classifier, but this method has a bidirectional LSTM *and* a multi-step attention loop in the forward path, and under that large a step the recurrences blow up and the output softmax collapses toward uniform. The contract exposes `LR_OVERRIDE`, so I set it to $10^{-3}$; the loop's gradient-norm clipping at $5.0$ also helps the LSTMs. `is_transductive` stays `False` — each query is classified on its own against the support, never letting queries see each other.

This is the most *intricate* of the metric methods — a per-point vote with two LSTMs — but intricacy is not strength here, and I expect the benchmarks to split on one axis: how cleanly the frozen cosine vote over individual points separates the classes. Where classes are visually distinct it should do reasonably and the LSTMs should help a little. Where the test distribution diverges from the broad training sampling — fine-grained classes that are mutually similar, one bird species against another — the per-point cosine vote should struggle most, because the query is asked to align with twenty-five scattered embeddings, five per class, all sitting close together, and the LSTM context cannot manufacture a margin the embedding-plus-fixed-metric does not already have. So I predict the weakest benchmark for this method is the fine-grained one, and that the next rung's job is to find a *cleaner per-class summary and a metric consistent with it* — which is precisely where collapsing each class to a single prototype, instead of voting over points, points.

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
