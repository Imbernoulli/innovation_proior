Let me start from what actually hurts. I have a classifier that is wonderful when I can pour thousands of labeled images per class into it, and useless the moment a brand-new class shows up with a single example. A child sees one giraffe in a book and then spots giraffes everywhere; my network sees one giraffe and shrugs, because the only way it knows how to learn is to take a gradient step, and one example is one tiny gradient step. So the deep failure is structural, not a tuning problem: in a parametric model the knowledge lives in the weights, and the weights move slowly, by stochastic gradient descent over many examples. One shot gives me one step. Worse, if I try to force the issue by hammering on those few new examples, I overwrite everything else — catastrophic forgetting. And the test classes only arrive at test time, as a tiny labeled support set, with the hard rule that I am not allowed to fine-tune on them. No gradient steps at evaluation. So whatever I build has to be able to absorb a never-before-seen class *instantly*, with no weight update at all, and still classify a query into it.

The thing that already absorbs new examples instantly is non-parametric: nearest neighbours, kernel density estimation, locally weighted learning. You store the example, and it's in the model — there is no training step, nothing to forget, and the capacity grows with the data because the model *is* the data. That's exactly the instant-assimilation property I need. But everyone knows the catch: a nearest-neighbour classifier is only as good as the metric it uses to decide what "near" means, and a raw-pixel or off-the-shelf-feature metric is weak. There's no learned representation tailoring the geometry to the task. So I have two halves of what I want, sitting at opposite ends of a trade-off — parametric models give me powerful learned features but can't absorb a class in one shot, non-parametric models absorb instantly but can't learn the geometry. The obvious move, and the one people already do, is to bolt them together by *borrowing features*: train a deep classifier on the base classes, throw away the softmax, and do nearest-neighbour matching in the penultimate-layer feature space. It works, moderately. But sit with *why* it's only moderate: those features were trained to make a softmax separate the *base* classes. They were never asked to make instance-to-instance comparison meaningful for *unseen* classes presented as a support set. The geometry I'm relying on at test time is a byproduct, not the objective. That mismatch is the thing I have to fix.

So let me try to make the metric learnable instead of borrowed, and the place to look is the one piece of prior work that already turned nearest-neighbour classification into something differentiable: Neighbourhood Components Analysis. Their problem was the same shape as mine — kNN is great but the metric is fixed — and their obstacle was that the leave-one-out kNN error is a *discontinuous* function of the metric: an infinitesimal change in the transformation can flip which point is the nearest neighbour and jump the classification by a finite amount, so you can't take a gradient of it. Their fix is the move I want to steal. Replace the hard "nearest neighbour" with a *soft, stochastic* neighbour: point i picks point j as its neighbour with a probability that is a softmax over distances in the embedding,

  p_ij = exp(−||A x_i − A x_j||²) / Σ_{k≠i} exp(−||A x_i − A x_k||²),  p_ii = 0,

and then the probability that i is classified correctly is the total mass it puts on its own class, p_i = Σ_{j ∈ C_i} p_ij, where C_i is the set of same-class points. Maximize Σ_i p_i, the expected number of points classified correctly, by gradient ascent. The discontinuity is gone because the softmax is smooth, and now the metric (their A) trains by backprop. That is exactly the bridge I need between "non-parametric classifier" and "learned by gradient descent." But two things about NCA don't fit my problem. First, A is a single *linear* transformation — a Mahalanobis metric — so the representational power is whatever a linear map of fixed features can buy, far short of a deep net. Second, and more important, the NCA objective is point-against-all-other-points over the whole fixed training set; it isn't shaped like my task at all. My task is: here is a small, freshly sampled labeled support set of N novel classes; here is a query; put a distribution over the N classes. The reference set isn't all of training — it's a tiny support set that changes every episode.

Let me carry the soft-neighbour idea over but re-aim it at *my* task directly. Forget pairwise-over-everything. I have a support set S = {(x_i, y_i)}, k examples, and a query x̂, and I want a distribution over labels. The simplest non-parametric thing I can write is: the predicted label is a weighted vote of the support labels, where the weights say how much the query matches each support example,

  ŷ = Σ_{i=1}^k a(x̂, x_i) y_i.

If the y_i are one-hot vectors and the a(x̂, x_i) are nonnegative and sum to one, then ŷ is literally a probability distribution over the classes present in S — the weight on each class is the total attention the query pays to support examples of that class. Stare at this for a second, because it's doing several jobs at once and I want to see all of them before I commit. If a is a kernel on X × X, this is a kernel density estimator — ŷ is a Nadaraya–Watson-style smoothing of the labels by similarity. If I let a be zero for all but the b nearest support points and constant on those, this is (k−b)-nearest-neighbours. So this one expression *subsumes* both KDE and kNN; I haven't thrown away the non-parametric behaviour, I've parameterized the kernel. And there's a third reading I like even more: a is an attention mechanism, the y_i are *memories bound to* the x_i, and given a query I "point" into the support set and read out the label of whatever I'm pointing at. It's an associative memory — exactly the content-based attention people use to read an external memory matrix, except the memory slots carry labels and the read-out is the prediction. Crucially it's non-parametric in the right way: as the support set grows, the memory grows with it; the classifier c_S(x̂) is re-defined by whatever S I feed it, with no change to the network. That's the instant-assimilation property, recovered.

Now I have to choose a, the attention. The form that connects cleanly to NCA's soft neighbours and to content-based attention is a softmax,

  a(x̂, x_i) = exp(c(f(x̂), g(x_i))) / Σ_{j=1}^k exp(c(f(x̂), g(x_j))),

where f embeds the query and g embeds the support examples — both deep neural networks, possibly the same one — and c is some similarity. The softmax gives me exactly the nonnegative-and-sum-to-one weights I assumed, it's differentiable, and it's the same smoothing trick NCA used to kill the discontinuity, now applied over the support set rather than the whole training set. So the metric is learned, but through f and g, which are deep — that fixes NCA's linear-A limitation. And the whole thing is differentiable end to end: f, g, and the comparison all train together by backprop. Good.

What should c be? NCA used negative squared Euclidean distance. Let me think about what goes wrong with that here. My embeddings f(x̂) and g(x_i) are outputs of a deep net with no constraint on their magnitude — one support embedding could come out with a much larger norm than another simply because of where it landed in feature space. If c is a raw dot product, a long support vector gets a large score for almost any query and dominates the softmax regardless of *direction*; if c is negative Euclidean, the absolute scale of the embeddings sets the softmax temperature and a class with tightly-clustered-but-far-out embeddings behaves differently from one near the origin. I don't want the magnitude of an embedding deciding the vote; I want its *direction* — does the query point the same way as this support example. So normalize it out: use the cosine similarity, c(u, v) = uᵀv / (||u|| ||v||). Cosine is bounded in [−1, 1], so the softmax sees scores on a fixed scale, well-behaved, and the comparison is about alignment, not length. That's the choice. (Later I'll want to be careful about *which* side I normalize, but the principle is: compare directions.)

There's a subtlety I should name about what kind of loss this is, because it tells me it'll work. The classifier defined by ŷ = Σ a(x̂, x_i) y_i is *discriminative*, not generative: for the query to be classified right, it's enough that it's sufficiently aligned with support pairs of its own class and misaligned with the rest — I never have to model the class densities. That's the same flavour as NCA, and as triplet losses and large-margin nearest-neighbour, but unlike those it's not a surrogate built from pairs or triplets; the objective I'll optimize *is* the multi-way one-shot classification I care about, end to end, with a simple differentiable loss. Which brings me to the second half of the problem, and the part I think people keep getting wrong.

I have a model that maps a support set to a classifier, S → c_S(x̂). How do I *train* it so it generalizes to support sets of classes it has never seen? Here is the trap everyone falls into: train on the base classes the ordinary way — say, an N-way softmax over all base classes, or NCA over the whole base training set — and then *hope* the learned features happen to support comparison on novel classes. But that's precisely the mismatch I diagnosed at the start with borrowed features. The training objective and the test task are different shapes. So state the principle baldly: the conditions I train under must match the conditions I test under. At test time I will be handed a small support set of a few unseen classes and a few queries, and asked to label the queries using only that support set. So I should *train* exactly that way — show the network a small support set, ask it to label a small query batch using only that support set, and switch which classes are in play from one minibatch to the next, so it never gets to memorize a fixed label set. If I want to be good at learning a classifier from a few examples, I should train by repeatedly being given a few examples and made to learn a classifier from them.

Let me make that an objective. Define a task T as a distribution over label sets L — say, uniformly pick a handful of classes (e.g. 5) with a few examples each (e.g. up to 5). To form one training *episode*: sample a label set L ∼ T (say {cats, dogs}); from L sample a support set S and a query batch B (both labeled examples of those classes); and train the model to predict the labels in B *conditioned on S*. Averaging the per-query log-likelihood over episodes,

  θ = argmax_θ E_{L ∼ T} [ E_{S ∼ L, B ∼ L} [ Σ_{(x,y) ∈ B} log P_θ(y | x, S) ] ].

This is meta-learning in the most literal sense: the outer expectation is over *tasks*, and within each task the network must turn a support set into a working classifier and be scored on held-out queries — so it is explicitly learning *to learn from a support set*. And because the support set's classes are resampled every episode and never bound to fixed output units, when I later hand it a support set S′ of genuinely novel classes, nothing about the network has to change — its non-parametric structure means S′ simply *is* the new classifier. No fine-tuning, by construction. The obvious caveat I should keep honest about: this only works while the test task distribution T′ resembles the training T; if the novel classes are drawn very differently (say, all fine-grained and similar, when training sampled them broadly and at random), the learned comparison won't transfer well. That's a real limitation of training-to-match, not a bug to paper over.

So the core model is: embed query and support with deep nets f and g, compute softmax-over-cosine attention from the query to each support example, read out the support labels by those weights to get a label distribution, and train the whole thing episodically with the log-likelihood above. Let me make sure I can actually compute it as a clean tensor operation. Run the backbone on the support images to get support features G of shape (k, d) and on the query images to get query features F of shape (n_query, d). The cosine-similarity matrix from queries to support is F̃ G̃ᵀ where ~ means L2-normalized rows; softmax that over the support axis to get attention weights of shape (n_query, k); multiply by the one-hot support labels Y of shape (k, N) to get, for each query, the total weight on each class, shape (n_query, N). That's ŷ. It's already a probability distribution per query — the attention weights are convex and the one-hots partition them by class — so to score it with the usual classification loss I take its log and apply negative-log-likelihood. One numerical care: if a query happens to put zero total weight on a class, log(0) is −∞; floor it with a tiny additive ε before the log. So the output is *log-probabilities*, and the matching loss is NLL, not cross-entropy-on-logits — cross-entropy would softmax my already-normalized probabilities a second time. That distinction matters and I'll carry it into the code.

I could stop here; this is a complete, trainable one-shot classifier. But there's a weakness in the embeddings I want to push on, because it's where the real subtlety of the method lives. Right now g embeds each support example *independently*: g(x_i) doesn't know what else is in the support set. Call that myopic. Why is that bad? Suppose two support examples x_i and x_j happen to land very close in feature space — maybe they're genuinely similar images of different classes, a hard pair. Embedding each in isolation, g has no way to *push them apart* to make the comparison cleaner; it would be useful for the embedding of x_i to shift given that x_j is sitting right next to it. And there's a symmetric problem on the query side: the way I embed the query x̂ ought to depend on *which* support set I'm comparing it against — the same image might want to emphasize different features when the candidate classes are {cat, dog} versus {truck, ship}. So both embeddings should be functions of the whole support set: g(x_i, S) and f(x̂, S), not g(x_i) and f(x̂). The classification is already conditioned on the full support set through P(·|x̂, S); I want the *features* it operates on to be conditioned too. Fully conditional embeddings.

Take g(x_i, S) first. I need a function that takes one support example *and* the whole set and returns a context-aware embedding. The set has no natural order, so I can't just feed it to a vanilla RNN and call it done — but I can treat S as a sequence anyway and use a *bidirectional* recurrence so that each element's encoding sees the elements on both sides, washing out the strong directionality of a one-way pass. Run a bidirectional LSTM over the support features. Let g′(x_i) be the per-example feature from the backbone. Going forward, h⃗_i, c⃗_i = LSTM(g′(x_i), h⃗_{i−1}, c⃗_{i−1}); going backward, h⃐_i, c⃐_i = LSTM(g′(x_i), h⃐_{i+1}, c⃐_{i+1}), with the backward recursion starting from i = |S|. Now each position has a forward hidden state that summarizes everything before it and a backward one that summarizes everything after it, so the contextual encoding of x_i depends on the entire set. Combine them, and — this is the piece that keeps it from breaking — add a skip connection back to the raw feature:

  g(x_i, S) = h⃗_i + h⃐_i + g′(x_i).

Why the skip? Because I don't want the LSTM to have to *reconstruct* the useful feature g′(x_i) from scratch; I want it to start from the raw feature and add a contextual *correction* on top. The identity path means at initialization g(x_i, S) ≈ g′(x_i) and the network only has to learn how the set should *modify* it — easier to optimize, and it can't do worse than the myopic embedding because it can always learn the LSTM contributions toward zero. (In practice it's worth checking whether contextualizing g helps at all on easy datasets, where the embeddings are already well separated; the value should show up on the harder tasks where support examples genuinely collide.)

Now f(x̂, S), the query side. I want the query's embedding to attend to the support set and be refined over a few steps — and I realize I've seen exactly the right machinery, the Process block from reading sets with iterated attention. The idea there is an LSTM that takes *no external input* and runs for a fixed number of steps, each step reading a memory by content attention and folding the read-out back into its state, so the final state is a permutation-invariant function of the set with "depth" set by the number of steps. Let me specialize it: the memory is the contextualized support {g(x_i, S)}, the thing being refined is the query, and I keep re-injecting the raw query feature f′(x̂) at every step so the LSTM never loses the query it's supposed to be embedding. Unroll K processing steps:

  ĥ_k, c_k = LSTM(f′(x̂), [h_{k−1}, r_{k−1}], c_{k−1}),
  h_k = ĥ_k + f′(x̂),
  r_{k−1} = Σ_{i=1}^{|S|} a(h_{k−1}, g(x_i)) g(x_i),
  a(h_{k−1}, g(x_i)) = softmax_i( h_{k−1}ᵀ g(x_i) ),

and f(x̂, S) = attLSTM(f′(x̂), g(S), K) = h_K. Let me read what each line does. The LSTM input is the fixed query feature f′(x̂) (constant across steps — this is the "no external input that changes" of the Process block, with the query held as a constant context); its recurrent state is the previous hidden state concatenated with the previous attention read-out r_{k−1}. That read-out is a content-attention summary of the support set as seen from the current query state h_{k−1}: score each support embedding by dot product with h_{k−1}, softmax over the support index, take the weighted sum. So each step the query "looks at" the support set, decides which support elements are relevant *given where it currently is*, reads them, and updates — iterated K times so the attention can sharpen over steps and the model can, if useful, learn to ignore some support elements. The skip connection h_k = ĥ_k + f′(x̂) plays the same role as on the g side: anchor to the raw query feature, learn only the contextual correction. The number of steps K is a depth knob — more steps, deeper attention computation; a natural and simple choice when I implement this is to take K equal to the number of support examples, so the read budget scales with the set I'm reading. The attention here is the unnormalized dot-product form rather than cosine, because inside this refinement loop I'm shaping an internal representation, not casting the final vote; the final classification vote is the cosine-softmax over f(x̂,S) and g(x_i,S), and that's where the magnitude-invariance argument matters.

Let me now actually assemble this into the code I'd ship, filling the one empty slot in the few-shot harness — the comparison rule, split across process_support_set (prepare from the support set) and forward (score the queries). The backbone gives pooled feature vectors. process_support_set runs the backbone, contextualizes the support features with the bidirectional LSTM, and stores them along with the one-hot labels. forward runs the backbone on the queries, refines them with the attention LSTM against the stored support, then does the cosine-softmax vote and returns log-probabilities. The support encoder is an nn.LSTM with bidirectional=True; its hidden state for one set comes out as (|S|, 2d), the first d columns the forward hidden state and the last d the backward, and I add both to the raw features for the skip — exactly h⃗_i + h⃐_i + g′(x_i). The query encoder is an nn.LSTMCell whose input width is 2d, because at each step I feed it the concatenation of the raw query feature (d) and the attention read-out (d).

```python
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CustomFewShotMethod(FewShotClassifier):
    """Map a support set to a non-parametric classifier: contextualize support and
    query features with LSTMs (full context embeddings), then classify each query by
    a cosine-similarity-weighted vote over the support labels. Output is
    log-probabilities, so the matching loss is NLL."""

    # Train this with Adam at 1e-3 (with gradient clipping): the bidirectional
    # support LSTM and the multi-step query attention loop are unstable under the
    # large plain-SGD rate used for the simpler metric baselines, collapsing the
    # softmax toward uniform.
    LR_OVERRIDE = 1e-3

    def __init__(self):
        backbone = make_backbone(use_pooling=True)            # ResNet-12 -> 640-dim vectors
        super().__init__(backbone=backbone, use_softmax=False)
        self.feature_dimension = FEATURE_DIMENSION

        # g(x_i, S): bidirectional LSTM contextualizes each support feature
        # in the context of the whole support set.
        self.support_features_encoder = nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # f(x_hat, S): attLSTM cell. Input width is 2d = [raw query feature ; attention read-out].
        self.query_features_encoding_cell = nn.LSTMCell(
            self.feature_dimension * 2, self.feature_dimension
        )
        self.softmax = nn.Softmax(dim=1)

        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)          # g'(x_i), shape (|S|, d)
        self.contextualized_support_features = self._encode_support(support_features)
        self.one_hot_support_labels = F.one_hot(support_labels).float()   # labels as memories

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)             # f'(x_hat)
        contextualized_query_features = self._encode_query(query_features)

        # Vote: softmax over cosine similarity from each query to each support example.
        # Normalize the SUPPORT side only -- keeps the query vectors "sharp" so the
        # softmax doesn't flatten toward uniform.
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                F.normalize(self.contextualized_support_features, dim=1).T
            )
        )
        # y_hat = sum_i a(x_hat, x_i) y_i, then log for NLL; +1e-6 floors log(0).
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def _encode_support(self, support_features: Tensor) -> Tensor:
        # nn.LSTM over the set as a (length-1 batch) sequence; hidden is (|S|, 2d).
        hidden_state = self.support_features_encoder(
            support_features.unsqueeze(0)
        )[0].squeeze(0)
        # g(x_i, S) = forward_h + backward_h + g'(x_i)   (skip connection)
        contextualized = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )
        return contextualized

    def _encode_query(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features                       # h_0 seeded with raw query feature
        cell_state = torch.zeros_like(query_features)

        # K processing steps; here K = number of support examples.
        for _ in range(len(self.contextualized_support_features)):
            # a(h_{k-1}, g(x_i)) = softmax_i( h_{k-1} . g(x_i) ); read-out r_{k-1}.
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            # LSTM input = [f'(x_hat) ; r_{k-1}]; query feature re-injected every step.
            lstm_input = torch.cat((query_features, read_out), 1)
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features    # h_k = h_hat_k + f'(x_hat) (skip)

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.nll_loss(scores, labels)                  # scores are log-probabilities
```

A couple of implementation choices in there I should justify rather than wave through. The cosine vote normalizes the support features but *not* the query features. Strictly, cosine would normalize both; but if I normalize both, every dot product gets squeezed into a narrow band around the same value, the softmax over support then comes out nearly uniform, and the vote loses its bite. Leaving the query unnormalized keeps the query vectors "sharp" so the similarities spread out and the softmax can actually concentrate on the right support examples — a deliberate trade of strict scale-invariance on the query side for a softmax that discriminates. The +1e-6 before the log is the floor against log(0) on a class that received zero attention mass. And the learning-rate note isn't cosmetic: the simpler metric baselines (prototype means, fixed distances) train fine under the large plain-SGD rate, but this method has a bidirectional LSTM *and* a multi-step attention loop in the forward path, and under that big a step those recurrences blow up and the output softmax collapses toward uniform; Adam at 1e-3 with gradient clipping is what keeps it trainable. is_transductive returns False because each query is classified on its own against the support set — I never let the queries see each other.

So the causal chain, start to finish. I began stuck: parametric deep nets can't absorb a novel class in one shot because knowledge is in slowly-moving weights, and I'm forbidden from fine-tuning at test time; non-parametric kNN absorbs instantly but has no learned metric; and the standard fix of borrowing base-class-softmax features is only moderate because those features were never optimized for comparing against a support set. NCA showed how to make a nearest-neighbour classifier differentiable — soft, stochastic neighbours via a softmax over embedding distances — but only with a linear metric and a pairwise-over-all-of-training objective. I transplanted the soft-neighbour idea onto my actual task: predict a label as an attention-weighted vote over the support labels, ŷ = Σ a(x̂, x_i) y_i, which simultaneously is a KDE, subsumes kNN, and reads as an associative memory that grows with the support set — so a new support set instantly *is* a new classifier, no weight update. I chose the attention to be a softmax over cosine similarity between deep embeddings f and g, picking cosine so an embedding's magnitude can't dominate the vote, and deep f, g to escape NCA's linear limit. Then I fixed the training itself: to generalize to unseen classes I train episodically, sampling a small support set and query batch from a resampled label set every step and maximizing the per-query log-likelihood conditioned on the support — learning to learn from a support set, with the train task shaped exactly like the test task. Finally I removed the myopia in the embeddings: a bidirectional LSTM with a skip connection makes each support embedding depend on the whole set (full context for g), and an attention LSTM that re-reads the support over K steps with a skip connection conditions the query embedding on the support set (full context for f), the latter being a specialized Process block for reading the set. The output is a log-probability vote, trained with NLL, run unchanged on any freshly sampled support set of novel classes.
</content>
