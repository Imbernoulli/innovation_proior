The classifier inside the episode is the whole point, and the scaffold hands me a default already —
mean prototypes, distance to prototypes — but I want to start from the weakest *principled* fill of
this contract and understand exactly where it stalls, because that failure is what every later rung
will react to. So I set the prototype shortcut aside and start from what actually hurts in this
problem, then build the simplest classifier that respects it.

What hurts is structural, not a tuning problem. I have a backbone that would be wonderful if I could
pour thousands of labeled images per class into it, and it is useless the moment a brand-new class
arrives with five examples. A child sees one giraffe in a book and spots giraffes everywhere; a
parametric net sees five giraffes and shrugs, because the only way it knows how to absorb a class is to
take gradient steps, and five examples are a few tiny steps. Worse, the rules of this task forbid the
move entirely: at test time I get a fresh support set of unseen classes and I am not allowed to update
weights on it. No fine-tuning at evaluation. And if I tried — hammering the backbone on twenty-five
images — I would overfit catastrophically and overwrite everything the base classes taught me. So
whatever I build has to absorb a never-before-seen class *instantly*, with no weight update, and still
classify a query into it. The knowledge that adapts to the task cannot live in the slowly-moving
weights; it has to live in the support set itself.

The thing that already assimilates new examples instantly is *non-parametric*: nearest neighbours,
kernel density estimation. You store the example and it is in the model — there is no training step,
nothing to forget, and capacity grows with the data because the model *is* the data. That is exactly
the instant-assimilation property I need, and it is the right half of the answer. The catch everyone
knows: a nearest-neighbour classifier is only as good as the metric that decides what "near" means, and
a raw-feature metric tailored to the base-class softmax is weak for comparing unseen classes. So I have
two halves at opposite ends of a trade-off — parametric deep nets give powerful *learned* features but
cannot absorb a class in one shot; non-parametric classifiers absorb instantly but cannot learn the
geometry. The standard compromise, borrowed-feature nearest neighbour, only works moderately, and it is
worth being precise about *why*: those features were optimized to make a softmax separate the base
classes, never to make instance-to-instance comparison meaningful for an unseen support set. The
geometry I lean on at test time is a byproduct of a different objective. That mismatch is the thing I
have to fix, and fixing it means making the metric *learnable* and training it under exactly the task I
will be judged on.

The one piece of prior work that already made nearest-neighbour classification differentiable is
Neighbourhood Components Analysis, and its trick is the one I want to steal. Hard kNN error is a
*discontinuous* function of the metric — an infinitesimal change can flip which point is nearest and
jump the classification by a finite amount — so you cannot take its gradient. NCA's cure is to replace
the hard neighbour with a *soft, stochastic* one: point i picks neighbour j with probability
p_ij ∝ exp(−‖Ax_i − Ax_j‖²), a softmax over embedding distances, and the probability i is classified
correctly is the mass it puts on its own class. Maximize that by gradient ascent and the metric A
trains by backprop; the discontinuity is gone because the softmax is smooth. That is exactly the bridge
between "non-parametric classifier" and "trained by gradient descent." But two things about NCA do not
fit this task. First, A is a single *linear* map — far short of the deep ResNet-12 the scaffold gives
me. Second, and more important, the NCA objective is point-against-all-of-training over a fixed set; my
task is not shaped like that at all. At test time I get a small, freshly sampled labeled support set of
unseen classes and a query, and I must put a distribution over those classes. The reference set is not
all of training — it is a tiny support set that changes every episode.

So I carry the soft-neighbour idea over but re-aim it at *my* task. Forget pairwise-over-everything. I
have a support set S = {(x_i, y_i)} and a query x̂, and I want a distribution over the labels present in
S. The simplest non-parametric thing I can write is a weighted vote of the support labels, where the
weight says how much the query matches each support example: ŷ = Σ_i a(x̂, x_i) y_i. If the y_i are
one-hot and the weights a(x̂, x_i) are non-negative and sum to one, then ŷ is *literally* a probability
distribution over the classes in S — the mass on each class is the total attention the query pays to
support examples of that class. This one expression is doing several jobs at once and I want to see all
of them before committing. If a is a kernel on the embedding space, this is kernel density estimation —
ŷ is a Nadaraya–Watson smoothing of the labels by similarity. If a is zero for all but the nearest
support points and constant on those, this is nearest neighbours. So the expression *subsumes* both KDE
and kNN; I have not thrown away the non-parametric behaviour, I have parameterized the kernel. And
there is a third reading I like even more: a is an *attention* mechanism, the y_i are memories bound to
the x_i, and given a query I "point" into the support set and read out the label of whatever I am
pointing at — an associative memory whose slots carry labels. Crucially it is non-parametric in the
right way: as the support set grows, the memory grows with it; the classifier is redefined by whatever
S I feed it, with no change to the network. That is the instant-assimilation property, recovered, and
it is what lets a freshly sampled support set of novel classes simply *be* a new classifier with no
weight update — exactly what this task demands at test time.

Now I choose a, the attention. The form that connects cleanly to NCA's soft neighbours and to
content-based attention is a softmax, a(x̂, x_i) = exp(c(f(x̂), g(x_i))) / Σ_j exp(c(f(x̂), g(x_j))),
where f embeds the query and g embeds the support examples — both the same deep backbone here — and c
is a similarity. The softmax gives me exactly the non-negative, sum-to-one weights I assumed, it is
differentiable, and it is the same smoothing that killed NCA's discontinuity, now applied over the
support set rather than all of training. The metric is learned through f and g, which are deep, fixing
NCA's linear limitation. What should c be? NCA used negative squared Euclidean. Let me think about what
goes wrong with that on top of an unconstrained deep embedding. The backbone's outputs have no
constraint on their magnitude — one support embedding can come out with a much larger norm than another
simply because of where it landed in feature space. If c is a raw dot product, a long support vector
scores high for almost any query regardless of *direction* and dominates the softmax; if c is negative
Euclidean, the absolute scale of the embeddings sets the softmax temperature, so a class with tightly
clustered but far-out embeddings behaves differently from one near the origin. I do not want an
embedding's *magnitude* deciding the vote; I want its *direction* — does the query point the same way
as this support example. So normalize the magnitude out: use cosine similarity, c(u, v) =
uᵀv/(‖u‖‖v‖), bounded in [−1, 1], so the softmax sees scores on a fixed scale and the comparison is
about alignment, not length. That is the choice, and it is a real divergence from the scaffold default,
which scores by L2 distance to a *mean* — I am voting over individual support points by cosine instead.

There is a subtlety about *what kind* of classifier this is that tells me it should work. The vote
ŷ = Σ a(x̂, x_i) y_i is *discriminative*, not generative: for a query to be classified right it only has
to be sufficiently aligned with support pairs of its own class and misaligned with the rest — I never
model class densities. That is the same flavour as NCA, but unlike triplet or large-margin surrogates,
the objective I will optimize *is* the multi-way classification I care about, end to end, with one
differentiable loss. Which brings me to the half of the problem I think the borrowed-feature baselines
get wrong. I have a model that maps a support set to a classifier, S → c_S(x̂). How do I train it so it
generalizes to support sets of classes it has never seen? The trap is to train the ordinary way — a
flat softmax over all base classes — and *hope* the features transfer. That is precisely the mismatch I
diagnosed: train objective and test task are different shapes. So state the principle baldly: the
conditions I train under must match the conditions I test under. At test time I am handed a small
support set of a few unseen classes and asked to label queries using only it. So I train exactly that
way — sample a handful of classes, draw a support set and a query batch, train the model to predict the
query labels *conditioned on* the support set, and resample which classes are in play every step so it
never memorizes a fixed label set. This is the episodic objective the scaffold already runs:
θ = argmax_θ E_{L∼T} E_{S,B∼L} Σ_{(x,y)∈B} log P_θ(y | x, S). Because the support classes are
resampled every episode and never bound to fixed output units, a genuinely novel support set at test
time simply *is* the new classifier — no fine-tuning, by construction. The honest caveat: this only
holds while the test task distribution resembles training; if novel classes are drawn very differently
(say, all fine-grained and mutually similar, when training sampled broadly), the learned comparison
transfers worse. I will want to remember that when I see the per-benchmark split.

I could stop here — a softmax-cosine vote over the support is a complete, trainable few-shot classifier
that fits the contract. But there is a myopia in the embeddings I want to remove, because it is where
the real subtlety lives. Right now g embeds each support example *independently*: g(x_i) does not know
what else is in the support set. Suppose two support examples of different classes land very close in
feature space — a hard pair. Embedding each in isolation, g has no way to push them apart to make the
vote cleaner; it would help for x_i's embedding to shift *given that* x_j sits next to it. And
symmetrically, the way I embed the query ought to depend on which support set I am comparing against —
the same image might emphasize different features when the candidates are {cat, dog} versus
{truck, ship}. The classification is already conditioned on the whole support set through P(·|x̂, S); I
want the *features* it operates on conditioned too. Fully conditional embeddings.

Take the support side, g(x_i, S). I need a function of one example *and* the whole set. The set has no
natural order, so a one-way RNN would impose a spurious directionality; a *bidirectional* recurrence
lets each element's encoding see both sides and washes that out. Run a bidirectional LSTM over the
backbone features. Going forward and backward gives each position a hidden state summarizing everything
before and after it, so the contextual encoding of x_i depends on the entire set. Combine the two
directions and — the piece that keeps it from breaking — add a *skip connection* back to the raw
feature: g(x_i, S) = h⃗_i + h⃐_i + g′(x_i). The skip matters because I do not want the LSTM to
reconstruct the useful feature from scratch; I want it to start from the raw feature and add a
contextual *correction*. At initialization g(x_i, S) ≈ g′(x_i), so the network only has to learn how
the set should *modify* the embedding, and it can never do worse than the myopic version because it can
drive the LSTM contributions to zero. (I should keep honest that on easy, well-separated benchmarks
this context may buy little — the value should show up where support examples genuinely collide.)

Now the query side, f(x̂, S). I want the query's embedding to attend to the support set and be refined
over a few steps. The right machinery is an attention LSTM that takes no changing external input and
runs a fixed number of steps, each step reading the support by content attention and folding the
read-out back into its state — a "Process block" for reading a set, where the number of steps sets the
depth. Specialize it: the memory is the contextualized support, the thing refined is the query, and I
re-inject the raw query feature f′(x̂) at every step so the LSTM never loses what it is supposed to be
embedding. Each step computes attention a(h_{k−1}, g(x_i)) = softmax_i(h_{k−1}ᵀ g(x_i)), reads
r_{k−1} = Σ_i a g(x_i), feeds [f′(x̂); r_{k−1}] to the LSTM cell, and adds the skip h_k = ĥ_k + f′(x̂);
after K steps f(x̂, S) = h_K. So each step the query "looks at" the support set, decides which elements
are relevant *given where it currently is*, reads them, and updates — iterated K times so the attention
can sharpen and the model can learn to ignore some support elements. A natural choice when I implement
it is K equal to the number of support examples, so the read budget scales with the set being read. The
attention *inside* this refinement is the unnormalized dot product, not cosine, because here I am
shaping an internal representation, not casting the final vote; the final vote stays the cosine-softmax
over f(x̂, S) and g(x_i, S), where the magnitude-invariance argument earns its keep.

Two implementation choices I should pin down rather than wave through, because they bite. First, the
cosine vote: strictly, cosine normalizes *both* sides, but if I L2-normalize the query too, every dot
product gets squeezed into a narrow band, the softmax over support comes out nearly uniform, and the
vote loses its bite. So I normalize the *support* side only and leave the query unnormalized — keeping
the query vectors "sharp" so similarities spread out and the softmax can actually concentrate. It is a
deliberate trade of strict query-side scale invariance for a softmax that discriminates. The output is
then a vote ŷ = Σ a(x̂, x_i) y_i, already a probability distribution per query (the weights are convex,
the one-hots partition them by class), so to score it with the usual loss I take its log and apply
*negative log-likelihood* — not cross-entropy, which would softmax an already-normalized distribution a
second time. One numerical care: a query that puts zero total mass on a class would give log(0) = −∞,
so I floor with a tiny additive ε before the log. Second, the learning rate. The scaffold defaults to
plain SGD at 1e-2, which is fine for a fixed-metric mean classifier but this method has a bidirectional
LSTM *and* a multi-step attention loop in the forward path; under that large a step those recurrences
blow up and the output softmax collapses toward uniform. The contract exposes `LR_OVERRIDE`, so I set
it to 1e-3 (Adam-style small step) — the gradient-norm clipping at 5.0 is already in the fixed loop and
helps the LSTMs. is_transductive stays False: each query is classified on its own against the support,
never letting queries see each other. The full scaffold module is in the answer.

So the delta from the scaffold default is concrete: where the default takes per-class *means* and
scores by L2 distance, I keep every support point, contextualize support and query with LSTMs, and vote
by cosine-softmax over the support labels with an NLL loss. Now let me reason about what this floor
should do, because that is the entire point of running it first. This is the most *intricate* of the
metric methods — a per-point vote with two LSTMs — but intricacy is not strength here. The vote keeps
one weight per support example and never forms a single concise entity per class, so at 5-shot each
class is a scatter of five embeddings the softmax has to integrate, and the FCE recurrences add
parameters that twenty-five images per episode can barely constrain. I expect the benchmarks to split
on one axis: how cleanly the *frozen cosine vote* over individual points separates the classes. Where
classes are visually distinct (generic objects, CIFAR-like and miniImageNet-like categories), the
cosine vote over points should do reasonably and the LSTMs should help a little. Where the test
distribution diverges from the broad training sampling — fine-grained classes that are mutually
similar, like one bird species against another — the per-point cosine vote should struggle most,
because cosine over individual scattered support points is exactly the comparison that has the least
margin when all classes look alike, and the LSTM context cannot manufacture a margin that the
embedding-plus-fixed-metric does not already have. So I predict the weakest benchmark for this method
is the fine-grained one, and that the next rung's job is to find a *cleaner per-class summary and a
metric consistent with it*, which is precisely where collapsing each class to a single prototype —
instead of voting over points — points.
