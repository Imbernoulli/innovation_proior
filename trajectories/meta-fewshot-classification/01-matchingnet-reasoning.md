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
weights on it. No fine-tuning at evaluation. And if I tried — hammering the backbone on the support —
I would overfit catastrophically and overwrite everything the base classes taught me. Let me put a
number on the regime, because it decides everything downstream. Five-way five-shot is twenty-five
labeled support images per episode, and the ResNet-12 the scaffold gives me is on the order of twelve
million parameters. Twenty-five constraints against twelve million degrees of freedom is not a fit, it
is an interpolation table; one or two SGD steps at any sane rate either move the weights not at all or
move them enough to wreck the base-class geometry. So whatever I build has to absorb a
never-before-seen class *instantly*, with no weight update, and still classify a query into it. The
knowledge that adapts to the task cannot live in the slowly-moving weights; it has to live in the
support set itself.

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
classes, never to make instance-to-instance comparison meaningful for an unseen support set. Concretely
what a base-class softmax rewards is that each training image land on the correct side of a fixed set of
class hyperplanes; nothing in that objective asks two images of a *held-out* class to sit near each
other, nor two images of different held-out classes to sit apart. The geometry I lean on at test time
is a byproduct of a different objective. That mismatch is the thing I have to fix, and fixing it means
making the metric *learnable* and training it under exactly the task I will be judged on.

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
NCA's linear limitation. What should c be? Let me actually walk the three candidates rather than reach
for one. NCA used negative squared Euclidean; a dot product is the other reflex; cosine is the third.
Take the raw dot product c(u,v) = uᵀv. The backbone's outputs have no constraint on their magnitude —
one support embedding can come out with a norm several times another's simply because of where it
landed in feature space — and under a dot product a long support vector scores high for almost any
query regardless of *direction*, so that one slot dominates the softmax and the vote reduces to "which
support point happens to have the biggest norm." That is a magnitude artefact, not a similarity, so the
dot product is out. Negative squared Euclidean is subtler but has the same disease from the other side:
−‖u−v‖² = −‖u‖² + 2uᵀv − ‖v‖², and the −‖u‖² term is the query's own norm, constant across the support
so it cancels in the softmax, but the −‖v‖² term is *per support point* and does not cancel, so again a
support example is scored partly on its own magnitude, and the absolute scale of the embeddings sets
the softmax temperature — a class of tightly clustered but far-out embeddings behaves differently from
one near the origin. I do not want an embedding's *magnitude* deciding the vote; I want its *direction*
— does the query point the same way as this support example. So normalize the magnitude out: use cosine
similarity, c(u, v) = uᵀv/(‖u‖‖v‖), bounded in [−1, 1], so the softmax sees scores on a fixed scale and
the comparison is about alignment, not length. That is the choice, and it is a real divergence from the
scaffold default, which scores by L2 distance to a *mean* — I am voting over individual support points
by cosine instead.

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
drive the LSTM contributions to zero. Let me be honest about the cost of this, though, because it is
exactly the ledger the hard benchmark will audit. A bidirectional LSTM with input and hidden width both
640 carries, per direction, four gates each with a 640×640 input-to-hidden matrix, a 640×640
hidden-to-hidden matrix, and two bias vectors: 4·(640² + 640² + 2·640) ≈ 3.28M parameters, times two
directions ≈ 6.56M. That is already half the size of the backbone itself, and it is bought so that each
support embedding can be reshaped using information from the other twenty-four support vectors in its
episode. Whatever contextualization it learns is fit to the support-set geometry of the *training*
classes; there is no guarantee that pattern transfers to a novel-class support set, and on a benchmark
where the classes barely differ it may be reshaping around noise. The skip connection is my only
insurance — it guarantees the module starts as a no-op and can be switched off — but insurance is not a
reason to expect a gain, so I will hold the FCE as "cannot hurt at init, must earn its keep," not as a
strength.

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
it is K equal to the number of support examples, so the read budget scales with the set being read; at
5-way 5-shot that is K = 25 unrolled cell steps. The attention *inside* this refinement is the
unnormalized dot product, not cosine, because here I am shaping an internal representation, not casting
the final vote; the final vote stays the cosine-softmax over f(x̂, S) and g(x_i, S), where the
magnitude-invariance argument earns its keep. The LSTM cell here reads a concatenation of width
640 + 640 = 1280 into a hidden width 640, which is another 4·(640·1280 + 640² + 2·640) ≈ 4.92M
parameters; so the two FCE modules together add roughly 11.5M parameters on top of the backbone —
nearly doubling the model, all of it sitting in front of a per-episode signal of at most a couple dozen
support vectors.

Before I commit I should trace the shapes end to end once, because a per-point vote with two recurrences
has several places a transpose or a normalization axis could be silently wrong. Support features come in
as 25×640; the bidirectional LSTM emits 25×1280, I split it into the two 640-wide halves and add both to
the raw 25×640, leaving contextualized support at 25×640. Query features are 75×640 (5 classes × 15
queries). The query encoder initializes its hidden and cell states at 75×640; each of the 25 loop steps
computes attention (75×640)·(640×25) = 75×25, reads (75×25)·(25×640) = 75×640, concatenates the
re-injected raw query to 75×1280, and the cell returns 75×640, plus the skip — still 75×640 after all 25
steps. The final vote takes the 75×640 query against the L2-normalized 25×640 support transposed to
640×25, giving a 75×25 similarity, softmaxed along the support axis so each query's 25 weights sum to
one, then multiplied by the 25×5 one-hot support labels to give 75×5. That last matmul is what turns 25
per-point weights into 5 per-class masses, and because the softmax rows sum to one and the one-hot
columns partition the 25 supports into 5 classes, each output row sums to one: it is a genuine
distribution over the 5 classes, shape (n_query, n_way) = (75, 5), exactly the contract. Good — no axis
is transposed wrong and the output is already normalized.

That "already normalized" is the fact that decides the loss, and it is worth doing the arithmetic
instead of asserting it. The output ŷ is a convex combination of one-hots, so ŷ is a probability
vector; I take its log and the natural loss is negative log-likelihood, −log ŷ_{true}. The reflex would
be cross-entropy, which the other fills use, so let me actually check what cross-entropy would do to a
log-probability input. PyTorch cross-entropy is log-softmax followed by NLL: it computes
log_softmax(x)_y = x_y − log Σ_j exp(x_j). Feed it x = log ŷ. Then exp(x_j) = ŷ_j and Σ_j exp(x_j) =
Σ_j ŷ_j = 1, so log_softmax(log ŷ)_y = log ŷ_y − log 1 = log ŷ_y, and cross-entropy returns exactly
−log ŷ_y — the *same* number NLL returns. So the two are not different values here; cross-entropy only
happens to agree because ŷ is already normalized, and softmax of the log of a normalized distribution is
the identity. The reason I still write NLL is semantic, not numeric: the output *is* a log-probability,
NLL is its matched loss, and I would rather not lean on the accident that a second softmax is a no-op —
the moment anything perturbs normalization, cross-entropy starts silently re-normalizing under me. And
one such perturbation is already in my code: to avoid log(0) = −∞ when a query puts zero total mass on a
class, I floor the vote with a tiny additive ε = 1e−6 before the log, which makes each row sum to
1 + 5ε instead of 1. Under NLL that just shifts every log by an imperceptible −log(1+5ε) ≈ −5e−6; under
cross-entropy the re-softmax would divide it back out. Both are negligible, but NLL is the honest
description of what the head emits, so NLL it is.

Two more implementation choices I should pin down rather than wave through, because they bite. First,
the cosine vote: strictly, cosine normalizes *both* sides, but if I L2-normalize the query too, the
softmax loses its bite, and I can see why with one line of arithmetic. Each logit is q·ŝ_i where ŝ_i is
a unit support vector, so logit_i = ‖q‖ · cos(q, ŝ_i): the query norm ‖q‖ multiplies every cosine and
acts as an *inverse temperature* on the support softmax. If I normalize the query too, ‖q‖ = 1 and the
logits live in [−1, 1]; a realistic best-vs-second cosine gap of maybe 0.1 becomes a logit gap of 0.1,
and softmax(…, gap 0.1) is nearly uniform — the vote cannot concentrate on the right class. Leave the
query unnormalized and ‖q‖ for a 640-d ReLU embedding is easily order 10, so the same 0.1 cosine gap
becomes a logit gap near 1.0 and the softmax actually peaks. So I normalize the *support* side only and
leave the query "sharp" — a deliberate trade of strict query-side scale invariance for a softmax that
discriminates, and the arithmetic says it is the difference between a vote and a shrug. Second, the
learning rate. The scaffold defaults to plain SGD at 1e-2, which is fine for a fixed-metric mean
classifier, but this method has a bidirectional LSTM *and* a K = 25-step attention loop in the forward
path, so backprop runs through a depth-25 recurrence stacked on the backbone — a much deeper and more
nonlinear computation graph than the plain net the 1e-2 default was chosen for. The global gradient-norm
clip at 5.0 is already in the loop and caps the *total* norm, but it does not cap the *per-step*
amplification a recurrence can build up, and I expect that under 1e-2 the LSTM states blow up and the
output softmax collapses toward uniform. The contract exposes `LR_OVERRIDE`, so I set it to 1e-3 — a
smaller step matched to the recurrent depth — and let the clip handle spikes. is_transductive stays
False: each query is classified on its own against the support, never letting queries see each other.
The full scaffold module is in the answer.

So the delta from the scaffold default is concrete: where the default takes per-class *means* and
scores by L2 distance, I keep every support point, contextualize support and query with LSTMs, and vote
by cosine-softmax over the support labels with an NLL loss. Now let me reason about what this floor
should do, because that is the entire point of running it first. This is the most *intricate* of the
metric methods — a per-point vote with two LSTMs and ~11.5M extra parameters — but intricacy is not
strength here. The vote keeps one weight per support example and never forms a single concise entity
per class, so at 5-shot each class is a scatter of five embeddings the softmax has to integrate, and the
FCE recurrences add capacity that a couple dozen support vectors per episode can barely exercise in a way
that transfers to novel classes. I expect the benchmarks to split on one axis: how cleanly the *frozen
cosine vote* over individual points separates the classes. Where classes are visually distinct — generic
objects, CIFAR-like and miniImageNet-like categories — the query's embedding points in a very different
direction from the wrong classes' support, the cosines spread, the softmax concentrates, and the LSTMs
should help a little. Where the test distribution diverges from the broad training sampling —
fine-grained classes that are mutually similar, one bird species against another — the per-point cosine
vote should struggle most: the embeddings of every class point in nearly the same direction, so a
query's cosine to all twenty-five scattered support points is uniformly high and nearly equal, the
softmax over support smears mass across all five classes, and the LSTM context cannot manufacture a
margin that the embedding-plus-fixed-metric does not already have — the skip connection guaranteed it
starts as a no-op, and there is no within-set signal telling it how to open a gap where none exists. So
I predict the weakest benchmark for this method is the fine-grained one, and that the next rung's job is
to find a *cleaner per-class summary and a metric consistent with it*, which is precisely where
collapsing each class to a single prototype — instead of voting over points — points.
