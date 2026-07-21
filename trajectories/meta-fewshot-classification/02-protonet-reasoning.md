The matching-vote run told me where the per-point classifier breaks, and it told me in numbers. On the
two generic benchmarks it was respectable — CIFAR-FS 0.769, miniImageNet 0.674 — but on the
fine-grained one it cratered to 0.625 on CUB, the lowest of the three by a wide margin and exactly the
benchmark I flagged as the danger. Before I read a mechanism into that I should check it is not seed
noise, because a wrong causal story here poisons the next step. The per-seed spreads on the
matching run are tiny: CUB std 0.004, CIFAR std 0.002, mini std 0.007. The CIFAR-to-CUB gap is
0.769 − 0.625 = 0.144, which is 0.144 / 0.004 ≈ 36 CUB standard deviations. That is not a fluctuation;
it is a structural deficit that no reseeding will close. So I can trust the split and read it the way I
predicted it: where classes are visually distinct, a cosine-softmax vote over individual support
embeddings has enough margin to work, and the two LSTMs help a little; where every class looks like
every other class — one bird species against another — voting over scattered individual support points
is the comparison with the *least* margin, because the query is being asked to align with twenty-five
separate embeddings, five per class, that all sit close together, and the softmax over points has
nothing crisp to concentrate on. The CUB number is the symptom of a deeper inefficiency: the matching
classifier never forms a single concise *entity* per class. A class is just whichever scattered support
points happen to wear its label, and the prediction is a sum over all of them, so the state the
classifier carries grows with the support set and there is no notion of "the class" to compare against.
And the FCE machinery I added — a bidirectional LSTM over support, a multi-step attention LSTM over the
query, some eleven million extra parameters — is capacity fit to the support-set geometry of the
*training* classes, and on the hard benchmark it is more likely to be reshaping around noise than
opening a margin, because the skip connection guaranteed it starts as a no-op and nothing in a set of
mutually-similar birds tells it how to pull them apart. So the diagnosis is sharp: I have *more*
representational machinery than this problem deserves, pointed at the wrong abstraction. The fix is not
a better metric on the same scatter of points; it is to collapse each class to its minimal summary and
compare against *that*.

So let me ask the question the matching vote refused to ask: what is the *minimal* summary of a class I
could compare a query against, independent of whether it has one example or fifty? If I had to describe
a class with a single object, the most natural single object for a cloud of vectors is its center. Take
the mean of the embedded support points for the class, one vector per class no matter the shot, nothing
to fit. Call it the prototype, c_k = (1/|S_k|) Σ_{(x_i,y_i)∈S_k} f_φ(x_i). Then classify a query by
which prototype it is closest to. This immediately fixes the two things the CUB number exposed: the
per-class state is one vector regardless of shot, and a "class" is now a concrete entity rather than a
bag of points the softmax has to integrate. The nearest-class-mean idea is not new — there is prior
work that represents each class by the mean of its examples and classifies by nearest mean, precisely
so a new class can be dropped in at near-zero cost by averaging its support features. But that work used
a *linear* embedding (a learned Mahalanobis metric on fixed features) for the many-examples regime, and
when it wanted non-linearity it bolted on a separate k-means partitioning step in input space,
decoupled from the metric learning. I want the embedding itself to be the deep ResNet-12 the scaffold
gives me, learned end-to-end by the episodic objective, with the mean computed *in the learned space*.
So: a deep f_φ, prototypes = means of embedded support, classify by distance — trained episodically so
f_φ learns to make the means separable. Everything that hurt on CUB — the scatter, the FCE parameters,
the absent class entity — goes away in one move, and I can count the move exactly: the matching head
carried roughly 11.5M fitted parameters in front of a 25-vector support set; the prototype head carries
*zero*. There is nothing to fit per episode at all. That is the reverse of the bet matching made, and the
36-sigma CUB deficit is my reason to make it.

Now I have to turn "nearest prototype" into something differentiable. A hard argmin over distances has
no gradient. The standard cure is a softmax: p_φ(y = k | x) = exp(−d(f_φ(x), c_k)) /
Σ_{k'} exp(−d(f_φ(x), c_{k'})), and minimize the negative log-probability of the true class,
J(φ) = −log p_φ(y = k | x), averaged over an episode's queries. The closer the query embedding to the
right prototype relative to the others, the smaller the loss; gradients flow through the distances,
through the prototypes, into f_φ. This is clean — it is just cross-entropy on logits that happen to be
negative distances, which means I do *not* repeat the matching method's NLL subtlety: there the output
was an already-normalized vote and the loss had to be NLL to avoid describing a second softmax, but here
the logits are raw negative distances, so cross-entropy (log-softmax then NLL) is exactly right and is
what the scaffold default uses. One care: the numerator's exponent uses c_k for the true class while
the denominator sums over all k′ including k — do not fuse those indices.

But which distance? Cosine is the lazy choice — it is what the matching vote used, and that vote just
posted 0.625 on CUB. Let me not pick by habit; let me pick by consistency, because I now have two
coupled design objects, the distance and the prototype, and I have committed the prototype to be the
*mean*. The right question is: for which distances is "the mean" actually the right representative of a
class — the point that best stands in for all its support points? That is precise and it has a precise
answer. Treat each class as a tiny clustering problem: given its support embeddings, the "best
representative" should be the point minimizing total distance to them. So I ask for which d we have
argmin_c Σ_{x∈S_k} d(f(x), c) = mean of the f(x). There is a known characterization. For the whole
family of *Bregman divergences* — d_φ(z, z′) = φ(z) − φ(z′) − (z − z′)ᵀ∇φ(z′) for strictly convex φ —
the minimizer of summed divergence to a set is *exactly* its arithmetic mean. Let me convince myself
rather than cite it. Differentiate g(c) = Σ_x d_φ(x, c): the φ(x) term is constant in c, and working
through the derivative of the inner product term gives ∇_c d_φ(x, c) = −∇²φ(c)(x − c). Sum over x, set
to zero: ∇²φ(c) Σ_x (x − c) = 0; since φ is strictly convex its Hessian is positive definite hence
invertible, so the only solution is Σ_x (x − c) = 0, i.e. c is the mean. So for *any* Bregman
divergence the mean is the optimal representative — precisely the prototype I already chose; the two
choices are made for each other. And there is a converse worth holding: among smooth distortions, this mean-as-minimizer
property characterizes exactly the Bregman divergences. That tells me my distance had better *be* a
Bregman divergence, or I am averaging points that the distance does not think the average represents.

Is cosine a Bregman divergence? No. Cosine distance, 1 − ⟨z,z′⟩/(‖z‖‖z′‖), cares only about angle,
throws away magnitude, and its best representative of a cluster is not the arithmetic mean — the
minimizer of summed cosine distance is a *direction* (the normalized sum of unit vectors), and
normalizing is exactly the step that breaks the "sum of deviations is zero" argument above, because ∇²φ
is no longer the clean positive-definite object the derivation needed. Let me make the disagreement
concrete with two points in the plane, (2, 0) and (0, 1). Their arithmetic mean is (1, 0.5), which
points at atan(0.5/1) ≈ 26.6°. Cosine first strips both to unit length, (1, 0) and (0, 1), and its best
representative direction is the normalized sum (1, 1)/√2, which points at 45°. So the mean points at
26.6° and cosine's optimal summary points at 45° — they disagree by nearly twenty degrees, and they
disagree precisely because the mean lets the longer vector (2, 0) pull harder while cosine has thrown
that length away. So if I average embeddings into a prototype and then score by cosine, I commit an
inconsistency: I summarized the class with the mean, but cosine does not agree the mean is the right
summary, and the query I score is being compared to a point cosine itself would not have chosen. Squared Euclidean distance, on the other hand, *is* a Bregman
divergence — take φ(z) = ‖z‖², and the construction collapses to ‖z − z′‖² exactly, so the mean is its
optimal representative. This predicts something concrete and testable against the matching run: for a
*mean-prototype* classifier, squared Euclidean should beat cosine, and the gap should be larger here
than it was for the point-wise vote, because it is the *averaging step* that cosine is inconsistent with
— and the matching vote never averaged. The CUB failure is partly this: matching voted over points with
cosine, which at least keeps cosine consistent with its (non-averaging) classifier, but it paid for the
scatter; here I average and so I must switch to the Euclidean partner of the mean to keep the summary
honest. So I will use squared Euclidean — and the scaffold's `l2_distance_to_prototypes` gives me the
negative *non-squared* L2 distance, which is a monotone re-parameterization of the same
nearest-prototype rule and trains fine; the clean Bregman story is stated for the squared form, but the
scaffold helper lands the same classifier.

There is a deeper reading of "squared Euclidean is a Bregman divergence" that surfaces the modeling
assumption I am silently making. There is a bijection between regular exponential-family densities and
regular Bregman divergences. Model each class in embedding space as one component of an
equally-weighted mixture; the posterior over which component generated z is exactly the
softmax-over-negative-divergence-to-prototypes I wrote down — for an equally-weighted mixture of
isotropic unit-variance Gaussians the class-independent (1/K) and (2π)^{−d/2} factors cancel in Bayes'
rule, leaving exactly the softmax over negative squared distance, the ½ nothing but an overall
temperature. So my classifier *is*, literally, the posterior of an exponential-family mixture
with one component per class, and the choice of d picks the class-conditional density. Squared Euclidean corresponds to a spherical, isotropic, unit-variance
Gaussian per class: every class a round blob of the same size around its mean. That is a strong, simple
assumption, and in this data-starved regime "strong and simple" is exactly the bias I want — it is the
opposite of the matching method's parameter-heavy FCE that CUB could not constrain. It also flags the
natural relaxation (a diagonal-Gaussian, i.e. per-dimension variances, which is another Bregman
divergence) so I know where to push if spherical proves too rigid; but I will argue it is redundant with
the embedding's own freedom, since f_φ can already rescale any coordinate before the distance is taken,
so learning per-dimension variances would only re-parameterize a stretch the embedding can apply itself.
One prior fewer to fit, and no loss of expressiveness.

Before I lock in squared Euclidean I should check it is not secretly a weak classifier, since it is
*quadratic* in the embedding. Expand −‖f_φ(x) − c_k‖² = −f_φ(x)ᵀf_φ(x) + 2 c_kᵀ f_φ(x) − c_kᵀ c_k. The
first term is the same for every class k, so it cancels top and bottom in the softmax. What survives,
2 c_kᵀ f_φ(x) − c_kᵀ c_k, is *linear* in f_φ(x) — a plain linear classifier with weights w_k = 2c_k and
bias b_k = −c_kᵀc_k, *induced* by the prototypes, no extra parameters. At first "only linear?" worried
me, but it is the right amount of expressiveness for this regime. Count what the head actually holds: at
5-way with a 640-d embedding the induced weight matrix is 5×640, but not one of those 3200 numbers is a
free parameter — every one is read off five-image means, so there is nothing to overfit. All the
non-linearity lives in f_φ, the deep ResNet-12, trained across 500 episodes per epoch for 200 epochs
where it *can* be constrained. That is the same division of labor every modern classifier uses, and it
is the cure for exactly what hurt the matching run: there, the FCE LSTMs put ~11.5M fitted parameters
into the *per-episode* head where twenty-five images could not pay for it; here the per-task head is
parameter-free and the capacity sits in the shared embedding that sees the whole training stream. I am
putting the parameters where the data can constrain them, and taking them out of where it cannot.

A sanity check that I have not strayed from the method I am reacting to. In the one-shot case the
prototype c_k is just the single support embedding — averaging one point returns it — so the classifier
becomes a soft nearest-neighbour over support points, which is the same *structure* the matching vote
reduces to in one-shot; the only residue of difference is the metric, cosine there against Euclidean
here. So even at K = 1 they are not literally the same classifier, but they share the soft-NN skeleton,
and the real divergence opens at K > 1, where I summarize by averaging and the matching vote does not.
At 5-shot, which is what every benchmark here uses, that averaging is the whole point, which is why I
expect the gain to show up most where the matching scatter hurt most. Should I allow more than one
prototype per class to capture multimodal classes? Tempting, but it would need a *partitioning* of each
class's support — k-means or the like — a separate step decoupled from the gradient updates of f_φ,
breaking the single end-to-end loop I value, and on five support points per class a k-means into, say,
two clusters is fitting a partition to two-or-three-points-per-cluster — more variance than signal. One
prototype per class keeps everything differentiable, and the deep embedding can warp space so a class
becomes unimodal around its mean. The same logic kills FCE outright: a per-episode contextual
re-embedding adds LSTM parameters and an arbitrary set ordering, and the CUB number is the evidence that
in this regime that over-parameterization does not pay. The simple inductive bias is the feature, not a
compromise.

Two episode-composition choices remain. First, training "way": should it match the test 5-way? The
naive answer is yes, but think about what the embedding has to learn. If I train on 5-way episodes, f_φ
only ever has to keep 5 prototypes mutually separable; train on higher way and it must keep many more
separable at once — a strictly harder constraint that forces finer-grained distinctions into f_φ, and
an embedding that separates many classes will comfortably separate 5 at test time. So I would tune
training way *upward* on validation rather than pin it to 5. The scaffold fixes 5-way episodes, so I do
not get that knob here — worth naming as a lever the harness removes, since it is one of the documented
ways to push prototype methods further and I cannot pull it. Second, the shot: here matching matters,
because a prototype's noise level changes with the number of support points averaged — averaging five
embeddings shrinks the per-class summary's variance by a factor of five relative to a single point — so
I want the network to see prototypes at test-time noise during training, and the scaffold's 5-shot
train / 5-shot test already gives me that match for free.

So the edit is the literal scaffold default, and that is the point: the matching run forced me back to
the simplest fill — mean prototypes, score by (negative) distance, cross-entropy — by showing that the
intricate per-point vote with FCE underperforms exactly where intricacy cannot buy margin. The full
scaffold module is in the answer. There is also a scoring reason to spend the change on CUB
specifically: the task score is the *geometric* mean of the three accuracies, dragged hardest by its
smallest factor, so a proportional gain on CUB (the 0.625 low factor) moves the product as much as one
anywhere but is far easier to obtain — lifting 0.625 by 20% needs 0.75, the same 20% on CIFAR would
demand 0.923, which no clean summary delivers. So the highest-leverage move is exactly the weakest
benchmark, which is also where the per-point scatter is the diagnosed cause. Collapsing each fine-grained
class to one prototype and scoring by the Euclidean partner of the mean removes that scatter, so I
expect CUB to jump the most — I will call the threshold past 0.70, a double-digit gain over the matching
vote. On the generic benchmarks the direction is confounded: the matching method ran at its own
LR_OVERRIDE of 1e-3 while prototypes run at the default 1e-2, so any mini/CIFAR change mixes the
classifier with the optimizer. Setting that aside, the mean summary is a strong but rigid prior — a
spherical Gaussian per class — and on already-separable generic classes the matching vote's per-point
flexibility may have bought something the rigid summary gives back, so I expect CIFAR flat or down a
little and mini in the same neighbourhood. A *directional* split — CUB up while CIFAR drops — is the
fingerprint no monotone optimizer change could produce, so it would be the classifier talking, not the
learning rate. And if even the rigid summary leaves the generic benchmarks where the per-point vote left
them, the diagnosis for the next step is already written: the bottleneck is not the *summary* but the
*fixed metric* comparing query to summary — the induced linear head cuts classes apart only with
hyperplanes, a weak knife when the classes are not round separable blobs — and the move is to stop
hand-choosing the comparison and learn it.
