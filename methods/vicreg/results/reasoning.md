I start with the part that is not controversial: two augmented views of the same image should land
near each other in embedding space. If I write only that, a constant encoder wins immediately. Every
view of every image agrees with every other view because all outputs are the same. So agreement is
necessary, but it is not a representation-learning objective by itself.

The usual fixes all add a force that keeps the representation from becoming constant. Contrastive
learning uses other images as negatives. That is conceptually clear, but it turns collapse prevention
into a large-sample repulsion estimate. Clustering methods use balanced assignments. That prevents an
empty-code solution, but the anti-collapse mechanism is now an assignment solver and a set of
prototypes. BYOL and SimSiam are appealing because they remove negatives, yet the scalar similarity
loss still has constant solutions; the method depends on asymmetry, a target network or a
stop-gradient, a predictor, and normalization effects. Barlow Twins and W-MSE feel closer to the
statistic I want because they talk about redundancy and whitening, but Barlow's statistic is a
cross-branch correlation matrix and W-MSE applies a whitening operator. What I would like instead is a
loss that names the forbidden event directly, as an ordinary statistic of one embedding batch, so that
no extra network, queue, or operator is doing the collapse-prevention work off to the side.

The most literal forbidden event is zero spread. Take one embedding batch \(Z\in R^{n\times d}\).
For coordinate \(j\), let \(z^j\) be that column across the minibatch. If the representation has
collapsed to a constant, every \(z^j\) has zero variance. So the first anti-collapse term should
penalize small spread. I do not want to maximize variance forever, because then the scale runs away
and the term competes with agreement indefinitely; I want each coordinate to keep at least a fixed
amount of spread and then be left alone. A hinge does exactly that:
\(v(Z)=d^{-1}\sum_j \max(0,\gamma-\sqrt{\operatorname{Var}(z^j)+\epsilon})\). Once a coordinate is
above the floor \(\gamma\), it stops receiving pressure from this term. With \(\gamma=1\), the scale
of the embeddings is set by this loss, so I do not also need l2 normalization or batch standardization
to pin it down.

Now, should the hinge be on the standard deviation or on the variance itself? Both are zero exactly
at collapse, so on the surface either would flag the constant batch. The difference is in the gradient
near collapse, which is where it matters, because that is the region the optimizer has to climb out of.
Let \(\delta_k=x_k-\bar x\) and use the unbiased sample variance
\(\operatorname{Var}(x)=(n-1)^{-1}\sum_k \delta_k^2\). In the active part of a standard-deviation hinge,
\(\gamma-\sqrt{\operatorname{Var}(x)+\epsilon}\), the derivative with respect to \(x_k\) is
\(-\delta_k/((n-1)\sqrt{\operatorname{Var}(x)+\epsilon})\). For a variance hinge,
\(\gamma-\operatorname{Var}(x)\), the derivative is \(-2\delta_k/(n-1)\). The variance-hinge gradient
carries a factor of \(\delta_k\) with no compensating denominator, so it shrinks toward zero as the
column approaches its mean; the standard-deviation hinge has a \(\sqrt{\operatorname{Var}}\) in the
denominator that partly cancels the shrinking numerator. I want to see how big that difference actually
is, so I take a near-collapsed column \(x=(0.51,0.49,0.50,0.50)\) with \(n=4\), mean \(0.5\), and
deviations \((0.01,-0.01,0,0)\), and \(\epsilon=10^{-4}\). Then
\(\operatorname{Var}(x)=6.67\times10^{-5}\), and \(\sqrt{\operatorname{Var}+\epsilon}=0.01291\). The
standard-deviation hinge gradient on the first sample is
\(-0.01/(3\cdot0.01291)=-0.2582\); the variance hinge gradient there is
\(-2\cdot0.01/3=-0.00667\). So the standard-deviation hinge pulls about \(39\) times harder on the
deviating sample. If I scale the deviations down by ten and by a hundred — closer to true collapse —
the ratios become \(49.8\) and \(50.0\). The amplification not only persists but grows as the batch
nears collapse and saturates near \(1/\sqrt{\epsilon}\) up to the constant. That is the behavior I want
from an anti-collapse term, and it settles the choice: standard deviation, not variance.

I have to be honest about one edge case so I do not overclaim. At an exactly constant column every
\(\delta_k=0\), so the gradient through the embedding values is exactly zero even though the loss is
positive (it equals \(\gamma\)). So the standard-deviation hinge does not magically push an exactly
constant batch off the constant point through this term's embedding gradient. What the computation
above does establish is the claim I actually need: the constant batch is not a minimum of the
objective — it has strictly positive loss — and any near-constant batch with even tiny nonzero
deviations gets a strongly amplified restoring gradient that grows as collapse approaches. With
\(\epsilon=0\) the standard-deviation derivative would be scale-invariant (the scale cancels between
numerator and denominator) and would not vanish under uniform shrinking of the deviation pattern; the
implemented \(\epsilon=10^{-4}\) keeps the gradient finite while preserving that amplification until
the variance falls well below \(\epsilon\). That distinction matters for mathematical honesty.

A spread floor still leaves a second failure mode. Every coordinate can have standard deviation one
while all coordinates copy the same scalar signal. That representation is not trivially constant, but
it is informationally collapsed into a low-dimensional subspace, and the spread floor cannot see it,
because each column individually looks healthy. To rule that out I need a statistic between
coordinates. The natural one is the covariance matrix
\(C(Z)=(n-1)^{-1}\sum_i (z_i-\bar z)(z_i-\bar z)^T\). Its diagonal is already handled by the spread
term; the off-diagonal entries tell me whether two coordinates move together. Penalizing
\(c(Z)=d^{-1}\sum_{i\ne j}C(Z)_{ij}^2\) drives redundant linear co-variation toward zero.

I want to check that this term actually fires on the failure mode the spread term misses. I build a
six-sample batch with three coordinates: coordinates 0 and 1 are the same unit-variance signal
\(s\), and coordinate 2 is essentially zero. The column standard deviations come out
\((1.0,\,1.0,\,6\times10^{-10})\), so the variance hinge reads \((0,0,1)\) — it sees only the dead
third coordinate and is perfectly satisfied with the two copied ones. The covariance matrix has
\(C_{01}=C_{10}=1\), and the off-diagonal sum of squares is \(2.0\), driven entirely by the copied
pair. So on this batch the variance term is blind to the redundancy and the covariance term reports it
exactly. The two statistics are genuinely complementary: neither one alone would penalize this
configuration on both fronts.

The complementarity runs the other way too, and this is why the covariance term cannot stand alone. If
all embeddings are constant, every off-diagonal covariance is zero, so the covariance loss is perfectly
happy with the collapsed solution — and the spread term is the thing that blocks it. Conversely, once
spread is forced and redundancy is penalized, the forced variance has to occupy many coordinates rather
than pile into one repeated direction. The two terms have different jobs, and the small examples above
show each one is satisfied by the other's failure mode, so neither substitutes for the other.

The agreement term then decides what the spread and decorrelation are about. Without view agreement,
a model could manufacture any sample-dependent high-variance decorrelated code that has nothing to do
with image content. With agreement, the code has to be stable across the two sampled augmentations. So
the loss has three pieces: plain MSE between paired embeddings, a per-branch standard-deviation floor,
and a per-branch off-diagonal covariance penalty.

I should also avoid normalization unless it is needed for optimization inside the head. L2-normalizing
the embeddings would constrain coordinate standard deviations to the unit-sphere scale, roughly
\(1/\sqrt d\), which fights the explicit scale floor of \(\gamma=1\). Batch-standardizing the final
embeddings would rescale every column to unit variance, which turns the covariance penalty into a
correlation penalty and hides the very scale that the variance term is meant to control. Batch
normalization can still be useful in hidden MLP layers, where it only helps conditioning, but the loss
itself should see unnormalized final embeddings.

There is one more thing I need to get right before writing code: the published equations and a faithful
implementation will not use identical normalizing constants, and I should not pretend they do. The
algebra writes the invariance term as \(n^{-1}\sum_i \|z_i-z'_i\|_2^2\), a mean over the \(n\) sample
pairs, and the variance part as \(v(Z)+v(Z')\). A direct PyTorch implementation reaches for
`F.mse_loss(x, y)`, which is an elementwise mean over the \(n d\) entries — off from the per-sample mean
by a factor of \(d\) — and it is natural to write the standard-deviation term as half the mean hinge
from one branch plus half from the other. Those are constant rescalings, and they get absorbed into the
tuned coefficients. So I have a decision: if I present code, it follows the implementation's constants;
if I present the algebra, I label it as the published form, and the coefficients differ accordingly.

The shape is now stable. Compute embeddings for both views, take their unnormalized MSE, gather
embeddings across distributed workers before estimating batch statistics, center each branch, apply the
standard-deviation hinge with \(\gamma=1\) and \(\epsilon=10^{-4}\), compute each branch's covariance
with denominator \(B-1\), square only the off-diagonal entries and divide by the embedding dimension,
then combine with coefficients tuned around 25, 25, and 1 in the implementation. What I end up with is
not a contrastive loss, not a whitening transform, and not a dynamics-only trick: it is an agreement
loss whose two explicit batch-statistic regularizers — verified above to fire precisely on the two
collapse modes, and on different ones — make zero spread and redundant coordinates costly.
