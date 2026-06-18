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
cross-branch correlation matrix and W-MSE applies a whitening operator. I want the loss itself to say
what is forbidden.

The most literal forbidden event is zero spread. Take one embedding batch \(Z\in R^{n\times d}\).
For coordinate \(j\), let \(z^j\) be that column across the minibatch. If the representation has
collapsed to a constant, every \(z^j\) has zero variance. So the first anti-collapse term should not
maximize variance forever; it should require each coordinate to keep at least a fixed standard
deviation. That gives a hinge
\(v(Z)=d^{-1}\sum_j \max(0,\gamma-\sqrt{\operatorname{Var}(z^j)+\epsilon})\). Once a coordinate is
above the floor, it stops receiving pressure from this term. With \(\gamma=1\), the scale is fixed by
the loss rather than by l2 normalization or batch standardization.

I need to be precise about why the hinge uses standard deviation and not variance. Let
\(\delta_k=x_k-\bar x\), and use the unbiased sample variance
\(\operatorname{Var}(x)=(n-1)^{-1}\sum_k \delta_k^2\). In the active part of the hinge,
\(\gamma-\sqrt{\operatorname{Var}(x)+\epsilon}\), the derivative with respect to \(x_k\) is
\(-\delta_k/((n-1)\sqrt{\operatorname{Var}(x)+\epsilon})\). Gradient descent therefore moves samples
above the mean farther upward and samples below the mean farther downward, increasing spread. If I
had used \(\gamma-\operatorname{Var}(x)\), the derivative would be
\(-2\delta_k/(n-1)\), which goes to zero linearly as the column approaches its mean.

There is an edge case I must not overstate. With \(\epsilon=0\), the standard-deviation derivative
does not vanish for a nonzero but tiny deviation pattern as that pattern is scaled toward zero; the
scale cancels between numerator and denominator. With the implemented \(\epsilon=10^{-4}\), the
gradient is finite and still amplifies small deviations until the variance is far below \(\epsilon\).
At an exactly constant column, however, the derivative through the embedding values is zero because
all \(\delta_k\) are zero. So the correct claim is not that the term magically pushes an exactly
constant batch away from the constant point. The correct claim is that the constant batch has positive
loss, is not a minimum of the objective, and near-collapse deviations receive a much stronger
restoring gradient from a standard-deviation hinge than from a variance hinge. That distinction
matters for mathematical honesty.

A spread floor still leaves a second failure mode. Every coordinate can have standard deviation one
while all coordinates copy the same scalar signal. That representation is not trivially constant, but
it is informationally collapsed into a low-dimensional subspace. To rule that out I need a statistic
between coordinates. The natural one is the covariance matrix
\(C(Z)=(n-1)^{-1}\sum_i (z_i-\bar z)(z_i-\bar z)^T\). Its diagonal is already handled by the spread
term; the off-diagonal entries tell me whether two coordinates move together. Penalizing
\(c(Z)=d^{-1}\sum_{i\ne j}C(Z)_{ij}^2\) drives redundant linear co-variation toward zero.

This covariance term cannot stand alone. If all embeddings are constant, every off-diagonal
covariance is zero, so the covariance loss is perfectly happy. The spread term blocks that cheap
solution. Conversely, the spread term cannot see copied coordinates, so the covariance term makes the
guaranteed variance occupy many coordinates rather than one repeated direction. The two terms have
different jobs, and neither substitutes for the other.

The agreement term then decides what the spread and decorrelation are about. Without view agreement,
a model could manufacture any sample-dependent high-variance decorrelated code. With agreement,
the code has to be stable across the two sampled augmentations. So the loss needs three pieces:
plain MSE between paired embeddings, a per-branch standard-deviation floor, and a per-branch
off-diagonal covariance penalty.

I should also avoid normalization unless it is needed for optimization inside the head. L2-normalizing
the embeddings would implicitly constrain coordinate standard deviations to the unit-sphere scale,
roughly \(1/\sqrt d\), which fights the explicit scale floor. Batch-standardizing the final embeddings
would turn the covariance penalty into a correlation penalty and hide the very scale that the variance
term is meant to control. Batch normalization can still be useful in hidden MLP layers, but the loss
itself should see unnormalized final embeddings.

There is one more implementation trap. The published equations and the official PyTorch code do not
use identical normalizing constants. The algebra writes the invariance term as \(n^{-1}\sum_i
\|z_i-z'_i\|_2^2\) and the variance part as \(v(Z)+v(Z')\). The official code uses
`F.mse_loss(x, y)`, which is an elementwise mean over the \(n d\) entries, and it computes the
standard-deviation term as half the mean hinge from one branch plus half from the other branch. Those
are constant rescalings absorbed into the tuned coefficients in the code. If I present code, it has
to follow the implementation; if I present the algebra, I have to label it as the published form.

The final shape is now stable. Compute embeddings for both views, take their unnormalized MSE,
gather embeddings across distributed workers before estimating batch statistics, center each branch,
apply the standard-deviation hinge with \(\gamma=1\) and \(\epsilon=10^{-4}\), compute each branch's
covariance with denominator \(B-1\), square only off-diagonal entries and divide by the embedding
dimension, then combine with coefficients 25, 25, and 1 in the official implementation. The result is
not a contrastive loss, not a whitening transform, and not a dynamics-only trick. It is an agreement
loss whose two explicit batch-statistic regularizers make zero spread and redundant coordinates
costly.
