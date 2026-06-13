The head is the whole game, and with nothing yet measured the floor I have to start from is the
simplest thing that could possibly map a 1280-dimensional ESM-2 embedding to a fitness score: a
linear predictor. The scaffold even hands it to me as the default fill — a bare `nn.Linear(1280, 1)`
over the raw mutant embedding. So my real question at step one is not "what is the fanciest head"
but "what is the *right* linear head," because a linear model on these particular features has a
specific, diagnosable failure mode that I should fix before I add any nonlinearity on top, and the
fix is so cheap that there is no excuse to skip it.

Let me write down what the linear head is actually doing. I have a design matrix `X` whose rows are
per-mutant feature vectors and a target `y` of fitness scores, and I want a weight vector `w` such
that `x'w` predicts `y`. The two-century-old recipe is ordinary least squares: minimize the residual
sum of squares `||y - Xw||^2`, set the gradient to zero, get the normal equations `X'X w = X'y`, and
if `X'X` inverts, `w = (X'X)^{-1}X'y`. Under Gaussian noise that is the maximum-likelihood estimate,
and Gauss-Markov crowns it the best linear *unbiased* estimator there is. By that theorem I should be
done. On a clean, orthogonal design I would be. The trouble is the design I actually have.

The features are 1280 ESM-2 dimensions, and they are nothing like orthogonal. A protein language
model's hidden coordinates are massively redundant — many directions encode overlapping structural
and evolutionary signal, so whole blocks of the 1280 are nearly proportional to each other. That
redundancy is exactly the disease least squares cannot survive, and I want to see *how* it kills the
estimate, because the shape of the failure dictates the shape of the cure. The quality of the OLS
estimate lives in its covariance `Cov(w) = sigma^2 (X'X)^{-1}`, so I diagonalize `X'X`. Write the
singular value decomposition `X = U D V'`; then `X'X = V D'D V' = sum_j d_j^2 v_j v_j'`, where the
`d_j` are the singular values of `X` and the `v_j` the right singular vectors. Inverting a symmetric
matrix inverts its eigenvalues on the same eigenvectors, so

  `Cov(w) = sigma^2 sum_j d_j^{-2} v_j v_j'`.

There it is in black and white: the variance of the estimate along direction `v_j` is
`sigma^2 / d_j^2`. A direction with a *small* singular value gets its variance blown up by
`1/d_j^2`, and a small singular value is precisely what redundant features create — a combination of
correlated ESM-2 coordinates along which `X` barely moves, so `d_j` is tiny, so the data carries
almost no information about `w` in that direction and the noise rushes in to fill the vacuum. That is
why an unregularized linear head on these embeddings produces coefficients of insane magnitude that
flip sign on the slightest perturbation of the training fold, and it is why the test-fold rank
correlation would be erratic: the predictor is reading mostly noise along the ill-conditioned
directions. Push the redundancy to the limit and some `d_j` hits zero, `(X'X)^{-1}` ceases to exist,
and OLS does not even return a unique answer. So the diagnosis is exact — tiny eigenvalues of `X'X`,
and the `1/d_j^2` amplification they cause — and any cure has to attack that amplification directly.

My first instinct is "just delete the offending directions" — drop correlated features, or project
onto the top principal components of `X` and regress only there. That does kill the `1/d_j^2`
blow-up because it deletes the small-`d_j` directions outright. But it is a guillotine, and two
things bother me. It is all-or-nothing, so the estimate is a discontinuous, high-variance function
of where I draw the cut. And worse, principal-component regression discards directions by their
variance *in `X`*, with no reference to whether they help predict `y`; a direction can have a small
`d_j` and still carry genuine fitness signal, and PCR throws it out with the noise. On embeddings as
correlated as these, where the predictive signal is spread across many partially-redundant
coordinates, a hard cut is the wrong tool. I want something that *damps* the unstable directions
smoothly, in proportion to how unstable they are, while leaving the stable ones alone. A dial, not a
switch.

So what is the smallest honest thing I can do to stop the `1/d_j^2` explosion? The explosion is
`d_j^{-2}` for small `d_j`. If I replace `d_j^2` by `d_j^2 + lambda` for some `lambda > 0` before
inverting, then for the healthy large-`d_j` directions the floor is negligible and nothing changes,
but for the sick small-`d_j` directions the denominator cannot fall below `lambda`, so the variance
is capped instead of diverging. Adding `lambda` to every eigenvalue is, in matrix terms, replacing
`X'X` by `X'X + lambda I`. That single move does a lot at once: `X'X` is positive semidefinite,
`lambda I` is positive definite, their sum is positive definite, so `X'X + lambda I` is *always*
invertible — even when `X'X` was singular, even when features outnumber samples — and along each
direction it inverts `d_j^2 + lambda` instead of `d_j^2`, the smooth proportional damping I wanted.
The estimator `w(lambda) = (X'X + lambda I)^{-1}X'y` is well-defined across the whole range from
orthogonal to super-collinear designs.

That looks like an ad-hoc patch on the linear algebra, and I do not trust patches, so I check what
objective it is the honest optimum of. Suppose I minimize not the plain residual sum of squares but a
penalized version that also charges me for large coefficients,
`L(w) = ||y - Xw||^2 + lambda ||w||^2`. Its gradient is `-2X'(y - Xw) + 2 lambda w`; set it to zero
and `(X'X + lambda I)w = X'y`, exactly the diagonal-loaded solve. So the patch is not a hack at all
— it is the unique global minimizer (the Hessian `2(X'X + lambda I)` is positive definite, so `L` is
strictly convex) of least-squares-plus-an-L2-penalty. The parameter `lambda` is reframed instantly:
not a fudge factor on a matrix, but the *strength of the penalty*, the exchange rate between fitting
the data and keeping the coefficients small. At `lambda = 0` I recover OLS; as `lambda` grows the
estimate shrinks toward zero.

Pushing the SVD through `w(lambda) = V (D'D + lambda I)^{-1} D' U' y` shows what the penalty does
direction by direction: each principal direction's coordinate is scaled by `d_j^2 / (d_j^2 + lambda)`,
a factor in `(0, 1]` — near 1 for the well-conditioned large-`d_j` directions, near 0 for the
unstable small-`d_j` ones. It is the soft version of dropping principal components, damping each
direction exactly in proportion to its instability. The estimator is biased — `E[w(lambda)] != w` —
but Gauss-Markov only protects the unbiased class, so stepping out of it deliberately is not refuted
by the theorem; the relevant question is total error `MSE = variance + bias^2`, and near `lambda = 0`
the variance saving is first-order in `lambda` while the bias cost is only second-order, so there is
always a positive penalty that strictly beats OLS in total mean squared error. That is the whole
justification for adding the penalty at all.

Now the part that matters for *this* task, where the README is blunt that a same-named baseline can
be a different method than the textbook one: I am not going to form `X'X + lambda I` and solve it in
closed form. The harness is fixed and it does not hand me a closed-form solver — it hands me an
`nn.Module` trained end-to-end by AdamW with MSE loss, a cosine schedule, and early stopping. So I
need the *training-loop* form of the ridge penalty, not the normal-equations form. And there is a
clean one: a single linear layer trained with squared-error loss while the optimizer applies decoupled
weight decay to the layer's weights. In an AdamW step, `weight_decay = lambda` multiplies the weights
by `(1 - lr * lambda)` each update independently of the gradient — that is exactly the gradient of the
`lambda ||w||^2` penalty applied as shrinkage, so AdamW weight decay *is* the ridge penalty realized
in the training loop. The fixed loop already uses AdamW and already exposes `weight_decay` through
`CONFIG_OVERRIDES`. So ridge regression here is literally: `MutationPredictor` is one `nn.Linear`,
and I set `weight_decay` to a value strong enough to tame the ill-conditioning of the embedding
features. I will use `weight_decay = 5e-2`, an order of magnitude above the scaffold's mild default,
because the diagnosis above says these correlated 1280-d features need real shrinkage, not a token
amount.

One design choice in the fill is load-bearing and worth stating: which input the linear layer reads.
The scaffold default reads the raw mutant `embedding`. But the quantity that actually encodes *what
the mutation did* is the `delta_embedding` — the mutant-minus-wild-type shift. The raw embedding is
dominated by the protein's identity, which is constant across all single mutants of one assay and
therefore carries no within-assay ranking signal; the linear head would have to subtract that constant
itself through its weights and bias, wasting capacity and worsening the conditioning. The delta has
the constant already removed, so a linear map from the delta to the score is a direct readout of the
mutation-induced shift in representation space. So the ridge head reads `delta_embedding`, not
`embedding`. This is a real divergence from the bare textbook "linear regression on the features":
the feature here is deliberately the *difference* feature, and that choice is part of what makes a
plain linear head competitive at all. The distilled module — one linear layer on the delta, plus the
`weight_decay = 5e-2` override — is in the answer.

With nothing measured yet, let me reason about what this floor should and should not do, because
that is the entire point of running it first. A linear readout of the delta is a strong, honest
baseline precisely when the fitness landscape is approximately linear in the embedding shift — when
"how far and in which direction the mutation moved the representation" maps roughly monotonically to
fitness. ESM-2's representation space is built so that functionally similar sequences sit near each
other, so I expect this linearity to hold *partly*: the head should capture the coarse direction of
effect and land a respectable Spearman on assays where the dominant signal is a smooth gradient, but
it should leave real correlation on the table wherever the mapping from delta to fitness is curved or
threshold-like — and biology is full of thresholds (a destabilizing mutation does nothing until it
crosses a stability cliff, after which the protein is dead). I therefore expect the three assays to
split on how linear-in-delta each one is. An assay whose fitness varies smoothly with the embedding
shift should give a solid Spearman; an assay dominated by a sharp stability threshold — where most
mutations are tolerated and a minority are catastrophic — is exactly where a single linear hyperplane
through delta-space should struggle most, because no linear function can bend to put the cliff in the
right place, and the rank correlation there could be far weaker, even near zero or negative if the
linear fit latches onto the wrong axis.

Whatever the precise split, the diagnosis is already pointed at the next rung. If the linear head
underperforms on the curved/threshold assay while holding up on the smooth ones, the problem is
representational — a hyperplane cannot bend — and the fix is to let the head learn a nonlinear
function of the same delta: insert a hidden layer with a nonlinearity between the input and the
readout, so the predictor can carve the delta-space into regions and put the stability cliff where it
belongs. That is precisely the move a multilayer head makes, and it is the natural step two. For now,
step one is the regularized linear floor: one linear layer on the delta, AdamW weight decay as the
ridge penalty, and a clear expectation that it will be respectable everywhere and weak exactly where
the fitness-versus-delta relationship is most curved.
