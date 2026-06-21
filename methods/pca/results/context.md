# Context: fitting and summarizing systems of correlated measurements (circa 1900-1933)

## Research question

We observe `n` individuals, each carrying `q` measured variables `x_1, ..., x_q` — stature and
leg-length on a person, scores on several mental tests, several physical readings on the same
object. The variables are correlated: knowing some tells you something about the others. Two
linked problems press on anyone working with such data.

The first is geometric. We want to draw the line, or the plane, or the lower-dimensional flat
that "fits" the cloud of points best — a compact stand-in for where the data actually lives in
`q`-space. Here every variable is measured with error; none of them is a privileged,
error-free input against which the others are predicted. A man at a given moment has one true
position; both his recorded time and his recorded position vary from trial to trial. We need a
criterion that treats all `q` coordinates on the same footing and returns one unique flat for
the cloud.

The second is reductive. When `q` is large and the variables are tangled together, the raw
coordinate axes are an arbitrary and redundant description — six camera readings of a ball that
really moves along one line, or a battery of tests that mostly reflect a couple of underlying
aptitudes. We would like to re-express the same data in a smaller number of new variables,
computed from the observed correlations, that are mutually uncorrelated and that, taken in
order, carry as much of the information in the original `q` variables as we can keep.

## Background

The field state is built on a few load-bearing pieces.

**Means, variances, correlations.** For `n` points fixed by `q` variables, the first- and
second-order summaries are the means `xbar_i = S(x_i)/n`, the standard deviations
`sigma_i^2 = S(x_i^2)/n - xbar_i^2`, and the correlation coefficients
`r_uv = [S(x_u x_v) - n*xbar_u*xbar_v] / (n*sigma_u*sigma_v)`. By the time of this work these
are the standard descriptors of a correlated system; everything analytic is expected to depend
on the data only through them (and through the means, which merely locate the centroid). The
matrix of correlations `R = [r_ij]` (ones on the diagonal) is the compact object that holds the
pairwise structure.

**The normal correlation surface and its contours.** For jointly normal variables the surface
of constant probability density is a quadric — concentric, similar ellipses in two dimensions,
concentric ellipsoids in `q` dimensions. In two variables, referred to the centroid, the
contour is
```
x^2/sigma_x^2 + y^2/sigma_y^2 - 2 r_xy xy/(sigma_x sigma_y) = const.
```
So the geometry of a correlated normal cloud is literally an ellipse/ellipsoid whose shape is
set by the sigmas and correlations. This much is established (Bravais; the multivariate normal;
the correlation memoirs). The ellipse has a definite orientation and definite axes — these are
properties of the cloud waiting to be characterized.

**The moment of inertia / second moment of a point set.** The sum of squared distances of a
point system from a line or plane is its second moment about that flat (the points "equally
loaded"). A classical fact of dynamics carries over directly: the second moment of a system
about a family of parallel flats is least for the one through the centroid. So any
least-squared-distance fit will be anchored at the mean.

**Quadrics, principal axes, and the secular/characteristic equation.** From celestial
mechanics (the secular perturbation equations of the planets) and the theory of quadric
surfaces comes a standard tool: to find the principal axes of a quadric `sum a_ij x_i x_j =
const` one forms a determinantal equation `det(A - k*I) = 0` of degree `q` in `k`. Cauchy
established that for a symmetric form the roots of this equation are all real, and that they
correspond to the lengths and directions of the principal axes. This "characteristic equation"
machinery — roots real, axes mutually perpendicular, each axis got by substituting a root back
into a set of homogeneous linear equations — is available off the shelf for any symmetric
matrix built from the data.

**Constrained quadratic optimization by Lagrange multipliers.** Extremizing a quadratic form
in the coefficients subject to a normalization constraint (e.g. that a direction be a unit
vector, `sum l_i^2 = 1`) is handled by adjoining the constraint with an undetermined
multiplier and differentiating. This is the routine way any "best direction subject to unit
length" problem gets turned into linear equations.

## Baselines

**The least-squares regression line (Legendre 1805; Gauss 1809; the Galton-Pearson regression
line).** Pick one variable as dependent, say `y`, and choose the line minimizing the sum of
squared residuals *in `y`*, `S(y' - y)^2`, where `y'` is the line's ordinate at the observed
`x`. This is the workhorse fit and it is exact and well understood. It is asymmetric: the
residual it minimizes is measured along one coordinate axis, so if instead you regress `x` on
`y` you minimize `S(x' - x)^2` and get a different line through the same cloud. The
construction presumes one variable is known exactly and the other carries all the error.

**Spearman's general-factor / factor-analytic model (Spearman 1904 and the line of work it
launched).** Posit that the observed scores arise from a small number of latent variables —
classically one dominant "general ability" factor plus specifics — `x_i = a_i*g + (specific)`.
For one common factor this implies the *tetrad* relations among correlations,
`r_ij*r_kl - r_il*r_kj = 0`, which can be tested. The appeal is interpretability: a few
meaningful latent traits. The number of latent factors is fixed in advance by hypothesis, and
the tetrad/consistency conditions are a testable constraint. When fewer factors than variables
are assumed, the latent axes can be rotated freely (a `q`-dimensional rotation's worth of
indeterminacy).

**The two regression lines as a description of cloud orientation.** One could report both the
`y`-on-`x` and `x`-on-`y` regression lines, or split the difference between them, as a
description of the cloud's orientation. The bisector of two regression lines has no extremal
property and no obvious generalization to `q > 2`.

## Evaluation settings

The natural material on which such a method is exercised, all available at the time:

- **Batteries of mental-test scores** on a sample of schoolchildren — e.g. reading speed,
  reading power, arithmetic speed, arithmetic power, given as a correlation matrix among the
  tests (corrected or uncorrected for attenuation). A few highly correlated tests on the order
  of a hundred or more individuals; the question is how many underlying dimensions the scores
  really occupy.
- **Physical/biometric measurements** on a population — several correlated bodily dimensions
  per individual — where one asks for the dominant directions of joint variation.
- **Any system of correlated readings on a common object** (the textbook framing: several
  sensors recording the same dynamics, redundantly).

The data enter only through the means, standard deviations, and correlations (or, with raw
units kept, the covariances). When variables have incomparable units, the standardized form
`z_i = (x_i - xbar_i)/sigma_i` (zero mean, unit variance) is used so the analysis runs on the
correlation matrix; when the units are comparable and meaningful, the covariance matrix is the
object instead. A fit is judged by the residual it leaves and by how much of the total
variation the retained new variables account for — the total being the sum of the per-variable
variances.

## Code framework

A solver is handed the data matrix and a target dimension, and must return the
low-dimensional re-expression. What already exists is the generic numerical-linear-algebra
substrate — forming means and centered data, matrix products, a routine to factor or
diagonalize a symmetric matrix (and, equivalently, to take a singular value decomposition of a
rectangular matrix), and the conventions for making such a factorization's signs reproducible.
What the new variables *are* — how to choose the directions onto which the data is
re-expressed — is exactly the slot to be filled.

```python
import numpy as np
from numpy.typing import NDArray


class DimReduction:
    """Re-express (n_samples, n_features) data in n_components new coordinates.

    The numerical substrate is assumed available: centering, matrix products, and a
    symmetric-eigensolver / SVD routine with a deterministic sign convention.
    """

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64]) -> "DimReduction":
        # X: (n_samples, n_features). Store the directions for later use.
        mean = X.mean(axis=0)
        Xc = X - mean
        embedding, directions = self._solve_centered(Xc)   # TODO
        self.mean_ = mean
        self.directions_ = directions[: self.n_components]
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Xc = X - self.mean_
        return Xc @ self.directions_.T

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # X: (n_samples, n_features). Return (n_samples, n_components).
        mean = X.mean(axis=0)
        Xc = X - mean
        embedding, directions = self._solve_centered(Xc)   # TODO
        self.mean_ = mean
        self.directions_ = directions[: self.n_components]
        return embedding[:, : self.n_components]

    def _solve_centered(
        self, Xc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # TODO: the centered-cloud directions and the coordinates in them.
        raise NotImplementedError
```

The one empty slot is `_solve_centered`: which directions in feature space the centered data
should be re-expressed onto, and what coordinates that choice gives. The rest is bookkeeping
that any such method shares.
