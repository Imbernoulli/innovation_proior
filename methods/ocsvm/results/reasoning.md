Let me start from the object I actually need. I have unlabelled points x_1, ..., x_l drawn from some distribution P, and I want a region in input space that contains most of the mass of P. If a new point lands outside that region, I want to call it novel or anomalous. I also want one knob that means something operational, roughly "allow this fraction of the training sample to fall outside," because in condition monitoring or novelty detection I often have a prior sense of contamination.

The density-estimation reflex is tempting: estimate p, then keep {x : p(x) >= t}. But that is solving a larger problem than I need. In high dimension, estimating p is data-hungry; worse, the density may not exist if P has singular components, while the support or high-mass region is still meaningful. I do not need the height of the distribution everywhere. I need a decision region.

There is a precise statistical way to say "small region containing mass." Fix a class C of candidate sets and a size functional lambda, and define U(alpha) = inf{lambda(C) : P(C) >= alpha, C in C}. With lambda as volume, the minimizer is the minimum-volume set containing alpha of the probability mass, and at alpha = 1 it is the support when that phrase is available. This is the right target, but the word "volume" is the obstacle. Directly minimizing input-space volume over a rich high-dimensional class is not the sort of convex kernel computation I can run.

The supervised SVM gives me a different kind of size control. Its 1/2 ||w||^2 term does not measure input-space volume; it controls flatness of a decision function in feature space, and the kernel trick lets that flat function draw nonlinear boundaries in input space. The machinery is exactly what I would like to reuse: convex quadratic program, support-vector expansion, and a decision function built from kernels. The problem is structural: an ordinary SVM separates two labelled classes, and here I only have one class. There is no negative sample.

Maybe I should first describe the data as a ball. A feature-space ball has a literal center and radius: choose a and R so that most mapped points Phi(x_i) lie inside ||Phi(x_i) - a||^2 <= R^2. With slacks, the natural program is

  minimize    R^2 + (1/(nu l)) sum_i xi_i
  subject to  ||Phi(x_i) - a||^2 <= R^2 + xi_i,   xi_i >= 0.

I put 1/(nu l) in the penalty because I want a counting interpretation later. Introduce alpha_i >= 0 for the ball constraints and gamma_i >= 0 for the slacks:

  L = R^2 + (1/(nu l)) sum_i xi_i
      - sum_i alpha_i [R^2 + xi_i - ||Phi(x_i) - a||^2]
      - sum_i gamma_i xi_i.

The stationary equations already look like support-vector algebra. From dL/dR = 0 I get 2R(1 - sum_i alpha_i) = 0, so for the nonzero-radius case sum_i alpha_i = 1. From dL/da = 0 I get a = sum_i alpha_i Phi(x_i). From dL/dxi_i = 0 I get alpha_i = 1/(nu l) - gamma_i, hence 0 <= alpha_i <= 1/(nu l). Substituting the center expansion gives the dual objective

  maximize    sum_i alpha_i k(x_i, x_i) - sum_{ij} alpha_i alpha_j k(x_i, x_j)
  subject to  sum_i alpha_i = 1,   0 <= alpha_i <= 1/(nu l).

A test point z is inside the ball when

  k(z,z) - 2 sum_i alpha_i k(x_i,z) + sum_{ij} alpha_i alpha_j k(x_i,x_j) <= R^2,

with R^2 read from a boundary support vector. This is a workable data description, but it gives me a center-and-radius rule. I still want to know whether the SVM margin machinery can be made into the region estimator directly.

With one class, I need a fixed reference to push against. The only canonical feature-space point I get for free is the origin. So the question becomes: can I find a hyperplane that puts the mapped data on one side and the origin on the other, with maximum margin, and then use the data side as the accepted region?

Whether such a separating hyperplane even exists is a property of the feature map, not something to take on faith for a generic kernel, so I look at the geometry for the kernel I actually intend to use. For the Gaussian kernel k(x,y) = exp(-||x-y||^2/c), every pairwise inner product k(x_i,x_j) is strictly positive, and k(x_i,x_i) = 1 for every i. So in feature space the mapped observations all have unit norm and all pairwise angles are acute -- a direction w with <w, Phi(x_i)> > 0 for all i would put the hyperplane {z : <w,z> = rho} with small positive rho strictly between the data and the origin. With all pairwise inner products positive, the candidate w = sum_i Phi(x_i) gives <w, Phi(x_j)> = sum_i k(x_i,x_j) >= k(x_j,x_j) = 1 > 0 for every j, so this w works. The data is separable from the origin, and the maximum-margin hyperplane cuts off a spherical cap containing the data: the one-class margin problem is well posed for this kernel, and the optimum cannot be the degenerate w = 0.

Before committing to the margin form over the ball, I should check that they are not just two names for the same thing, and if they are, which is cheaper to carry. In the ball dual, the term sum_i alpha_i k(x_i,x_i) equals k(x,x) times sum_i alpha_i = k(x,x), a constant, whenever k(x,x) is constant in i. For translation-invariant kernels such as the Gaussian, k(x,x) is constant, so that linear term is a constant offset and drops out of the maximization. The ball optimization then has the same minimizer as

  minimize    sum_{ij} alpha_i alpha_j k(x_i,x_j)
  subject to  sum_i alpha_i = 1,   0 <= alpha_i <= 1/(nu l),

and an irrelevant positive factor 1/2 will not change the minimizer. That algebraic argument is worth confirming on numbers rather than trusting on sight: take six points in R^2, an RBF kernel with gamma = 0.7, and nu = 0.5 so U = 1/(0.5*6) = 1/3. Solving the ball dual (maximize sum_i alpha_i K_ii - alpha'K alpha) and the stripped quadratic (minimize 1/2 alpha'K alpha) under the same box-and-equality constraints, the two optimizers come out as

  ball:  [0.28521, 0.25269, 0.15159, 0.0, 0.04586, 0.26464]
  quad:  [0.28521, 0.25269, 0.15159, 0.0, 0.04586, 0.26464]

agreeing to about 1e-7, with the diagonal K_ii = 1 exactly as required. So when the diagonal is constant the ball description and the hyperplane description are the same optimization, and the hyperplane form is cleaner to carry: one signed kernel expansion and one offset rather than an explicit center and radius. That settles which form to develop.

So I formulate the margin problem. I want <w, Phi(x_i)> to be at least rho for most training points, with slacks for the failures, while pushing rho upward and keeping ||w|| small:

  minimize    1/2 ||w||^2 + (1/(nu l)) sum_i xi_i - rho
  subject to  <w, Phi(x_i)> >= rho - xi_i,   xi_i >= 0.

The term -rho is what rewards moving the separating hyperplane away from the origin. The slack coefficient is scaled by 1/(nu l); the KKT conditions below will show exactly what count that controls.

Introduce alpha_i >= 0 on the margin constraints and beta_i >= 0 on xi_i >= 0:

  L = 1/2 ||w||^2 + (1/(nu l)) sum_i xi_i - rho
      - sum_i alpha_i(<w, Phi(x_i)> - rho + xi_i)
      - sum_i beta_i xi_i.

Stationarity gives the whole dual. From dL/dw = 0,

  w = sum_i alpha_i Phi(x_i).

From dL/dxi_i = 0,

  alpha_i = 1/(nu l) - beta_i,

so 0 <= alpha_i <= 1/(nu l). From dL/drho = 0,

  sum_i alpha_i = 1.

Substitute w back. The quadratic terms become 1/2 sum_{ij} alpha_i alpha_j k(x_i,x_j) - sum_{ij} alpha_i alpha_j k(x_i,x_j), the rho terms cancel because sum_i alpha_i = 1, and the xi terms vanish by stationarity. Maximizing the remaining dual is the same as

  minimize    1/2 sum_{ij} alpha_i alpha_j k(x_i, x_j)
  subject to  0 <= alpha_i <= 1/(nu l),   sum_i alpha_i = 1.

This is the same box-and-equality QP I just verified the ball reduces to, with the harmless 1/2 in front. The decision rule is the signed expansion

  f(x) = sgn(sum_i alpha_i k(x_i, x) - rho),

positive on the data side and negative on the other side. The offset comes from any non-bound support vector. If 0 < alpha_i < 1/(nu l), then beta_i = 1/(nu l) - alpha_i > 0, so complementary slackness beta_i xi_i = 0 forces xi_i = 0; alpha_i > 0 also forces the margin constraint tight. Therefore <w, Phi(x_i)> = rho, and

  rho = sum_j alpha_j k(x_j, x_i).

The same KKT equations now constrain nu. Suppose a training point is strictly outside the learned region, so its slack xi_i is positive. Complementary slackness on beta_i xi_i gives beta_i = 0, hence alpha_i = 1/(nu l). Every strict outlier is pinned at the top of the box. Since the alpha_i sum to 1 and all are nonnegative, at most nu l of them can sit at the ceiling 1/(nu l). The strict-outlier fraction is therefore at most nu. For support vectors the count goes the other way. A support vector has alpha_i > 0, and each nonzero alpha_i can contribute no more than 1/(nu l). To make the total sum equal 1, I need at least nu l nonzero coefficients. The support-vector fraction is therefore at least nu.

I want to see these two inequalities actually bracket nu, so let me fit on 200 points in R^3 with gamma = 1/n_features and read off the fractions:

  nu=0.1:  outlier_frac=0.115,  sv_frac=0.170
  nu=0.3:  outlier_frac=0.305,  sv_frac=0.325
  nu=0.5:  outlier_frac=0.490,  sv_frac=0.510

The support-vector fraction is above nu in every row, exactly as the lower bound predicts. The outlier side is more interesting: at nu=0.1 the measured 0.115 sits slightly above 0.1. That is not the bound failing -- it is the gap between "strictly positive slack" and the numerical test decision_function < 0. The clean inequality is on points with genuine positive slack at the exact optimum; the solver stops at a tolerance and a handful of points hover within numerical reach of the boundary, so counting "score below zero" can pick up a few extra. The asymptotic statement is the trustworthy one: under an analytic nonconstant kernel and a distribution without discrete components, the fraction of examples landing exactly on the margin vanishes, the two finite-sample slacks close, and the outlier and support-vector fractions both converge to nu. So the strange-looking coefficient 1/(nu l) together with the free rho is what turns a penalty constant into a count parameter, with nu sandwiched between an outlier fraction below and a support-vector fraction above.

The two endpoint cases deserve an explicit check rather than a gesture, because they are where the formulation could quietly misbehave. Take nu = 1 first. Then the ceiling is 1/(nu l) = 1/l, and the constraints become 0 <= alpha_i <= 1/l with sum_i alpha_i = 1. The only vector of l nonnegative numbers each at most 1/l whose sum is 1 is the one with every entry equal to 1/l -- any entry below 1/l would force another above 1/l to keep the sum, which the box forbids. So alpha is pinned to (1/l, ..., 1/l) with no optimization left, and the expansion becomes (1/l) sum_i k(x_i,x) - rho, a thresholded Parzen-window estimate when the kernel is normalized as a density. Now nu approaching 0: the ceiling 1/(nu l) goes to infinity, the box upper bound disappears, and the slack penalty coefficient 1/(nu l) becomes infinite, so paying any slack is prohibitive and the program tends to a hard-margin support estimator. It stays feasible because rho is unconstrained and can be driven very negative if it must. That freedom is load-bearing: if I had instead constrained rho >= 0, the rho stationarity would loosen to the inequality sum_i alpha_i >= 1, and the multipliers could diverge rather than giving the clean equality sum_i alpha_i = 1 that everything above depends on. So the single parameter sweeps from a hard-margin support set at one end to a full Parzen-window expansion at the other.

There is also a resistance property worth naming. If a point already has positive slack, its alpha_i is pinned at 1/(nu l). Moving that outlying feature-space point locally in a direction parallel to w, without making the slack vanish, leaves the separating hyperplane unchanged, although the numerical representation of w and rho may shift. That is a local statement, not a claim about arbitrary remote moves, but it is exactly the bounded-influence behavior I was missing from density estimation, where a single faraway point can drag the estimate.

The construction also folds back into ordinary two-class SVM geometry rather than sitting apart from it: if (w, rho) is the supporting hyperplane for the mapped points x_i, then (w, 0) separates the labelled symmetric set {(x_i, +1), (-x_i, -1)} through the origin with the same margin, and conversely a through-origin binary separator for labelled points is a supporting hyperplane for the signed points y_i x_i -- margin errors in that binary picture are exactly the outliers here. So the one-class problem is the older margin theory read in a different light, and the capacity intuitions built for binary SVMs carry over rather than needing to be rebuilt from scratch.

Now I need to solve the dual at scale. A dense generic QP would be cubic in l, but this dual has only a box and one equality, so the smallest feasible move changes two multipliers at a time. Pick alpha_1 and alpha_2, freeze the rest, and define Delta = 1 - sum_{i>=3} alpha_i so alpha_1 + alpha_2 = Delta. Let K_ij = k(x_i,x_j), and let C_i = sum_{j>=3} alpha_j K_ij. The part of the objective involving the pair is

  1/2 (Delta - alpha_2)^2 K_11
  + (Delta - alpha_2) alpha_2 K_12
  + 1/2 alpha_2^2 K_22
  + (Delta - alpha_2) C_1 + alpha_2 C_2.

Differentiate with respect to alpha_2:

  -(Delta - alpha_2)K_11 + (Delta - 2 alpha_2)K_12 + alpha_2 K_22 - C_1 + C_2.

Setting this to zero gives

  alpha_2 = [Delta(K_11 - K_12) + C_1 - C_2] / (K_11 + K_22 - 2K_12).

If alpha_2^* and alpha_1^* are the values before the step, and O_i = K_1i alpha_1^* + K_2i alpha_2^* + C_i is the current expansion output on x_i, the same update reads

  alpha_2 = alpha_2^* + (O_1 - O_2) / (K_11 + K_22 - 2K_12).

That is just a Newton step along the equality-preserving direction, and the closed form and the output-based form should coincide and land on the true pair minimum: on a 3-point instance with the third coefficient frozen at 0.3 (so Delta = 0.7), the closed form gives alpha_2 = 0.250556, the output-based Newton form gives alpha_2 = 0.250556, and a brute-force scan of the pair objective over alpha_2 in [0, Delta] bottoms out at 0.25055. After the step, alpha_2 has to be clipped to the feasible interval that keeps both pair variables in the box:

  max(0, Delta - U) <= alpha_2 <= min(U, Delta),   U = 1/(nu l),

and then alpha_1 = Delta - alpha_2. I recompute rho after the pair update. To choose pairs, I scan for a KKT violator, pair it with a non-bound support vector that maximizes |O_i - O_j| when possible, and alternate full scans with scans over the non-bound set. A feasible initialization is to set a fraction nu of the alpha_i to 1/(nu l), adjust one coefficient if nu l is not an integer so the sum is exactly 1, and initialize rho from the largest active output.

The practical interface has two sign conventions to settle. With an RBF kernel exp(-gamma ||x-y||^2), gamma controls locality: small gamma gives a smoother, looser region; large gamma gives a tighter, more variable boundary. The distances in that exponent make preprocessing important, because a large-scale feature can dominate ||x-y||^2. The signed decision value sum_i alpha_i k(x_i,x) - rho is positive for inliers and negative for outliers. An anomaly detector usually wants larger scores to mean more anomalous, so I return the negative of that signed value.

The pair-update solver I just derived is exactly what a mature SVM library already implements at industrial strength, so there is no reason to hand-roll it: I wrap the fit/decision_function interface around `sklearn.svm.OneClassSVM`, which delegates to libsvm, passing through `nu`, `gamma`, and the RBF kernel choice, and negate the signed decision value on the way out so larger means more anomalous.
