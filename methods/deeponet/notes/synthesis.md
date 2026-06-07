# DeepONet — synthesis (grounded in arXiv 1910.03193 main.tex + DeepXDE pytorch deeponet)

## Verified arXiv id: 1910.03193 (Lu, Jin, Pang, Zhang, Karniadakis 2019; Nature MI 2021)
Canonical impl: DeepXDE (github.com/lululxvi/deepxde), deepxde.nn.pytorch.deeponet (DeepONet / DeepONetCartesianProd). Original: github.com/lululxvi/deeponet.

## Pain point / research question
Learn an OPERATOR G: u -> G(u), a mapping from a function (input function u) to a function (output G(u)), from data. NN universal approximation is usually stated for functions (Cybenko 1989, Hornik 1989). Less-known but stronger: a single-hidden-layer NN can approximate any nonlinear continuous FUNCTIONAL or OPERATOR (Chen & Chen 1995). Want to realize that theorem in PRACTICE: small approximation error is guaranteed by the theorem for large nets, but optimization + generalization errors (the dominant ones in practice) are not addressed. Need an architecture that learns operators accurately/efficiently from a relatively small dataset with small total (approx+opt+gen) error.

## Problem setup
G takes input function u, outputs function G(u); for point y in domain of G(u), G(u)(y) is a real number. NN input = two parts: u (represented discretely) and y; output = G(u)(y).
Represent u discretely by its values at m fixed "sensors" {x_1,...,x_m}: [u(x_1),...,u(x_m)].
Training data: triplet (u, y, G(u)(y)). Same sensors for all u (not necessarily a lattice/grid); NO constraint on output locations y. One u -> many (y, G(u)(y)) data points.

## Universal Approximation Theorem for Operators (Chen & Chen 1995) — THE seed (exact)
sigma continuous non-polynomial; X Banach; K1 subset X, K2 subset R^d compact; V compact in C(K1); G nonlinear continuous operator V -> C(K2). Then for any eps>0 there exist positive integers n,p,m and constants c_i^k, xi_ij^k, theta_i^k, zeta_k in R, w_k in R^d, x_j in K1 (i=1..n,k=1..p,j=1..m) such that:
  | G(u)(y) - sum_{k=1}^p [ sum_{i=1}^n c_i^k sigma( sum_{j=1}^m xi_ij^k u(x_j) + theta_i^k ) ]_{branch} * [ sigma(w_k . y + zeta_k) ]_{trunk} | < eps
for all u in V and y in K2.
Structure: a sum over k=1..p of (branch_k) x (trunk_k). branch_k = shallow net taking [u(x_1..x_m)] -> scalar; trunk_k = sigma(w_k . y + zeta_k) taking y -> scalar.

## Architecture (the method)
- Why >=2 sub-nets: in high-d, y is a d-vector, dimension doesn't match u(x_i); can't treat u(x_i) and y equally -> separate sub-nets for [u(x_1..m)] and y. Naive baseline = FNN on concatenated [u(x_1),...,u(x_m), y] (no structure -> no reason to use CNN/RNN; FNN baseline).
- Theorem gives the structure Eq (sum_k branch_k * trunk_k) but only for SHALLOW (one hidden layer). DEEPEN both sub-nets (more expressivity).
- Trunk net: input y -> output [t_1,...,t_p] in R^p. Note trunk applies activation in its LAST layer: t_k = sigma(.). (consistent with theorem's sigma(w_k.y+zeta_k))
- Branch net(s): input [u(x_1..m)] -> output b_k in R, k=1..p.
- Merge: G(u)(y) approx sum_{k=1}^p b_k t_k. (dot product of branch vector and trunk vector over the p latent dim)
- Interpretation: trunk-branch net = trunk net whose last-layer weights are each parameterized by a branch net rather than being free scalars. (b_k plays role of c_i^k aggregation.)
- Bias: theorem has NO bias in last layer of b_k; adding bias b_0 (and biases in branch last layer) NOT necessary by theorem but REDUCES generalization error & training uncertainty: G(u)(y) approx sum_k b_k t_k + b_0.

## Stacked vs Unstacked
- Stacked DeepONet (Fig C, = literal theorem): ONE trunk + p SEPARATE branch nets stacked in parallel, branch_k outputs scalar b_k.
- Unstacked DeepONet (Fig D): ONE trunk + ONE branch net that outputs the whole vector [b_1,...,b_p] in R^p.
- Why unstacked: p is at least ~10; p separate branch nets is compute/memory expensive. Merge into one branch net.
- Empirics (motivating): unstacked has LARGER training error but SMALLER test error (smaller GENERALIZATION error: tighter train-test MSE correlation, ~linear). Fewer params, faster, less memory. => use unstacked with bias. (These are comparisons among the proposed variants -> keep as design rationale, not as "the method beats baselines".)

## Inductive bias (the WHY it generalizes)
G(u)(y) has two independent inputs u and y; explicitly splitting into branch (handles u) and trunk (handles y) encodes that prior -> strong inductive bias -> good generalization even with plain FNN sub-nets (like CNN's bias for images, RNN's for sequences). G(u)(y) = a function of y conditioned on u; branch provides the conditioning. DeepONet is a high-level architecture; inner sub-nets can be FNN (chosen here, simplest), CNN (needs grid), attention.

## Number-of-sensors theory (Sec, Theorem on m)
For ODE operator G mapping u to solution s. L_m = operator mapping u to u_m (poly interpolation/reconstruction at m sensors). Use compactness: U_m, W_m = V union U_m, W = union W_i compact (Lemma). G continuous -> G(W) compact. Then there exist m and a (shallow) net s.t. for all u: ||(Gu)(d) - (W2 . sigma(W1.[u(x_0)...u(x_m)]^T + b1) + b2)||_2 < eps. I.e. enough sensors m make the discretized input represent u well enough to hit accuracy eps.
Empirical (motivating, on existing-system behaviour): error decays ~exponentially with m at first (MSE ~ 4.6^{-#sensors}), then plateaus; transition ~10 sensors.

## Data generation / spaces (context-level facts)
Input function spaces: Gaussian random field GRF G(0, k_l) with RBF kernel k_l(x1,x2)=exp(-||x1-x2||^2/2l^2), length-scale l (larger l = smoother u); orthogonal (Chebyshev T_i) polynomials V_poly = {sum a_i T_i(x): |a_i|<=M}. Solve ODE by Runge-Kutta(4,5), PDE by 2nd-order finite difference for reference G(u)(y).
Example operators: antiderivative (linear), ODE g=-s^2+u (nonlinear), gravity pendulum.

## Load-bearing ancestors
- Universal approximation for FUNCTIONS (Cybenko 1989, Hornik 1989): NN approximates any continuous function. Doesn't cover function->function maps.
- Chen & Chen 1995 universal approximation for OPERATORS: the seed structure (branch x trunk sum). Shallow only.
- FNN baseline: concatenate [u(x_1..m), y] -> FNN. Works (approx) but large generalization error.
- Image-to-image CNN PDE approaches (Winovich 2019, Zhu 2019): require equispaced grid for both u sensors and y. DeepONet doesn't.
- Parametrized-PDE coefficient ID (Brunton 2016, Rudy 2017): only identify coefficients, not general operators.
- GMLS-net (Trask 2019): unstructured but only LOCAL operators (not integral). DeepONet learns integral ops too.

## Design decisions -> why
- Discretize u at fixed sensors: must feed a function to a net; finite values at shared sensors is the weakest-constraint encoding.
- Two sub-nets (branch/trunk) not one concatenated FNN: u-values and y are different objects (dims, role); the split encodes the (u,y) independence -> inductive bias -> lower generalization error. Theorem also dictates the product structure.
- Branch x trunk via dot product sum_k b_k t_k: directly the theorem's sum-of-products; p latent basis functions t_k(y) with input-dependent coefficients b_k(u).
- Trunk activation in last layer: matches theorem's sigma(w_k.y+zeta_k); t_k are basis functions.
- Deepen sub-nets beyond the shallow theorem: expressivity.
- Bias b_0 (and branch last-layer bias): not needed for approximation but lowers generalization error and training variance.
- Unstacked (single branch outputting [b_1..b_p]) over stacked (p branches): fewer params, faster, less memory, AND smaller generalization/test error.
- FNN sub-nets: simplest; CNN needs grid; attention possible for general settings.

## Code (DeepXDE pytorch) structure
branch = FNN(layer_sizes_branch); trunk = FNN(layer_sizes_trunk); trunk has activation on last layer; b = learnable bias param init 0.
forward: x_func = branch([u(x_1..m)]); x_loc = activation(trunk(y));  (both R^p)
merge: y = einsum("bi,bi->b", x_func, x_loc) -> unsqueeze -> += b   (single point)
       or einsum("bi,ni->bn", ...) for cartesian-product grid (each u against all y).
