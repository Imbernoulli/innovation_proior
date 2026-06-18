I start with the obstruction. If I freeze the features of a two-layer network at random initialization, I have a kernel method. That can fit functions in the span of its fixed random features, but it has no mechanism for discovering that `f*(x)` really depends only on `S* = span(u_1,...,u_r)`. For a degree-`p` polynomial this is exactly the wrong scaling: the ambient polynomial space has size like `d^p`, while the promised structure only has dimension `r`. So the only useful question is whether the gradient on the first layer contains a recoverable signal for `S*`.

The target is a polynomial, and the input is Gaussian, so the average Hessian is the first object I should inspect. If `f*(x)=g(<u_1,x>,...,<u_r,x>)`, every derivative of `f*` lies in `S*`. Thus `H = E[grad^2 f*(x)]` has column span inside `S*`. If `H` has rank exactly `r`, then it contains the whole subspace. This rank condition is the door the method walks through. If `H` is zero or rank deficient, the second-order signal cannot recover all of `S*`, and I should not pretend that the same argument still works; that is exactly why the lower-bound side matters.

Before I touch the network gradient, I need to remove the easy pieces of the target. The constant piece and linear piece do not require learned nonlinear features, and in the Hermite expansion they appear before the Hessian term. So I compute `alpha = mean_i y_i` and `beta = mean_i y_i x_i`, then work with residual labels `y_i - alpha - beta.x_i`. The final predictor must add `alpha + beta.x` back. If I skip this preprocessing, the derivation is not the same derivation: the `C_0` and `C_1` terms can contaminate the first-layer gradient.

Now I use the symmetric initialization. I pair neurons so that `a_j = -a_{m-j}`, `w_j = w_{m-j}`, and `b_j = b_{m-j}=0`, with `a_j` a sign and `w_j ~ N(0,I_d/d)`. Then the network output at initialization is exactly zero for every input. With the square loss `L = n^{-1} sum_i (f_theta(x_i)-y_i)^2`, the empirical gradient of one first-layer row at initialization is

`grad_{w_j} L(theta_0) = -2 a_j n^{-1} sum_i y_i^res x_i 1_{w_j.x_i >= 0}`.

The sign is important. The residual label appears with a minus sign because `f_theta0=0`, and the ReLU derivative contributes the gate. If I define

`g_n(w) = n^{-1} sum_i y_i^res x_i sigma'(w.x_i)`,

then `grad_{w_j} L(theta_0) = -2 a_j g_n(w_j)`. The first update uses weight decay with `lambda_1 = eta_1^{-1}`, so

`w_j^(1) = w_j^(0) - eta_1(grad_{w_j}L + eta_1^{-1} w_j^(0)) = -eta_1 grad_{w_j}L = 2 eta_1 a_j g_n(w_j)`.

So the decay does not merely regularize the step; it cancels the initial row and leaves a scaled empirical gradient feature. A code implementation that overwrites rows with a separately estimated direction is not this algorithm. It must compute this gated empirical gradient, or an exactly equivalent quantity.

I then check the population object behind `g_n`. Let `sigma(z)=ReLU(z)=sum_k c_k He_k(z)/k!`, so `sigma'(z)=sum_k c_{k+1} He_k(z)/k!`. Stein's identity gives the expansion, after preprocessing,

`E[y^res x sigma'(w.x)] = (C_1-beta)/2 + w(C_0-alpha)/sqrt(2*pi) + H w/sqrt(2*pi) + higher terms`.

Pushing the expansion further, the higher terms are even-Hermite contractions:

`sum_{k>=2} c_{2k} C_{2k}(w^{otimes(2k-1)})/(2k-1)! + w sum_{k>=1} c_{2k+2} C_{2k}(w^{otimes 2k})/(2k)!`.

The constant on the Hessian term is `1/sqrt(2*pi)` because the ReLU coefficient satisfies `c_2 = 1/sqrt(2*pi)`. The preprocessing concentration makes the `(C_0-alpha)` and `(C_1-beta)` terms small, and the next real terms are smaller by powers of the random overlap of `w` with `S*`. Thus the leading informative population feature is `H w / sqrt(2*pi)`.

A random row has overlap `O(d^{-1/2})` with any fixed relevant direction, so `H w` is only `O(d^{-1/2})` in norm when the target scale is constant. The empirical vector average fluctuates at the scale controlled by a uniform concentration bound, and making the Hessian signal dominate costs `n >= O~(d^2 kappa^2 r)`. This is the first price: enough full-batch data to see the feature signal. It also explains why the real first-layer step should be full batch in the scaffold. A mini-batch update is not the object analyzed here.

The second price is the step size. Since the signal in a row is `O(d^{-1/2})`, an `O(1)` learning rate barely moves the row. To make the first layer enter the feature-learning regime in one step, I need `eta_1 = O~(sqrt(d))`. There is no width-normalization correction here because this initialization uses `a_j in {+-1}`, not `a_j = 1/sqrt(m)`. There is also no branch on an information exponent in this method. The method assumes the Hessian signal is present and analyzes that signal.

After the first step, the rows are scaled versions of `g_n(w_j)`, whose population part lies in `S*`. Because `H` has rank `r`, many random probes `w_j` produce enough projected directions to span `S*`. Now I need a rich set of scalar nonlinear features on that learned subspace, so I reinitialize the biases as independent `N(0,1)` draws. That random threshold spread is part of the construction. A deterministic linspace may be a reasonable engineering approximation, but it is not the source algorithm.

At this point the first layer is fixed. The remaining problem is linear regression over the ReLU features `sigma(W^(1)x+b)`, with weight decay on the head. I use the equivalence between weight-decayed least squares and a norm-constrained head to justify that some `lambda` gives a small-norm solution. In code I can train the head with weight-decayed gradient descent to stay literal to the algorithm, or I can explicitly label a closed-form ridge solve as the equivalent fixed-feature solver. What I should not do is present ridge plus extra row-overwrite heuristics as the canonical implementation.

The final theorem follows the shape of this construction. The first-step concentration requires `n >= O~(d^2 kappa^2 r)` and `d >= O~(kappa r^{3/2})`. Then the fixed-feature head gives

`E|f_thetaT(x)-y| - varsigma <= O~(sqrt(d r^p kappa^{2p}/n) + sqrt(r^p kappa^{2p}/m) + n^{-1/4})`.

Informally, the source task costs `d^2 r` samples to learn the representation and `d r^p` more through the reused-data head analysis. If I resample fresh target data after learning the representation, retraining only the head has target complexity independent of `d`, with the bound depending on `r^p/min(m,N)` and `N^{-1/4}`. If the Hessian rank condition fails, I do not get to reuse this proof by reaching for a third-Hermite estimator; the CSQ lower-bound argument shows that such classes can require much larger tolerance or sample complexity.

So the faithful scaffold implementation is now clear. The mini-batch callback is a no-op because the analyzed first step is full batch. In `finalize` I compute `alpha` and `beta`, form residual labels, compute the gated empirical first-layer gradient at the symmetric initialization, apply `W1 = W0 - eta_1(grad_W + eta_1^{-1}W0)`, resample Gaussian biases, train only the head with weight decay, and ensure prediction adds back `alpha + beta.x`. That last affine residual is easy to miss because the fixed `TwoLayerMLP` forward has no skip connection, so the strategy has to wrap or patch the forward method after finalization.
