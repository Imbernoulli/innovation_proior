I start with the discriminator, because in a GAN it is the only channel through which the generator learns. The generator does not see the data distribution directly. It moves because the discriminator assigns scores to generated samples and the gradient of those scores flows back through `G`. If that discriminator becomes badly conditioned, the whole game becomes badly conditioned. Making `D` stronger as a classifier is not automatically good; if it becomes a perfect separator too quickly, it can stop being a useful teacher.

The sharp failure is the support argument. Real images and generated images live near low-dimensional sets in a huge ambient space, and early in training those sets need not overlap in any meaningful way. If the two supports are disjoint, a discriminator can separate them perfectly. On the generated side it can be essentially flat, so the derivative with respect to the input is zero on the region where the generator is asking for guidance. The generator then receives no useful direction even though the discriminator is "right." So I do not want the best possible unconstrained discriminator. I want the best discriminator inside a smoother function class.

The optimal discriminator for the original GAN objective gives me the same diagnosis from another angle. For fixed `G`,

    D*_G(x) = q_data(x) / (q_data(x) + p_G(x)) = sigmoid(f*(x)),
    f*(x) = log q_data(x) - log p_G(x).

Differentiating the log-density ratio gives

    grad_x f*(x) = grad_x q_data(x) / q_data(x) - grad_x p_G(x) / p_G(x).

Those ratios can blow up near the edge of support or become undefined where the density vanishes. The dangerous object is not just the value of `D`; it is the input derivative of the discriminator score. So I need a way to bound how fast the discriminator can change with its input.

That is what the Lipschitz constant measures. If `||f||_Lip <= K`, then `||f(x)-f(x')|| <= K ||x-x'||`, so no part of the score network can develop arbitrarily steep transitions. This does not promise nonzero generator gradients everywhere, and I should not pretend it does. It is an upper control on sensitivity, not a lower control on signal. But it removes the family of arbitrarily sharp separators that causes the most brittle behavior, and it is exactly the kind of restriction WGAN-style methods already need. So a reasonable thing to aim for is

    argmax_{||f||_Lip <= K} V(G,D).

Now the question is enforcement. I need the condition to apply to an actual neural network made of linear or convolutional maps and pointwise activations. For a differentiable map, the local stretching is measured by the largest singular value of its Jacobian:

    ||g||_Lip = sup_h sigma(grad g(h)).

For a linear layer `g(h)=W h`, the Jacobian is just `W`, so the layer's Lipschitz constant is `sigma(W)`, the spectral norm. Composition is submultiplicative, and ReLU or leaky ReLU with slope at most one contributes a factor of one. So for the score network

    f(x) = W^{L+1} a_L(W^L ... a_1(W^1 x)),

I get the bound

    ||f||_Lip <= product_l sigma(W^l).

That is a useful reduction, because it tells me I do not need to control every entry of every matrix, fix a Frobenius norm, or regularize the input gradient at a finite set of sampled points. If I can keep the top singular value of each weight matrix under control, I control the standard product upper bound on the network's Lipschitz constant. The question becomes how to bound `sigma(W^l)` per layer, cheaply, without crippling the layer.

Before I commit to bounding `sigma` directly, I want to be honest about the older weight constraints, because each one already does *something* to the spectrum and I should see exactly what. Weight clipping constrains entries in a box, not the actual operator stretch. It is cheap, but it is a very indirect way to get a Lipschitz condition, and the critic can pay for sensitivity by lining up weights into a low-dimensional effective map. The more interesting case is the pure row-normalized version of weight normalization, with the learned scale removed, so that every output row has unit norm. I want to know what that does to the singular values. The claim I keep seeing is that it fixes a budget, `sum_t sigma_t(W)^2 = tr(W W^T) = d_out`. Let me actually check this rather than take it on faith. I take a random `3 x 4` matrix, normalize each row to unit norm, and compute the singular values:

    rows have norms [1, 1, 1]
    sum_t sigma_t^2 = 3.000000
    d_out = 3

So the identity holds: row-normalization pins the *sum* of squared singular values to `d_out`, not the *top* one. That changes how I read the method. If the discriminator wants to maximize `||W h||` for some unit direction `h` under that fixed budget, the cheapest way is to dump most of the mass into one singular value and starve the rest. Frobenius normalization has the same shape with budget one. These methods are not bounding the top stretch; they are tying together the whole spectrum, so pressure for sensitivity becomes pressure toward low effective rank. Orthonormal regularization over-corrects in the opposite direction by trying to make all singular values one. That prevents collapse, but it also forces directions that may be useless to receive the same scale as useful directions. What I want is narrower than all of these: the largest singular value controlled, and the rest of the spectrum left alone.

The most direct thing to try is to divide the weight by its own top singular value,

    W_bar_SN = W / sigma(W),

and just see what that does to the spectrum. Spectral norm is homogeneous, `sigma(c W) = |c| sigma(W)`, so I expect `sigma(W_bar_SN) = 1` and I expect the ratios `sigma_t/sigma_1` to be untouched, but "expect" is not "checked," so I work a concrete case. I take a deliberately ill-conditioned `2 x 2` layer

    W = [[3, 1],
         [1, 1]],

whose singular values come out as `sigma_1 = 3.414214`, `sigma_2 = 0.585786`, a ratio of `0.171573`. Dividing by `sigma_1`:

    sigma(W / sigma_1) = [1.000000, 0.171573].

The top singular value is exactly one and the ratio `0.171573` is preserved to all printed digits. I push the same test on a random `5 x 5` matrix: the singular values of `W/sigma_1` come out equal to `S/sigma_1` to machine precision, `np.allclose` returns true. So normalizing by the top singular value collapses only the overall scale; the *shape* of the spectrum — how high-rank the layer is, which directions it amplifies relative to its strongest — survives untouched. That is precisely the property the budget-fixing methods destroy. If I apply this to every matrix layer, each matrix factor in the product bound becomes one, so the standard upper bound on `||f||_Lip` is one. If I want a different global Lipschitz scale I can put that scale outside or allocate constants across layers; the normal case is simply to use one.

There is a convolution caveat I must keep straight. A convolutional kernel is stored as a tensor `W in R^{d_out x d_in x k_h x k_w}`. The practical proxy is to reshape it into a matrix of shape `d_out x (d_in k_h k_w)` and control the largest singular value of that matrix. The true operator norm of the discrete convolution also depends on stride, padding, and input size, and I should not claim the flattened-kernel norm equals it. I will use the flattened-kernel norm as the layer quantity and treat the remaining difference as a predefined architecture constant. So for fully connected layers I can speak exactly about the matrix operator norm; for convolutional layers I should say I am controlling the standard flattened-kernel proxy that bounds the intended layer scale up to architecture constants, not that the discrete convolution operator has been exactly normalized in every setting.

The remaining obstacle is computational. `sigma(W)` is the largest singular value, and a full SVD for every layer at every discriminator update is not acceptable in a GAN loop. But I only need the top singular value, so power iteration is the candidate. Starting from a vector `u`, I alternate

    v <- W^T u / ||W^T u||_2,
    u <- W v / ||W v||_2.

Equivalently, `u` is repeatedly acted on by `W W^T`. In the singular-vector basis, the component along `u_t` is multiplied by `sigma_t^2` each round, so the top component should dominate when the largest singular value is separated and the initialization is not orthogonal to the top vector, and then `u^T W v` should approximate `sigma(W)`. How fast is "should" here? On the same `2 x 2` matrix above, starting from `u = (1, 0)`, I run the iteration and watch the estimate:

    iter 1: sigma_est = 3.405877   (error 8.3e-03)
    iter 2: sigma_est = 3.414206   (error 7.3e-06)
    iter 3: sigma_est = 3.414214   (error 6.3e-09)
    iter 4: sigma_est = 3.414214   (error 5.5e-12)

It reaches the true `sigma_1 = 3.414214` to six digits by the third iteration. The error contracts by roughly the factor `(sigma_2/sigma_1)^2 = 0.0294` per step, which is exactly the quadratic decay of the second component I predicted — and even when the conditioning is far less favorable than this, one step still moves the estimate substantially toward `sigma_1`. So a single power-iteration step is not a high-accuracy solver, but it is a cheap and directionally-correct estimate, which is what matters.

That observation changes the algorithm. Restarting the iteration from a fresh random vector every step would throw away exactly the warm start that makes one step useful. The weight changes only by one optimizer step at a time, so the leading singular vector should also move gradually. I can store the current estimate of `u` with the layer and reuse it at the next update. Then one power-iteration step per update is a practical tracker rather than a from-scratch solve. This is not a theorem that one step is always enough; it is the empirical and algorithmic reason the method is cheap, and the convergence trace above is why I believe the warm-started single step tracks well in practice. The layer only needs to persist `u`; it can recompute `v`, update `u`, estimate `sigma`, and use the normalized weight for the layer call.

Now I want to make sure the gradient through the normalization has the right behavior, because the forward pass uses `W_bar = W / sigma(W)` but the trainable parameter is the raw `W`, and `sigma(W)` itself depends on `W`. The standard fact for a simple (non-degenerate) top singular value is

    d sigma(W) / d W = u_1 v_1^T.

This is the kind of identity that is easy to misremember the transpose or sign of, so I check it by finite differences on the `2 x 2` `W`. The leading singular pair is `u_1 = v_1 = (-0.92388, -0.382683)`, so `u_1 v_1^T` predicts

    [[0.853553, 0.353553],
     [0.353553, 0.146447]].

Perturbing each entry of `W` by `1e-6` and recomputing `sigma_1` gives a finite-difference gradient matching that to within `2.5e-10`. So the identity is right as written.

Now the chain rule through `W_bar`. Entrywise,

    d W_bar / d W_ij
      = (1/sigma) E_ij - (1/sigma^2) (d sigma / d W_ij) W
      = (1/sigma)(E_ij - [u_1 v_1^T]_ij W_bar).

Let `h` be the layer input and let `delta := (d V / d (W_bar h))^T` be the backpropagated output signal. Without the normalization, the minibatch gradient for the weight would be `Ehat[delta h^T]`. With the normalization, threading the chain rule through gives

    dV/dW = (1/sigma(W)) ( Ehat[delta h^T] - lambda u_1 v_1^T ),
    lambda = Ehat[delta^T W_bar h].

I do not trust this until I have checked it numerically, because the structure of the correction term is the whole point of the method. I take a linear surrogate objective `V = delta^T (W_bar h)` with a fixed input `h = (0.7, -0.4)` and output signal `delta = (0.5, 0.3)`, evaluate the analytic formula, and compare against a finite-difference gradient of `V` with respect to `W`:

    lambda = delta^T W_bar h = 0.275320
    predicted dV/dW =
      [[ 0.033683, -0.087089],
       [ 0.032997, -0.046957]]
    finite-difference dV/dW =
      [[ 0.033683, -0.087089],
       [ 0.032997, -0.046957]]

They agree to `3.3e-11`. So the formula is correct, including the data-dependent scalar `lambda`. Now I can read off what the correction *does*. The sign in front of `u_1 v_1^T` is negative, and `lambda` here is `+0.275`, the average alignment between the backprop signal and the layer output. When that alignment is positive, the correction subtracts mass from the leading singular direction — it pushes against further growth of the dominant singular component. I should phrase this as an adaptive pressure, not an absolute prevention theorem: the fixed-point condition is only that the ordinary weight gradient aligns with `u_1 v_1^T` up to a scalar, which is far less restrictive than driving the whole matrix to rank one. But it is a real, automatic, sample-dependent regularization that falls out of the parameterization for free.

This also clarifies why a soft spectral-norm penalty is genuinely different and not just a notational variant. A penalty adds something like a coefficient times `sigma(W)` to the objective; it pushes through a fixed regularization weight and competes with the task loss. Dividing by `sigma(W)` instead changes the parameterization used by the forward pass: it sets the layer scale directly per forward, and the correction term I just verified appears in the raw-weight gradient automatically with a coefficient that adapts to the data. So the older regularizer hands me the numerical tool (power iteration) and the right matrix quantity (`sigma(W)`), but not the hard per-forward scale control.

The general normalizer formula makes the contrast precise. For `W_bar = W / N(W)`,

    dV/dW = (1/N) ( grad_{W_bar} V
             - trace((grad_{W_bar} V)^T W_bar) grad_W N ).

If `N` is the Frobenius norm, `grad_W N = W_bar`, so the correction follows the whole matrix direction — it touches every singular direction at once, which is exactly the budget-coupling I measured earlier and want to avoid. If `N` is the spectral norm, `grad_W N = u_1 v_1^T`, so the correction targets only the leading singular pair. That is the design goal stated in one line: control the top stretch while leaving the rest of the spectrum as free as possible. (Specializing this general formula with `N = sigma` reproduces the `lambda u_1 v_1^T` correction I checked above, which is a reassuring consistency.)

There is one optional relaxation. I can write `W_tilde = gamma W_bar_SN` with a learned scalar `gamma`. This gives the layer back a trainable scale, so the layer is no longer fixed to a 1-Lipschitz matrix factor by itself, and some other mechanism such as a gradient penalty must then handle the global Lipschitz control. But as a reparameterization it is still useful: the direction and spectrum shape stay normalized by the top singular value, while the model learns an overall scale.

Now I translate the method into the layer interface. The implementation should not be an optimizer step that permanently overwrites weights after training; it should be a layer computation. Store the raw trainable parameter `W`. Store a persistent estimate `u` as non-parameter state. On each training forward, flatten `W` if it is convolutional, run the configured number of power iterations, update `u`, compute `sigma = u^T W_mat v`, and call the underlying linear or convolution operation with `W / sigma`. The optional `gamma` multiplies this normalized weight. A factor parameter can divide `sigma` before the final division if a different target scale is desired.

The subtle implementation detail is gradient flow. The power-iteration updates for `u` and `v` can use the raw array value of `W`, so the singular-vector tracking itself is bookkeeping and does not need to carry gradients. But `sigma` must then be computed with the current `W` as a differentiable variable, so that the division by `sigma` contributes the adaptive gradient term I verified — if I detach `W` from the `sigma` computation, I lose the `-lambda u_1 v_1^T` correction entirely. In other frameworks this often appears as a forward pre-hook with `u` and `v` buffers; that is equivalent at the conceptual level, but the source-faithful shape is a layer with a `W_bar` property and persistent `u`, not a generic hook object.

The whole chain now holds together, and I have checked the load-bearing pieces rather than asserted them. The discriminator can fail by becoming an arbitrarily sharp or perfectly separating teacher. Lipschitz control is the right family restriction, and for the network its constant is upper-bounded by the product of layer spectral norms. The budget-fixing alternatives couple the whole spectrum (I measured `sum sigma_t^2 = d_out` for row-normalization); dividing each weight by its largest singular value instead pins only the top scale to one and leaves the spectrum shape intact (verified on the `2 x 2` and `5 x 5` cases). The top singular value is estimated by one warm-started power-iteration step, which I watched converge to six digits within a few iterations, flattening convolutional kernels to the standard `d_out x (d_in k_h k_w)` matrix proxy. The gradient through the normalized weight carries a verified adaptive correction `-lambda u_1 v_1^T` along the leading singular direction, and the implementation is just a raw weight, a persistent `u`, and a per-forward `W_bar = W / sigma(W)`.
