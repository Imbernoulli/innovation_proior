I start with the discriminator, because in a GAN it is the only channel through which the generator learns. The generator does not see the data distribution directly. It moves because the discriminator assigns scores to generated samples and the gradient of those scores flows back through `G`. If that discriminator becomes badly conditioned, the whole game becomes badly conditioned. Making `D` stronger as a classifier is not automatically good; if it becomes a perfect separator too quickly, it can stop being a useful teacher.

The sharp failure is the support argument. Real images and generated images live near low-dimensional sets in a huge ambient space, and early in training those sets need not overlap in any meaningful way. If the two supports are disjoint, a discriminator can separate them perfectly. On the generated side it can be essentially flat, so the derivative with respect to the input is zero on the region where the generator is asking for guidance. The generator then receives no useful direction even though the discriminator is "right." So I do not want the best possible unconstrained discriminator. I want the best discriminator inside a smoother function class.

The optimal discriminator for the original GAN objective gives me the same diagnosis from another angle. For fixed `G`,

    D*_G(x) = q_data(x) / (q_data(x) + p_G(x)) = sigmoid(f*(x)),
    f*(x) = log q_data(x) - log p_G(x).

Differentiating the log-density ratio gives

    grad_x f*(x) = grad_x q_data(x) / q_data(x) - grad_x p_G(x) / p_G(x).

Those ratios can blow up near the edge of support or become undefined where the density vanishes. The dangerous object is not just the value of `D`; it is the input derivative of the discriminator score. I need a way to bound how fast the discriminator can change with its input.

That is the Lipschitz constant. If `||f||_Lip <= K`, then `||f(x)-f(x')|| <= K ||x-x'||`, so no part of the score network can develop arbitrarily steep transitions. This does not promise nonzero generator gradients everywhere, and I should not pretend it does. It is an upper control on sensitivity, not a lower control on signal. But it removes the family of arbitrarily sharp separators that causes the most brittle behavior, and it is exactly the kind of restriction WGAN-style methods already need.

So the target is clear: search over

    argmax_{||f||_Lip <= K} V(G,D).

Now the question is enforcement. I need the condition to apply to an actual neural network made of linear or convolutional maps and pointwise activations. For a differentiable map, the local stretching is measured by the largest singular value of its Jacobian:

    ||g||_Lip = sup_h sigma(grad g(h)).

For a linear layer `g(h)=W h`, the Jacobian is just `W`, so the layer's Lipschitz constant is `sigma(W)`, the spectral norm. Composition is submultiplicative, and ReLU or leaky ReLU with slope at most one contributes a factor of one. Therefore for the score network

    f(x) = W^{L+1} a_L(W^L ... a_1(W^1 x)),

I get the bound

    ||f||_Lip <= product_l sigma(W^l).

This is the key reduction. I do not need to control every entry of every matrix. I do not need to fix a Frobenius norm. I do not need to regularize the input gradient at a finite set of sampled points. If I can keep the top singular value of each weight matrix under control, I control the standard product upper bound on the network's Lipschitz constant.

That also tells me what is wrong with the older weight constraints. Weight clipping constrains entries in a box, not the actual operator stretch. It is cheap, but it is a very indirect way to get a Lipschitz condition, and the critic can pay for sensitivity by lining up weights into a low-dimensional effective map. For the pure row-normalized version of weight normalization, with the learned scale removed, every output row has unit norm. Then

    sum_t sigma_t(W)^2 = tr(W W^T) = d_out.

That is a fixed budget over all singular values. If the discriminator wants to maximize `||W h||` for some unit direction `h` under that budget, the easiest way is to put most of the mass into one singular value. Frobenius normalization has the same shape with budget one. These methods are not merely bounding the top stretch; they are tying together the whole spectrum, so pressure for sensitivity can become pressure toward low effective rank. Orthonormal regularization over-corrects in the opposite direction by trying to make all singular values one. That prevents collapse, but it also forces directions that may be useless to receive the same scale as useful directions. I want the largest singular value controlled and the rest of the spectrum left alone.

The direct operation is almost embarrassingly simple:

    W_bar_SN = W / sigma(W).

For a matrix layer, `sigma(W_bar_SN)=1`, because the spectral norm is homogeneous. The ratios `sigma_2/sigma_1`, `sigma_3/sigma_1`, and so on are unchanged. The layer can still be high rank; I have only fixed the top scale. If I apply this to every matrix layer in the score network, the product bound becomes one for those matrix factors. If I want a different global Lipschitz scale, I can put that scale outside or allocate constants across layers. The normal case is simply to use one.

There is a convolution caveat I must keep straight. A convolutional kernel is stored as a tensor `W in R^{d_out x d_in x k_h x k_w}`. The practical proxy is to reshape it into a matrix of shape `d_out x (d_in k_h k_w)` and control the largest singular value of that matrix. The true operator norm of the discrete convolution also depends on stride, padding, and input size. I will use the flattened-kernel norm as the layer quantity and treat the remaining difference as a predefined architecture constant. So for fully connected layers I can speak exactly about the matrix operator norm; for convolutional layers I should say I am controlling the standard flattened-kernel proxy that bounds the intended layer scale up to architecture constants, not that the discrete convolution operator has been exactly normalized in every setting.

The remaining obstacle is computational. `sigma(W)` is the largest singular value. A full SVD for every layer at every discriminator update is not acceptable in a GAN loop. But I only need the top singular value, so power iteration is the right tool. Starting from a vector `u`, I alternate

    v <- W^T u / ||W^T u||_2,
    u <- W v / ||W v||_2.

Equivalently, `u` is repeatedly acted on by `W W^T`. In the singular-vector basis, the component along `u_t` is multiplied by `sigma_t^2` each round, so the top component dominates when the largest singular value is separated and the initialization is not orthogonal to the top vector. Once `u` and `v` approximate the leading singular pair, the scalar

    u^T W v

approximates `sigma(W)`.

Restarting this iteration from a fresh random vector every step would be wasteful. The weight changes only by one optimizer step at a time, so the leading singular vector should also move gradually. I can store the current estimate of `u` with the layer and reuse it at the next update. Then one power-iteration step per update is a practical tracker rather than a from-scratch solve. This is not a theorem that one step is always enough; it is the empirical and algorithmic reason the method is cheap. The layer only needs to persist `u`; it can recompute `v`, update `u`, estimate `sigma`, and use the normalized weight for the layer call.

Now I want to make sure the gradient through the normalization has the right sign. The raw parameter is `W`, but the forward pass uses `W_bar = W / sigma(W)`. For a simple top singular value, the derivative of `sigma(W)` is

    d sigma(W) / d W = u_1 v_1^T.

Entrywise,

    d W_bar / d W_ij
      = (1/sigma) E_ij - (1/sigma^2) (d sigma / d W_ij) W
      = (1/sigma)(E_ij - [u_1 v_1^T]_ij W_bar).

Let `h` be the layer input and let

    delta := (d V / d (W_bar h))^T

be the backpropagated output signal. Without the normalization, the minibatch gradient for the weight would be

    Ehat[delta h^T].

With the normalization, the chain rule gives

    dV/dW = (1/sigma(W)) ( Ehat[delta h^T] - lambda u_1 v_1^T ),
    lambda = Ehat[delta^T W_bar h].

The sign is negative in the leading singular direction. The coefficient is data-dependent: it is the average alignment between the backprop signal and the layer output. When that coefficient is positive, the correction pushes against increasing the dominant singular component. I should phrase this as a pressure or adaptive correction, not as an absolute prevention theorem. The fixed-point condition is that the ordinary weight gradient must align with `u_1 v_1^T` up to a scalar; that is much less restrictive than driving the entire matrix to rank one.

This also explains why a soft spectral-norm penalty is different. A penalty adds something like a coefficient times `sigma(W)` to the objective, which always pushes through a regularization weight and competes with the task loss. Dividing by `sigma(W)` changes the parameterization used by the forward pass. It sets the layer scale directly and produces a sample-dependent correction in the raw-weight gradient. So the older regularizer gives me the numerical tool and the right matrix quantity, but it does not give the hard per-forward scale control.

The general normalizer formula is a useful check. For `W_bar = W / N(W)`,

    dV/dW = (1/N) ( grad_{W_bar} V
             - trace((grad_{W_bar} V)^T W_bar) grad_W N ).

If `N` is the Frobenius norm, `grad_W N = W_bar`, so the correction follows the whole matrix direction. If `N` is the spectral norm, `grad_W N = u_1 v_1^T`, so the correction targets only the leading singular pair. That matches the design goal: control the top stretch while leaving the rest of the spectrum as free as possible.

There is one optional relaxation. I can write

    W_tilde = gamma W_bar_SN

with a learned scalar `gamma`. This gives the layer back a trainable scale, so the layer is no longer fixed to a 1-Lipschitz matrix factor by itself. That means some other mechanism, such as a gradient penalty, must handle the desired global Lipschitz control. But as a reparameterization it can be useful: the direction and spectrum shape are still normalized by the top singular value, while the model can learn an overall scale.

Now I translate the method into the layer interface. The implementation should not be an optimizer step that permanently overwrites weights after training; it should be a layer computation. Store the raw trainable parameter `W`. Store a persistent estimate `u` as non-parameter state. On each training forward, flatten `W` if it is convolutional, run the configured number of power iterations, update `u`, compute `sigma = u^T W_mat v`, and call the underlying linear or convolution operation with `W / sigma`. The optional `gamma` multiplies this normalized weight. A factor parameter can divide `sigma` before the final division if a different target scale is desired.

The subtle implementation detail is gradient flow. The power-iteration updates for `u` and `v` can use the raw array value of `W`, so the singular-vector tracking itself is bookkeeping. But `sigma` must then be computed with the current `W` as a differentiable variable, so the division by `sigma` still contributes the adaptive gradient term. In other frameworks this often appears as a forward pre-hook with `u` and `v` buffers; that is equivalent at the conceptual level, but the source-faithful shape is a layer with a `W_bar` property and persistent `u`, not a generic hook object.

The whole chain now holds together. The discriminator can fail by becoming an arbitrarily sharp or perfectly separating teacher. Lipschitz control is the right family restriction. For the network I can upper-bound the Lipschitz constant by the product of layer spectral norms. Instead of clipping entries, fixing a Frobenius budget, or pushing all singular values to one, I divide each weight by its largest singular value. I estimate that value with one warm-started power-iteration update, flattening convolutional kernels to the standard `d_out x (d_in k_h k_w)` matrix proxy. The gradient through the normalized weight has the correct negative correction along `u_1 v_1^T`, and the implementation is just a raw weight, a persistent `u`, and a per-forward `W_bar = W / sigma(W)`.
