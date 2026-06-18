I am staring at the parameters of a transformer, and the annoying fact is that the bulk of them are not scalars in spirit. They sit in 2D hidden weight matrices. Yet the optimizer I usually trust, AdamW, acts as if every entry were its own little coordinate. If I turn off Adam's moving averages and ignore epsilon, the update is `g / sqrt(g^2) = sign(g)`. That is a useful sanity check: the core geometry is entrywise. Every coordinate gets normalized independently, and the matrix structure disappears.

That feels wrong for a hidden linear layer. A weight matrix maps an input activation vector to an output activation vector. The thing that matters is not just how large an individual entry moves, but how much the whole linear operator changes what it does to vectors. If the update matrix has almost all its mass in a few singular directions, an entrywise rule may look well-scaled coordinate by coordinate while still starving most operator directions. The empirical spectra make this worry concrete: SGD-momentum and Adam updates for transformer matrices are nearly low-rank, with a few large singular values and a long tail of tiny ones. I want a unit step that is unit-sized as a matrix operator, not as a bag of entries.

Let me phrase the optimizer step as a small local problem. Around the current weight, the loss changes to first order like `<G, Delta W>`, where `G` is the gradient or a momentum estimate. I do not trust this linear model arbitrarily far, so I penalize the step size:

`min_Delta <G, Delta> + (lambda/2) ||Delta||^2`.

Write `Delta = c T`, with `c >= 0` and `||T|| = 1`. Then the direction and scale separate:

`min_c c min_{||T||=1}<G,T> + (lambda/2)c^2`.

The direction is the unit object most aligned with `G`, with a minus sign for descent, and the scale is the dual norm of `G` divided by `lambda`. So the whole question reduces to the norm. Which norm should a hidden weight matrix use?

If I choose the flattened infinity norm, I recover Adam's skeleton. The maximizer of `<G,T>` subject to `|T_ij| <= 1` is `T_ij = sign(G_ij)`, and the dual norm is the entrywise `l1` norm. That is exactly the coordinatewise sign update. It is not a bad derivation; it just reveals the assumption. The step is controlled by its largest entry, not by its action as an operator.

For a dense hidden linear map, the natural spaces on both sides are Euclidean, or RMS-scaled Euclidean. The induced operator norm is therefore the spectral norm, up to the fan-in/fan-out RMS factor. So I should solve

`argmax_{||T||_2 <= 1} <G,T>`.

Let `G = U Sigma V^T = sum_i sigma_i u_i v_i^T`. Then

`<G,T> = sum_i sigma_i u_i^T T v_i`.

If `||T||_2 <= 1`, each `u_i^T T v_i <= 1`, so `<G,T> <= sum_i sigma_i`. This upper bound is attained by `T = U V^T`, because then every active singular pair contributes one. The dual norm is the nuclear norm, and the steepest unit spectral-norm direction is the polar factor `U V^T`.

That is exactly the equalization I was looking for. The update keeps the singular vectors of the momentum, so it still follows the directions suggested by the gradient history, but it replaces the singular values by ones. The dominant singular directions no longer consume the whole step. The small but possibly useful directions are lifted to comparable size.

There is a second way to land on the same object. Among semi-orthogonal matrices, minimizing `||O-G||_F^2` is the same as maximizing `<O,G>`, because `||O||_F^2 = min(A,B)` is fixed. The same SVD bound gives `O = U V^T` when the matrix is full rank. If `G` is rank deficient, the null singular subspace is not unique, which is fine: the natural reduced polar factor still says what to do on the directions that have signal.

This also explains the relationship to Shampoo. In the matrix case, Shampoo uses left and right preconditioners and updates with `L^(-1/4) G R^(-1/4)`. If I remove accumulation for a single gradient, `L = G G^T` and `R = G^T G`, so with `G = U Sigma V^T` the update becomes

`(G G^T)^(-1/4) G (G^T G)^(-1/4) = U Sigma^(-1/2) Sigma Sigma^(-1/2) V^T = U V^T`.

So the target update is not arbitrary. It is what accumulation-free Shampoo computes. The problem is that explicit SVD or inverse roots are too expensive and too precision-hungry for every hidden matrix on every training step.

The cheap GPU primitive is matrix multiplication, especially in bfloat16. I need a matmul-only way to push singular values toward one while leaving singular vectors alone. Odd matrix polynomials do that. If `X = U S V^T`, then `X X^T X = U S^3 V^T`, and higher odd powers behave the same way. A scalar polynomial applied through this form changes only the singular values.

The clean Newton-Schulz cubic is

`X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k`,

which applies `f(s) = 1.5s - 0.5s^3` to each singular value. The fixed point at `s=1` is attracting for positive singular values in the basin below `sqrt(3)`. So I normalize first. Dividing by the Frobenius norm guarantees `sigma_max <= 1`, safely inside the basin, and scaling by a positive constant does not change the polar direction.

But the cubic is slow near zero, and the updates I care about have many tiny singular values. The speed near zero is the derivative of the scalar map at zero. A quintic gives another degree of freedom:

`g(s) = a s + b s^3 + c s^5`.

If I insist on exact convergence to one everywhere, I cannot make `a` very large. But exact orthogonality is not the practical requirement; roughly equalizing singular values is. So I can choose coefficients with a large slope at zero and accept that the iteration produces something like `U S' V^T`, with the diagonal of `S'` clustered around one rather than exactly equal to one. The tuned coefficients are `(a,b,c) = (3.4445, -4.7750, 2.0315)`. Five bfloat16 iterations are enough in the reference implementations.

Now I have the core update: maintain momentum, feed a Nesterov-style momentum estimate into the Newton-Schulz orthogonalizer, and subtract the result from the hidden weight matrix. I should apply it only where the geometry matches the derivation: hidden 2D linear weights, and convolutional filters only after flattening them into matrices. Embeddings are 2D, but they are lookups from one-hot indices, not dense hidden operators. The final head is also 2D, but empirically it does better with the usual adaptive rule. Gains and biases have no matrix singular geometry. Those stay on AdamW.

There is still a scale problem. A semi-orthogonal update does not have the same entry RMS for every shape. For an update `O = U_{:,:r} V_{:r,:}^T` of shape `[A,B]`, the squared RMS is

`(1/(A B)) sum_ij (sum_k U_ik V_kj)^2`.

When I expand the square, orthonormality kills the cross terms and leaves

`RMS(O)^2 = r/(A B)`.

Only in the common full-rank case, `r = min(A,B)`, does this simplify to `1/max(A,B)`. So a bare orthogonalized update is smaller on matrices with a larger long side. That means a large MLP matrix under-steps, while splitting a parameter into small pieces can over-step. Multiplying by `sqrt(max(A,B))` cancels the full-rank shape dependence. Then multiplying by `0.2` brings the update RMS into the empirical AdamW range I am trying to match, so one learning rate and one weight decay can be shared with the AdamW-managed parameters.

Weight decay has to be decoupled. Folding decay into the gradient would send it through the orthogonalizer or the adaptive denominator and change its meaning. The stable form is the AdamW-style one:

`W <- (1 - eta lambda) W - eta update`.

That matters at long scale because without decay the weight RMS and layer-output RMS drift upward, and bfloat16 precision becomes a practical constraint.

Finally, sharding changes the implementation but not the math. An AdamW shard can update elementwise by itself. This orthogonalized update cannot, because the singular directions of a shard are not the singular directions of the full matrix. In a ZeRO-1-style setup the local shard updates its momentum, the momentum shards are gathered into the full matrix, Newton-Schulz runs on the full matrix in bfloat16, and each rank keeps only its local slice of the resulting update. That preserves the matrix operation while keeping one momentum buffer instead of AdamW's two.

So the method is: use momentum to smooth the hidden-matrix gradient, replace that matrix by an approximate polar factor using the tuned Newton-Schulz quintic, scale it to remove the full-rank shape RMS dependence and match AdamW's update RMS, apply decoupled weight decay, and leave embeddings, heads, gains, and biases to AdamW.
