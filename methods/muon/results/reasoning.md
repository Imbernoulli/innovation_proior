I am staring at the parameters of a transformer, and the annoying fact is that the bulk of them are not scalars in spirit. They sit in 2D hidden weight matrices. Yet the optimizer I usually trust, AdamW, acts as if every entry were its own little coordinate. Let me make that precise rather than just feel it: if I turn off Adam's moving averages and ignore epsilon, the update is `g / sqrt(g^2)`, which for any nonzero `g` is `sign(g)`. So the core geometry is entrywise — every coordinate gets normalized independently, and the matrix structure disappears entirely.

That feels wrong for a hidden linear layer. A weight matrix maps an input activation vector to an output activation vector. The thing that matters is not just how large an individual entry moves, but how much the whole linear operator changes what it does to vectors. If the update matrix has almost all its mass in a few singular directions, an entrywise rule may look well-scaled coordinate by coordinate while still starving most operator directions. The empirical spectra make this worry concrete: SGD-momentum and Adam updates for transformer matrices are nearly low-rank, with a few large singular values and a long tail of tiny ones. So I want a unit step that is unit-sized as a matrix operator, not as a bag of entries — but I do not yet know what "unit-sized as an operator" should buy me, so let me derive it instead of guessing.

Let me phrase the optimizer step as a small local problem. Around the current weight, the loss changes to first order like `<G, Delta W>`, where `G` is the gradient or a momentum estimate. I do not trust this linear model arbitrarily far, so I penalize the step size:

`min_Delta <G, Delta> + (lambda/2) ||Delta||^2`.

Write `Delta = c T`, with `c >= 0` and `||T|| = 1`. Then the direction and scale separate:

`min_c c min_{||T||=1}<G,T> + (lambda/2)c^2`.

The direction is the unit object most aligned with `G`, with a minus sign for descent, and the scale is the dual norm of `G` divided by `lambda`. So the whole question reduces to which norm I impose on the step. Different choices of norm will hand back different directions; I want to see which one corresponds to "operator-sized."

Start with the choice that should reproduce what I already have. If I take the flattened infinity norm, the constraint is `|T_ij| <= 1`, and `<G,T> = sum_ij G_ij T_ij` is maximized term by term at `T_ij = sign(G_ij)`. Let me sanity-check that this is actually the maximizer and recover its value: for a random `3x3` `G`, `<G, sign(G)> = sum_ij |G_ij|`, and indeed numerically `<G,T> = 8.4243` equals `sum|G| = 8.4243`. So the dual norm is the entrywise L1 norm and the direction is the coordinatewise sign update. That is Adam's skeleton, and it is not a bad derivation — it just exposes the assumption. The step is controlled by its largest entry, not by its action as an operator.

For a dense hidden linear map, the natural spaces on both sides are Euclidean, or RMS-scaled Euclidean. The induced operator norm is therefore the spectral norm, up to the fan-in/fan-out RMS factor. So the matrix-flavored version of the same problem is

`argmax_{||T||_2 <= 1} <G,T>`.

Let `G = U Sigma V^T = sum_i sigma_i u_i v_i^T`. Then

`<G,T> = sum_i sigma_i u_i^T T v_i`.

If `||T||_2 <= 1`, each `u_i^T T v_i <= 1`, so `<G,T> <= sum_i sigma_i`, the nuclear norm. A clean upper bound, but a bound is only useful if something attains it. The obvious candidate is `T = U V^T`, because each `u_i^T (U V^T) v_i = e_i^T e_i = 1`, which would make every term contribute its full `sigma_i`. Let me not trust the algebra blindly and actually check it. With `Sigma = diag(3, 1, 0.2)` and random orthogonal `U, V`, `T = U_{:,:3} V^T` has spectral norm `1.0000` (feasible), and `<G,T> = 4.2000`, which is exactly `sigma_1+sigma_2+sigma_3 = 4.2`. And to make sure nothing feasible beats it, I draw 20000 random `M` with `||M||_2 <= 1` and take the best `<G,M>`; the best is `3.0509 < 4.2`. So `U V^T` is not merely feasible — it is the maximizer, and the dual norm is the nuclear norm.

What this direction does is striking once I read it back. The update keeps the singular vectors of the momentum, so it still follows the directions the gradient history suggests, but it sets every singular value to one. The dominant singular directions no longer eat the whole step; the small-but-possibly-useful directions are lifted to the same size. That is precisely the equalization my low-rank worry was asking for, and it fell out of the spectral-norm constraint rather than being imposed by hand.

There is a second way to land on the same object, and it is worth confirming they agree. Among semi-orthogonal matrices, minimizing `||O-G||_F^2` is the same as maximizing `<O,G>`, because `||O||_F^2 = min(A,B)` is fixed. The same SVD bound gives `O = U V^T` in the full-rank case. If `G` is rank deficient the null singular subspace is not unique, which is fine: the reduced polar factor still says what to do on the directions that carry signal. So the steepest spectral-norm direction is also the nearest semi-orthogonal matrix — same target from two angles.

This should also clarify the relationship to Shampoo, which uses left and right preconditioners and updates with `L^(-1/4) G R^(-1/4)`. If I strip the accumulation down to a single gradient, `L = G G^T` and `R = G^T G`. Substituting `G = U Sigma V^T` formally, `(G G^T)^(-1/4) = U Sigma^(-1/2) U^T` and `(G^T G)^(-1/4) = V Sigma^(-1/2) V^T`, so the product telescopes to `U Sigma^(-1/2) Sigma Sigma^(-1/2) V^T = U V^T`. The Sigma factors cancel cleanly on paper, but the inverse roots make me nervous on near-singular matrices, so I verify numerically with `Sigma = diag(3,1,0.2)`: the assembled update `L^(-1/4) G R^(-1/4)` matches the reduced polar factor `U_{:,:3} V^T` to a max abs difference of `1.1e-12`, and its own singular values come back `[1, 1, 1]`. So the target I derived from the spectral norm is exactly what accumulation-free Shampoo computes. The trouble is that an explicit SVD or these inverse roots are too expensive and too precision-hungry to run on every hidden matrix every step.

The cheap GPU primitive is matrix multiplication, especially in bfloat16. So I need a matmul-only way to push singular values toward one while leaving singular vectors alone. Odd matrix polynomials in `X` have exactly this property: if `X = U S V^T`, then `X X^T X = U S^3 V^T`, and any sum of odd powers built from `(X X^T)^k X` acts as a scalar polynomial on the singular values while leaving `U` and `V` fixed. So I can design a scalar map and apply it through this form.

The simplest such map is the Newton-Schulz cubic

`X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k`,

which applies `f(s) = 1.5 s - 0.5 s^3` to each singular value. I want `s = 1` to be a stable fixed point. Checking: `f(1) = 1.5 - 0.5 = 1`, good, and `f'(s) = 1.5 - 1.5 s^2` gives `f'(1) = 0`, so `s=1` is not just attracting but super-attracting — quadratic convergence near the target. But the iteration can also diverge or collapse: `f(0) = 0`, and `f(sqrt(3)) = 1.5*sqrt(3) - 0.5*3*sqrt(3) = 0`, so any singular value at or beyond `sqrt(3)` is driven to zero or past it. To stay safe I should make sure every singular value starts in `(0, sqrt(3))`. Dividing `X` by its Frobenius norm forces `sigma_max <= 1`, comfortably inside the basin, and scaling by a positive constant does not touch the direction `U V^T`. Let me trace it: from `s0 = 1.2` the cubic goes `1.2 -> 0.936 -> 0.994 -> 0.9999 -> 1.0000`, and from `s0 = 1.6` it goes `1.6 -> 0.352 -> 0.506 -> ... -> 0.9992 -> 1.0000`. Both converge. And the endpoint matters: from exactly `sqrt(3) = 1.73205` the iteration is `1.73205 -> 0 -> 0 -> ...`, dead. So the basin is the open interval, and Frobenius normalization keeps me well clear of it.

Now the problem with the cubic. The updates I care about have many tiny singular values, and the speed of lifting those is governed by the slope at zero, `f'(0) = 1.5`. That is barely above one, so tiny values crawl. Tracing from `s0 = 0.01`: the cubic gives `0.01 -> 0.015 -> 0.0225 -> 0.0337 -> 0.0506 -> 0.0758`. After five steps a singular value that started at `0.01` has only reached `0.076` — nowhere near one. With a fixed budget of five bfloat16 iterations, the long tail of small directions would barely move, which defeats the whole point of equalizing them.

So I want a larger slope at zero. A quintic gives the extra degree of freedom:

`g(s) = a s + b s^3 + c s^5`,

with `g'(0) = a`. If I demanded exact convergence to `s = 1` for every singular value, the constraints `g(1) = 1` and a stable fixed point would pin `a` down near the cubic's value and I gain little. But exact orthogonality is not the real requirement; roughly equalizing the singular values inside five steps is. So I can spend the freedom on a large `a` and accept an iterate of the form `U S' V^T` with the diagonal of `S'` merely clustered around one. A known tuned choice is `(a, b, c) = (3.4445, -4.7750, 2.0315)`, giving `g'(0) = 3.4445`, more than double the cubic's slope.

Let me see what that actually buys and what it costs, because "clustered around one" is doing real work in that sentence. Tracing the quintic from `s0 = 0.01`: `0.01 -> 0.0344 -> 0.1184 -> 0.4001 -> 1.0931 -> 0.6989`. By step four the tiny value has been lifted past `0.4` and then overshoots one, where the cubic from the same start was still at `0.05`. From `s0 = 0.05` the quintic reaches `0.567` by step three. So the small directions are genuinely rescued within the budget — that is the win. The cost is visible in the same trace: the quintic does not settle at exactly one, it oscillates in the `0.7`-to-`1.09` band. That is fine. It confirms the honest reading of this iteration: it is an approximate orthogonalizer that returns `U S'V^T` with `S'` near one, not an exact polar factor, and five iterations is the practical budget.

Now I have the core update: maintain momentum, feed a Nesterov-style momentum estimate into this Newton-Schulz orthogonalizer, and subtract the result from the hidden weight matrix. I should apply it only where the geometry matches the derivation — hidden 2D linear weights, and convolutional filters only after flattening them into matrices. Embeddings are 2D but they are lookups from one-hot indices, not dense hidden operators, so the spectral-norm story does not apply. The final head is also 2D but empirically does better on the usual adaptive rule. Gains and biases have no matrix singular geometry. Those stay on AdamW.

There is still a scale problem, and it could quietly break the "share one learning rate" property I am after. A semi-orthogonal update does not have the same entry RMS for every shape. For an update `O = U_{:,:r} V_{:r,:}^T` of shape `[A,B]`, the squared RMS is

`(1/(A B)) sum_ij (sum_k U_ik V_jk)^2`.

Expanding the square gives `sum_k sum_l U_ik U_il V_jk V_jl`; summing over `i` uses `sum_i U_ik U_il = delta_kl` (columns of `U` orthonormal) and likewise over `j` for `V`, so the cross terms drop and the double sum collapses to `sum_k 1 = r`. Hence

`RMS(O)^2 = r/(A B)`.

I want to be sure I did the index bookkeeping right, so I check it numerically on a few shapes with random column-orthonormal `U, V`. For `(A,B)=(4,6)` I get `RMS^2 = 0.16667` against `r/(AB) = 4/24 = 0.16667`; for `(6,4)`, `0.16667` again; for `(5,5)`, `0.20000 = 5/25`; for `(2,10)`, `0.10000 = 2/20`. All match. In the common full-rank case `r = min(A,B)`, this is `1/max(A,B)`, also confirmed by the same numbers. So a bare orthogonalized update is smaller on matrices with a larger long side: a large MLP matrix under-steps, while splitting a parameter into small pieces can over-step. Multiplying by `sqrt(max(A,B))` cancels the full-rank shape dependence, lifting `RMS` to `1`. Then multiplying by `0.2` brings the update RMS into the empirical AdamW range (~0.2-0.4) I am trying to match, so one learning rate and one weight decay can be shared with the AdamW-managed parameters.

Weight decay has to be decoupled. Folding decay into the gradient would send it through the orthogonalizer — and since the orthogonalizer discards singular-value magnitude, a decay term hidden inside `G` would lose its scale entirely. The stable form is the AdamW-style one:

`W <- (1 - eta lambda) W - eta update`.

That matters at long scale because without decay the weight RMS and layer-output RMS drift upward, and in bfloat16 that drift becomes a real precision constraint.

Finally, sharding changes the implementation but not the math, and there is one subtlety I have to respect. An AdamW shard can update elementwise by itself because its rule is per-coordinate. This orthogonalized update cannot, because the singular directions of a shard are not the singular directions of the full matrix — orthogonalizing each slice separately would compute the wrong `U, V`. So in a ZeRO-1-style setup the local shard updates its momentum, the momentum shards are gathered into the full matrix, Newton-Schulz runs on the full matrix in bfloat16, and each rank keeps only its local slice of the resulting update. That preserves the matrix operation while still holding one momentum buffer instead of AdamW's two.

So the method is: use momentum to smooth the hidden-matrix gradient, replace that matrix by an approximate polar factor using the tuned Newton-Schulz quintic, scale it by `0.2 * sqrt(max(A,B))` to remove the full-rank shape RMS dependence and match AdamW's update RMS, apply decoupled weight decay, and leave embeddings, heads, gains, and biases to AdamW.
