I start from the uncomfortable fact that the training objective is underdetermined. I can minimize `L_S(w)` until it is tiny, but in an overparameterized network that only tells me that this point fits this sample. It does not tell me whether I am in a stable basin or on a narrow spike. If two solutions have the same training loss and different test loss, then the missing information is not the value at the point. It has to be in the neighborhood around the point. So whatever I add to the objective has to be a property of the surface near `w`, not of `w` alone.

The old flat-minimum intuition is the first thing that comes to mind: prefer a solution whose loss remains low when the weights are imprecise. A flat solution can tolerate a coarse description; a sharp solution needs many bits because the loss changes quickly under tiny moves. That is a real signal, but I have to be careful with it. Raw flatness is not a theorem by itself. Deep networks have scale symmetries — for a ReLU layer I can multiply incoming weights by `c` and outgoing weights by `1/c` and represent the same function — and that rescaling changes the local curvature without changing anything observable. So a measure that is purely "how fast does loss rise under a fixed-size weight step" can be inflated or deflated for free. I cannot let a bare flatness term carry the whole argument; it needs to be paired with something that controls parameter scale or description cost.

PAC-Bayes is the framework I know that pairs those two things automatically. If I choose a posterior distribution over weights centered at `w`, the bound controls the population loss of a stochastic predictor by the empirical loss averaged under that posterior plus a KL cost to a prior. The two pieces are exactly the two I said I needed: a neighborhood loss term and a scale/complexity term. If the posterior is Gaussian around `w`, the empirical term becomes an average of `L_S(w + epsilon)`. That is already a neighborhood quantity. It says I should not merely fit at `w`; I should fit under perturbations of `w`.

The naive thing to do now is to optimize that expected perturbed loss directly: sample Gaussian `epsilon`, average `L_S(w + epsilon)`, descend. Before committing to it I want to know whether the average is actually sensitive to the kind of sharpness I care about. Suppose there is one steep wall near `w` along a single direction `u`, and the surface is flat in the other `d-1` directions. A random Gaussian `epsilon` has its component along `u` distributed like a single standard normal coordinate scaled by the per-coordinate width. In `d = 10^6` parameters, the squared length of `epsilon` is spread over a million directions, so the expected squared overlap with `u` is a fraction `1/d` of the total. The average loss therefore feels the wall weighted by something like `1/d` — essentially not at all. The dangerous direction is real but the random average dilutes it into invisibility. Generalization failure from sharpness is about vulnerability to *some* small change, not the typical one, so an average over random directions is measuring the wrong statistic. I want the local worst case.

PAC-Bayes lets me trade the average for the worst case without abandoning the bound. Most of the mass of a Gaussian perturbation of radius set by its standard deviation lies inside a ball of comparable radius, and on that high-probability event the loss is at most the maximum loss inside the ball. So the averaged local term is upper-bounded by

`max_{||epsilon||_2 <= rho} L_S(w + epsilon)`

up to the small-probability tail, and replacing the average by this maximum keeps the generalization guarantee while making the objective sensitive to the single bad direction the average was hiding. That is the term I want to minimize.

It is worth writing the maximum as a sum of two pieces, because it tells me what the practical objective should contain:

`max L_S(w + epsilon) = L_S(w) + [max L_S(w + epsilon) - L_S(w)]`.

The first piece is ordinary training loss. The bracket is the rise in loss under the worst nearby move — a sharpness term that is zero on a perfectly flat patch and large on a spike. The PAC-Bayes complexity side, meanwhile, scales with something like `||w||^2 / rho^2`, so a norm penalty belongs in the full objective as well. So the practical objective has two distinct jobs: keep the local worst-case training loss small, and keep the weights controlled. Weight decay is the cheap surrogate for that second job; it is not the new idea. The new pressure is on the bracket — training the whole neighborhood, not the point.

Now the obvious obstacle. The term I want contains an inner maximization over the weights, and `L_S` is nonconvex. If I try to solve that inner problem exactly at every optimizer step, the method is hopeless — I would be running an adversarial optimization inside each training step. I need the cheapest adversary that still points at the steep direction.

The neighborhood radius `rho` is small, so a first-order expansion of the loss should be accurate enough to locate the worst direction even if it is not exact about the worst value:

`L_S(w + epsilon) ~= L_S(w) + epsilon^T g`, with `g = grad_w L_S(w)`.

The constant `L_S(w)` does not depend on `epsilon`, so the inner maximization reduces to a linear function over a norm ball:

`argmax_{||epsilon||_p <= rho} epsilon^T g`.

This is a problem with a known closed form, and it is where the choice of norm enters. Holder's inequality gives the ceiling:

`epsilon^T g <= ||epsilon||_p ||g||_q <= rho ||g||_q`, with `1/p + 1/q = 1`,

and the maximizer that attains equality is

`epsilon_hat = rho * sign(g) * |g|^(q - 1) / (||g||_q^q)^(1/p)`.

I do not want to take that formula on faith, because it is the load-bearing step — if it is wrong, the whole adversary is wrong. Two things have to hold: `epsilon_hat` must lie on the constraint boundary (`||epsilon_hat||_p = rho`, using all the budget), and it must achieve the Holder ceiling (`epsilon_hat^T g = rho ||g||_q`). Let me check both on a concrete vector `g = (3, -1, 0.5, 2)` with `rho = 0.05`, for several exponents:

```
p=2.0, q=2.0000:  ||eps||_p = 0.050000   eps^T g = 0.188746   rho*||g||_q = 0.188746
p=3.0, q=1.5000:  ||eps||_p = 0.050000   eps^T g = 0.222355   rho*||g||_q = 0.222355
p=1.5, q=3.0000:  ||eps||_p = 0.050000   eps^T g = 0.165287   rho*||g||_q = 0.165287
```

In every case the perturbation sits exactly on the boundary and the inner product hits the Holder bound to all the digits I printed. So the closed form is the genuine maximizer, not just a plausible candidate. For `p = 2` the exponent `q - 1 = 1` and the denominator is `(||g||_2^2)^{1/2} = ||g||_2`, so the formula collapses to

`epsilon_hat = rho * g / ||g||_2`,

which I confirmed numerically equals the general expression at `p = 2`. The inner maximization is now trainable: I take one gradient, normalize it to length `rho`, and that is the worst direction. No nonlinear search.

I need to get the sign right, because it is easy to flip and the flip silently inverts the whole method. The inner problem is a *maximum*, so `epsilon_hat` points uphill — toward higher loss. I add it to `w` to stand on the locally bad point. The outer problem is a *minimum*, so I then descend using the gradient measured at that uphill point. If I instead subtracted the perturbation, I would step to an easier nearby point and measure the gradient where the surface is gentler — exactly hiding the wall I built this to expose. Climb first, then descend.

Now I have to actually differentiate the local loss to know what the optimizer should apply. Write

`L_S^local(w) ~= L_S(w + epsilon_hat(w))`.

By the chain rule, the gradient is not simply the gradient evaluated at `w + epsilon_hat`. There is a second contribution from the dependence of `epsilon_hat(w)` on `w`, and since `epsilon_hat` is built from `grad L_S(w)`, that contribution carries a derivative of the gradient — a Hessian. So the exact gradient is

`grad L_S^local = (I + d epsilon_hat / dw)^T grad L_S |_{w + epsilon_hat}`,

and the `d epsilon_hat / dw` block contains `grad^2 L_S`. It is not intractable — it can be realized through Hessian-vector products rather than a dense Hessian — but it doubles the conceptual cost and turns a wrapper into a second-order method.

Before deciding to drop it, I want to see what that second-order term actually does, so that dropping it is a judgment and not a hope. Take the cleanest case where I can write the Hessian term in closed form: a quadratic loss `L(w) = (1/2)(a w_0^2 + b w_1^2)` with a sharp direction (`a = 50`) and a flat one (`b = 1`). For a quadratic, `grad(w + epsilon) = grad(w) + H epsilon` exactly, so the gradient I would apply — the one evaluated at the ascended point — is itself the interesting quantity. At `w = (0.02, 0.5)`, `rho = 0.05`:

```
plain gradient  g      = [1.0,  0.5]
SAM gradient    g_sam  = [3.236, 0.522]
g + H eps              = [3.236, 0.522]   (matches g_sam exactly)
componentwise ratio    = [3.236, 1.045]
```

The gradient at the ascended point equals the plain gradient plus `H * eps`, and that extra term is large precisely where curvature is large: the sharp coordinate's gradient is amplified 3.2x while the flat coordinate barely changes (1.05x). So even *before* I add the explicit chain-rule Hessian term, the gradient-at-the-ascended-point already carries a curvature-weighted correction `H eps` that pushes hardest along the steep directions. That is the dominant signal I wanted from sharpness, and it is present in the simple two-pass quantity. The extra `d epsilon_hat / dw` term only refines how the adversary's *direction* shifts as `w` moves — a second-order correction on top of a quantity that already responds to curvature. Keeping it would buy a small refinement at the price of the scalability that motivated the whole derivation.

So I drop it:

`grad_w L_S^local(w) ~= grad_w L_S(w) |_{w + epsilon_hat(w)}`.

This is the decisive simplification, and the toy computation is what makes me comfortable with it: the cheap quantity is not a crude stand-in for sharpness, it *is* curvature-weighted. The method becomes a wrapper around any base optimizer. On a minibatch, I compute the ordinary gradient at `w`, form `epsilon_hat`, temporarily move to `w + epsilon_hat`, compute the ordinary gradient there, restore `w`, and let the base optimizer apply that second gradient. The cost is about two backpropagations per update — no Hessian, no inner loop.

The same computation also separates this from two things it could be confused with. It is not noise injection: noise would pick `epsilon` without consulting the loss slope, and the `H eps` push would point in a random direction rather than along the curvature. Random noise asks "is the basin usually fine"; the adversarial step asks "does the basin have an exposed wall." And it is not weight decay: weight decay pulls the parameter norm down at the current point and never evaluates `L_S(w + epsilon)` at all. The norm penalty stays in the objective because the bound needs scale control, but the distinctive pressure comes from descending the loss measured at the ascended point.

The norm choice deserves one more look in light of the closed form. For `p = 2` the perturbation is `rho * g / ||g||_2`, which keeps the relative magnitude structure of the gradient — every coordinate contributes in proportion to its slope. At `p = infinity` the maximizer is `rho * sign(g)`, a pure sign step that discards magnitudes; at `p = 1` it concentrates the whole budget on the single largest-magnitude coordinate. The L2 perturbation is the one the Gaussian PAC-Bayes argument hands me in the first place and the one that respects the full gradient direction, so I keep `p = 2` as the default and `rho` as the neighborhood radius.

One subtlety remains about minibatching. The clean objective is over the whole training set, but in practice the perturbation is computed on a batch, and in data-parallel training often on each accelerator shard independently. If each shard builds its own `epsilon_hat` from its own gradient and only the final descent gradients are averaged, I am not measuring one global sharpness — I am measuring the sharpness of each smaller subset and averaging the responses. That is not obviously a defect: each small subset has its own local wall, and a per-shard worst case can be a sharper diagnostic than one diluted global maximum. I will treat this as a deliberate variant rather than something to correct.

The final method is therefore simple enough to drop into existing training. It adds one hyperparameter, `rho`, and reuses the existing loss, model, and base optimizer. Its update on a batch `B` is:

`g = grad L_B(w)`,

`epsilon_hat = rho * g / (||g||_2 + small_eps)`,

`g_local = grad L_B(w + epsilon_hat)`,

then update `w` with `g_local`.

What this amounts to is sharpness-aware minimization. The reframing is the substance: generalization is treated as a local property of the loss surface rather than the loss at an isolated point; the hard local adversary is made cheap by the first-order dual-norm maximizer I checked; and the outer update is made cheap by discarding the chain-rule Hessian term, which the toy quadratic showed is a refinement on top of an already curvature-aware two-pass gradient.
