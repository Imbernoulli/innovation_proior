OK, let me think this through from scratch. The thing that keeps bothering me is the gap between two pictures of a neural network that refuse to be reconciled. On one side, training is gradient descent on the parameters, and the loss as a function of those parameters is hopelessly non-convex — a high-dimensional surface strewn with saddle points, no convexity to lean on, and yet gradient descent reliably drives the training loss to zero on networks far wider than the data demands. On the other side, the same overparametrized network, which can be made to memorize random labels, generalizes when the labels are real. Capacity arguments explain neither: capacity says it should overfit, and it does interpolate the training set, but it still generalizes. And the curious thing is that plain kernel methods do exactly this too — they interpolate and still generalize. So maybe I'm staring at the wrong object. Maybe I shouldn't be watching the weights at all.

A network is a map from parameters theta to a function f_theta that sends inputs to outputs. Training moves theta, but what I actually care about — what generalizes, what the loss is really a function of — is f_theta itself. So let me try to track f_theta directly, as a point moving in a space of functions, and ask: under gradient descent on theta, what differential equation does f_theta obey? If I can write that down, I might find that the function-space view is convex even when the parameter-space view is not. The cost C, after all, is just (say) a squared error between f and a target f*; as a functional of f that's convex. The non-convexity is entirely an artifact of the parametrization F: theta -> f_theta being nonlinear. So let me push the dynamics through that parametrization and see what comes out.

Set it up carefully. Continuous-time gradient descent — gradient flow — on the composite cost C ∘ F is

    theta_dot_p = - partial_{theta_p} (C ∘ F)(theta).

The cost only depends on f through its values on the data x_1, ..., x_N, so the functional derivative partial_f C is a linear form on the function space — it pairs a function g against some fixed function d via the data inner product <g, h>_{pin} = E_{x~pin}[g(x)^T h(x)], where pin is the empirical distribution on the dataset. Call d the dual of partial_f C; for squared error C(f) = (1/2) ||f - f*||^2 we get d = f - f* immediately. By the chain rule,

    partial_{theta_p}(C ∘ F) = <d, partial_{theta_p} f_theta>_{pin},

so

    theta_dot_p = - <d, partial_{theta_p} f_theta>_{pin}.

Now the move I actually wanted: how does the function change? f_theta changes only because theta changes, so

    partial_t f_theta = sum_p (partial_{theta_p} f_theta) theta_dot_p
                      = - sum_p <d, partial_{theta_p} f_theta>_{pin} (partial_{theta_p} f_theta).

Stare at the right-hand side. It is a sum, over parameters, of (a scalar measuring how much partial_{theta_p} f overlaps with the error direction d) times (that same partial_{theta_p} f). That is exactly the form of an operator built from the family of functions {partial_{theta_p} f}. Define

    Theta(x, x') = sum_p partial_{theta_p} f_theta(x) ⊗ partial_{theta_p} f_theta(x') = sum_p <grad_theta f(x), grad_theta f(x')>,

a kernel — for scalar output it's a scalar-valued kernel, in general an n_L x n_L matrix kernel. Then the right-hand side is precisely the kernel applied to the error:

    partial_t f_theta(x) = - (1/N) sum_{j} Theta(x, x_j) d(x_j),

i.e. partial_t f = - nabla_Theta C, the *kernel gradient* of the cost with respect to Theta. For squared error, evaluated on the data, this collapses to the clean linear-looking statement

    f_dot = - Theta (f - f*)   (the vector of values on the dataset),

and off the data the same Theta extends the update to new inputs. So the function follows gradient descent in function space, against this particular kernel that I built out of the parameter-gradients of the network. I'll call it the tangent kernel of the network — it is literally the Gram matrix of the tangent map of F at theta.

This already buys something. Watch the cost evolve: partial_t C = <d, partial_t f>_{pin} = - <d, nabla_Theta C>_{pin} = - ||d||^2_Theta, where ||d||^2_Theta = E_{x,x'}[d(x)^T Theta(x,x') d(x')]. That's non-positive, and it's strictly negative whenever d is nonzero *as seen by the kernel*. So if Theta is positive definite — meaning ||f||_{pin} > 0 implies ||f||_Theta > 0 — the cost strictly decreases until d = 0, and since C is convex and bounded below in function space, f converges to a global minimum. The non-convex parameter landscape never entered; convergence is now a statement about positive-definiteness of a kernel. That's a real prize. But it's hollow until I understand Theta, because Theta = Theta(theta) depends on the current parameters: it is random at initialization (theta is random) and it moves as theta moves. A kernel that wanders is not something I can solve against.

Let me get intuition from the one case where this works cleanly. Take Rahimi and Recht's random-feature picture: sample P random functions f^(p) with E[f^(p)(x) f^(p)(x')] = K(x, x'), and form the *linear* model f_theta = (1/sqrt(P)) sum_p theta_p f^(p). Here partial_{theta_p} f_theta = (1/sqrt(P)) f^(p), which does not depend on theta at all — the model is linear in its parameters. So the tangent kernel is

    K-tilde = sum_p partial_{theta_p} f ⊗ partial_{theta_p} f = (1/P) sum_p f^(p) ⊗ f^(p),

constant in time, and by the law of large numbers K-tilde -> K as P -> infinity. Gradient descent on this model *is* kernel gradient descent against a fixed kernel, exactly. So the obstruction with a real network is precisely its nonlinearity: F is not linear in theta, the feature map partial_{theta_p} f wiggles as theta moves, and Theta is random and time-varying. If only I could argue that for a wide network the feature map barely moves — that the network behaves, near initialization, like a linear-in-parameters model — then Theta would be (asymptotically) a fixed kernel and I'd inherit the whole clean story. Two things to establish, then: that Theta at initialization converges to some deterministic kernel as width grows, and that Theta stays put during training.

Before training, I need a parametrization. Layers 0 (input) to L (output), widths n0, ..., n_L. The natural thing people do is W a + b with W, b small. But I want a clean width-to-infinity limit, so write the pre-activation as

    a-tilde^(ell+1) = (1/sqrt(n_ell)) W^(ell) a^(ell) + beta b^(ell),   a^(ell) = sigma(a-tilde^(ell)),   a^(0) = x,

with all entries of W^(ell), b^(ell) iid N(0, 1), and f_theta = a-tilde^(L). Why the 1/sqrt(n_ell)? Because a^(ell) has n_ell coordinates each O(1), so without the factor W a would be a sum of n_ell terms of order 1 and would blow up like sqrt(n_ell); the 1/sqrt(n_ell) renormalizes the pre-activation to O(1) so the limit exists. It represents the same set of functions as the textbook small-weight init (it's a reparametrization), but — and this is the load-bearing observation — the *derivatives* are different: partial_{W^(ell)_{ij}} f picks up a 1/sqrt(n_ell). So each individual weight in a wide layer has a gradient that vanishes like 1/sqrt(n), which is exactly the mechanism I hoped for: each parameter moves negligibly. There's a side effect to fix. With connection-weight gradients shrunk by 1/sqrt(n), the bias gradients (no such factor) would dominate the dynamics; the scalar beta is there to rebalance the influence of biases against connections so neither is starved. (In practice something like beta = 0.1 with a correspondingly large learning rate behaves like a classical moderate-width net.)

Now the first limit: what is Theta at initialization, as n_1, ..., n_{L-1} -> infinity? I'll take the widths to infinity one at a time, n_1 first, then n_2, and so on — it makes the induction honest, each limit landing before the next is taken. And I'll need, alongside Theta, the covariance of the network function itself, because the kernel recursion will turn out to feed on it.

Start with the function's covariance Sigma. For L = 1 there are no hidden layers: f_theta(x) = (1/sqrt(n_0)) W^(0) x + beta b^(0), an affine function of iid Gaussians, so each output coordinate is centered Gaussian with

    Sigma^(1)(x, x') = (1/n_0) x^T x' + beta^2,

and distinct output coordinates are independent (disjoint rows of W^(0)). For the induction step, view an (L+1)-network as: an L-network producing pre-activations a-tilde^(L), then sigma applied elementwise, then one more random affine map to the output. By the induction hypothesis the n_L pre-activations a-tilde^(L)_i become iid centered Gaussians with covariance Sigma^(L). Condition on the values of a^(L) = sigma(a-tilde^(L)); then the output

    f_{theta,i} = (1/sqrt(n_L)) W^(L)_i a^(L) + beta b^(L)_i

is, over the randomness of the last layer, centered Gaussian with covariance

    Sigma-tilde^(L+1)(x, x') = (1/n_L) a^(L)(x)^T a^(L)(x') + beta^2
                             = (1/n_L) sum_{i=1}^{n_L} sigma(a-tilde^(L)_i(x)) sigma(a-tilde^(L)_i(x')) + beta^2.

That's an average over the n_L hidden units of a function of iid Gaussian pre-activations, so by the law of large numbers as n_L -> infinity it concentrates on its expectation:

    Sigma^(L+1)(x, x') = E_{f ~ N(0, Sigma^(L))}[ sigma(f(x)) sigma(f(x')) ] + beta^2.

The limit is deterministic, so it no longer depends on the particular a^(L); the conditional Gaussian becomes the unconditional one, and the outputs are iid centered Gaussians with covariance Sigma^(L+1). Good — that's the network-as-Gaussian-process fact, recovered as a byproduct, with the covariance recursion I'll lean on.

Now the tangent kernel itself, at initialization. L = 1: Theta is a sum over the entries of W^(0) and b^(0) of partial-products. With f the affine map above, partial_{W^(0)_{ij}} f_k = (1/sqrt(n_0)) x_i delta_{jk} and partial_{b^(0)_j} f_k = beta delta_{jk}, so

    Theta^(1)_{kk'}(x, x') = (1/n_0) sum_i x_i x'_i delta_{kk'} + beta^2 delta_{kk'} = ((1/n_0) x^T x' + beta^2) delta_{kk'} = Sigma^(1)(x, x') delta_{kk'}.

So at the bottom, the tangent kernel equals the covariance kernel. For the induction step, split the parameters of an (L+1)-network into theta-tilde (the first L layers) and the last layer's (W^(L), b^(L)). The tangent kernel is the sum of the contributions from each group.

Last-layer contribution. partial_{W^(L)_{ij}} f_{theta, k} = (1/sqrt(n_L)) a^(L)_i delta_{jk} and partial_{b^(L)_j} f_{theta,k} = beta delta_{jk}, so summing the outer products over those parameters gives, for output indices k = k',

    (1/n_L) sum_i a^(L)_i(x) a^(L)_i(x') + beta^2 = Sigma-tilde^(L+1)(x, x') -> Sigma^(L+1)(x, x')

by exactly the LLN argument above. So the last layer contributes Sigma^(L+1) delta_{kk'}. This is the "the top layer's own weights learn" term.

Lower-layer contribution. By the chain rule, the output's derivative with respect to a first-L-layers parameter theta-tilde_p threads through the last layer:

    partial_{theta-tilde_p} f_{theta, k}(x) = (1/sqrt(n_L)) sum_{i=1}^{n_L} partial_{theta-tilde_p} a-tilde^(L)_i(x) * sigma-dot(a-tilde^(L)_i(x)) * W^(L)_{ik}.

Form the contribution to Theta^(L+1)_{kk'} by summing partial_{theta-tilde_p} f_k(x) partial_{theta-tilde_p} f_{k'}(x') over theta-tilde_p. The sum over p of partial_{theta-tilde_p} a-tilde^(L)_i(x) partial_{theta-tilde_p} a-tilde^(L)_{i'}(x') is exactly the *smaller network's* tangent kernel Theta^(L)_{ii'}(x, x'). So the contribution is

    (1/n_L) sum_{i, i'} Theta^(L)_{ii'}(x, x') sigma-dot(a-tilde^(L)_i(x)) sigma-dot(a-tilde^(L)_{i'}(x')) W^(L)_{ik} W^(L)_{i'k'}.

By the induction hypothesis, as the lower widths go to infinity, Theta^(L)_{ii'} -> Theta^(L)_inf delta_{ii'} — it becomes diagonal in the hidden index and deterministic. The off-diagonal i != i' terms drop, leaving

    (1/n_L) sum_i Theta^(L)_inf(x, x') sigma-dot(a-tilde^(L)_i(x)) sigma-dot(a-tilde^(L)_i(x')) W^(L)_{ik} W^(L)_{ik'}.

Now n_L -> infinity. The factors W^(L)_{ik} W^(L)_{ik'} are iid with mean delta_{kk'} (variance 1, independent across i), independent of the pre-activations, so the average over i tends by the LLN to Theta^(L)_inf(x, x') times the expectation of sigma-dot(f(x)) sigma-dot(f(x')) over the limiting Gaussian pre-activations, times delta_{kk'}. Define

    Sigma-dot^(L+1)(x, x') = E_{f ~ N(0, Sigma^(L))}[ sigma-dot(f(x)) sigma-dot(f(x')) ].

Then the lower-layer contribution converges to Theta^(L)_inf(x, x') Sigma-dot^(L+1)(x, x') delta_{kk'}. Adding the two contributions,

    Theta^(L+1)_inf(x, x') = Theta^(L)_inf(x, x') Sigma-dot^(L+1)(x, x') + Sigma^(L+1)(x, x'),

with Theta^(1)_inf = Sigma^(1). So at initialization the tangent kernel converges in probability to an explicit deterministic kernel given by a layer-wise recursion, and it factorizes as (scalar kernel) ⊗ Id over the output coordinates. It depends only on the depth, the nonlinearity, and the initialization variance — nothing about the particular random draw survives. The structure is telling: the second summand Sigma^(L+1) is the learning done by the top layer's own weights; the first summand Theta^(L)_inf Sigma-dot^(L+1) is the learning back-propagated into all the lower layers, the previous tangent kernel modulated by how the nonlinearity passes gradients (Sigma-dot).

That handles initialization. The harder claim is that Theta doesn't move during training, because if it drifts I'm back to a wandering kernel. Let me set up training a little more generally than plain gradient descent — the parameters follow a "training direction" d_t in function space,

    theta_dot_p = <partial_{theta_p} F^(L), d_t>_{pin},

which for gradient descent is d_t = -d|_{f_theta} = -(f - f*) for squared error, but I'll keep it general (it could come from another network, as in adversarial training). The only assumption I need is that the accumulated forcing integral_0^T ||d_t||_{pin} dt stays stochastically bounded as width grows. For squared error on any fixed finite interval this is automatic: ||d_t||_{pin} = ||f - f*||_{pin} is decreasing, so the integral is at most T times its initial value. Good.

Now I want: as n_1, ..., n_{L-1} -> infinity, Theta(t) -> Theta^(L)_inf uniformly on [0, T]. Induction on depth again. L = 1: Theta = Sigma^(1) does not depend on the parameters, so it's literally constant in time. Done. For L+1, split off the last layer as before. The first-L subnetwork is being driven by a back-propagated direction; differentiating through the last layer, the smaller network sees

    d'_t = sigma-dot(a-tilde^(L)(t)) ((1/sqrt(n_L)) W^(L)(t))^T d_t.

To apply the induction hypothesis to the subnetwork, I must check its forcing integral is bounded, i.e. integral ||d'_t|| dt is bounded. Since sigma is c-Lipschitz, |sigma-dot| <= c, so

    ||d'_t||_{pin} <= c ||(1/sqrt(n_L)) W^(L)(t)||_op ||d_t||_{pin}.

So I need to bound the operator norm of (1/sqrt(n_L)) W^(L)(t) along the whole trajectory. Split it: W^(L)(t) = W^(L)(0) + (W^(L)(t) - W^(L)(0)). The initial part is fine — each of the (fixed number n_{L+1} of) rows of W^(L)(0) has norm controlled by the LLN, so ||(1/sqrt(n_L)) W^(L)(0)||_op is bounded. The variation is the content of a lemma I'll need:

    lim ... lim sup_{t in [0,T]} ||(1/sqrt(n_ell)) (W^(ell)(t) - W^(ell)(0))||_op = 0,

in probability, for every layer ell. Grant the lemma for a moment; then ||(1/sqrt(n_L)) W^(L)(t)||_op is bounded uniformly in t, the subnetwork's forcing is bounded, and by induction the subnetwork's pre-activations evolve under the *constant* kernel Theta^(L)_inf:

    partial_t a-tilde^(L)_i(t) = (1/sqrt(n_L)) Phi_{Theta^(L)_inf}( <sigma-dot(a-tilde^(L)_i(t)) (W^(L)_i(t))^T d_t, .>_{pin} ).

And the last layer's own weights evolve by

    partial_t W^(L)_{ij}(t) = (1/sqrt(n_L)) <a^(L)_i(t), d_{t,j}>_{pin}.

The crucial feature of both: an explicit 1/sqrt(n_L) out front. So the per-parameter and per-activation motion is order 1/sqrt(n_L). Let me make that quantitative and couple the two, because the activation drift feeds the weight bound and vice versa. From the displays above, using the Cauchy-Schwarz inequality and partial_t ||.|| <= ||partial_t .||,

    partial_t ||W^(L)_i(t) - W^(L)_i(0)||_2 <= (1/sqrt(n_L)) ||a^(L)_i(t)||_{pin} ||d_t||_{pin},
    partial_t ||a-tilde^(L)_i(t) - a-tilde^(L)_i(0)||_{pin} <= (1/sqrt(n_L)) ||Theta^(L)_inf||_op ||sigma-dot(a-tilde^(L)_i(t))||_inf ||W^(L)_i(t)||_2 ||d_t||_{pin},

where I used that the operator norm of Phi_{Theta^(L)_inf} equals ||Theta^(L)_inf||_op. To bound both at once, bundle them into

    A(t) = ||a^(L)_i(0)||_{pin} + c ||a-tilde^(L)_i(t) - a-tilde^(L)_i(0)||_{pin} + ||W^(L)_i(0)||_2 + ||W^(L)_i(t) - W^(L)_i(0)||_2.

Its derivative: using |sigma-dot| <= c for the activation term and noting both ||W^(L)_i(t)||_2 and ||a^(L)_i(t)||_{pin} are each <= A(t) (the first because ||W(t)|| <= ||W(0)|| + ||W(t)-W(0)||, the second because ||a(t)|| <= ||a(0)|| + c||a-tilde(t)-a-tilde(0)|| by Lipschitz sigma),

    partial_t A(t) <= (1/sqrt(n_L)) ( c^2 ||Theta^(L)_inf||_op ||W^(L)_i(t)||_2 + ||a^(L)_i(t)||_{pin} ) ||d_t||_{pin}
                  <= (max{ c^2 ||Theta^(L)_inf||_op, 1 } / sqrt(n_L)) ||d_t||_{pin} A(t).

That's a linear differential inequality; Grönwall gives

    A(t) <= A(0) exp( (max{ c^2 ||Theta^(L)_inf||_op, 1 } / sqrt(n_L)) integral_0^t ||d_s||_{pin} ds ).

Since the forcing integral is stochastically bounded and ||Theta^(L)_inf||_op is constant, the exponent goes to 0 in probability as n_L -> infinity, so A(t) -> A(0). The activation drift ||a-tilde^(L)_i(t) - a-tilde^(L)_i(0)||_{pin} is at most c^{-1}(A(t) - A(0)) and the weight drift ||W^(L)_i(t) - W^(L)_i(0)||_2 at most A(t) - A(0), so both vanish at rate O(1/sqrt(n_L)). Each individual weight and each individual activation barely moves — quantitatively, like 1/sqrt(width).

Now translate "parameters barely move" into "Theta barely moves." Walk through the pieces of Theta^(L+1). The bias derivatives partial_{b^(L)_j} f_{theta, j'} = delta_{jj'} are exactly constant, no motion. The top connection-weight derivatives partial_{W^(L)_{ij}} f_{theta, j'} = (1/sqrt(n_L)) a^(L)_i delta_{jj'}; each is already O(1/sqrt(n_L)) and drifts at rate 1/sqrt(n_L) (since a^(L)_i does), so each summand of Theta of the form partial f ⊗ partial f varies at rate n_L^{-3/2}, and summed over the n_L of them, the top-layer block of Theta varies at rate 1/sqrt(n_L). For the lower-layer block, recall it is

    (1/n_L) sum_i Theta^(L)_inf(x, x') sigma-dot(a-tilde^(L)_i(x)) sigma-dot(a-tilde^(L)_i(x')) W^(L)_{ij} W^(L)_{ij'},

after the induction hypothesis collapsed the smaller-network kernel to Theta^(L)_inf delta_{ii'}. Two time-varying factors: the connection weights W^(L)_{ij}, which drift at 1/sqrt(n_L) as just shown; and the sigma-dot(a-tilde^(L)_i) terms. For the latter, since sigma has bounded second derivative, partial_t sigma-dot(a-tilde^(L)_i(t)) = sigma-ddot(...) partial_t a-tilde^(L)_i(t) = O(partial_t a-tilde^(L)_i(t)), and the pre-activation drifts at 1/sqrt(n_L), so sigma-dot also moves at 1/sqrt(n_L). Every moving factor in Theta moves at 1/sqrt(n_L), so the whole kernel does:

    Theta^(L+1)(t) -> Theta^(L+1)_inf   uniformly on [0, T].

The kernel is asymptotically frozen. So in the limit, the function obeys a *linear* ODE in function space against a fixed kernel:

    partial_t f_theta(t) = Phi_{Theta^(L)_inf ⊗ Id}( <d_t, .>_{pin} ),

and for gradient descent on squared error, partial_t f = -Theta^(L)_inf (f - f*) on the data. The non-convex parameter problem has become a fixed-kernel linear flow — the random-feature picture, realized by a genuine deep nonlinear network in the wide limit.

I owe the lemma — the uniform control ||(1/sqrt(n_ell))(W^(ell)(t) - W^(ell)(0))||_op -> 0 for every layer at once, not just the top. The two-quantity Grönwall above did the top layer assuming the subnetwork was already controlled; to get all layers simultaneously I need to back-propagate the same bookkeeping through the whole stack. Define, for each layer, the back-propagated training direction

    d^(ell)_t = d_t                                         if ell = L+1,
              = sigma-dot(a-tilde^(ell)) ((1/sqrt(n_ell)) W^(ell))^T d^(ell+1)_t   if ell <= L,

so that the preactivations and weights of every layer evolve as

    partial_t a-tilde^(ell) = Phi_{Theta^(ell)}( <d^(ell)_t, .>_{pin} ),
    partial_t W^(ell)       = (1/sqrt(n_ell)) <a^(ell), d^(ell+1)_t>_{pin},

with Theta^(ell) the subnetwork tangent kernels obeying their own recursion
Theta^(1) = [(1/sqrt(n_0)) a^(0)]^T[(1/sqrt(n_0)) a^(0)] ⊗ Id + beta^2 ⊗ Id and
Theta^(ell+1) = (1/sqrt(n_ell)) W^(ell) sigma-dot(a-tilde^(ell)) Theta^(ell) sigma-dot(a-tilde^(ell)) (1/sqrt(n_ell)) W^(ell) + [(1/sqrt(n_ell)) a^(ell)]^T[(1/sqrt(n_ell)) a^(ell)] ⊗ Id + beta^2 ⊗ Id. Now abbreviate w^(k)(t) = ||(1/sqrt(n_k)) W^(k)(t)||_op and a^(k)(t) = ||(1/sqrt(n_k)) a^(k)(t)||_{pin}. The c-Lipschitz nonlinearity gives the recursive direction bound ||d^(ell)_t|| <= c w^(ell)(t) ||d^(ell+1)_t||, hence ||d^(ell)_t|| <= c^{L+1-ell} (prod_{k=ell}^L w^(k)(t)) ||d_t||. The kernel recursion gives ||Theta^(1)||_op <= (a^(0))^2 + beta^2 and ||Theta^(ell+1)||_op <= c^2 (w^(ell))^2 ||Theta^(ell)||_op + (a^(ell))^2 + beta^2, so ||Theta^(ell+1)||_op is bounded by a polynomial P in the w's and a's (coefficients depending only on ell, c, beta, pin). Define the variations a-tilde^(k)(t) = ||(1/sqrt(n_k))(a-tilde^(k)(t) - a-tilde^(k)(0))||_{pin} and w-tilde^(k)(t) = ||(1/sqrt(n_k))(W^(k)(t) - W^(k)(0))||_op, and the aggregate A(t) = sum_{k=1}^L [ a^(k)(0) + c a-tilde^(k)(t) + w^(k)(0) + w-tilde^(k)(t) ]. From the evolution equations,

    partial_t a-tilde^(ell)(t) <= (1/sqrt(n_ell)) ||Theta^(ell)(t)||_op ||d^(ell)_t||,
    partial_t w-tilde^(ell)(t) <= (1/sqrt(n_ell)) a^(ell)(t) ||d^(ell+1)_t||,

so summing and substituting the polynomial bounds on ||Theta^(ell)|| and ||d^(ell+1)|| in terms of the w's and a's,

    partial_t A(t) <= (1/sqrt(min{n_1, ..., n_L})) Q( w^(1), ..., a^(L) ) ||d_t||,

with Q a positive-coefficient polynomial; and using a^(k)(t) <= a^(k)(0) + c a-tilde^(k)(t), w^(k)(t) <= w^(k)(0) + w-tilde^(k)(t) to fold everything into A,

    partial_t A(t) <= (1/sqrt(min{n})) Q-tilde( A(t) ) ||d_t||.

At t = 0, A(0) is stochastically bounded in the width limit: the scaled Gaussian operator norms w^(ell)(0) are tight, and the normalized activations a^(ell)(0) converge by the Gaussian-process limit. I do not need those operator norms to vanish; boundedness is the load-bearing fact. The 1/sqrt(min n) prefactor and a nonlinear Grönwall inequality then keep A(t) uniformly bounded on [0, tau] with tau -> T as min n -> infinity, and force partial_t A -> 0 uniformly, so A(t) -> A(0). In particular every w-tilde^(ell)(t) -> 0: every layer's weights, scaled by 1/sqrt(n_ell), barely move relative to initialization. That's the lemma, and it's exactly what I borrowed above to bound the top layer's operator norm.

One thing I want to make sure I believe, because it sounds wrong: I've just argued each hidden activation barely changes during training. Isn't the whole point of hidden layers to learn representations? Resolve it by counting. Each individual a^(ell)_i drifts at 1/sqrt(n_ell), but there are n_ell of them, and the network function depends on their *aggregate* through the 1/sqrt(n_ell)-weighted sums. A 1/sqrt(n) drift in each of n coordinates, combined coherently, is an O(1) effect on the function — that is precisely the first summand Theta^(L)_inf Sigma-dot^(L+1) in the kernel recursion, the lower layers' collective contribution to learning. So the lower layers do learn, in aggregate, even though no single neuron's preactivation moves appreciably. (A side consequence: since the preactivations stay near their initial Gaussian, they remain Gaussian throughout training, with the same covariance Sigma.)

With Theta frozen, return to the convergence question I raised at the very start. Convergence of kernel gradient descent to a global optimum needs Theta^(L)_inf positive definite. Intuitively it should be: positive-definiteness amounts to the span of the parameter-gradients becoming dense in function space as width grows, and the last layer's preactivations already form a rich family (universal-approximation-flavored). Let me prove it cleanly in the case that matters for high-dimensional data — inputs on the unit sphere S^{n0-1}, where all points have equal norm so the kernel is a function of the dot product x^T x'. Claim: for a non-polynomial Lipschitz sigma and L >= 2, Theta^(L)_inf restricted to the sphere is positive definite.

Unwind the recursion: Theta^(L+1)_inf = Sigma-dot^(L+1) Theta^(L)_inf + Sigma^(L+1) (writing the product of kernels pointwise). The product Sigma-dot^(L+1) Theta^(L)_inf of two positive-semidefinite kernels is positive semidefinite (the Schur/Hadamard product of PSD kernels is PSD). So if Sigma^(L+1) is positive definite, the sum is positive definite. Thus it suffices that the covariance kernels Sigma^(L) are positive definite for L >= 2. Push that down: is positive-definiteness of Sigma^(L) inherited by Sigma^(L+1)? For coefficients c_1, ..., c_d and distinct points,

    sum_{ij} c_i c_j Sigma^(L+1)(x_i, x_j) = E[ (sum_i c_i sigma(f(x_i)))^2 ] + (beta sum_i c_i)^2,

a sum of two non-negative terms. It vanishes only if sum_i c_i sigma(f(x_i)) = 0 almost surely. If Sigma^(L) is positive definite, the Gaussian vector (f(x_1), ..., f(x_d)) is non-degenerate, and since sigma is non-constant the only way that linear combination is a.s. zero is c_1 = ... = c_d = 0. So Sigma^(L) PD implies Sigma^(L+1) PD, and by induction it's enough to establish Sigma^(2) PD on the sphere.

For Sigma^(2): the layer-1 preactivations are jointly Gaussian with the 2x2 covariance built from Sigma^(1)(x,x) = 1/n0 + beta^2 (on the sphere) and Sigma^(1)(x,x') = (1/n0) x^T x' + beta^2, so

    Sigma^(2)(x, x') = E_{(X,Y) ~ N(0, that 2x2)}[ sigma(X) sigma(Y) ] + beta^2.

Rescale to unit variance: with mu(z) = sigma(z sqrt(1/n0 + beta^2)), the pair (X, Y) normalized has correlation rho = (n0 beta^2 + x^T x') / (n0 beta^2 + 1), and

    Sigma^(2)(x, x') = mu-hat(rho) + beta^2,

where mu-hat is the Gaussian dual of mu in the sense of Daniely et al. (2016): expand mu = sum_i a_i h_i in Hermite polynomials, then mu-hat(rho) = sum_i a_i^2 rho^i. Because sigma is non-polynomial, mu is non-polynomial, so infinitely many a_i are nonzero and the dual has infinitely many positive coefficients before the final change of variables. I have to be careful about parity here: non-polynomial alone does not mean the Hermite support itself has both even and odd indices (an even sigma would give only even-indexed coefficients). What rescues the Schoenberg step is the beta > 0 shift. Substituting rho = (n0 beta^2 + x^T x')/(n0 beta^2 + 1), every nonzero high-degree term a_i^2 rho^i expands as a positive binomial polynomial in x^T x' with powers of both parities below i, because the shift n0 beta^2 is positive. Infinitely many nonzero Hermite coefficients therefore give the dot-product series nu(x^T x') infinitely many positive even and positive odd coefficients. By the Schoenberg/Gneiting criterion (Gneiting, 2013) — a dot-product kernel on S^{n0-1} is positive definite for every n0 iff its power-series coefficients are strictly positive for infinitely many even and infinitely many odd powers — Sigma^(2) is positive definite on the sphere. Climbing back up, all Sigma^(L) for L >= 2 are positive definite, hence so are all Theta^(L)_inf for L >= 2. (The non-polynomial condition is sharp: a polynomial sigma gives only finitely many nonzero coefficients and the kernel fails to be PD for some input dimensions.) So for beta > 0 and L >= 2 with a non-polynomial nonlinearity, training converges to a global optimum of the function-space cost.

Now I can actually solve the dynamics for least squares and read off generalization. With C(f) = (1/2) ||f - f*||^2_{pin}, the function obeys

    partial_t f_t = Phi_K( <f* - f_t, .>_{pin} ),   K = Theta^(L)_inf ⊗ Id.

Introduce the linear operator Pi: f -> Phi_K(<f, .>_{pin}); on the finite dataset, Pi(f)_k(x) = (1/N) sum_i sum_{k'} f_{k'}(x_i) K_{kk'}(x_i, x). The ODE is partial_t f_t = -Pi(f_t - f*), a linear ODE, with solution

    f_t = f* + e^{-t Pi}(f_0 - f*).

Diagonalize Pi by its eigenfunctions f^(i) with eigenvalues lambda_i — these are exactly the kernel principal components of the data with respect to K, and lambda_i is the variance the component captures; there are at most N n_L positive ones. Decompose f_0 - f* = Delta^0 + sum_i Delta^i along the eigenspaces (Delta^0 in the null space of Pi). Then

    f_t = f* + Delta^0 + sum_i e^{-t lambda_i} Delta^i.

Each component relaxes exponentially at its own rate lambda_i: the function converges fastest along the top kernel principal components, slowest along the small-eigenvalue directions — which, for a smooth kernel, are the high-frequency, typically noisier directions. That immediately motivates early stopping: cut training off and you have fit the large-lambda directions while leaving the small-lambda (noisy) ones untouched. The eigenvalues of the kernel literally set a per-direction convergence schedule.

Since e^{-t Pi} is linear and f_0 is Gaussian (the infinite-width init), f_t is Gaussian at all times. As t -> infinity, assuming K is positive definite on the data so the N n_L x N n_L Gram matrix K-tilde is invertible, the surviving piece is

    f_inf, k(x) = kappa_{x,k}^T K-tilde^{-1} y* + ( f_0(x) - kappa_{x,k}^T K-tilde^{-1} y_0 ),

with kappa_{x,k} the vector of kernel values (K_{kk'}(x, x_i)), y* the targets on the data, y_0 the network's own outputs on the data at init. The first term — the mean — is precisely kernel ridge regression with kernel Theta^(L)_inf in the ridgeless (regularization -> 0) limit, equivalently the maximum-a-posteriori estimate under a Gaussian-process prior f_k ~ N(0, Theta^(L)_inf) conditioned on interpolating the data. The second term is a centered Gaussian fluctuation that vanishes on the training points. So a wide network trained to convergence on squared error *is* a kernel machine: it generalizes off the data exactly as ridgeless kernel regression with the deterministic kernel Theta^(L)_inf, plus a mean-zero wobble pinned to zero on the training set. The generalization mystery dissolves into the spectrum of a fixed, explicit kernel.

Let me make this concrete in code, both to compute the limiting kernel from the recursion and to check it really is the limit of a finite network's tangent kernel. I'll take ReLU, where the Gaussian expectations have closed forms — the arc-cosine kernels of Cho & Saul (2009). For a 2x2 covariance with variances Sigma_xx, Sigma_x'x' and correlation rho = Sigma_xx' / sqrt(Sigma_xx Sigma_x'x'), and theta = arccos(rho),

    E[relu(X) relu(X')]  = sqrt(Sigma_xx Sigma_x'x') / (2 pi) * ( sin theta + (pi - theta) cos theta ),
    E[relu'(X) relu'(X')] = (pi - theta) / (2 pi),

the second being just P(X > 0, X' > 0) for the centered Gaussian pair. These are Sigma^(L+1) (minus beta^2) and Sigma-dot^(L+1), so the recursion is direct.

```python
import numpy as np
import torch


def relu_dual(cov_xx, cov_xpxp, cov_xxp):
    # Gaussian expectations for ReLU: E[relu(X)relu(X')] and E[relu'(X)relu'(X')].
    denom = np.sqrt(cov_xx * cov_xpxp)
    rho = np.clip(cov_xxp / np.maximum(denom, 1e-12), -1.0, 1.0)
    angle = np.arccos(rho)
    nngp = denom / (2.0 * np.pi) * (
        np.sin(angle) + (np.pi - angle) * np.cos(angle)
    )
    nngp_dot = (np.pi - angle) / (2.0 * np.pi)
    return nngp, nngp_dot


def infinite_ntk(X, Xp, depth, beta=0.1):
    # Scalar fully connected ReLU specialization of the Neural Tangents Dense/Relu recursion.
    if depth < 1:
        raise ValueError("depth counts affine layers and must be at least 1")
    n0 = X.shape[1]
    beta2 = beta ** 2
    sig = X @ Xp.T / n0 + beta2
    sig_xx = (X * X).sum(axis=1) / n0 + beta2
    sig_pp = (Xp * Xp).sum(axis=1) / n0 + beta2
    theta = sig.copy()
    for _ in range(depth - 1):
        nngp, nngp_dot = relu_dual(sig_xx[:, None], sig_pp[None, :], sig)
        sig_xx = relu_dual(sig_xx, sig_xx, sig_xx)[0] + beta2
        sig_pp = relu_dual(sig_pp, sig_pp, sig_pp)[0] + beta2
        sig = nngp + beta2
        theta = theta * nngp_dot + sig
    return theta, sig


class WideMLP(torch.nn.Module):
    # NTK parametrization: preactivation = (1/sqrt(n_in)) W a + beta b.
    def __init__(self, n0, width, depth, beta=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth counts affine layers and must be at least 1")
        self.beta = beta
        sizes = [n0] + [width] * (depth - 1) + [1]
        self.Ws = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim, in_dim))
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
        )
        self.bs = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim)) for out_dim in sizes[1:]
        )
        self.scales = [in_dim ** -0.5 for in_dim in sizes[:-1]]

    def forward(self, x):
        a = x
        last = len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = self.scales[i] * (a @ W.T) + self.beta * b
            if i != last:
                a = torch.relu(a)
        return a.squeeze(-1)


def _jacobian_rows(net, X):
    params = list(net.parameters())
    rows = []
    for xi in X:
        out = net(xi.unsqueeze(0)).squeeze()
        grads = torch.autograd.grad(out, params)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)


def empirical_ntk(net, X, Xp=None):
    # Jacobian contraction, matching the canonical empirical NTK definition.
    J = _jacobian_rows(net, X)
    Jp = J if Xp is None else _jacobian_rows(net, Xp)
    return (J @ Jp.T).detach().cpu().numpy()


def kernel_regression(K_train, K_test, y, ridge=0.0):
    # f_inf = K(x*, X) K(X, X)^{-1} y; set ridge > 0 only for conditioning.
    A = K_train + ridge * np.eye(K_train.shape[0])
    return K_test @ np.linalg.solve(A, y)
```

The recap, as a single causal chain: I stopped tracking the weights and tracked the function, and gradient flow on theta turned into kernel gradient descent against the tangent kernel Theta = sum_p grad_theta f ⊗ grad_theta f; on a finite dataset the empirical operator includes the 1/N average over training points. With the 1/sqrt(width) parametrization, two things happen as width grows — Theta at initialization concentrates on a deterministic kernel given by the layer recursion Theta^(L+1)_inf = Theta^(L)_inf Sigma-dot^(L+1) + Sigma^(L+1), and Theta stays frozen during training because every hidden preactivation and scaled weight displacement is small (Grönwall on the coupled drift). So least-squares training is a linear ODE in function space against a fixed kernel; when the sphere positive-definiteness condition applies, it converges globally, fastest along the top kernel principal components, and at convergence the mean predictor is ridgeless kernel regression with Theta^(L)_inf.
