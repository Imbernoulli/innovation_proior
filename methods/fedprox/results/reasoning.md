Let me start from what actually breaks when I try to train one model across a network of phones. I have a global objective that is a weighted sum of per-device local risks, f(w) = sum_k p_k F_k(w) with p_k = n_k/n, and the brutal constraints are that I cannot move the data off any device, communication is slow and rare, and each round I only get to hear from a small handful of devices. The recipe everyone uses is the obvious one: broadcast the current model w^t, let each selected device run a few epochs of SGD on its own F_k, send the resulting model back, and average the models the server gets. That averaging is what folds the local work into a global step, and the reason it can work at all is that everyone started this round from the same w^t — if I average models that were trained from a common initialization on different data, they tend to sit in the same low-loss basin and their average is a sensible model, whereas averaging models trained from different initializations lands me somewhere worse than either parent. So the shared per-round starting point is doing real load-bearing work; the local solutions have to stay in a common basin for their average to mean anything.

Now here is the thing that ruins it. The devices' data is non-identically distributed — each device's distribution D_k is whatever that user generated — so the local risks F_k genuinely differ, and the minimizer of any one F_k can be far from the minimizer of f. When I let a device run more local epochs, I am letting it solve its *own* F_k more exactly, which means I am walking it toward *its* optimum, not the global one. The more local work I do per round (which is exactly what I want, to save communication), the further each device drifts toward its own private solution, the further apart the returned models spread, and the less their average is in any shared basin. People have watched this happen: crank up the local epochs on non-IID data and convergence slows, oscillates, and on the genuinely heterogeneous splits it can outright diverge. So the single knob I have — number of local epochs E — is fighting itself: turn it up to save rounds and you amplify the drift that wrecks the average.

And there is a second, orthogonal mess. Devices are wildly different in compute, network, battery. If I fix E for everyone, the slow devices can't finish in the round's time window, and the standard move is to just drop them. But dropping is ugly twice over: I throw away whatever partial computation the straggler did, and if the devices that tend to drop have characteristic data — say, a particular hardware demographic — then by systematically excluding them I am quietly changing the distribution I'm averaging over, i.e. changing the objective I'm actually optimizing. So "set a uniform amount of work and drop whoever can't keep up" is both wasteful and biased.

Let me also be honest that the method I'm starting from has no convergence guarantee in the regime I actually live in. Plenty of analyses cover local-update schemes — but they assume either IID data, so that every device is running a copy of the same stochastic process and the local solutions are statistically interchangeable, or that *all* devices participate every round, or they pin the local solver to be SGD. The realistic combination — non-IID, only a fraction of devices active, an arbitrary and possibly inexact local solver — falls through all of those. So I want two things at once: a practical fix for the drift-and-stragglers mess, and a method I can actually prove something about.

Let me look hard at *why* the drift happens, because the fix should come out of the mechanism. The disease is that more local steps = a more-exactly-solved local F_k = a longer walk away from w^t along the local gradient field, and that walk is unbounded — nothing stops a device from going as far as its own optimum demands. If I want the average to stay meaningful I need to keep every device's returned model from straying too far from the shared w^t it started at. The crude way is to limit E, but E is coarse, fixed per round, and the same value is wrong for different devices and different rounds. What I really want is to cap the *displacement* ||w_k - w^t|| directly, regardless of how many steps the device takes or which solver it uses — a soft tether back to the round's starting model that the optimization itself respects.

A tether on displacement that the optimizer respects is exactly what a quadratic penalty buys me. Instead of having device k minimize F_k(w), have it minimize

  h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2.

The added term is a spring anchored at the broadcast model w^t: it costs the device to move away from where it started, and the bigger mu, the stiffer the spring. I'm not capping E and I'm not changing the solver — I'm changing the *objective*, so however the device optimizes and for however many steps, it is now optimizing something that penalizes drift. Let me check it does what I think to the local solver. The gradient of the penalty is mu(w - w^t), so a local SGD step on h_k is

  w <- w - eta * (∇F_k(w) + mu (w - w^t)),

which is just ordinary SGD on F_k with an extra restoring force pulling w back toward w^t every step. When w has drifted far from w^t the spring is strong and pulls it back; near w^t the spring is weak and it behaves like plain local SGD. That's the tether, and it costs nothing but a frozen copy of w^t and one elementwise term per step. Set mu = 0 and I recover exactly the local SGD the baseline does — so whatever I build is a strict generalization, and I can dial heterogeneity-robustness with a single continuous knob instead of the integer E.

Now I want to know whether this quadratic is doing more than just "feels like it should help," because the second thing I wanted was provability, and the quadratic is suspiciously the same object that makes non-convex problems analyzable. F_k is non-convex — it's a neural net — but suppose its negative curvature is bounded: ∇^2 F_k ≥ -L_ I for some L_ > 0. Then the Hessian of h_k is ∇^2 F_k + mu I ≥ (mu - L_) I. So if I pick mu > L_, h_k is (mu - L_)-strongly convex *even though F_k isn't convex at all*. Write bar_mu = mu - L_ > 0. That's a real gift: the local subproblem each device solves is now strongly convex, hence has a unique minimizer and a clean displacement bound. The exact minimizer hat_w_k = argmin_w h_k(w; w^t) satisfies, by bar_mu-strong convexity evaluated against the gradient at w^t (where ∇h_k(w^t; w^t) = ∇F_k(w^t) since the spring contributes nothing at its anchor),

  ||hat_w_k - w^t|| ≤ (1/bar_mu) ||∇F_k(w^t)||.

So the displacement of the exactly-solved local problem is bounded by the local gradient size divided by bar_mu — the spring really does cap how far a device can travel, and stiffer mu means a smaller cap. The same quadratic that tethers the drift is the thing that convexifies the subproblem and gives me a displacement bound to build a proof on. Good — these are the same move, not two.

Now the straggler half. I refused to drop them and I refused to mandate a fixed E. So I need a way to *say*, formally, "device k did some variable amount of work this round and returned a partial solution," and then aggregate that partial solution safely. The natural measure is inexactness of the local minimization: device k returns some w_k that need not be the exact minimizer of h_k, and I'll quantify how far it got by how small the residual gradient of h_k is at w_k relative to where it started. Say w_k is a gamma-inexact solution if

  ||∇h_k(w_k; w^t)|| ≤ gamma * ||∇h_k(w^t; w^t)|| = gamma * ||∇F_k(w^t)||,

with gamma in [0,1]: gamma = 0 means w_k is the exact minimizer (the gradient of h_k vanishes), gamma = 1 means it did essentially no useful work, and in between gamma interpolates. The number of local iterations a device runs is a proxy for gamma — more iterations push the residual gradient down. The point of phrasing it this way is that gamma can vary by device and by round, gamma_k^t, and a slow device just has a larger gamma_k^t. I don't drop it; I aggregate its partial w_k and I account for its inexactness in the analysis. That is the systems-heterogeneity fix, and it rides on the same proximal subproblem: the residual-gradient definition only makes sense because h_k has a well-defined gradient ∇h_k(w; w^t) = ∇F_k(w) + mu(w - w^t) and (under mu > L_) a unique minimizer to be inexact about.

Before I try to prove anything, let me make sure I'm not reinventing something with a known fatal flaw, because there are two relatives that look close. One is elastic-averaging SGD from the data center: it also couples each worker to a center by a quadratic force. But it maintains a *separate* moving-average center variable with its own update rule, it's locked to SGD as the solver, and it was only ever analyzed for quadratics — none of which is what I need; my anchor is just this round's broadcast w^t, no extra state, any solver, and I want a non-convex guarantee. The more dangerous relative is DANE/AIDE, the distributed Newton-type methods, because their local subproblem has the *same* proximal quadratic plus a linear correction term, F_i(w) - (∇F_i(w_{t-1}) - eta ∇f(w_{t-1}))^T w + (mu/2)||w - w_{t-1}||^2. That gradient-correction term is what gives DANE its data-center edge — it corrects each local gradient by the gap to the global gradient. But it needs ∇f(w_{t-1}), the *full* global gradient over all devices, every round. In my world only a handful of devices report each round, so I can only estimate ∇f from the active fraction; the correction I can afford is noisy and stale, so I should not build the local subproblem around it. I deliberately do *not* carry a gradient-correction term — the bare proximal anchor, no linear correction, is the right call for low participation, precisely because it needs nothing global beyond the w^t I already broadcast.

So the method is: broadcast w^t; each selected device approximately minimizes h_k(w; w^t) = F_k(w) + (mu/2)||w - w^t||^2 to whatever gamma_k^t-inexactness it can afford; send back w_k; the server averages the w_k exactly as before. Server aggregation is unchanged — this is a lightweight modification to the existing method, which also means I can reason about the existing method, since it's the mu = 0 case. Now let me actually prove the thing converges on non-convex F_k under non-IID data with only K devices per round, because that's the part nobody has had.

I need to quantify heterogeneity, since with constant step sizes and non-IID data I can't expect convergence without bounding how much the devices disagree. The right object is a dissimilarity: say the F_k are B-locally dissimilar at w if

  E_k ||∇F_k(w)||^2 ≤ B^2 ||∇f(w)||^2,

with the expectation over devices weighted by p_k = n_k/n, and B(w) = sqrt(E_k||∇F_k(w)||^2 / ||∇f(w)||^2). If all the local functions were identical then E_k||∇F_k||^2 = ||∇f||^2 and B = 1; the more the local gradients scatter around the global gradient, the bigger B. So B ≥ 1 is a clean scalar measuring statistical heterogeneity, and B = 1 is the homogeneous limit. (I'll only need it to hold on the non-stationary region where ||∇f(w)||^2 > epsilon, since near a stationary point the ratio is ill-defined and I don't need it there.) As a sanity sibling, if I instead assume bounded variance E_k||∇F_k(w) - ∇f(w)||^2 ≤ sigma^2, then E_k||∇F_k||^2 = ||∇f||^2 + E_k||∇F_k - ∇f||^2 ≤ ||∇f||^2 + sigma^2 (the cross term vanishes because E_k[∇F_k] = ∇f), so on ||∇f||^2 ≥ epsilon I get B ≤ sqrt(1 + sigma^2/epsilon). The variance picture is a special case of dissimilarity, which is what I'll use because it's exactly the per-step quantity I keep generating.

Let me carry one round and bound the expected decrease in f. The inexactness condition says each returned w_k^{t+1} leaves a residual gradient; write it as e_k^{t+1} defined by

  ∇F_k(w_k^{t+1}) + mu (w_k^{t+1} - w^t) - e_k^{t+1} = 0,   ||e_k^{t+1}|| ≤ gamma ||∇F_k(w^t)||,

i.e. e_k^{t+1} = ∇h_k(w_k^{t+1}; w^t) is exactly the residual gradient of the local problem, and gamma-inexactness bounds it. (I'll write a single gamma for now and put back the per-device gamma_k^t at the end.) Define the average of *all* devices' returned models, bar_w^{t+1} = E_k[w_k^{t+1}] — this is the "ideal" full-participation aggregate, the thing the K-device sample is approximating. Solving the stationarity condition for w_k^{t+1} - w^t = -(1/mu)(∇F_k(w_k^{t+1}) - e_k^{t+1}) and taking E_k,

  bar_w^{t+1} - w^t = -(1/mu) E_k[∇F_k(w_k^{t+1})] + (1/mu) E_k[e_k^{t+1}].

I want this in the form "a step along -∇f plus an error," because then the descent lemma takes over. Write

  bar_w^{t+1} - w^t = -(1/mu)(∇f(w^t) + M_{t+1}),   so M_{t+1} = E_k[∇F_k(w_k^{t+1}) - ∇F_k(w^t) - e_k^{t+1}],

using E_k[∇F_k(w^t)] = ∇f(w^t). M_{t+1} is the deviation from a clean gradient step caused by (a) evaluating ∇F_k at the moved-and-inexact w_k^{t+1} rather than at w^t, and (b) the residual e_k. So bar_w^{t+1} is, up to M_{t+1}, a gradient-descent step on f with step 1/mu. The whole proof is about controlling M_{t+1} and the sampling gap.

First the displacements, because everything is in terms of ||∇F_k(w^t)||. The exact minimizer hat_w_k^{t+1} satisfies ||hat_w_k^{t+1} - w^t|| ≤ (1/bar_mu)||∇F_k(w^t)|| as I derived, and the inexact w_k^{t+1} is close to it: bar_mu-strong convexity gives ||hat_w_k^{t+1} - w_k^{t+1}|| ≤ (1/bar_mu)||∇h_k(w_k^{t+1})|| = (1/bar_mu)||e_k^{t+1}|| ≤ (gamma/bar_mu)||∇F_k(w^t)||. (The first inequality is the standard strong-convexity bound: for a bar_mu-strongly convex function the distance to the minimizer is at most 1/bar_mu times the gradient norm.) Triangle inequality:

  ||w_k^{t+1} - w^t|| ≤ ||w_k^{t+1} - hat_w_k^{t+1}|| + ||hat_w_k^{t+1} - w^t|| ≤ ((1 + gamma)/bar_mu) ||∇F_k(w^t)||.

So each device's drift is bounded by (1+gamma)/bar_mu times its local gradient — the spring cap, loosened slightly by inexactness. Now lift this to the average using Jensen and then dissimilarity:

  ||bar_w^{t+1} - w^t|| ≤ E_k ||w_k^{t+1} - w^t|| ≤ ((1+gamma)/bar_mu) E_k||∇F_k(w^t)||
                       ≤ ((1+gamma)/bar_mu) sqrt(E_k||∇F_k(w^t)||^2) ≤ (B(1+gamma)/bar_mu) ||∇f(w^t)||,

where the middle step is Jensen (E||X|| ≤ sqrt(E||X||^2)) and the last is the dissimilarity bound. Call this (1). The full-participation aggregate moves by at most B(1+gamma)/bar_mu times the global gradient — and notice B appears here: more heterogeneity means a looser cap, exactly the trade-off I expect.

Next bound M_{t+1}. Each term is ∇F_k(w_k^{t+1}) - ∇F_k(w^t) - e_k^{t+1}; by L-Lipschitz smoothness of F_k the gradient difference is at most L||w_k^{t+1} - w^t||, and ||e_k|| ≤ gamma||∇F_k(w^t)||, so

  ||M_{t+1}|| ≤ E_k[L||w_k^{t+1} - w^t|| + ||e_k^{t+1}||] ≤ (L(1+gamma)/bar_mu + gamma) E_k||∇F_k(w^t)||
            ≤ (L(1+gamma)/bar_mu + gamma) B ||∇f(w^t)||,

using the per-device displacement bound and then dissimilarity. Now the descent lemma on bar_w^{t+1}, since f is L-smooth:

  f(bar_w^{t+1}) ≤ f(w^t) + <∇f(w^t), bar_w^{t+1} - w^t> + (L/2)||bar_w^{t+1} - w^t||^2.

Substitute bar_w^{t+1} - w^t = -(1/mu)(∇f + M_{t+1}). The inner product gives <∇f, -(1/mu)(∇f + M)> = -(1/mu)||∇f||^2 - (1/mu)<∇f, M>, and -(1/mu)<∇f, M> ≤ (1/mu)||∇f|| ||M|| ≤ (1/mu)(L(1+gamma)/bar_mu + gamma)B ||∇f||^2. The quadratic term uses bound (1): (L/2)||bar_w^{t+1}-w^t||^2 ≤ (L/2)(B(1+gamma)/bar_mu)^2 ||∇f||^2 = (L(1+gamma)^2 B^2/(2 bar_mu^2))||∇f||^2. Collecting,

  f(bar_w^{t+1}) ≤ f(w^t) - ((1 - gamma B)/mu - LB(1+gamma)/(bar_mu mu) - L(1+gamma)^2 B^2/(2 bar_mu^2)) ||∇f(w^t)||^2.

Call this (3). Read it: the leading good term is (1 - gamma B)/mu — a decrease proportional to ||∇f||^2 with coefficient roughly 1/mu — eaten into by two penalty terms scaling with L and B. If I make mu large enough (so 1/mu dominates the penalty terms... wait, 1/mu shrinks with mu too, so it's not that simple, but the structure is that there is a regime of mu, B where the net coefficient is positive). The first penalty, gamma B/mu, demands gamma B < 1 for the leading term to even stay positive — heterogeneity B and inexactness gamma trade off against each other, which is intuitively right: sloppy local solves are only safe if the network isn't too heterogeneous.

But (3) is for bar_w^{t+1}, the all-device average. The algorithm only has w^{t+1} = (1/K) sum_{k in S_t} w_k^{t+1}, the average over K randomly sampled devices. So I have to pay for the sampling gap ||w^{t+1} - bar_w^{t+1}||. Local Lipschitz continuity of f gives

  f(w^{t+1}) ≤ f(bar_w^{t+1}) + L_0 ||w^{t+1} - bar_w^{t+1}||,   L_0 ≤ ||∇f(w^t)|| + L(||bar_w^{t+1}-w^t|| + ||w^{t+1}-w^t||),

where L_0 is a local Lipschitz constant valid on the segment between w^{t+1} and bar_w^{t+1} (both near w^t), bounded by the gradient at w^t plus L times the radius of the neighborhood. Take expectation over the device sample S_t. The key fact is that w^{t+1} is the empirical mean of K i.i.d. draws (each w_k^{t+1} drawn with the device masses) whose population mean is bar_w^{t+1}, so its variance is at most (1/K) times the population second moment about bar_w^{t+1}:

  E_{S_t}||w^{t+1} - bar_w^{t+1}||^2 ≤ (1/K) E_k||w_k^{t+1} - bar_w^{t+1}||^2 ≤ (2/K) E_k||w_k^{t+1} - w^t||^2,

where the first inequality is the variance of an empirical mean of K independent device draws, and the second is a loose second-moment bound around w^t. The exact identity around the population mean would even remove the 2, but I keep the uniform factor 2 because that is the constant that propagates into rho. Then plug the per-device displacement bound and dissimilarity:

  E_{S_t}||w^{t+1} - bar_w^{t+1}||^2 ≤ (2/K)(1+gamma)^2/bar_mu^2 E_k||∇F_k(w^t)||^2 ≤ (2 B^2/K)(1+gamma)^2/bar_mu^2 ||∇f(w^t)||^2.

So the sampling deviation shrinks like 1/K (more devices per round = tighter aggregate) and grows like B^2 (heterogeneity makes the device-to-device spread larger). Now assemble Q_t = E_{S_t}[L_0 ||w^{t+1} - bar_w^{t+1}||]:

  Q_t ≤ (||∇f(w^t)|| + 2L||bar_w^{t+1}-w^t||) E_{S_t}||w^{t+1}-bar_w^{t+1}|| + L E_{S_t}||w^{t+1}-bar_w^{t+1}||^2,

where I bounded ||w^{t+1}-w^t|| ≤ ||w^{t+1}-bar_w^{t+1}|| + ||bar_w^{t+1}-w^t|| and folded one copy of ||w^{t+1}-bar_w^{t+1}|| into the squared expectation, picking up the 2L. Use E_{S_t}||w^{t+1}-bar_w^{t+1}|| ≤ sqrt(E_{S_t}||w^{t+1}-bar_w^{t+1}||^2) ≤ B(1+gamma)sqrt(2)/(bar_mu sqrt(K)) ||∇f|| for the first term, ||bar_w^{t+1}-w^t|| ≤ B(1+gamma)/bar_mu ||∇f|| from (1), and the squared bound for the last term:

  Q_t ≤ (B(1+gamma)sqrt(2)/(bar_mu sqrt(K)) + L B^2 (1+gamma)^2/(bar_mu^2 K)(2 sqrt(2K) + 2)) ||∇f(w^t)||^2.

Let a = B(1+gamma)/bar_mu. Then ||bar_w^{t+1}-w^t|| ≤ a||∇f||, E||w^{t+1}-bar_w^{t+1}|| ≤ a sqrt(2/K)||∇f||, and E||w^{t+1}-bar_w^{t+1}||^2 ≤ 2a^2/K ||∇f||^2. Substituting into Q_t gives a sqrt(2/K)||∇f||^2 from the first ||∇f|| term, plus 2L a^2 sqrt(2/K)||∇f||^2 from 2L||bar_w-w^t|| times the square-root sampling deviation, plus 2L a^2/K||∇f||^2 from the squared sampling deviation. The last two pieces combine as L a^2(2 sqrt(2K)+2)/K, which is exactly the second coefficient above. Call this (4).

Now combine. f(w^{t+1}) ≤ f(bar_w^{t+1}) + Q_t, take E_{S_t}, and substitute (3) and (4):

  E_{S_t}[f(w^{t+1})] ≤ f(w^t) - rho ||∇f(w^t)||^2,

with the coefficient written out as

  rho = 1/mu - gamma B/mu - B(1+gamma)sqrt(2)/(bar_mu sqrt(K)) - LB(1+gamma)/(bar_mu mu)
        - L(1+gamma)^2 B^2/(2 bar_mu^2) - L B^2 (1+gamma)^2/(bar_mu^2 K)(2 sqrt(2K) + 2).

There it is — one round of the method decreases f in expectation by rho ||∇f(w^t)||^2, with rho assembled from exactly the heterogeneity B, the inexactness gamma, the participation K, the smoothness L, and the proximal stiffness through mu and bar_mu = mu - L_. For this to be a genuine decrease I need the whole displayed rho to be positive. Two terms show unavoidable qualitative constraints: the leading 1/mu must beat gamma B/mu, so I need gamma B < 1; and the sampling term B/(bar_mu sqrt(K)) has to be controlled, which points to B/sqrt(K) < 1, i.e. enough devices per round relative to how heterogeneous the network is. Those constraints are the qualitative heart of it: sloppy local solves are only plausible when the network is homogeneous, and heterogeneity needs enough sampled devices. The proximal term enters through bar_mu in every denominator — a stiffer spring (larger mu, hence larger bar_mu) shrinks all the penalty terms, which is the precise sense in which the anchor stabilizes convergence.

The rate falls right out by telescoping. If rho > 0 holds at each round, take total expectation and sum the per-round decrease from t = 0 to T-1: sum_t rho E||∇f(w^t)||^2 ≤ f(w^0) - f(w^T) ≤ f(w^0) - f^* = Delta. So (1/T) sum_t E||∇f(w^t)||^2 ≤ Delta/(rho T), which is ≤ epsilon after T = O(Delta/(rho epsilon)) rounds. That's convergence to an epsilon-approximate stationary point for general non-convex F_k under non-IID data with only K devices reporting per round and inexact local solves — the guarantee the bare local-averaging method didn't have.

Let me sanity-check the convex special case to make sure the constants are sane and to see what mu the theory wants. Take F_k convex, so L_ = 0 and bar_mu = mu, and gamma = 0 (exact local solves). With 1 << B ≤ 0.5 sqrt(K), the small terms (the 1/K ones, and gamma-terms which vanish) are dominated and the decrease is approximately

  E_{S_t}[f(w^{t+1})] ⪅ f(w^t) - (1/(2mu)) ||∇f(w^t)||^2 + (3LB^2/(2 mu^2)) ||∇f(w^t)||^2.

The 1/(2mu) is the surviving fraction of the leading 1/mu after the B/sqrt(K) ≤ 1/2 sampling term eats up to half of it; the 3LB^2/(2mu^2) collects the L(1+gamma)^2 B^2/(2 bar_mu^2) curvature term and its 1/K sibling under B ≤ 0.5 sqrt(K). Now optimize over mu: the net coefficient is 1/(2mu) - 3LB^2/(2mu^2), maximized when its derivative -1/(2mu^2) + 3LB^2/mu^3 = 0, i.e. mu = 6LB^2. Plug mu ≈ 6LB^2 back: 1/(2·6LB^2) - 3LB^2/(2·(6LB^2)^2) = 1/(12LB^2) - 3LB^2/(72 L^2 B^4) = 1/(12LB^2) - 1/(24LB^2) = 1/(24LB^2). So rho ≈ 1/(24LB^2), and the number of rounds to reach a gradient-squared below epsilon is O(LB^2 Delta/epsilon). This tells me two real things: to solve to higher accuracy (smaller epsilon forces a larger B_epsilon since the dissimilarity bound has to hold on a larger non-stationary region) I should *increase* mu — the spring should stiffen as I demand a more accurate solution; and if I substitute the bounded-variance bound B_epsilon ≤ sqrt(1 + sigma^2/epsilon), the round count becomes O(LDelta/epsilon + LDelta sigma^2/epsilon^2), which is exactly the SGD complexity. So the method doesn't beat distributed SGD asymptotically — it matches it — and indeed under genuinely non-IID data a local-update scheme can be *worse* than distributed SGD. I should be clear-eyed: the analysis gives *sufficient* conditions for convergence and a characterization of how heterogeneity hurts, not a proof of superiority or a better asymptotic rate.

Now the variable-work extension I promised, which is almost free. Everywhere I used a single gamma, I can let it be gamma_k^t per device per round, because the only place gamma entered the expectations over devices was through E_k[(1+gamma)||∇F_k(w^t)||], and E_k[(1+gamma_k^t)||∇F_k(w^t)||] ≤ (1 + max_{k in S_t} gamma_k^t) E_k||∇F_k(w^t)||. So define gamma^t = max_{k in S_t} gamma_k^t and the entire bound goes through with gamma replaced by gamma^t. Concretely: a fast device solves h_k more exactly (small gamma_k^t), a straggler solves it less exactly (large gamma_k^t, e.g. by running fewer local iterations), every device contributes its partial w_k, and the round still decreases f as long as the *worst* inexactness gamma^t satisfies the rho^t > 0 condition. Nobody is dropped; the straggler's partial work is aggregated and paid for honestly in the bound through gamma^t. That is the systems-heterogeneity story, and it cost one line of the proof because I set up inexactness as residual-gradient size from the start.

Let me put the whole thing into the code I'd actually run, filling the local optimizer slot in the federated harness. The theorem uses p_k sampling and a simple average for the clean analysis; the implementation path I want to mirror uses the baseline engineering choice of uniform client sampling and a sample-weighted server mean. The local objective is F_k + (mu/2)||w - w^t||^2, realized by adding mu(w - w^t) to the gradient at every local step, with w^t stored as a frozen slot; variable work is just running each device's local solver for however many epochs it can afford before returning its partial solution.

```python
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import optimizer
from .fedbase import BaseFedarated


class PerturbedGradientDescent(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        return control_flow_ops.group(
            state_ops.assign_sub(var, lr_t * (grad + mu_t * (var - vstar)))
        )

    def set_params(self, global_params, client):
        with client.graph.as_default():
            for variable, value in zip(tf.trainable_variables(), global_params):
                self.get_slot(variable, "vstar").load(value, client.sess)


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        self.inner_opt = PerturbedGradientDescent(
            params["learning_rate"], params["mu"]
        )
        super(Server, self).__init__(params, learner, dataset)

    def train_round(self, round_num):
        _, selected_clients = self.select_clients(
            round_num, num_clients=self.clients_per_round
        )
        np.random.seed(round_num)
        active_clients = np.random.choice(
            selected_clients,
            round(self.clients_per_round * (1 - self.drop_percent)),
            replace=False,
        )
        self.inner_opt.set_params(self.latest_model, self.client_model)

        client_solutions = []
        for client in selected_clients.tolist():
            client.set_params(self.latest_model)
            if client in active_clients:
                solution, stats = client.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
            else:
                partial_epochs = np.random.randint(low=1, high=self.num_epochs)
                solution, stats = client.solve_inner(
                    num_epochs=partial_epochs, batch_size=self.batch_size
                )
            client_solutions.append(solution)
            self.metrics.update(rnd=round_num, cid=client.id, stats=stats)

        self.latest_model = self.aggregate(client_solutions)
        self.client_model.set_params(self.latest_model)

    def aggregate(self, weighted_solutions):
        total_weight = 0.0
        base = [0] * len(weighted_solutions[0][1])
        for weight, solution in weighted_solutions:
            total_weight += weight
            for i, value in enumerate(solution):
                base[i] += weight * value.astype(np.float64)
        return [value / total_weight for value in base]
```

Let me trace the causal chain back to be sure it holds together. I start stuck: averaging locally-trained models works only when they stay in a shared basin, but non-IID data makes more local work drive each device toward its own optimum, so the one knob I have (local epochs) buys communication efficiency at the cost of drift that wrecks the average and can diverge — and meanwhile heterogeneous devices cannot all do the same work, so stragglers get dropped, wasting their effort and biasing the sample. I want to cap how far each device strays from the broadcast model directly, regardless of solver or step count, which is exactly what a quadratic spring (mu/2)||w - w^t||^2 added to the local objective does: it tethers the drift, reduces to plain local averaging at mu = 0, and — because adding mu I to the Hessian convexifies F_k whenever mu exceeds its negative curvature L_ — simultaneously turns the local subproblem strongly convex, which hands me the displacement bound ||w_k - w^t|| ≤ (1+gamma)/bar_mu ||∇F_k(w^t)|| that the proof runs on. I refuse to drop stragglers by measuring local work as gamma-inexactness in the residual gradient of that same subproblem, which lets work vary per device and round while still being aggregated. I deliberately drop the DANE-style gradient-correction term because it needs the full global gradient and fails under low participation. Then, bounding the all-device average's drift, the deviation M_{t+1} from a clean gradient step, the L-smooth descent on that average, and the 1/K sampling gap from K random devices, all in terms of the dissimilarity B, I get a per-round expected decrease rho ||∇f||^2 whenever the full rho expression is positive, with gamma B < 1 and B/sqrt(K) < 1 as the visible constraints that explain the trade-offs. Telescoping gives O(Delta/(rho epsilon)) rounds to an epsilon-stationary point; the convex case pins mu ≈ 6LB^2 and matches SGD's complexity; the variable-gamma version costs one extra line. In code, the whole thing is the existing federated harness with the local gradient changed to grad + mu*(w - w^t) and the sample-weighted server average left alone.
