Let me start from what actually hurts. I have a robot with a known but strongly nonlinear model, and a cost over a short horizon that I want to minimize while acting in real time — re-deciding the controls every twenty milliseconds, executing one step, and re-optimizing. The system is control-affine with noise that enters through the actuators, dx = (f(x,t) + G(x,t) u) dt + B(x,t) dw, and the cost has a quadratic control term (1/2) u^T R u plus an arbitrary state cost q(x,t) and a terminal cost phi(x_T). The trouble is the kind of cost I actually want to write. On a quadrotor I want to say "pay an impulse the instant you touch an obstacle" — a discontinuous indicator, not a smooth barrier. On a race car I want a nonlinear tire model and a cost that rewards sliding through a corner. The fast trajectory optimizers everyone runs, the DDP/iLQG family, work by Taylor-expanding the dynamics to first or second order along a nominal trajectory and approximating the cost as a quadratic, then doing a backward Riccati sweep to get a local feedback law. That is elegant and converges fast near a good nominal, but it is built entirely on derivatives. A discontinuous crash cost has no derivative and no honest quadratic model; the best DDP can do is replace it with a smooth surrogate, a sum of exponential barriers around every nearby obstacle, which distorts the objective and scales with the obstacle count. And solving the problem exactly — the stochastic Hamilton-Jacobi-Bellman PDE for the value function V(x,t) — is hopeless above a few state dimensions, because discretizing V over the state space is the curse of dimensionality. I want something derivative-free, that swallows arbitrary nonlinear dynamics and discontinuous costs whole, and that is principled rather than a made-up score.

So let me look hard at the HJB itself before I give up on it. For this control-affine system with quadratic control cost, the stochastic HJB reads, schematically, -partial_t V = q + f^T V_x - (1/2) V_x^T G R^{-1} G^T V_x + (1/2) tr(B B^T V_xx), and the optimal control falls out as u* = -R^{-1} G^T V_x. The thing that kills it is that quadratic term V_x^T G R^{-1} G^T V_x — it is nonlinear in V, so I cannot solve the PDE by any linear method. But stare at that nonlinearity. It is exactly the gradient-squared term that shows up in physics when you have a diffusion with a potential, and there is a classical trick for exactly that shape: the logarithmic, Cole-Hopf-type change of variables. Suppose I write V = -lambda log(psi) for some positive constant lambda and a new function psi. Then V_x = -lambda psi_x / psi, and V_x^T (...) V_x picks up a 1/psi^2 and a psi_x psi_x^T; meanwhile the second-derivative term V_xx, when I differentiate V_x = -lambda psi_x/psi again, produces a -lambda psi_xx/psi term and a +lambda psi_x psi_x^T / psi^2 term. The second of those is also quadratic in psi_x over psi^2. So there is a real chance the two quadratic-in-(psi_x/psi) terms cancel — if I can match their coefficients. The coefficient on the HJB's nonlinear term is set by G R^{-1} G^T, and the coefficient on the diffusion term is set by B B^T. They cancel precisely when I demand a relationship between the noise and the control cost:

  B_c(x,t) B_c(x,t)^T = lambda G_c(x,t) R(x,t)^{-1} G_c(x,t)^T,

where the subscript c is the actuated block of the state — the noise only acts where the controls act. This is a real assumption with a cost: it ties the noise covariance to R^{-1} through a single scalar lambda, and it forces B to have the same rank as R, meaning noise can only hit directly actuated states. Many robots fit (the noise is on the commanded input, passed through a low-level controller); some don't. I accept it, because of what it buys. With this assumption the quadratic terms cancel and the HJB for V becomes a *linear* PDE for psi — the backward Chapman-Kolmogorov equation, partial_t psi = (q/lambda) psi - f^T psi_x - (1/2) tr(Sigma psi_xx), with Sigma = B_c B_c^T. I traded a nonlinear PDE in V for a linear PDE in psi.

A linear backward PDE with a terminal condition is a thing I know how to evaluate without ever discretizing the state space, by the Feynman-Kac lemma. Feynman-Kac says the solution of exactly this kind of backward parabolic PDE equals an expectation over trajectories of the corresponding forward diffusion. The terminal condition is psi(x_T,T) = exp(-phi(x_T)/lambda), the exponentiated terminal cost, and Feynman-Kac gives

  psi(x_0, t_0) = E_P[ exp( -(1/lambda) integral q dt ) psi(x_T,T) ] = E_P[ exp( -(1/lambda) S(tau) ) ],

with S(tau) = phi(x_T) + integral q dt the total state cost of a trajectory, and — this is the crucial subscript — the expectation is taken with respect to P, the *uncontrolled* dynamics, the system with u set to zero. The value function has become an integral over all trajectories of the passive system, weighted by the exponential of their negative cost. That is the path integral, and it is exactly the free-energy / partition-function object from statistical mechanics with lambda in the role of temperature: low-cost trajectories get large exp(-S/lambda) weight, high-cost ones get suppressed, and lambda controls how sharply. To get the control rather than just psi, differentiate with respect to the initial state; after a lengthy but mechanical computation the optimal control comes out as a ratio of trajectory expectations,

  u* dt = M · E_P[ exp(-S/lambda) B dw ] / E_P[ exp(-S/lambda) ],

with M = R^{-1} G_c^T (G_c R^{-1} G_c^T)^{-1}, reducing to G_c^{-1} when the actuated dynamics are square. The whole structure of classical control has inverted: no backward sweep, no value-function PDE — the optimal control is an *expectation* I can approximate by forward simulation. Turn the machine on, let trajectories happen, average exp(-S/lambda) times the realized noise.

Now I discretize so I can actually sample on a computer. One step is x_{t+1} = x_t + (f + G u) Delta t + B epsilon sqrt(Delta t) with epsilon standard Gaussian; for the uncontrolled system drop the G u term. The state cost discretizes to S(tau) = phi(x_T) + sum_i q Delta t. Each one-step transition is Gaussian, so the probability of a whole discrete trajectory tau = (x_0,...,x_N), by the Markov property, is a product of Gaussians, P(tau) = Z(tau)^{-1} exp( -(Delta t/2) sum_i (z_i - mu_i)^T Sigma_i^{-1} (z_i - mu_i) ), where z_i = Delta x_i^{(c)}/Delta t - f^{(c)} is the realized rate after subtracting passive drift, mu_i = G_c u_i is the controlled rate shift, and Sigma_i = B_c B_c^T is the diffusion covariance in that rate-space quadratic (equivalently, the state-increment covariance is Sigma_i Delta t). Good — I have an explicit trajectory density and a Monte-Carlo recipe: sample K trajectories from the uncontrolled dynamics, evaluate exp(-S/lambda), form the weighted average.

And here is where it falls apart in practice. The expectation is under P, the *uncontrolled* dynamics. Sampling from P means switching the system on and waiting for its natural noise to produce something useful. For a well-engineered machine the natural noise is tiny, so essentially every trajectory I draw from P is high-cost, every weight exp(-S/lambda) is numerically zero, and the estimator is either dominated by one freak lucky sample or is the indeterminate 0/0. The variance of the Monte-Carlo estimate is enormous because I am evaluating an expectation under a distribution that almost never visits the region that determines its value. Wall.

This is a textbook importance-sampling situation: the expectation is under a bad distribution P, so draw from a better distribution q and reweight by the likelihood ratio. Take the integral form of the control expectation and multiply numerator and denominator by q(tau)/q(tau) = 1; the integrals become expectations under q, each integrand multiplied by p(tau)/q(tau):

  u* ∝ E_q[ exp(-S/lambda) (...) p/q ] / E_q[ exp(-S/lambda) p/q ].

Now I get to *choose* q — the distribution I actually sample from. What do I want to choose about it? Prior path-integral work answered: change the mean. Shift the controls I sample around toward a promising trajectory, so the rollouts cluster where the cost is low instead of wherever the passive noise wanders. That change of mean is exactly Girsanov's theorem on the drift of a diffusion, and it gives an iterative scheme — sample around the current control guess, reweight, update the guess, repeat — which is what the generalized path-integral reinforcement-learning methods (Theodorou, Buchli & Schaal's PI-squared, the reward-weighted averaging of control variations) do. So mean-shifting is solved and principled and derivative-free; it already clears most of my requirements.

But let me look at what mean-shifting alone leaves on the table, because there is a stubborn failure I keep seeing. Picture the cart-pole that has to swing up. I start with a control guess that does not swing it up, I shift the mean toward whatever the weighted rollouts suggest — but the rollouts are all drawn with the *natural* system variance, which for this well-behaved actuator is small, so every sampled trajectory is a tiny wiggle around the current one. They are all bunched tightly together; none of them dares to take the big, committed swing that would actually get the pole over the top. The weighted average of a cluster of near-identical mediocre trajectories is another mediocre trajectory. The mean creeps but never escapes, and the controller simply never swings the pole up — it stalls at a high cost. The thing locking me in is that across all of this prior work the *variance* of the sampling distribution was held fixed at the system's natural noise. I can move where I sample but not how widely. The patches people use are telling: either inject extra fake noise into the system and optimize that noisier system — but then I have optimized the wrong dynamics — or just sample from whatever distribution happens to work and quietly ignore that it no longer matches the measure my path-integral derivation assumed. Both betray the principled estimator. So the real question sharpens: can I change the variance of the sampling distribution too, *and keep the importance-sampling identity exact* — get the right likelihood ratio for a q whose covariance differs from P's?

That is the term I have to compute. I need p(tau)/q(tau) when q has both a shifted mean (controls u, so the rate shift is mu_i = G_c u_i, while the state increment shifts by mu_i Delta t) and a *scaled* covariance. Let me write the scaled diffusion as B_E = (0; A_t B_c), i.e. multiply the actuated noise block by some matrix A_t at each step, and call the covariance that appears in q's Gaussian quadratic Lambda_i. With the convention I am using here, Lambda_i = A_{t_i}^T Sigma_i A_{t_i}; the only thing that matters in the proof is that Lambda_i is q's covariance and Sigma_i is p's covariance. Both densities are products of Gaussians of the form Z^{-1} exp(-(Delta t/2) sum (z - m)^T C^{-1} (z - m)). Dividing them, the determinant prefactors give a clean product prod_i |A_{t_i}|, the change-of-variables Jacobian of scaling the noise, and the exponent is the difference of two quadratics in z_i:

  zeta_i = z_i^T Sigma_i^{-1} z_i - (z_i - mu_i)^T (A_t^T Sigma_i A_t)^{-1} (z_i - mu_i).

A difference of two quadratics in the same variable is itself a single quadratic, so I complete the square. Let me define Gamma_i by Gamma_i^{-1} = Sigma_i^{-1} - Lambda_i^{-1} with Lambda_i = A_t^T Sigma_i A_t — the inverse-covariance difference of the two distributions. Completing the square on zeta_i in z_i, the quadratic part is z_i^T Gamma_i^{-1} z_i (the difference of the two inverse covariances, by construction), and there are linear and constant pieces. Carefully: expand (z_i - mu_i)^T Lambda_i^{-1} (z_i - mu_i) = z_i^T Lambda_i^{-1} z_i - 2 mu_i^T Lambda_i^{-1} z_i + mu_i^T Lambda_i^{-1} mu_i, so

  zeta_i = z_i^T (Sigma_i^{-1} - Lambda_i^{-1}) z_i + 2 mu_i^T Lambda_i^{-1} z_i - mu_i^T Lambda_i^{-1} mu_i
         = z_i^T Gamma_i^{-1} z_i + 2 mu_i^T Lambda_i^{-1} z_i - mu_i^T Lambda_i^{-1} mu_i.

Now I want this in terms of the *deviation from the controlled mean*, tilde z_i = z_i - mu_i, because that is what I actually sample (z_i is mu_i plus a noise draw). Substitute z_i = tilde z_i + mu_i. The quadratic term gives tilde z_i^T Gamma_i^{-1} tilde z_i + 2 mu_i^T Gamma_i^{-1} tilde z_i + mu_i^T Gamma_i^{-1} mu_i. The cross term 2 mu_i^T Lambda_i^{-1} z_i becomes 2 mu_i^T Lambda_i^{-1} tilde z_i + 2 mu_i^T Lambda_i^{-1} mu_i. The constant -mu_i^T Lambda_i^{-1} mu_i stays. Collect:

  zeta_i = tilde z_i^T Gamma_i^{-1} tilde z_i + 2 mu_i^T Gamma_i^{-1} tilde z_i + mu_i^T Gamma_i^{-1} mu_i + 2 mu_i^T Lambda_i^{-1} tilde z_i + mu_i^T Lambda_i^{-1} mu_i.

This still has Gamma in it, and I would rather express it through Sigma and Lambda, the two distributions' own covariances, since Gamma is just their difference. Use Gamma_i^{-1} = Sigma_i^{-1} - Lambda_i^{-1} to split every Gamma_i^{-1}: the linear term 2 mu_i^T Gamma_i^{-1} tilde z_i becomes 2 mu_i^T Sigma_i^{-1} tilde z_i - 2 mu_i^T Lambda_i^{-1} tilde z_i, and the cross-constant mu_i^T Gamma_i^{-1} mu_i becomes mu_i^T Sigma_i^{-1} mu_i - mu_i^T Lambda_i^{-1} mu_i. Now watch the bookkeeping. The -2 mu_i^T Lambda_i^{-1} tilde z_i exactly cancels the +2 mu_i^T Lambda_i^{-1} tilde z_i that came from the cross term. The -mu_i^T Lambda_i^{-1} mu_i exactly cancels the +mu_i^T Lambda_i^{-1} mu_i constant. Everything in Lambda evaporates. What survives is

  zeta_i = tilde z_i^T Gamma_i^{-1} tilde z_i + 2 mu_i^T Sigma_i^{-1} tilde z_i + mu_i^T Sigma_i^{-1} mu_i,

that is, writing tilde z_i = z_i - mu_i back out,

  Q_i = (z_i - mu_i)^T Gamma_i^{-1} (z_i - mu_i) + 2 mu_i^T Sigma_i^{-1} (z_i - mu_i) + mu_i^T Sigma_i^{-1} mu_i.

There is the generalized likelihood ratio: p/q = (prod_i |A_{t_i}|) exp(-(Delta t/2) sum_i Q_i). Let me read it, because the structure tells me what I gained. The last two terms, 2 mu_i^T Sigma_i^{-1}(z_i - mu_i) + mu_i^T Sigma_i^{-1} mu_i, are precisely the terms Girsanov's theorem produces for a pure change of mean — they are the price of shifting the drift. The first term, (z_i - mu_i)^T Gamma_i^{-1} (z_i - mu_i), is entirely new, and it is exactly the term that carries the variance change, because Gamma_i^{-1} = Sigma_i^{-1} - Lambda_i^{-1} is the *difference of inverse covariances* — it is zero exactly when A_t = I, when q's variance equals p's. So a pure-mean-change derivation is the special case A_t = I of this, and the new quadratic is a penalty that grows when q explores with larger variance than the natural noise — a penalty on over-aggressive exploration. I can now change the variance and still have the exact reweighting. The wall is down.

Now I fold this into the algorithm rather than carrying it as a separate factor. The likelihood ratio appears in both numerator and denominator of the control expectation, so any factor that does not depend on the trajectory — in particular the numerically nasty determinant prod_i |A_{t_i}| — cancels between them. Only the sum of Q_i survives, and a Q_i is a function of the realized trajectory, so I can add it into the running cost. Since Sigma = lambda H with H = G R^{-1} G^T, the same Q_i can be written as (1/lambda)[(z - mu)^T tildeGamma^{-1}(z - mu) + 2 mu^T H^{-1}(z - mu) + mu^T H^{-1} mu]. That means the exponent -S/lambda - (Delta t/2)sum_i Q_i is the exponent of an augmented running cost with

  tilde q = q + (1/2)(z - mu)^T tildeGamma^{-1}(z - mu) + mu^T H^{-1}(z - mu) + (1/2)mu^T H^{-1}mu.

With tilde S(tau) = phi + sum_i tilde q_i Delta t, the control update becomes the same exponentiated-cost weighted form but now sampling from q,

  u_t* = M G u_t + M · E_q[ exp(-tilde S/lambda) B (epsilon/sqrt(Delta t)) ] / E_q[ exp(-tilde S/lambda) ].

I should specialize to the case I will actually run, because the general form is heavier than I need. Take G_c square and invertible, so M reduces to G_c^{-1}; and take the variance scaling to be a single scalar, A = sqrt(nu) I, so I just multiply the natural noise by sqrt(nu) — one exploration-variance knob nu. Then rewrite the noise as a random perturbation of the control: define delta u = (1/sqrt(rho)) epsilon / sqrt(Delta t), so that B epsilon/sqrt(Delta t) = G delta u, the noise is literally a random kick to the commanded control. The G_c^{-1} and G cancel, and the update collapses to something I can read at a glance:

  u_t* = u_t + E_q[ exp(-tilde S/lambda) delta u ] / E_q[ exp(-tilde S/lambda) ],

which as a Monte-Carlo sum over K rollouts is

  u_{t_i}* ≈ u_{t_i} + ( sum_k exp(-tilde S(tau_{i,k})/lambda) delta u_{i,k} ) / ( sum_k exp(-tilde S(tau_{i,k})/lambda) ).

This is a reward-weighted average of random control variations: each rollout k perturbs the control by delta u_{i,k}, and I move the control toward the perturbations that produced low cost, with the exponential giving them the weight. And in this scalar-A special case the augmented cost tilde q simplifies beautifully. Since H = G R^{-1} G^T, the square invertible case gives H^{-1} = G^{-T} R G^{-1}; with z - mu = G delta u and tildeGamma^{-1} = (1 - nu^{-1}) H^{-1}, tilde q turns into q + (1 - nu^{-1})/2 delta u^T R delta u + u^T R delta u + (1/2) u^T R u. In words: introducing the likelihood ratio just re-injects the original quadratic *control* cost into the sampling cost, which until now only saw state-dependent terms. That is reassuring — the bookkeeping I did is not adding something foreign, it is restoring the control cost the path-integral expectation had absorbed.

Let me confirm I understand the weighting itself by deriving it the other way, from the free-energy / KL side, because that is where the intuition for the temperature and the numerics lives. Define the free energy F = log E_P[exp(-S(V)/lambda)] over control sequences V drawn from a base distribution P. By Jensen's inequality on the concave log, -lambda F lower-bounds the control objective: -lambda F <= E_{Q}[S(V)] + lambda D_KL(Q || P) for any sampling distribution Q. The right side is "expected cost plus a KL control penalty pulling Q toward the base P," which is the control objective. When is the bound tight? When Q equals the optimal distribution q*(V) = (1/eta) exp(-S(V)/lambda) p(V), the base measure tilted by the exponentiated cost — substitute it and the KL term collapses the inequality to equality. So the optimal sampling distribution is the Gibbs distribution over trajectories: weight each by exp(-cost/lambda). Now I cannot sample q* directly (it is defined through the unknown normalizer), but I can importance-sample it from my current Gaussian Q_U: the optimal control is u_t = E_{Q_U}[ w(V) v_t ] with weight w(V) ∝ exp(-(1/lambda) S(V)) (plus a term that, with the uncontrolled base, encourages samples back toward the base). That is the same exponentiated-cost weight I just derived from the likelihood ratio, now with a crisp reading: I am pushing my Gaussian toward the Gibbs-optimal trajectory distribution by reweighting its own samples.

This reframes lambda. It is a temperature controlling how peaked the optimal distribution is. As lambda -> 0 the weight exp(-S/lambda) puts essentially all its mass on the single lowest-cost trajectory — greedy, winner-take-all. As lambda -> infinity every trajectory gets equal weight — the update is a plain unweighted average that ignores cost entirely. In between, lambda trades exploitation against robustness to Monte-Carlo noise: too small and a single lucky sample dictates the answer, too large and I average good and bad together. And the free-energy view immediately flags a numerical landmine: exp(-S/lambda) over raw costs either underflows to zero (costs too high) or overflows (costs unbounded below). The fix is to shift the costs so the *best* sampled trajectory has shifted cost zero — multiply numerator and denominator by exp(rho/lambda) where rho is the minimum sampled cost. This is multiplying by 1, so it changes nothing about the optimum, but it guarantees the best sample has weight exp(0) = 1 and every other exponent is non-positive. So the weight I will actually compute is exp((rho - S)/lambda) = exp(-(S - min S)/lambda): exponentiate the *cost gap from the best sample*, not the raw cost.

Now I have to put this in the model-predictive loop, and translate it to the setting I am actually planning in: a fixed, pre-trained latent dynamics model that forward-rolls a batch of action sequences and gives me a scalar cost per sequence (distance of the predicted final latent to a goal). I do not have an analytic noise model here, no B, no R — I have a black-box rollout and a cost. But the *update* I derived survives that abstraction completely, because in the end it only ever uses (rollout costs) and (the action perturbations). The natural way to carry the construction over is to maintain, per planning step, a Gaussian over the action sequence — a mean and a standard deviation per timestep — and on each iteration sample a population of action sequences from it, roll them all out through the model, score them, and refit the Gaussian. The mean refit is exactly the reward-weighted average I derived: new mean = sum_k w_k * action_k with w_k = exp((min cost - cost_k)/lambda) normalized to sum to one. That is the path-integral control update, with the Gaussian mean playing the role of the nominal control u and the samples playing u + delta u.

And here is the piece that mean-only methods could not do, now landing naturally: I also refit the *standard deviation*. The variance-change capability I fought for becomes, in this sampling form, simply estimating the spread of the good samples and using it as next iteration's exploration width — new std = sqrt( sum_k w_k * (action_k - new mean)^2 ). When the weighted samples are tightly clustered (the cost surface is sharp and I am near a good solution) the std shrinks and I exploit; when they are spread out (still searching) it stays wide and I keep exploring. The exploration variance is now adapted online from the rollouts themselves, instead of being frozen at a fixed natural noise — which is exactly the failure I diagnosed on the cart-pole. I initialize the std deliberately large (max_std around 2) so the first iteration explores aggressively and does not get trapped in a local basin of the cost — the same pathology, addressed at initialization.

One practical truncation: rather than soft-weight all N samples, first keep the top-k lowest-cost "elites" and apply the exponential weighting *within that elite set*. Why bother, when the soft weight already down-weights bad samples? Because with a finite population the far-out high-cost samples still carry tiny but nonzero weight, and they contribute noise to the mean-and-variance fit, especially to the variance estimate which is sensitive to outliers; restricting to the elites cleans the fit while the soft exp-weighting preserves the graded "how much better" information among the survivors. This is the precise distinction from the cross-entropy method, which I should be honest about because the two look superficially identical — sample a Gaussian, score, refit. CEM keeps the top-k elites and refits to their *unweighted* mean and standard deviation: every elite counts equally, the rest are thrown away, and all information about how much better the best elite is than the worst is discarded at the hard threshold. The update here keeps the elites but weights them by exp((min cost - cost)/lambda), so a markedly better elite pulls the mean and tightens the variance more than a marginal one. CEM is the hard-selection cousin: the same elite truncation, but uniform weights inside the elite set. The soft version is the one that keeps the path-integral / Gibbs shape and uses the cost magnitudes the objective actually reports. In code I store the inverse temperature as a cost-gap multiplier: score = exp( temperature * (min_cost - cost) ), so temperature = 1/lambda; a small temperature (here 0.005) is a large lambda — a relatively soft, broad weighting that averages over the elites rather than collapsing onto a single one, which is the robust choice when the model rollouts are themselves noisy.

The model-predictive wrapper is the standard receding-horizon scheme: each control step, run these sampling iterations to optimize the action sequence over the horizon, execute the first action from the selected plan, then re-observe and re-plan. Warm-starting from the previous step's un-executed tail is the textbook MPC trick; in this fixed-horizon planning code each call re-optimizes from a zero mean and then samples one final elite according to the path-integral weights, which is the simplification the latent-planning setting uses.

Let me write the planner, filling the one empty slot — the rule that turns a scored population of action sequences into the next sampling distribution. The two state quantities per planning step are the per-timestep mean and std of the action Gaussian; the per-iteration work is sample, roll out and score, pick elites, exp-weight by the cost gap from the best, refit mean and std, repeat; then commit an action by sampling an elite according to the weights.

```python
import numpy as np
import torch
from einops import rearrange


class MPPIPlanner(Planner):
    """Model Predictive Path Integral planner over a learned rollout model.

    Maintains a per-timestep Gaussian over action sequences; each iteration
    samples a population, scores it by the model rollout cost, and refits the
    Gaussian to the cost-exponentiated (Gibbs) weighted samples -- the
    path-integral control update, refining BOTH mean and std."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=500, n_iters=15,
                 max_std=2.0, num_elites=64, temperature=0.005, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.max_std = max_std                 # large initial exploration width
        self.num_elites = num_elites           # top-k kept before soft weighting
        self.temperature = temperature         # = 1/lambda; small => soft, broad weighting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # action-sequence Gaussian: per-timestep mean and std
        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)

        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            # sample a population of action sequences from the current Gaussian
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)             # T B A

            # score every sequence by rolling it through the (learned) model
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init).unsqueeze(1)         # B 1
            losses.append(cost.min().item())

            # keep the top-k elites (clean the weighted fit of far-out samples)
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # path-integral / Gibbs weights: exp over the cost gap from the best
            # sample (cost-shift = numerical conditioning, multiplies by 1).
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))      # B'
            score /= score.sum(0)

            # refit the Gaussian: reward-weighted average (mean) ...
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9)                                                # T A
            # ... and the weighted spread of the good samples (the variance change)
            std = torch.sqrt(torch.sum(
                score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                dim=1) / (score.sum(0) + 1e-9))

        # commit a plan: draw one final elite according to its path-integral weight
        score_np = score.cpu().numpy()
        selected = elite_actions[:, np.random.choice(np.arange(score_np.shape[0]), p=score_np)]

        return PlanningResult(
            actions=selected,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```

Let me trace the causal chain that got me here. I needed a real-time controller for nonlinear dynamics and discontinuous, non-convex costs, which ruled out HJB (curse of dimensionality) and derivative-based DDP (no derivatives of a crash impulse, no faithful quadratic cost model). Assuming the noise enters only through the actuated channels with a covariance tied to the control cost by a scalar lambda, the logarithmic change of variables V = -lambda log psi linearized the stochastic HJB, and Feynman-Kac turned the value function into a path integral — an expectation of exp(-cost/lambda) over forward-sampled trajectories, evaluated by simulation rather than a backward PDE. That expectation was under the useless uncontrolled dynamics, so I imported it via importance sampling, which let me reshape the sampling distribution — and the question that prior path-integral work left open was whether I could reshape its *variance*, not just its mean, since the fixed natural variance kept the rollouts too timid to ever swing the cart-pole up. Computing the likelihood ratio between two discrete diffusion processes that differ in both mean and covariance, by completing the square on the difference of their Gaussian exponents, produced a single quadratic whose Girsanov mean-change terms were the old result and whose extra term — the difference of inverse covariances penalizing over-aggressive exploration — was exactly the variance-change capability I wanted, exact rather than a hack. Folding that ratio into the running cost left a clean exponentiated-cost weighted average of random control perturbations, which the free-energy / KL view independently confirmed is importance sampling toward the Gibbs-optimal trajectory distribution, with lambda the temperature and a cost-shift by the best sample for numerical safety. Carried into the latent-planning setting, this becomes a per-step Gaussian over action sequences whose mean is refit as the reward-weighted average of sampled rollouts and whose std is refit as the weighted spread of the good ones — the online variance adaptation that mean-only path-integral control and hard-threshold CEM both lacked.
