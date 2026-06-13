# Synthesis — KL-UCB (Garivier & Cappé, COLT 2011; arXiv:1102.2490)

## Identification from MLS-Bench
- Task: `optimization-online-bandit`. Edit `kl_ucb.edit.py` defines the index
  `U_a(t) = sup{ q in [0,1] : N_a · d(mu_hat_a, q) <= c·log(t) }`, with `c=1`,
  `d` the Bernoulli KL, computed by bisection. Docstrings cite Garivier & Cappé COLT 2011
  (arXiv:1102.2490) and Cappé/Garivier/Maillard/Munos/Stoltz, Annals of Statistics 41 (2013).
- The published method title: "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond."
- FIXED helpers in custom_bandit.py: `kl_bernoulli` (rel_entr) and `kl_ucb_bound`
  (max q in [mu_hat,1] s.t. n·KL(mu_hat,q) <= c·log(t), via scipy brentq). Matches the paper.

## The pain point / problem
- Stochastic K-armed bandit; rewards bounded in [0,1] (or rescaled). Minimize regret
  R_n = sum over suboptimal arms of Delta_a · E[N_a(n)], Delta_a = mu* - mu_a.
- The frontier: get an *index* policy (per-arm UCB, no horizon tuning, no problem tuning)
  whose per-suboptimal-arm draw count matches the Lai-Robbins lower bound for Bernoulli,
  AND keeps a *distribution-free* guarantee for all [0,1]-bounded rewards.

## Lower bound (the target to hit) — Lai & Robbins 1985, generalized Burnetas-Katehakis 1997
- For any uniformly-good (consistent) policy:
  liminf_n E[N_a(n)]/log n >= 1 / Kinf(nu_a, mu*),  Kinf = inf{ KL(nu_a, nu') : E[nu'] > mu* }.
- Bernoulli case: Kinf = d(mu_a, mu*) = Bernoulli KL. So the target constant per arm is
  1/d(mu_a, mu*). Change-of-measure / "most confusing instance" argument.
- d(p,q) = p log(p/q) + (1-p) log((1-p)/(1-q)), convention 0log0=0, x log x/0 = +inf.

## Baselines (ancestors) and their gaps
- UCB1 (Auer, Cesa-Bianchi, Fischer 2002): index mu_hat_a + sqrt(2 log t / N_a) (Hoeffding /
  subgaussian). Regret <= sum 8 log n / Delta_a + C. Distribution-free, no tuning, but the
  Hoeffding bonus uses only the [0,1] range (variance proxy 1/4); constant is 8, far from the
  1/2 optimal, and far from the KL lower bound. Symmetric, range-only confidence width that
  does NOT tighten near 0 or 1.
- UCB2 (Auer et al. 2002): tighter constant (1+eps)/2, but needs an alpha tuned to horizon.
- UCB-Tuned (Auer et al. 2002): plugs an empirical variance estimate into the bonus; great
  empirically but NO theoretical guarantee, and observed to be "risky" (heavy upper tail).
- UCB-V (Audibert, Munos, Szepesvári 2009): empirical Bernstein bonus with a non-asymptotic
  correction 3 log t / N_a. The correction term does not vanish for suboptimal arms (N_a ~
  log t) on moderate horizons -> disappointing finite-time.
- MOSS (Audibert & Bubeck 2010): distribution-free minimax-optimal rate, log(t/(K N_a)) style
  exploration; still range-based, not Lai-Robbins-optimal.
- DMED (Honda & Takemura 2010): large-deviations / arm-elimination; first-order optimal for
  bounded support, uses d(mu_hat_a, max_b mu_hat_b) < log t to keep a "to-play" list. But it
  is an *elimination* policy (compares to the empirical best, not to that arm's own UCB), and
  empirically index policies beat their elimination variants; also requires the rate function.

## The central move (what reasoning.md must discover, not context.md)
- Replace the additive Hoeffding bonus with a KL ball: the UCB is the largest mean q whose
  Bernoulli-KL "distance" from mu_hat costs at most (log t + c log log t)/N_a in deviation
  budget. I.e. U_a = max{ q > mu_hat_a : N_a · d(mu_hat_a, q) <= log t + c log log t }.
- Why d = Bernoulli KL even for arbitrary [0,1] rewards: Lemma (bounded->Bernoulli):
  for X in [0,1], mu=E[X], all lambda: E[exp(lambda X)] <= 1 - mu + mu·exp(lambda)
  = MGF of Bernoulli(mu). Proof: f(x)=exp(lambda x) - x(exp(lambda)-1) - 1 is convex,
  f(0)=f(1)=0, so f<=0 on [0,1]; take E. => Bernoulli is the *least concentrated* [0,1]
  variable with given mean, so its Chernoff rate d(.,.) upper-bounds deviations of any
  [0,1] variable. This is exactly why the Bernoulli-KL index is *distribution-free*.

## Deviation core (self-normalized, peeling) — Theorem 2/3 of the paper
- Self-normalized supermartingale: phi_mu(lambda) = log E[exp(lambda X1)] <= log(1-mu+mu e^lambda).
  W_t^lambda = exp(lambda S(t) - N(t) phi_mu(lambda)) is a supermartingale w.r.t. F_t (uses
  predictable/previsible epsilon_t in {0,1}, the "was arm pulled at s" indicator). E[W_0]=1.
- Peeling trick (Massart): N(n) ranges over {1..n}; slice into geometric blocks t_k=floor((1+eta)^k),
  eta = 1/(delta-1), D = ceil(log n / log(1+eta)) slices. On each slice Markov gives
  exp(-delta/(1+eta)); union over D slices and log(1+1/(delta-1)) >= 1/delta gives
  P(u(n) < mu) <= e·ceil(delta log n)·exp(-delta).
- With delta = log t + c log log t, c=3: P(mu1 > u1(t)) <= e ceil(log t^2 + 3 log t log log t)/(t log t^3),
  summable -> O(log log n). This is the c>=3 requirement.

## Regret proof skeleton (Section 6 of the paper)
- WLOG a*=1. Decompose E[N_n(a)] <= sum_t P(mu1 > u1(t)) + E[ sum_s 1{ s d+(mu_hat_{a,s}, mu1)
  < log n + 3 log log n } ], d+(x,y) = d(x,y) 1{x<y}. (Lemma majN reduces the on-policy count
  to a per-sample-count count.)
- First sum: O(log log n) from the deviation bound above.
- Second sum: split at K_n = floor( (1+eps)/d+(mu_a,mu1) (log n + 3 log log n) ). For s > K_n,
  d+(mu_hat_{a,s}, mu1) < d(mu_a,mu1)/(1+eps) implies mu_hat_{a,s} > r(eps) with
  d(r(eps),mu1)=d(mu_a,mu1)/(1+eps); Chernoff P(mu_hat_{a,s} > r) <= exp(-s d(r(eps),mu_a)),
  geometric sum -> C2(eps)/n^beta(eps). r(eps)=mu_a+O(eps), C2=O(eps^-2), beta=O(eps^2).
- => E[N_n(a)] <= (log n)/d(mu_a,mu*) (1+eps) + C1 log log n + C2(eps)/n^beta(eps).
- => limsup E[R_n]/log n <= sum_{a:mu_a<mu*} (mu*-mu_a)/d(mu_a,mu*).  Matches Lai-Robbins (Bernoulli).
- By-product (Prop): the same proof with quadratic divergence 2(p-q)^2 (Pinsker) gives a
  correctly tuned UCB the optimal 1/2 constant: E[N_n(a)] <= log n/(2 Delta_a^2)(1+eps)+...

## Exponential-family extension (Section 4)
- Canonical 1-param exp family p_theta(x) = exp(x theta - b(theta) + c(x)); mu(theta)=b'(theta),
  b''=Var>0 so mu is one-to-one. Replace d by d(x, mu(theta)) = sup_lambda { lambda x - phi(lambda) }
  (Cramér / Legendre transform of cumulant gen function). Lemma: this rate function equals
  KL(p_beta, p_theta) = mu(beta)(beta-theta) - b(beta) + b(theta) with x=mu(beta). All proofs go
  through unchanged (they only used the MGF bound), so the same regret bound and Lai-Robbins
  optimality hold for that family. Examples: Exponential d(x,y)=x/y-1-log(x/y); Poisson
  d(x,y)=y-x+x log(x/y); Gaussian fixed var -> recovers UCB-style sqrt bonus.
- Practical: may use an *upper bound* on the true d for simplicity (small perf loss).

## Design decisions -> why
- Index = sup of a KL ball, not additive bonus: makes the confidence width *asymmetric* and
  *self-tightening* near 0/1 (where d is steep), so it adapts to each arm's variance for free,
  unlike Hoeffding's fixed sqrt width. This is what closes the constant from 8 to Lai-Robbins.
- Exploration function log t + c log log t, c=3 for the theorem; c=0 recommended in practice
  (the (1+eps) log t alternative needs t > 2e51 to dominate). SMPyBandits default c=1
  (the index uses c·log t / N). The task scaffold uses c=1, log(t). Newton/bisection inverts
  d(mu_hat, .) which is strictly convex increasing on [mu_hat,1]; SMPyBandits uses bisection
  with a Gaussian/Pinsker seed for the upper end, the task uses scipy brentq.
- Why "max q > mu_hat": d(mu_hat, .) is U-shaped with min 0 at q=mu_hat; the relevant branch
  for an *upper* bound is the increasing one to the right, so we take the right root.
- KL-UCB+ heuristic: replace log t by log(t/N_a) (a la MOSS) — better mean, riskier tail.
  (This is the SMPyBandits klUCBPlus; Cappé et al. 2013.)

## Canonical implementation (grounding for the code)
- SMPyBandits Policies/klUCB.py: computeIndex(arm) = klucbBern(rewards[a]/pulls[a],
  c·log(t)/pulls[a], tolerance); +inf if never pulled. c=1 default.
- kullback.py: klucb(x, d, kl, upperbound, precision, lowerbound) = bisection on q for
  sup{q : kl(x,q) <= d}. klucbBern seeds upperbound = min(1, klucbGauss(x,d)) = min(1, x+sqrt(d/2))
  (Pinsker inversion) then bisects with klBern. klBern truncates to [eps,1-eps], eps=1e-15.
- The MLS task: kl_ucb_bound(mu_hat,n,t,c=1) = max q s.t. n·KL(mu_hat,q) <= c log t via brentq.

## Self-account note
- No first-person "how I discovered KL-UCB" memoir found for Garivier/Cappé. The paper itself
  states the lineage ("elaborates on ideas in Sections 3 and 4 of Lai-Robbins 1985"; advocated
  by Filippi 2010; concurrent Maillard-Munos-Stoltz 2011). Lattimore-Szepesvári Ch. 10 is the
  canonical re-derivation used as the explainer. Flagged: no author self-account; reconstruct
  from primary + ancestors + book.
