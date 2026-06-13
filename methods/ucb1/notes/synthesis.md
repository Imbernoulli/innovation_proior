# Synthesis: UCB1 (Auer, Cesa-Bianchi, Fischer 2002) — the discovery path

## What the method is (identification from the task)
MLS-Bench baseline `ucb1` in task `optimization-online-bandit`. Authoritative definition:
edits/ucb1.edit.py implements `mu_hat_a + sqrt(2 log t / N_a)` (the SW-UCB fallback was a red
herring the edit itself disables, calling vanilla UCB1 the correct version). Task description
attributes it to "Auer, Cesa-Bianchi, and Fischer, 'Finite-time Analysis of the Multiarmed
Bandit Problem', Machine Learning 47, 2002" with index `mu_hat + sqrt(2 log t / n_a)`,
"paper-default exploration constant c=2 in the sqrt(c log t / n_a) term". This is UCB1.
Canonical code: SMPyBandits Policies/UCB.py and UCBalpha.py.

PRIMARY = Auer, Cesa-Bianchi, Fischer, "Finite-time Analysis of the Multiarmed Bandit
Problem," Machine Learning 47(2-3):235-256, 2002 (Kluwer). Preliminary version: Proc. 15th
ICML, pp. 100-108, Morgan Kaufmann, 1998. Pre-arXiv classic (no e-print). Read in full,
refs/acbf2002.{pdf,txt}.

## Pain point / research question
K-armed stochastic bandit. Arm i gives i.i.d. rewards X_{i,1}, X_{i,2}, ... with unknown mean
mu_i, support [0,1]. A policy A picks the next arm from past plays+rewards. T_i(n) = #pulls of
arm i in first n plays. Regret after n plays:
  mu* n - mu_? ... actually: R_n = mu* n - sum_j mu_j E[T_j(n)] = sum_{j: mu_j<mu*} Delta_j E[T_j(n)]
with mu* = max_i mu_i, Delta_i = mu* - mu_i.  (Eq. 5 of paper.)
So regret = sum over suboptimal arms of (gap x expected #times pulled). The ONLY thing to
control is E[T_j(n)] for suboptimal arms.

The pain: prior art (Lai-Robbins, Agrawal) achieved the optimal LOGARITHMIC regret but only
(a) ASYMPTOTICALLY (n -> infinity), and (b) Lai-Robbins via indices that depend on the WHOLE
reward sequence / are "generally hard" to compute, restricted to single-parameter reward
families. Goal: a policy that is (i) finite-time (a bound that holds for EVERY n, not just in
the limit), (ii) simple/efficient to compute, (iii) works for ALL reward distributions with
bounded support [0,1] (distribution-free, no parametric family).

## Background concepts (load-bearing, pre-method, all sourced)
- Exploration-exploitation dilemma; regret as the yardstick (Berry & Fristedt 1985 stats;
  Sutton & Barto 1998 RL).
- Lai & Robbins (1985): for single-parameter reward families, policies with
  E[T_j(n)] <= (1/D(p_j||p*) + o(1)) ln n  (Eq. 1), where D(p_j||p*) = integral p_j ln(p_j/p*)
  is KL divergence. And the matching asymptotic LOWER bound:
  E[T_j(n)] >= (ln n)/D(p_j||p*) for any allocation strategy (under mild assumptions).
  These work via an "upper confidence index" attached to each machine; computing the index is
  generally hard, relying on the entire reward sequence.  [Primary paywalled - grounded via
  ACBF §1 which states it exactly, plus explainers. FLAG in notes.]
- Agrawal (1995): a family of policies whose index can be expressed as a SIMPLE function of
  the total reward obtained so far from the machine -> much easier to compute than
  Lai-Robbins; regret keeps optimal log order (larger leading constant in some cases). UCB1's
  index is "derived from the index-based policy of Agrawal (1995)". [Grounded via ACBF §1.]
- Chernoff-Hoeffding bound (Fact 1 in the paper). For X_1..X_n in [0,1] with
  E[X_t | X_1..X_{t-1}] = mu and S_n = sum X_t:
    P{S_n >= n mu + a} <= e^{-2 a^2 / n}   and   P{S_n <= n mu - a} <= e^{-2 a^2 / n}.
  (Equivalently for the average X_bar = S_n/n: P{X_bar >= mu + eps} <= e^{-2 n eps^2}.)
  Bernstein (Fact 2) used for the eps-greedy proof only. Source: paper Fact 1/2 (cites
  Pollard 1984 appendix; classic Hoeffding 1963).

## The central object & difficulty
Want to bound E[T_j(n)] for a suboptimal arm. The difficulty: you must pull a suboptimal arm
ENOUGH to be statistically confident it is worse, but no more. Too little exploration -> you
might never discover the best arm (linear regret); too much -> you waste pulls on known-bad
arms (also extra regret). Need the SMALLEST amount of exploration that still rules out a bad
arm being best with overwhelming probability.

## The chain (theory -> the algorithm)
1. Pure greedy on empirical mean fails: a bad early sample can permanently bury the best arm
   (it never gets pulled again, so the estimate never corrects). Need to keep some optimism for
   under-sampled arms.
2. Optimism in the face of uncertainty: replace mu_j by an UPPER confidence bound on mu_j,
   = empirical mean + a radius that is the half-width of a one-sided confidence interval, and
   play the largest. (Explainer: Jeremy Kun. Paper §2: "the size ... of the one-sided
   confidence interval for the average reward within which the true expected reward falls with
   overwhelming probability".)
3. Size of the radius from Chernoff-Hoeffding. With n_j samples, P(mu_j > x_bar_j + a) <=
   e^{-2 n_j a^2}. Choose the failure prob = t^{-4} (t = total plays). Then 2 n_j a^2 = 4 ln t
   => a = sqrt(2 ln t / n_j). WHY t^{-4}: in the union bound the bad events are summed over the
   triple (t, s, s_i) ~ t^3 terms each O(t^{-?}); t^{-4} per Hoeffding event makes
   sum_t t^3 * t^{-4} = sum_t 1/t... actually the proof yields sum_t sum_s sum_{s_i} 2 t^{-4}
   <= 2 sum_t t * t * t^{-4}... see exact proof. Net: the two over-/under-estimate events each
   get prob t^{-4}, and the double sum collapses to 2 sum_t t^{-2} = pi^2/3.
4. UCB1 (Figure 1): Initialization - play each machine once. Loop - play j maximizing
     x_bar_j + sqrt(2 ln n / n_j),
   where x_bar_j = average reward from j, n_j = #pulls of j, n = total plays so far.
   (NB: the algorithm uses n=total plays; the proof writes c_{t,s}=sqrt(2 ln t / s) with t the
   round index. At decision time t and arm-sample-count s these coincide.)

## Theorem 1 (finite-time regret) and full proof
THEOREM 1. For K>1, UCB1 on arms with arbitrary [0,1] reward distributions has expected
regret after any n plays at most
  [ 8 sum_{i: mu_i<mu*} (ln n / Delta_i) ] + (1 + pi^2/3)(sum_{j=1}^K Delta_j).
Equivalently, per suboptimal arm (Eq. 2):
  E[T_j(n)] <= (8/Delta_j^2) ln n + 1 + pi^2/3.

PROOF (do it fully in reasoning.md).
Let c_{t,s} = sqrt(2 ln t / s). x_bar_{i,s} = average of first s samples of arm i. Star =
optimal arm. ell = arbitrary positive integer (will set ell = ceil(8 ln n / Delta_i^2)).

T_i(n) = 1 + sum_{t=K+1}^n 1{I_t = i}
       <= ell + sum_{t=K+1}^n 1{I_t = i, T_i(t-1) >= ell}
       <= ell + sum_{t=K+1}^n 1{ x_bar*_{T*(t-1)} + c_{t-1,T*(t-1)} <= x_bar_{i,T_i(t-1)} + c_{t-1,T_i(t-1)},  T_i(t-1) >= ell }
       <= ell + sum_{t=K+1}^n 1{ min_{0<s<t} x_bar*_s + c_{t-1,s} <= max_{ell<=s_i<t} x_bar_{i,s_i} + c_{t-1,s_i} }
       <= ell + sum_{t=1}^infty sum_{s=1}^{t-1} sum_{s_i=ell}^{t-1} 1{ x_bar*_s + c_{t,s} <= x_bar_{i,s_i} + c_{t,s_i} }.    (Eq. 6)

The event x_bar*_s + c_{t,s} <= x_bar_{i,s_i} + c_{t,s_i} implies at least one of:
  (7) x_bar*_s <= mu* - c_{t,s}                       [optimal arm under-estimated]
  (8) x_bar_{i,s_i} >= mu_i + c_{t,s_i}               [bad arm over-estimated]
  (9) mu* < mu_i + 2 c_{t,s_i}.                       [confidence interval too wide / not enough samples]
(If none of 7,8,9 held, then x_bar*_s + c > mu* >= mu_i + 2c > x_bar_i ... contradiction.)

Bound (7),(8) by Hoeffding (Fact 1): P(7) <= e^{-4 ln t} = t^{-4}, P(8) <= t^{-4}.
(9) is FALSE once s_i >= (8 ln n)/Delta_i^2: then
  mu* - mu_i - 2 c_{t,s_i} = mu* - mu_i - 2 sqrt(2 ln t / s_i) >= mu* - mu_i - Delta_i = 0
(using ln t <= ln n for t<=n and s_i >= 8 ln n/Delta_i^2 so 2 sqrt(2 ln t/s_i) <= Delta_i).
So set ell = ceil(8 ln n / Delta_i^2) and (9) never fires; only (7),(8) contribute:
  E[T_i(n)] <= ceil(8 ln n/Delta_i^2) + sum_{t=1}^infty sum_{s=1}^{t-1} sum_{s_i=ceil(...)}^{t-1} ( P(7)+P(8) )
            <= 8 ln n/Delta_i^2 + 1 + sum_{t=1}^infty sum_{s=1}^t sum_{s_i=1}^t 2 t^{-4}
            <= 8 ln n/Delta_i^2 + 1 + 2 sum_{t=1}^infty t^{-2}
            = 8 ln n/Delta_i^2 + 1 + pi^2/3.   (sum 1/t^2 = pi^2/6)
Multiply by Delta_i and sum over suboptimal arms (plus the +1+pi^2/3 over all arms) -> Theorem 1.

## Why 8 and not the optimal constant; relation to Lai-Robbins
8/Delta_j^2 is WORSE than Lai-Robbins' 1/D(p_j||p*). In fact (paper, §after Eq.2) one can show
D(p_j||p*) >= 2 Delta_j^2 with constant 2 best possible (Pinsker-type). So UCB1 has the right
O(ln n / Delta^2) ORDER but a loose constant. UCB2 (Figure 2: epochs, radius
a_{n,r} = sqrt((1+alpha) ln(e n/tau(r)) / (2 tau(r))), tau(r)=ceil((1+alpha)^r)) brings the
main constant arbitrarily close to 1/(2 Delta_j^2) as alpha->0 (Theorem 2). eps_n-GREEDY
(Figure 3: eps_n = min{1, cK/(d^2 n)}) also gives log regret if you KNOW a gap lower bound d
(Theorem 3). UCB1-NORMAL (Figure 4) handles unknown-mean-and-variance Gaussian arms via
index x_bar_j + sqrt(16 (q_j - n_j x_bar_j^2)/(n_j-1) * ln(n-1)/n_j) and the rule "play any arm
pulled < ceil(8 log n) times" (Theorem 4: regret <= 256 (log n) sum sigma_i^2/Delta_i + ...).
[All from the paper, read in full. These are the SAME paper's siblings - present them as the
family the construction naturally produces, not as later work.]

## Worst-case (minimax) corollary
Distribution-INDEPENDENT: choosing the worst gaps, sum log n / Delta gives O(sqrt(K n log n))
(the gaps that maximize the bound are ~ sqrt(K log n / n)). [Standard reading; explainer
states O(sqrt(KT log T)). Mention as the gap-free reading of Theorem 1.]

## Canonical implementation (SMPyBandits)
- BasePolicy: holds t (internal clock, incremented on each getReward), pulls[k] (N_k),
  rewards[k] (cumulative sum). startGame() zeros everything.
- IndexPolicy(BasePolicy): adds index[] array; choice() = argmax index (random tie-break);
  computeAllIndex() loops computeIndex(arm).
- UCB(IndexPolicy): computeIndex(arm) = rewards[arm]/pulls[arm] + sqrt(2 log(t)/pulls[arm]),
  returns +inf if pulls[arm]<1 (forces initial round-robin so each arm pulled once).
- UCBalpha(UCB): index = mean + sqrt(alpha log t / (2 N_k)). With alpha=4 this equals UCB1's
  sqrt(2 log t/N_k); ALPHA default set to 4 in the file (alpha=2 -> 1-subgaussian variant).
The task harness (custom_bandit.py BanditPolicy) wants __init__(K, context_dim), reset(),
select_arm(t, context)->int, update(arm, reward, context). For UCB1: initial round-robin
(if t < K return t), else argmax(mean + sqrt(2 log(t+1)/N)); update accumulates count and sum.
This is the artifact answer.md lands on (faithful to SMPyBandits + the paper).

## Design decisions -> why (table)
- Add an UPPER bound, not the mean: pure greedy can permanently bury the best arm on a bad
  early sample; optimism self-corrects (a wrong-high bound drops fast when the arm is pulled).
- Radius = sqrt(2 ln t / n_j): inverted Hoeffding tail at failure level t^{-4}. The 2 inside is
  the 1/2 in Hoeffding's exponent (e^{-2 n a^2}); the ln t numerator makes under-pulled arms
  regain optimism over time so none is dismissed forever; the 1/n_j shrinks the radius for
  well-sampled arms so good arms get exploited.
- Failure level t^{-4}: chosen so the union bound over the ~t^3 (t,s,s_i) triples leaves a
  CONVERGENT sum (2 sum t^{-2} = pi^2/3) - a constant additive cost, not growing in n. A milder
  level (e.g. t^{-2}) would not converge after the triple union bound.
- n = total plays (not n_j) in the log: it is the global "how confident must I be by now" knob;
  ln grows slowly so exploration cost is only logarithmic.
- Initialization play each arm once: every n_j >= 1 so the index/radius is defined (no 0/0).
- Constant 8 (= 8 ln n/Delta^2): forced by requiring 2 c_{t,s_i} <= Delta_i, i.e.
  2 sqrt(2 ln n/s_i) <= Delta_i <=> s_i >= 8 ln n/Delta_i^2. Not chosen for tightness; it is the
  smallest exploration count that makes event (9) impossible under THIS radius.

## Self-account?
No first-person discovery memoir for UCB1 located this run. The 1998 ICML preliminary version +
the 2002 journal version are the record; reconstruct the path from the primary + antecedents +
explainer as usual. (Flag in notes.) Lai-Robbins 1985 primary PDF paywalled (ScienceDirect) /
on aggregators only - its load-bearing content (Eq.1, the 1/KL constant, the "upper confidence
index" framing, change-of-measure lower bound) is grounded via the ACBF paper's own §1 and the
explainers; flagged.
