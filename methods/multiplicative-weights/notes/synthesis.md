# Legacy Synthesis — Multiplicative Weights Update

This orientation note is superseded by `notes/source_matrix.md` and `notes/discovery_synthesis.md` for strict evidence. The mandatory ancestor/background grounding is the full-read primary source plus the full-read Freund-Schapire source recorded there.

## Source grounding
- Primary survey: Arora, Hazan, Kale, "The Multiplicative Weights Update Method: A Meta-Algorithm and Applications," Theory of Computing 8(1):121-164, 2012. PDF read pages 1-28 (refs/mw-survey.pdf). This is the META survey; in-frame we re-derive the method, not cite this paper.
- Strict ancestor evidence: Freund & Schapire 1997, JCSS 55:119-139 — Hedge (exp factor) + AdaBoost, removes factor 2 via randomization. Deterministic weighted-majority background is retained only as summarized in the full-read primary source. Plotkin, Shmoys, Tardos 1995 — packing/covering LP framework, width ρ. von Neumann 1928 minimax. Garg-Könemann (width reduction). Hannan 1957 / Kalai-Vempala (FTPL). Winnow (Littlestone 1988).
- Code: github.com/j2kun/mwua — generic MWUA (gains form w_i *= 1+η·reward_i) and LP solver (linear-programming/linear.py) reducing LP to experts game with oracle + binary search. Jeremy Kun blog (jeremykun.com 2017/02/27) for LP-as-game intuition.

## The pain point / research question
Repeatedly choose one of n decisions; after choosing, an adversary reveals a cost vector m^(t) ∈ [-1,1]^n (costs depend on our choice possibly). We pay cost of the decision we picked. Goal: total cost comparable to the best single fixed decision in hindsight — low *regret* — even though the adversary is arbitrary and we commit before seeing costs. Naive "follow the majority" deterministically fails (majority can be wrong every day). Naive "average decision" / uniform random pays the average, no learning.

## Core algorithm (Figure 1)
Init: η ≤ 1/2, w_i^(1) = 1 for all i.
Each round t:
1. p^(t) = w^(t)/Φ^(t), Φ^(t) = Σ_i w_i^(t). Sample decision i ~ p^(t).
2. Observe cost vector m^(t) ∈ [-1,1]^n.
3. Update: w_i^(t+1) = w_i^(t)(1 - η m_i^(t)).
Expected cost in round t = m^(t)·p^(t).

## Regret bound (Theorem 2.1)
For all costs in [-1,1], η ≤ 1/2, any decision i:
  Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t |m_i^(t)| + ln(n)/η.
Corollary 2.2: for any distribution p, Σ_t m^(t)·p^(t) ≤ Σ_t (m^(t)+η|m^(t)|)·p + ln(n)/η.

### Proof via potential Φ = Σ w_i (the heart)
Upper bound: 
  Φ^(t+1) = Σ w_i^(t)(1-η m_i^(t)) = Φ^(t) - η Φ^(t) Σ m_i^(t) p_i^(t) = Φ^(t)(1 - η m^(t)·p^(t)) ≤ Φ^(t) exp(-η m^(t)·p^(t)),
using p_i = w_i/Φ and 1+x ≤ e^x. Induction: Φ^(T+1) ≤ n·exp(-η Σ_t m^(t)·p^(t)).  (2.2)
Lower bound: Φ^(T+1) ≥ w_i^(T+1) = Π_t (1-η m_i^(t)) ≥ (1-η)^{Σ_{≥0} m_i} (1+η)^{-Σ_{<0} m_i},  (2.4)
using (1-η)^x ≤ 1-ηx for x∈[0,1] and (1+η)^{-x} ≤ 1-ηx for x∈[-1,0].
Take logs of (2.2) and (2.4), combine:
  ln n - η Σ_t m^(t)·p^(t) ≥ Σ_{≥0} m_i ln(1-η) - Σ_{<0} m_i ln(1+η).
Negate, divide by η, use ln(1/(1-η)) ≤ η+η² and ln(1+η) ≥ η-η² (for η≤1/2):
  Σ_t m^(t)·p^(t) ≤ ln(n)/η + Σ_t m_i^(t) + η Σ_t |m_i^(t)|.  ∎
With |m_i|≤1, Σ|m_i|≤T, optimize η = sqrt(ln n / T): regret ≤ 2 sqrt(T ln n). Sublinear → average regret → 0.

## Hedge (Theorem 2.3)
Update w_i^(t+1) = w_i^(t) exp(-η m_i^(t)). Uses exp(-ηx) ≤ 1-ηx+η²x² for |ηx|≤1. Bound:
  Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t (m^(t))²·p^(t) + ln(n)/η. The η-term depends on algorithm's own distribution, not best decision.

## Weighted Majority deterministic background (Theorem 1.1)
Init w_i=1; on each mistake by expert i, w_i ← (1-η)w_i; predict weighted majority. Bound M^(T) ≤ 2(1+η) m_i^(T) + 2 ln(n)/η — factor 2 unavoidable for any DETERMINISTIC alg. Proof: each algorithm mistake means ≥ half the weight was on wrong side, so Φ drops by ≥ (1-η/2): Φ^(t+1) ≤ Φ^(t)(1/2 + (1-η)/2) = Φ^(t)(1-η/2). Induction Φ^(T+1) ≤ n(1-η/2)^{M}; and Φ ≥ w_i = (1-η)^{m_i}; combine with -ln(1-η) ≤ η+η². The factor 2 is removed by randomizing (predict up w.p. proportional to weight) — that's exactly MW with sampling.

## Gains version (Theorem 2.5)
Run with cost = -gain. w_i *= (1+η m_i) where m_i is gain. Bound: Σ m·p ≥ Σ m_i - η Σ|m_i| - ln(n)/η. This is the form the code uses.

## Application: zero-sum games / von Neumann minimax (Theorem 3.1)
Payoff matrix A, A(i,j)∈[0,1]. von Neumann: min_p max_j A(p,j) = max_q min_i A(i,q) = λ* (value). MWUA gives constructive proof: row player runs MW over rows; each round ORACLE returns best-response column j^(t) = argmax_j A(p^(t),j); cost vector m^(t) = j^(t)-th column of A. Apply Corollary 2.2: 
  λ* ≤ (1/T)Σ A(p^(t),j^(t)) ≤ (1/T)Σ A(p*,j^(t)) + η + ln(n)/(ηT).
Set η=ε/2, T=⌈4 ln(n)/ε²⌉ → within ε of λ*. Average row strategy p̄=(1/T)Σ p^(t) is ε-optimal; q* = empirical frequency of columns played is ε-optimal. O(log(n)/ε²) oracle calls. The two no-regret players' time-averaged play converges to equilibrium ⇒ minimax theorem holds.

## Application: packing/covering LPs (PST, Theorem 3.3)
Feasibility ∃x∈P: Ax≥b. Decisions = m constraints. Cost of constraint i at point x: m_i = (1/ρ)(A_i x - b_i), where width ρ = bound on |A_i x - b_i|; lies in [-1,1]. ORACLE: given distribution p on constraints, find x∈P with p^T A x ≥ p^T b (one averaged constraint — easy). If oracle fails, p is a certificate of infeasibility. Each round m^(t)·p^(t) = (1/ρ)(p^T A x - p^T b) ≥ 0. Apply Theorem 2.1 to a tight constraint i∈I; after T=⌈8 ℓ ρ ln(m)/ε²⌉ rounds with η=ε/(4ℓ), the average point x̄=(1/T)Σx^(t) satisfies A_i x̄ ≥ b_i - ε. O(ℓρ log(m)/ε²) oracle calls. Width ρ controls iteration count (Garg-Könemann reduce width to 1 for flow by routing only c^(t)=min-capacity flow).

## Application: set cover greedy (η=1)
Decisions = universe elements; cost of element i for set C_j is 1 if i∈C_j else 0. Run MW with η=1: covered elements get weight 0, uncovered have weight 1 → p^(t) = uniform on uncovered. Maximizing m·p = picking set covering most uncovered = GREEDY. Φ^(t+1) = Φ^(t)(1-m·p) < Φ^(t) e^{-1/OPT} (since m·p ≥ 1/OPT, as OPT sets cover all, one covers ≥1/OPT fraction). After T=⌈ln n⌉OPT rounds Φ < n e^{-ln n} ≤ 1 ⇒ all covered. ln(n)-approximation.

## Application: AdaBoost / boosting (Section 3.6)
Decisions = training samples S, |S|=N. Round t: present distribution p^(t) on samples to γ-weak learner, get h^(t) with error ≤ 1/2-γ under p^(t), i.e. m^(t)·p^(t) ≥ 1/2+γ where cost m_x = 1-|h^(t)(x)-c(x)| (=1 if correct). Increase weight of MIS-classified samples (low cost). Final hypothesis = majority vote of h^(1..T). After T=⌈(2/γ²)ln(1/ε)⌉ rounds, fraction of training errors ≤ ε. Uses Theorem 2.4 (restricted distributions / RE potential) to bound: any misclassified x has Σ_t m_x ≤ T/2; with the (1/2+γ)T lower bound on cumulative cost vs the T/2 per-error, get |E|/N ≤ ε.

## Design-decision → why table
- Multiply (not add): additive updates can't make a bad expert irrelevant fast; multiplicative ⇒ weight = product of per-round penalties = (1-η)^{#mistakes}, geometric ⇒ best expert dominates Φ within ln(n)/η.
- Sample ∝ weight (randomize): removes the factor-2 that's provably unavoidable for any deterministic rule; expected cost = m·p is linear in p, which is what the potential argument needs.
- Potential Φ = Σw_i (NOT entropy/KL): exponential calc using 1+x≤e^x and (1-η)^x≤1-ηx is cleaner than entropy proofs; sandwiches cumulative cost between two computable bounds on Φ.
- Linear factor (1-ηm) vs exponential exp(-ηm) (Hedge): linear keeps the η-penalty tied to best-decision loss (Thm 2.1); exponential ties it to algorithm's own second moment (Thm 2.3) — different applications prefer different bounds. For LPs/games the linear MW bound is the one that gives width-ρ guarantees.
- η = sqrt(ln n / T): balances the two regret terms ln(n)/η (cost of initial ignorance) and ηT (cost of over-reacting). Equal at η=sqrt(ln n/T) ⇒ regret 2 sqrt(T ln n).
- Width ρ / normalizing costs by ρ to land in [-1,1]: the bound has a Σ|m_i| term; if costs range over [-ρ,ρ], iterations blow up by ρ². Garg-Könemann's reweighting reduces ρ.
- Cost = "how well constraint satisfied" (counterintuitive): we DECREASE weight on penalty; a well-satisfied constraint should get low weight so the algorithm focuses adversarial effort on the violated/hard constraints. The oracle then must satisfy the weighted-average constraint, which forces progress on the currently-hardest constraints.

## In-frame discipline
- Never name "Arora-Hazan-Kale" or "the survey/this paper." Cite Freund-Schapire 1997, Plotkin-Shmoys-Tardos 1995, von Neumann 1928, and Hannan 1957 only where they are grounded in retrieved evidence for the strict artifact.
- The method name "Multiplicative Weights Update" may appear in answer.md as the thing being built.
- Code grounded in j2kun/mwua: gains form w_i *= (1+η reward_i), draw ∝ weights, LP solver via oracle+binary search.
