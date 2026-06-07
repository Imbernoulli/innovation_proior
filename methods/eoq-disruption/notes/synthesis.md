# EOQD synthesis

## Pain point / research question
Classic EOQ (Harris 1913) assumes the supplier is ALWAYS available: order arrives instantly whenever you place it, so zero-inventory ordering (ZIO) is optimal and Q*=sqrt(2KD/h). Real suppliers go OFF (fire, strike, earthquake) for random durations. When you run out and the supplier is OFF you wait -> lost sales / backorders. Question: how big should Q be when the supplier alternates ON/OFF? Intuition: bigger than EOQ (buffer against disruption).

## Model (continuous review, deterministic demand D, zero lead time, ZIO)
- Supplier = 2-state CTMC. ON->OFF rate lambda (disruption rate), OFF->ON rate psi (recovery rate). Mean ON = 1/lambda, mean OFF = 1/psi. Steady-state disrupted prob = lambda/(lambda+psi).
- phi(t) = prob supplier OFF t after a cycle start (which is by def an ON moment) = (lambda/(lambda+psi))(1 - e^{-(lambda+psi)t}).  [Ross]
- ZIO: order Q, deplete at rate D over Q/D, hit 0. If supplier ON -> instant reorder. If OFF -> wait expected w = 1/psi (memoryless) then order.
- Cycle = renewal between consecutive successful shipments.

## Renewal-reward (Berk-Arreola-Risa corrected; = Qi-Shen-Snyder with alpha=0)
General QSS (disruptions at supplier AND retailer, alpha=retailer disruption, beta=retailer recovery):
  Abar = lam(alpha+beta)/(beta*psi*(alpha+lam+psi)),  Bbar = 1/alpha + 1/beta
  w = (1/psi)[1 + alpha(lam+psi)/(beta(alpha+beta+lam+psi))]
  E[T] = Abar(1 - e^{-(alpha+lam+psi)Q/D}) + Bbar(1 - e^{-alpha Q/D})   ... eq (4)
  I(Q) = piD + [F + (a+h/alpha)Q - (1-e^{-alpha Q/D})(hD/alpha^2 + piD/alpha)] / E[T]   ... eq (10)
  Quasiconvex in Q (Property 4). alpha=0 -> Berk-Arreola-Risa EOQD (Property 2). alpha=lam=0 -> classic EOQ (Property 3).

EOQD (alpha=0) clean reduction (verified numerically as alpha->0 limit, diff scales O(alpha)):
  A0 = lam/(psi(lam+psi))     [= Abar at alpha=0]
  E[T] = Q/D + A0(1 - e^{-(lam+psi)Q/D})        (expected cycle length: production time Q/D + expected extra wait when supplier OFF at reorder)
  E[C] = F + aQ + hQ^2/(2D) - piQ + piD*E[T]
  I(Q) = piD + [F + aQ + hQ^2/(2D) - piQ] / E[T]
  - F fixed order cost (=K), a variable unit cost, h holding, pi per-unit stockout (lost sales).
  - holding term hQ^2/2D = classic triangle. shortage: stockout occurs only when you hit 0 AND supplier OFF; expected shortage per cycle = piD(E[T]-Q/D); -piQ + piD E[T] groups it.
  Quasiconvex; minimize by bisection/golden section. NOT closed form (transcendental e^{-(lam+psi)Q/D}).

## Snyder's tight approximation (alpha=0, Snyder 2014; Property 7 of QSS)
- "In realistic instances psi is large, so e^{-(lam+psi)Q/D} ~ 0." Replace the exponential with 0 (constant).
- Then E[T] ~ Q/D + A0  (constant additive buffer A0 added to the EOQ cycle Q/D).
- minimize I_hat(Q) = piD + [F+aQ+hQ^2/2D-piQ]/(Q/D + A0). First-order condition N'g = N g', g'=1/D, gives a QUADRATIC:
    (h/2D) Q^2 + (h A0) Q + ((a-pi)A0 D - F) = 0
  Q_hat = [ -h A0 + sqrt( (h A0)^2 - 4(h/2D)((a-pi)A0 D - F) ) ] / (2 * h/2D)
        = [ -h A0 + sqrt( h^2 A0^2 + (2h/D)(F + (pi-a)A0 D) ) ] / (h/D)
  (the general QSS closed form (15) reduces to this at alpha=0; verified Q_hat -> Q* as alpha->0)
- This is the "tight approximation": closed form, convex I_hat, recovers classic EOQ when lam=0 (A0=0 -> Q_hat=sqrt(2FD/h)).

## Numerical anchor instance (verified)
D=1000, F=K=6, a=2, h=0.2, pi=10, lambda=1, psi=12 (supplier OFF ~1 month avg, ON ~1 yr avg)
  Q* exact (bisection)   = 750.26   I=2150.09
  Q_hat Snyder closed    = 750.47   I=2150.09   (errQ 0.028%, errC ~0%)
  Q_E classic EOQ        = 244.95   I=2243.57   (3x too small, +4.3% cost)
Disruption inflates EOQ ~3x. Robust across regimes (errors <0.03% Q, ~0% cost).

## Key intellectual moves (for reasoning.md, discovery order)
1. EOQ breaks because reorder isn't instant when supplier OFF -> cycle length becomes RANDOM.
2. Cycle = renewal; use renewal-reward: long-run rate = E[cost per cycle]/E[cycle length].
3. Need phi(t): supplier OFF prob t after an ON start -> 2-state CTMC -> (lam/(lam+psi))(1-e^{-(lam+psi)t}).
4. E[T] = Q/D + expected wait; wait happens with prob phi(Q/D), expected length 1/psi (memoryless) -> A0(1-e^{-(lam+psi)Q/D}).
5. Costs per cycle: fixed F, variable aQ, holding hQ^2/2D (classic triangle since ZIO and demand deterministic up to 0), shortage piD(E[T]-Q/D) (you starve exactly during the wait).
6. I(Q) is quasiconvex but transcendental -> numerical (golden section). Wall: no closed form, can't embed.
7. Approximation: psi large => e^{-(lam+psi)Q/D}~0 => E[T]~Q/D + A0 constant buffer => quadratic => closed form Q_hat. Tight.
8. Sanity: lam=0 => A0=0 => recover Harris EOQ. Disruption makes A0>0 => bigger Q.

## Berk-Arreola-Risa correction (background, NOT the proposed method here)
Parlar-Berkin 1991 first did this but with two errors: (i) assumed a stockout in EVERY off period (false if disruption ends before inventory hits 0), (ii) charged shortage per-unit-TIME instead of per-unit for lost sales. Berk-Arreola-Risa corrected -> redefined the cycle, got the quasiconvex (not proven convex) cost above. This is the "established prior art" the EOQD anchor builds on. In-frame we may cite Harris; the EOQD papers themselves are the target (in-frame, don't cite as artifacts) -- but Parlar-Berkin/Berk-Arreola-Risa ARE the target lineage, so treat the EOQD itself as the thing being discovered.

NOTE on in-frame: the task IS the EOQD model. So Harris is the ancestor (citable). The renewal-reward theorem and CTMC are background tools (citable as standard). The EOQD/Berk-Arreola-Risa/Snyder results are what we're deriving -> do not cite them as papers.

## Sources retrieved this run
- PRIMARY math: Qi, Shen, Snyder "A Continuous-Review Inventory Model with Disruptions at Both Supplier and Retailer" (POM) -- full renewal-reward derivation, exact I(Q) eq(4)(10), Snyder approx eq(15), reductions to Berk-Arreola-Risa (Prop 2) and EOQ (Prop 3). coral.ise.lehigh.edu/larry/files/pubs/qi-shen-snyder-pom.pdf  [READ IN FULL]
- SECONDARY/lineage: Snyder et al. "OR/MS Models for Supply Chain Disruptions: A Review" -- EOQD history, Parlar-Berkin/Berk-Arreola-Risa correction, ON/OFF Markov modeling, disruption/recovery rates. lehigh tech-papers 12W_005.pdf [READ relevant sections]
- BACKGROUND: Harris 1913 EOQ (sqrt(2KD/h)); renewal-reward theorem (Ross); 2-state CTMC.
- NOT obtained in full: Snyder 2014 IJPE original (paywalled SSRN/ScienceDirect), Atan-Snyder 2014 Springer chapter (paywalled), Parlar-Berkin 1991 & Berk-Arreola-Risa 1994 originals (paywalled NRL). Their exact equations are recovered faithfully via the Qi-Shen-Snyder primary-author restatement (Properties 2,3,7 give the exact reductions) + verified numerically. Flag in report.
