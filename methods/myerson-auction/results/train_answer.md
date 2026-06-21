I would describe the canonical method as the Myerson optimal auction. It gives the revenue-maximizing way to sell a single indivisible item to independent private-value bidders when truthful reporting and voluntary participation must be guaranteed.

The setting is this. There are n risk-neutral bidders. Bidder i privately knows a value v_i drawn independently from a distribution F_i with density f_i. A direct mechanism asks each bidder for a report b_i, then chooses an allocation probability x_i(b) in [0,1] and a payment p_i(b). Feasibility means at most one bidder can receive the item, so the allocation probabilities must sum to at most one. The bidders' utilities are quasilinear: bidder i receives utility v_i x_i(b) minus p_i(b). Incentive compatibility means reporting the true value is optimal for every bidder no matter what others do or report. Individual rationality means every bidder's expected utility from truthfully participating is nonnegative. The seller's goal is to maximize expected revenue, which is the expectation of the sum of payments.

The right place to start is not a familiar auction format but the constraints themselves. Fix one bidder and hold the others fixed, or take expectations over them to obtain an interim allocation rule q(v) and interim expected payment P(v). If the bidder's true value is v, truthful utility is U(v) equals v q(v) minus P(v). Incentive compatibility says that for any two reports v and z, the type v bidder should not prefer to report z and the type z bidder should not prefer to report v. Combining these two inequalities for v greater than z gives U(v) minus U(z) is at least (v minus z) q(z) and at most (v minus z) q(v). For both to hold for all pairs, q(v) must be at least q(z). So monotonicity of the interim allocation rule is not an extra assumption; it is forced by incentive compatibility. Any implementable mechanism must give a higher type a weakly higher chance of receiving the good.

Once q is monotone, the inequalities also pin down the shape of U. In the smooth case the derivative of U is q(v), and more generally the envelope identity says U(v) equals U(0) plus the integral from 0 to v of q(t) dt. Substituting back, the payment identity becomes P(v) equals v q(v) minus U(0) minus the integral from 0 to v of q(t) dt. This is a powerful reduction: every implementable allocation rule has its payments determined up to a single constant, the utility of the lowest type. Individual rationality then tells us what to do with that constant. Because q is nonnegative, U is nondecreasing, so the lowest type is the hardest to satisfy. Setting U(0) to zero is both individually rational and revenue-maximizing, because any positive U(0) would simply reduce every type's payment by the same amount without helping incentives. So after imposing U(0) equals zero, the entire mechanism design problem collapses to choosing a feasible monotone allocation rule.

The next step is to rewrite expected revenue in terms of that allocation rule. For bidder i with value distribution F, expected payment is the expectation of V q(V) minus the expectation of the integral from 0 to V of q(t) dt. Swapping the order of integration in the second term turns it into the integral from 0 to infinity of q(t) times the probability that V is at least t, which is q(t) times one minus F(t). Putting both terms under the same integral gives the expectation of q(V) times the quantity V minus (one minus F(V)) over f(V). This quantity is the virtual value phi(V) equals V minus (one minus F(V)) over f(V). The bidder's expected contribution to revenue is therefore not value times allocation probability but virtual value times allocation probability. Summing across independent bidders, expected revenue equals expected virtual surplus, assuming lowest-type utilities are zero.

This virtual-surplus representation changes the optimization problem dramatically. Instead of choosing payments and allocations jointly subject to incentive constraints, the seller simply chooses a feasible allocation to maximize the sum of virtual values. The remaining constraint is monotonicity. A distribution is called regular when its virtual value phi(v) is nondecreasing in v. For regular distributions, pointwise maximization of virtual surplus is automatically monotone: raising a bidder's report raises its virtual score, so a winner cannot lose by reporting a higher value. The optimal rule is therefore to allocate the item to the bidder with the highest nonnegative virtual value, and to withhold the item if all virtual values are negative. Payments are the threshold payments implied by the allocation rule: the winner pays the smallest report at which it would still win.

For identical regular bidders, the virtual transformation preserves the ranking of bids, so the optimal auction simplifies to a second-price auction with a reserve price r satisfying phi(r) equals zero, or equivalently r equals (one minus F(r)) over f(r). This reserve is the monopoly price. The winner pays the maximum of the reserve and the second-highest bid. It is important that this format emerges from the derivation rather than being assumed at the start. The reserve is simply the cutoff where serving the bidder switches from having positive to negative marginal expected revenue.

When bidders are asymmetric but regular, ranking by raw bids is no longer optimal. A lower value from a distribution with a thinner upper tail can have a higher virtual value than a higher value from another distribution. The optimal allocation ranks virtual values, not bids, and the critical payment is computed by converting the virtual threshold back into that bidder's value space through its own phi function.

If a distribution is irregular, meaning phi decreases somewhere, the pointwise virtual-surplus rule can be nonmonotone, and no payment rule can implement it. The repair uses quantile space. Let q equal one minus F(v), so smaller q corresponds to a stronger type, and let v(q) be the inverse demand curve. The revenue curve is R(q) equals q times v(q). Its derivative with respect to q is marginal revenue in quantile coordinates, and it corresponds to the virtual value. When R is concave, marginal revenue is nonincreasing in q, which matches the regularity condition that virtual value increases with value. When R is not concave, the marginal revenue slopes move the wrong way and produce a nonmonotone allocation.

The least invasive fix is to replace R by its concave envelope. On intervals where the original revenue curve lies below a chord, the envelope has a constant slope. Translating back to value space, all types in such an interval receive the same ironed virtual value. Maximizing ironed virtual surplus can bunch those types together, treating them as tied for allocation purposes, with randomization or tie-breaking chosen so the allocation probability remains monotone across the interval. The seller gives up the attempt to distinguish types in regions where doing so would violate incentive compatibility, while keeping the best concave revenue frontier available under monotonicity. Payments are again recovered from the same envelope and payment identity.

The Myerson optimal auction is therefore best understood as a three-step reduction. First, incentive compatibility and individual rationality reduce the design space to monotone allocation rules with lowest-type utility zero. Second, integration by parts turns expected revenue into expected virtual surplus. Third, regularity or ironing lets the seller maximize virtual surplus pointwise subject to feasibility. The theorem, the mechanism, and the payment rule all follow from this reduction.

```python
import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar

np.random.seed(0)

# Regular example: values Uniform[0,1]
# virtual value phi(v) = v - (1 - v)/1 = 2v - 1
# reserve r solves phi(r)=0 => r = 0.5

# Two asymmetric bidders:
# bidder 1: Uniform[0,1]
# bidder 2: Beta(2,1) with CDF F(v)=v**2 on [0,1]
# density f(v)=2v, virtual phi2(v)=v - (1 - v**2)/(2v)

def phi1(v):
    return 2 * v - 1.0

def phi2(v):
    return v - (1.0 - v**2) / (2.0 * v + 1e-12)

def optimal_allocation(b1, b2):
    p1 = phi1(b1)
    p2 = phi2(b2)
    if p1 < 0 and p2 < 0:
        return None, 0.0
    if p1 >= p2:
        return 1, p1
    else:
        return 2, p2

def payment_threshold(winner, other_report):
    # Winner pays smallest report at which it would still win.
    if winner == 1:
        # bidder 1 wins if phi1(b1) >= max(0, phi2(other_report))
        threshold = max(0.0, phi2(other_report))
        # phi1 is linear: phi1(v) = 2v - 1
        return (threshold + 1.0) / 2.0
    else:
        threshold = max(0.0, phi1(other_report))
        # bidder 2 wins if phi2(b2) >= threshold
        # solve v - (1 - v**2)/(2v) = threshold on (0,1]
        def loss(v):
            return (phi2(v) - threshold)**2
        res = minimize_scalar(loss, bounds=(1e-6, 1.0), method='bounded')
        return res.x

# Simulate reports drawn from true value distributions
def simulate(n=200000):
    v1 = np.random.rand(n)
    v2 = np.random.beta(2, 1, size=n)
    revenue = 0.0
    alloc_count = [0, 0]
    for b1, b2 in zip(v1, v2):
        winner, _ = optimal_allocation(b1, b2)
        if winner is None:
            continue
        other = b2 if winner == 1 else b1
        pay = payment_threshold(winner, other)
        revenue += pay
        alloc_count[winner - 1] += 1
    return revenue / n, alloc_count

rev, counts = simulate()
print(f"Simulated expected revenue: {rev:.4f}")
print(f"Allocations: bidder1={counts[0]}, bidder2={counts[1]}")

# Check that truthful reporting is interim-IC for bidder 1 by estimating
# expected utility of bidding z when value is v.
def estimate_utility(v, z, n=50000):
    # bidder 2 draws from Beta(2,1)
    b2 = np.random.beta(2, 1, size=n)
    # use bidder 1 report z, value v
    winner, _ = optimal_allocation(z, b2)
    utility = 0.0
    for i in range(n):
        if winner[i] == 1:
            pay = payment_threshold(1, b2[i])
            utility += v - pay
    return utility / n

v_test = 0.75
truth_util = estimate_utility(v_test, v_test)
shade_util = estimate_utility(v_test, 0.6)
print(f"Bidder 1 value {v_test}: truthful utility={truth_util:.4f}, shade-to-0.6 utility={shade_util:.4f}")
```
