## Research question

A seller has one indivisible item and several risk-neutral bidders. Bidder i privately knows a value v_i drawn independently from a known distribution F_i with density f_i. The seller wants a direct mechanism that maximizes expected revenue while making truthful reporting incentive compatible and participation individually rational.

The seller's objective is money, not total value. Under private information, the allocation rule and payment rule cannot be chosen point by point independently: a high type must prefer its own outcome to a lower type's outcome, and every type who participates must receive nonnegative utility. The useful formulation keeps IC and IR constraints explicit enough to prevent cheating, but simple enough to optimize over.

## Background

The private-values auction model treats each bidder's value as information known only to that bidder. Utilities are quasilinear: if bidder i receives allocation probability x_i and pays p_i, utility is v_i x_i - p_i. Direct revelation is the natural design language because incentive compatibility can be stated as truth-telling being optimal among all reports.

In a single-parameter environment, the allocation probability faced by one bidder, holding the others fixed or taking expectations over them, is the central object. If a higher report can reduce a bidder's chance of receiving the good, no payment rule can make truthful reporting robust: the bidder with the higher value would sometimes want to shade down. Conversely, if allocation is monotone in the report, payments can be chosen so that the bidder's utility is the area under the allocation curve up to its value. This is the envelope-theorem view of incentive compatibility.

Individual rationality fixes the remaining transfer degree of freedom. Once the allocation curve is fixed, the payment identity determines payments up to the utility assigned to the lowest type. A revenue-maximizing seller sets the lowest type's utility to zero whenever that is feasible, because leaving extra utility with all types lowers revenue without improving incentives.

The classical second-price auction already gives a clean example of this logic. The highest bidder receives the item, the allocation rule is monotone, and the winner pays the smallest bid that would still have won. The price is not an arbitrary add-on; it is the critical value implied by the allocation rule.

For profit rather than surplus, withholding the item can be valuable. A single posted price r yields expected revenue r(1-F(r)) for one buyer, so a revenue-maximizing price solves the monopoly-pricing tradeoff between a higher payment and a lower sale probability. In a multi-bidder auction, competition sometimes makes a reserve irrelevant, but when competition is weak the same single-agent tradeoff reappears.

## Baselines

**Posted price for one buyer.** A seller facing one buyer can announce a price r. The buyer accepts if v >= r, so expected revenue is r(1-F(r)). This directly handles the no-sale option and gives the monopoly reserve.

**Second-price auction.** Allocating to the highest bidder and charging the second-highest bid is dominant-strategy incentive compatible and individually rational. It maximizes realized surplus among bidders.

**Second-price auction with a reserve.** Adding a reserve price combines competition with the possibility of no sale. For identical regular distributions, this format links the reserve to the distribution.

**Surplus maximization with externality payments.** In general single-parameter environments, the VCG/externality-pricing template maximizes total value subject to truthful incentives. The allocation that maximizes surplus is monotone and the payments are critical values.

**Arbitrary direct mechanisms.** A direct mechanism can specify any feasible allocation probability and payment for every bid profile, with IC coupling all reports by one bidder.

## Evaluation settings

The theoretical yardstick is expected seller revenue under independent private values, subject to truthful reporting and interim or ex post individual rationality. The mechanism must be feasible for every report profile, and its allocation rule must be implementable by a payment identity.

For a single item, natural special cases include one bidder, identical bidders, asymmetric independent bidders, and distributions with monotone or nonmonotone revenue curves. The relevant checks are: whether the allocation is monotone in each bidder's own report; whether the payment charged to a winner is the appropriate threshold or identity payment; whether the seller withholds the item when selling has negative marginal revenue; and whether bunching or random tie-breaking is needed when the distribution is irregular.

The clean benchmark mechanisms are posted pricing, second-price auction, second-price auction with reserve, and distribution-aware direct allocation rules. The output is a theorem and mechanism characterization rather than an empirical performance table.

## Code framework

For this theory artifact, the scaffold is a direct-mechanism specification rather than executable code. The primitives are a feasible set for allocations, independent value distributions, allocation probabilities, payments, and expected revenue.

```text
Inputs:
  bidders i = 1,...,n
  independent distributions F_i with densities f_i
  feasible allocations x in X

Direct mechanism skeleton:
  receive reports b = (b_1,...,b_n)
  choose a feasible allocation x(b)
  choose payments p(b)

Implementability checks:
  for each bidder i, the interim allocation q_i(z) must be nondecreasing in z
  payments must make truthful reporting optimal
  lowest-type utility must satisfy individual rationality
```
