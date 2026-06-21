# Myerson Auction

## Setting

There are n risk-neutral bidders and one indivisible item. Bidder i has private value v_i drawn independently from distribution F_i with density f_i. A direct mechanism specifies an allocation x_i(b) in [0,1] and payment p_i(b). It must be incentive compatible and individually rational, and it must maximize expected seller revenue.

Define the virtual value

```text
phi_i(v) = v - (1 - F_i(v)) / f_i(v).
```

## Theorem

For independent single-parameter bidders, every IC and IR mechanism has expected revenue equal to expected virtual surplus minus the lowest-type utilities:

```text
E[revenue] = E[sum_i x_i(v) phi_i(v_i)] - sum_i U_i(0).
```

Revenue maximization sets U_i(0)=0 whenever the lowest type can be given zero utility. If every phi_i is nondecreasing, an optimal auction allocates the item to a bidder with the highest nonnegative virtual value and withholds the item if all virtual values are negative. Payments are the threshold payments implied by the monotone allocation rule.

If a distribution is irregular, replace its revenue curve by its concave envelope and use the corresponding ironed virtual values. The optimal implementable allocation maximizes ironed virtual surplus, with bunching or random tie-breaking on ironed intervals, and payments again come from the payment identity.

## Proof

Fix bidder i and hold the other reports fixed, or work with interim allocation and payment after averaging over the other bidders. Let q(v) be bidder i's allocation probability and P(v) its expected payment. Truthful utility is

```text
U(v) = v q(v) - P(v).
```

IC implies that for v > z,

```text
U(v) >= U(z) + (v-z)q(z)
U(v) <= U(z) + (v-z)q(v).
```

These inequalities force q(v) >= q(z), so implementable allocation is monotone. With monotonicity, the envelope identity gives

```text
U(v) = U(0) + int_0^v q(t) dt
P(v) = v q(v) - U(0) - int_0^v q(t) dt.
```

Expected payment is therefore

```text
E[P(V)]
= int v q(v) f(v) dv - int q(t)(1-F(t)) dt - U(0)
= int q(v)[v f(v) - (1-F(v))] dv - U(0)
= E[q(V) phi(V)] - U(0).
```

Summing over bidders gives the revenue-virtual-surplus identity. Once payments are eliminated this way, the only remaining design choice is a feasible monotone allocation rule. For regular distributions, pointwise virtual-surplus maximization is monotone, so it is implementable and optimal.

## Mechanism

For regular distributions:

1. Ask each bidder to report b_i.
2. Compute phi_i(b_i) for every bidder.
3. If max_i phi_i(b_i) < 0, do not sell.
4. Otherwise allocate the item to a bidder with maximum nonnegative phi_i(b_i).
5. Charge the winner the smallest report at which it would still win; equivalently, convert the relevant virtual threshold back into value space.

For identical regular bidders, phi is common and monotone, so the winner is the highest bidder provided its bid exceeds the reserve r satisfying

```text
phi(r) = 0.
```

The payment is

```text
max(reserve price, second-highest bid).
```

For irregular distributions:

1. Write the distribution in quantile space q=1-F(v) and its revenue curve R(q)=q v(q).
2. Replace R by its concave envelope.
3. Use the slope of that envelope as the ironed virtual value.
4. Allocate to maximize ironed virtual surplus subject to feasibility.
5. Treat types in flat ironed intervals as tied or bunched so the allocation remains monotone.
6. Recover payments from the same envelope/payment identity.
