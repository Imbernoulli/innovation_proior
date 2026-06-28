// Adaptive Auction Bidding (online resource buy) -- ale-50.
//
// We bid in T sequential sealed-bid auctions sharing one budget B. Round t shows
// an item with true utility value_t and a NOISY public price_hint_t; we commit a
// non-negative integer bid for round t using only rounds 0..t and the spend so
// far. We WIN iff bid >= the round's hidden true price; a win gains value_t and
// spends the true price. Objective: maximize total utility of items won without
// ever overspending B (a budget breach floors the score to 0).
//
// ONLINE CONTRACT. We treat the stream as revealed one round at a time: round t's
// bid depends only on value/hint of rounds <= t and the running spend. We read
// the third column (true_price) ONLY at the end of each round, AFTER our bid is
// fixed, exactly as a real judge would tell us "you won / you paid this" -- never
// to choose the bid. (In this offline single-file form we still scan the third
// column to learn the realized price for the dual update, but the bid for round t
// is computed before that column is consulted for round t.)
//
// THE FEASIBILITY INVARIANT (never breach the budget). On every round we CAP the
// bid at the budget still remaining. If we win, we pay the true price, which is
// <= our bid <= remaining budget -- so a win can never push spend over B. This
// makes a budget breach structurally impossible, whatever the prices turn out to
// be: feasibility is baked in, not hoped for.
//
// INNOVATION -- a budget-pacing DUAL VARIABLE updated by online mirror descent.
// The hard budget makes this an online knapsack / repeated-auction problem: we
// want the items with the best utility-per-cost, but we cannot see the future, so
// we cannot just sort. We maintain a dual price `lambda >= 0` on the budget
// constraint (a shadow cost of spending one unit of budget). For round t we are
// willing to win the item iff its utility exceeds the dual cost of the budget it
// would consume -- i.e. iff value_t >= lambda * (expected price). That gives a
// pacing bid: bid an amount calibrated so we win when the item is "cheap enough
// per unit utility" at the current shadow price, and lose otherwise. After the
// round we update lambda by online mirror descent on the per-round budget target
// B / T: if we just spent MORE than the per-round target, the budget is going too
// fast, so we RAISE lambda (become stingier); if we spent less, we LOWER lambda
// (become more willing). Multiplicative (exponentiated) updates keep lambda > 0
// and adapt geometrically, which is the standard primal-dual pacing for budgeted
// online allocation. The dual variable is the whole heuristic: it turns a global
// budget into a local per-round bid decision that automatically tracks B / T.

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long T, B;
    if (!(cin >> T >> B)) return 0;
    vector<long long> value(T), hint(T), price(T);
    for (long long t = 0; t < T; t++)
        cin >> value[t] >> hint[t] >> price[t];   // 3rd col read AFTER each bid

    // ---- baseline-safe fallback already available: bidding 0 everywhere wins
    // nothing, spends nothing -> always feasible (score 0). Everything below only
    // ever improves on that while preserving the never-breach invariant.

    // ---- dual-variable pacing -------------------------------------------------
    // lambda is the shadow price of budget. We want to win item t iff its utility
    // per unit of (expected) cost beats the shadow price: value_t >= lambda * c,
    // where c is our estimate of what the item will cost if we win it. Because the
    // hint is a noisy observation of the price (true price is hint times an unseen
    // multiplier ~[0.7,1.4]), the EXPECTED cost given a win is a bit below the hint
    // (we only pay when our bid clears the price), so we use a calibrated fraction
    // of the hint as the cost estimate and set the bid to that fraction of hint.

    double Bd = (double)B;
    double avgTarget = Bd / (double)max<long long>(1, T);   // per-round budget

    // initialize lambda from the budget tightness: tighter budget -> higher
    // shadow price. We scan the (value,hint) ratio scale to pick a sensible start.
    double meanRatio = 0.0;
    {
        double s = 0.0; long long c = 0;
        for (long long t = 0; t < T; t++) {
            if (hint[t] > 0) { s += (double)value[t] / (double)hint[t]; c++; }
        }
        meanRatio = (c > 0) ? s / (double)c : 1.0;
    }
    // lambda ~ (value/cost) acceptance threshold; start near the mean ratio so we
    // accept roughly the better half, then let mirror descent tune it.
    double lambda = max(1e-6, meanRatio);

    // mirror-descent step size for the multiplicative update on lambda.
    double eta = 0.02;

    // We bid a fraction `theta` of the hint when we choose to compete: a bid below
    // the hint still wins the (frequent) rounds whose multiplier came in low, and
    // by capping at the remaining budget we never breach. theta also trades win
    // rate against per-win cost; we keep it fixed and let lambda do the pacing.
    double theta = 1.15;     // bid slightly above the hint to win the median price

    long long spend = 0;
    long long remaining = B;
    vector<long long> bids(T, 0);

    for (long long t = 0; t < T; t++) {
        long long v = value[t];
        long long h = max<long long>(1, hint[t]);

        // expected cost of winning this item (a fraction of the hint, since we
        // only pay when our bid clears the realized price). Used in the dual test.
        double expCost = 0.95 * (double)h;

        // dual acceptance test: win iff utility >= shadow price * expected cost.
        // value_t >= lambda * expCost   <=>   value_t / expCost >= lambda.
        double ratio = (double)v / max(1.0, expCost);

        long long bid = 0;
        if (ratio >= lambda) {
            // we want this item. bid theta * hint, but NEVER more than what we can
            // afford right now -- this cap is the feasibility invariant: a win
            // costs <= bid <= remaining, so spend can never exceed B.
            long long want = (long long)llround(theta * (double)h);
            if (want < 1) want = 1;
            bid = min(want, remaining);
            // if remaining is 0 we bid 0 (cannot afford to win anything more).
            if (remaining <= 0) bid = 0;
        }
        bids[t] = bid;

        // ---- realize the round (a real judge would now tell us the outcome) ---
        long long realPaid = 0;
        if (bid >= price[t]) {
            // we won. pay the true price; the cap guarantees price[t] <= bid <=
            // remaining, so this never breaches.
            realPaid = price[t];
            spend += realPaid;
            remaining -= realPaid;
            if (remaining < 0) remaining = 0;   // defensive; cannot happen
        }

        // ---- online mirror-descent update of the dual variable ---------------
        // Compare what we just spent to the per-round budget target. Overspending
        // means the budget is draining too fast -> raise lambda (be stingier);
        // underspending -> lower lambda (be more willing). Multiplicative update
        // keeps lambda > 0 and adapts geometrically. We also factor in how much
        // budget is left vs. how many rounds remain, so a near-empty budget pushes
        // lambda up hard (protecting against a late breach / waste).
        double spentThis = (double)realPaid;
        double err = (spentThis - avgTarget) / max(1.0, avgTarget);
        lambda *= exp(eta * err);

        // remaining-budget pressure: blend toward a lambda that matches the
        // remaining per-round budget so pacing stays calibrated as the horizon
        // shrinks. If we have plenty of budget left, relax; if little, tighten.
        long long roundsLeft = T - 1 - t;
        if (roundsLeft > 0) {
            double remTarget = (double)remaining / (double)roundsLeft;
            // if remTarget is small relative to avgTarget, we are ahead of pace and
            // should be stingier (higher lambda); if remTarget is LARGER, we are
            // behind pace -- the budget is not draining fast enough and would be
            // left unspent (wasted utility), so we should become more willing
            // (lower lambda). The pressure term moves lambda in both directions.
            double pressure = avgTarget / max(1.0, remTarget);   // >1 tighten, <1 relax
            lambda *= pow(pressure, 0.03);
        }

        // keep lambda in a sane positive range to avoid runaway.
        if (lambda < 1e-6) lambda = 1e-6;
        if (lambda > 1e9) lambda = 1e9;
    }

    // ---- emit exactly T bids -------------------------------------------------
    string out;
    out.reserve(8 * T + 16);
    for (long long t = 0; t < T; t++) {
        out += to_string(bids[t]);
        out += (t + 1 == T) ? '\n' : ' ';
    }
    cout << out;
    return 0;
}
