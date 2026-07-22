#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Guild Forecast Tournament: Liquidity Allocation Under
// the Market-Scoring Rule"  (family: lmsr-liquidity-allocator)
//
// Input:  N B lambda
//         for q=1..N: "M_q outcome_q" then M_q lines "belief_i budget_i"
//           (belief_i integer percent 1..99, budget_i positive integer,
//            printed in trading order)
//
// Output: N lines, beta_q (real, 0 or in [0.01,B]), sum_q beta_q <= B.
//
// Replay (exact, closed-form; see statement.txt for the derivation): market
// state is tracked as outstanding shares (q_yes,q_no), starting (0,0). LMSR
// cost function C(q_yes,q_no) = beta*ln(e^{q_yes/beta}+e^{q_no/beta}). Writing
// x = (q_yes-q_no)/beta (the price's logit) and h(x) = ln(1+e^x) (softplus):
//   moving x up   from x0 to x1 (buying YES, q_no fixed) costs beta*(h(x1)-h(x0))
//   moving x down from x0 to x1 (buying NO,  q_yes fixed) costs beta*(h(-x1)-h(-x0))
// Trader i moves toward belief_i's logit target: if the FULL move to the
// target costs <= their budget, they land exactly on the target; otherwise
// they spend the whole budget and land as far as it reaches (inverse-softplus
// closed form -- no search needed).
//
// Baseline B_ref (checker-computed, naive): split B uniformly, beta_q=B/N for
// every question, replayed with the exact same rules. solutions/trivial.cpp
// reproduces this construction exactly (-> ratio ~0.1).
// Score (max): sc = min(1000, 100*F/max(eps,B_ref)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static inline double softplus(double x) {
    if (x > 0) return x + log1p(exp(-x));
    return log1p(exp(x));
}
static inline double logitOf(double p) { return log(p / (1.0 - p)); }

struct Trader { int beliefPct; long long budget; };

int N;
long long Btot;
double LAMBDA;
vector<vector<Trader>> traders; // 0-indexed questions
vector<int> outcomeOf;

// Replay question qi at liquidity beta (beta==0 => market never opens).
// Returns (accuracyTerm, subsidy).
static pair<double,double> replayQuestion(int qi, double beta) {
    if (beta <= 0.0) {
        // price stuck at 0.5: log(0.5)-log(0.5) = 0, nothing traded.
        return {0.0, 0.0};
    }
    double qy = 0.0, qn = 0.0;
    const double LOGIT_LO = -6.0, LOGIT_HI = 6.0; // belief in [1,99]% => |logit|<=~4.6, safe margin
    for (auto& tr : traders[qi]) {
        double belief = tr.beliefPct / 100.0;
        double xt = logitOf(belief);
        xt = max(LOGIT_LO, min(LOGIT_HI, xt));
        double x = (qy - qn) / beta;
        double budget = (double)tr.budget;
        if (fabs(xt - x) < 1e-13) continue;
        if (xt > x) {
            double costNeeded = beta * (softplus(xt) - softplus(x));
            double xnew;
            if (budget >= costNeeded) {
                xnew = xt;
            } else {
                double y = softplus(x) + budget / beta;
                double e = exp(y);
                double m = max(e - 1.0, 1e-300);
                xnew = log(m);
            }
            qy = qn + beta * xnew;
        } else {
            double costNeeded = beta * (softplus(-xt) - softplus(-x));
            double xnew;
            if (budget >= costNeeded) {
                xnew = xt;
            } else {
                double y = softplus(-x) + budget / beta;
                double e = exp(y);
                double m = max(e - 1.0, 1e-300);
                xnew = -log(m);
            }
            qn = qy - beta * xnew;
        }
    }
    double xf = (qy - qn) / beta;
    double pf = 1.0 / (1.0 + exp(-xf));
    // revenue collected = C(final) - C(0,0), via a stable log-sum-exp
    double mx = max(qy, qn);
    double Cfinal = mx + beta * log(exp((qy - mx) / beta) + exp((qn - mx) / beta));
    double C0 = beta * log(2.0);
    double revenue = Cfinal - C0;
    int outcome = outcomeOf[qi];
    double payout = (outcome == 1) ? qy : qn;
    double subsidy = payout - revenue;
    double acc;
    if (outcome == 1) acc = log(max(pf, 1e-300)) - log(0.5);
    else acc = log(max(1.0 - pf, 1e-300)) - log(0.5);
    return {acc, subsidy};
}

static double totalObjective(const vector<double>& betas) {
    double F = 0.0;
    for (int q = 0; q < N; q++) {
        auto pr = replayQuestion(q, betas[q]);
        F += pr.first - LAMBDA * pr.second;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    Btot = inf.readLong();
    LAMBDA = inf.readDouble();

    traders.assign(N, {});
    outcomeOf.assign(N, 0);
    for (int q = 0; q < N; q++) {
        int Mq = inf.readInt();
        int outcome = inf.readInt();
        outcomeOf[q] = outcome;
        traders[q].reserve(Mq);
        for (int i = 0; i < Mq; i++) {
            int bp = inf.readInt();
            long long bud = inf.readLong();
            traders[q].push_back({bp, bud});
        }
    }

    // ---- internal baseline: uniform split beta_q = B/N for every question ----
    vector<double> uniformBetas(N, (double)Btot / (double)N);
    double Bref = totalObjective(uniformBetas);
    if (!isfinite(Bref) || Bref <= 0.0) Bref = 1e-6;

    // ---- read participant allocation ----
    vector<double> betas(N);
    double sumBeta = 0.0;
    for (int q = 0; q < N; q++) {
        double b = ouf.readDouble(0.0, (double)Btot, "beta_q");
        if (!isfinite(b)) quitf(_wa, "beta_%d is not finite", q);
        if (b > 0.0 && b < 0.01) quitf(_wa, "beta_%d = %.9f must be exactly 0 or >= 0.01", q, b);
        betas[q] = b;
        sumBeta += b;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after beta list");

    double tol = 1e-6 * (double)Btot + 1e-6;
    if (sumBeta > (double)Btot + tol) {
        quitf(_wa, "sum of beta_q = %.6f exceeds budget B = %lld (tolerance %.6f)", sumBeta, Btot, tol);
    }

    double F = totalObjective(betas);
    if (!isfinite(F)) quitf(_wa, "objective is not finite");

    double sc = max(0.0, min(1000.0, 100.0 * F / Bref));
    quitp(sc / 1000.0, "OK F=%.6f Bref=%.6f sumBeta=%.6f Ratio: %.6f", F, Bref, sumBeta, sc / 1000.0);
    return 0;
}
