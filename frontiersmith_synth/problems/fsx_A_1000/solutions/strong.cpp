// TIER: strong
// Insight: replaying the printed trader sequence is exact and closed-form, so
// for each question the value-vs-beta curve V_q(beta) = accuracy(beta) -
// lambda*subsidy(beta) can be sampled on a grid WITHOUT any external solver --
// turning the whole problem into an explicit resource-allocation optimization.
// V_q can have more than one local optimum (a cheap "whoever trades last"
// peak near beta->0, and a separate, usually better, peak where the
// well-funded correct faction's budget is just enough to prevail); a
// monotone/water-filling/ternary-search allocator can get trapped on the
// wrong one. Instead: discretize the shared budget B into UNITS liquidity
// units, discretize each question's curve on a log-spaced beta grid
// (reserving budget CONSERVATIVELY via ceil-rounding so nothing is ever
// over-spent), and solve the resulting multiple-choice knapsack with a DP
// across all N questions -- picking, simultaneously and optimally within the
// discretization, one grid point per question subject to the shared cap.
#include <bits/stdc++.h>
using namespace std;

static inline double softplus(double x) {
    if (x > 0) return x + log1p(exp(-x));
    return log1p(exp(x));
}
static inline double logitOf(double p) { return log(p / (1.0 - p)); }

struct Trader { int beliefPct; long long budget; };

int N;
long long Btot;
double LAMBDA;
vector<vector<Trader>> traders;
vector<int> outcomeOf;

static pair<double,double> replayQuestion(int qi, double beta) {
    if (beta <= 0.0) return {0.0, 0.0};
    double qy = 0.0, qn = 0.0;
    const double LOGIT_LO = -6.0, LOGIT_HI = 6.0;
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
            if (budget >= costNeeded) xnew = xt;
            else {
                double y = softplus(x) + budget / beta;
                double m = max(exp(y) - 1.0, 1e-300);
                xnew = log(m);
            }
            qy = qn + beta * xnew;
        } else {
            double costNeeded = beta * (softplus(-xt) - softplus(-x));
            double xnew;
            if (budget >= costNeeded) xnew = xt;
            else {
                double y = softplus(-x) + budget / beta;
                double m = max(exp(y) - 1.0, 1e-300);
                xnew = -log(m);
            }
            qn = qy - beta * xnew;
        }
    }
    double xf = (qy - qn) / beta;
    double pf = 1.0 / (1.0 + exp(-xf));
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

static double valueAt(int qi, double beta) {
    auto pr = replayQuestion(qi, beta);
    return pr.first - LAMBDA * pr.second;
}

int main() {
    if (!(cin >> N >> Btot >> LAMBDA)) return 0;
    traders.assign(N, {});
    outcomeOf.assign(N, 0);
    for (int q = 0; q < N; q++) {
        int Mq, outcome;
        cin >> Mq >> outcome;
        outcomeOf[q] = outcome;
        traders[q].reserve(Mq);
        for (int i = 0; i < Mq; i++) {
            int bp; long long bud;
            cin >> bp >> bud;
            traders[q].push_back({bp, bud});
        }
    }

    const int UNITS = 400;
    const int K = 48; // grid points per question (excluding the explicit 0)
    double unit = (double)Btot / (double)UNITS;

    // Per-question: best value achievable while reserving AT MOST u units,
    // and the actual beta that achieves it (monotone-padded so "reserve more
    // than needed" is always allowed for free).
    vector<vector<double>> bestVal(N, vector<double>(UNITS + 1, -1e18));
    vector<vector<double>> bestBeta(N, vector<double>(UNITS + 1, 0.0));

    for (int q = 0; q < N; q++) {
        vector<double> arr(UNITS + 1, -1e18);
        vector<double> betaOf(UNITS + 1, 0.0);
        arr[0] = max(0.0, valueAt(q, 0.0));
        betaOf[0] = 0.0;
        for (int k = 0; k < K; k++) {
            double frac = (K == 1) ? 0.0 : (double)k / (double)(K - 1);
            double beta = exp(log(0.01) + (log((double)Btot) - log(0.01)) * frac);
            if (beta > (double)Btot) beta = (double)Btot;
            int u = (int)ceil(beta / unit);
            if (u < 1) u = 1;
            if (u > UNITS) continue;
            double v = valueAt(q, beta);
            if (v > arr[u]) { arr[u] = v; betaOf[u] = beta; }
        }
        // monotone forward-fill: reserving more units is never worse
        for (int u = 1; u <= UNITS; u++) {
            if (arr[u - 1] > arr[u]) { arr[u] = arr[u - 1]; betaOf[u] = betaOf[u - 1]; }
        }
        bestVal[q] = arr;
        bestBeta[q] = betaOf;
    }

    // Knapsack DP across questions, capacity UNITS.
    const double NEG = -1e18;
    vector<double> dp(UNITS + 1, NEG);
    dp[0] = 0.0;
    vector<vector<int>> choice(N, vector<int>(UNITS + 1, 0));
    for (int q = 0; q < N; q++) {
        vector<double> ndp(UNITS + 1, NEG);
        for (int used = 0; used <= UNITS; used++) {
            if (dp[used] <= NEG / 2) continue;
            double base = dp[used];
            int room = UNITS - used;
            for (int add = 0; add <= room; add++) {
                double v = bestVal[q][add];
                if (v <= NEG / 2) continue;
                int nu = used + add;
                double cand = base + v;
                if (cand > ndp[nu]) { ndp[nu] = cand; choice[q][nu] = add; }
            }
        }
        dp = ndp;
    }

    int bestU = 0;
    double bestF = NEG;
    for (int u = 0; u <= UNITS; u++) if (dp[u] > bestF) { bestF = dp[u]; bestU = u; }

    vector<double> outBeta(N, 0.0);
    int rem = bestU;
    for (int q = N - 1; q >= 0; q--) {
        int add = choice[q][rem];
        outBeta[q] = bestBeta[q][add];
        rem -= add;
    }

    // Safety net: by construction sum(outBeta) <= B already (each grid point
    // reserved ceil(beta/unit) units and the DP never exceeds UNITS total
    // units), but trim defensively against floating-point rounding by taking
    // any tiny excess off the single largest allocation (never pushes a
    // value into the forbidden (0,0.01) gap).
    double sumBeta = 0.0;
    for (double b : outBeta) sumBeta += b;
    if (sumBeta > (double)Btot) {
        double excess = sumBeta - (double)Btot;
        int hi = (int)(max_element(outBeta.begin(), outBeta.end()) - outBeta.begin());
        double nb = outBeta[hi] - excess - 1e-9;
        outBeta[hi] = (nb < 0.01) ? 0.0 : nb;
    }

    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    for (int q = 0; q < N; q++) cout << outBeta[q] << "\n";
    return 0;
}
