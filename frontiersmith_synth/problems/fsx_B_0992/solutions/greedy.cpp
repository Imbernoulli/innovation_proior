// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The "obvious" single-pass heuristic an average coder writes first: every
// day, rank every available source (return scrap, each scrap lot, virgin
// metal) by a FLAT myopic margin  value = P0 - cost_per_ton  (return scrap is
// free so it always ranks first; virgin ranks wherever P0-CV happens to
// fall). Fill the furnace from best value down, respecting the daily
// capacity and the hard ppm cap for THAT DAY ONLY via the standard blend
// inequality. There is no look-ahead: it does not know or care that the
// FLAT ranking ignores the nonlinear quality-discount coupling (running the
// blend close to the cap taxes the price on every ton sold that day), and it
// never spends anything to keep the internal return-scrap pool clean for
// later -- it draws the pool to the hilt every day because "free" always
// looks best today. This is exactly the trap: myopic, per-day-optimal
// decisions that let contamination concentrate in the recycle loop and
// erode both today's price and tomorrow's capacity to exploit cheap scrap.
struct Cand { double avail, ppm, cost; int type, idx; }; // type 0=return 1=lot 2=virgin

int main() {
    int T;
    cin >> T;
    double CAP, V, RF, PPM_CAP, P0, CV, BETA;
    int LAG;
    cin >> CAP >> V >> RF >> LAG >> PPM_CAP >> P0 >> CV >> BETA;

    vector<double> histMass(T + 2, 0.0), histPpm(T + 2, 0.0);
    double remV = V;
    double poolMass = 0.0, poolPpm = 0.0;

    cout << setprecision(9) << fixed;
    for (int t = 1; t <= T; t++) {
        int K;
        cin >> K;
        vector<double> av(K), pp(K), pr(K);
        for (int j = 0; j < K; j++) cin >> av[j] >> pp[j] >> pr[j];

        if (t - LAG >= 1) {
            double massIn = RF * histMass[t - LAG];
            if (massIn > 1e-12) {
                double ppmIn = histPpm[t - LAG];
                double newMass = poolMass + massIn;
                poolPpm = (poolMass * poolPpm + massIn * ppmIn) / newMass;
                poolMass = newMass;
            }
        }

        vector<Cand> cand;
        if (poolMass > 1e-9) cand.push_back({poolMass, poolPpm, 0.0, 0, -1});
        for (int j = 0; j < K; j++) cand.push_back({av[j], pp[j], pr[j], 1, j});
        double virginRoom = min(CAP, remV);
        if (virginRoom > 1e-9) cand.push_back({virginRoom, 0.0, CV, 2, -1});

        sort(cand.begin(), cand.end(), [&](const Cand &a, const Cand &b) {
            double va = P0 - a.cost, vb = P0 - b.cost;
            if (fabs(va - vb) > 1e-9) return va > vb;
            if (a.ppm != b.ppm) return a.ppm < b.ppm;
            return a.type < b.type;
        });

        double virginUsed = 0.0, returnUsed = 0.0;
        vector<double> xUsed(K, 0.0);
        double curMass = 0.0, curPpmSum = 0.0;

        for (auto &c : cand) {
            double remCap = CAP - curMass;
            if (remCap <= 1e-12) break;
            double add;
            if (c.ppm <= PPM_CAP + 1e-9) {
                add = min(c.avail, remCap);
            } else {
                double RHS = PPM_CAP * curMass - curPpmSum;
                double denom = c.ppm - PPM_CAP;
                double maxByPpm = (RHS <= 0) ? 0.0 : RHS / denom;
                add = min({c.avail, remCap, maxByPpm});
            }
            if (add < 0) add = 0;
            curMass += add;
            curPpmSum += add * c.ppm;
            if (c.type == 0) returnUsed += add;
            else if (c.type == 1) xUsed[c.idx] += add;
            else virginUsed += add;
        }

        remV -= virginUsed;
        poolMass -= returnUsed;
        if (poolMass < 0) poolMass = 0.0;

        double ppmToday = (curMass > 1e-9) ? (curPpmSum / curMass) : 0.0;
        histMass[t] = curMass;
        histPpm[t] = ppmToday;

        cout << virginUsed;
        for (int j = 0; j < K; j++) cout << ' ' << xUsed[j];
        cout << ' ' << returnUsed << '\n';
    }
    return 0;
}
