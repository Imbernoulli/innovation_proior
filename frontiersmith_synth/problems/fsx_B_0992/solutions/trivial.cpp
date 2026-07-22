// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Do-nothing-clever baseline: melt ONLY virgin metal every day, up to
// min(daily CAP, remaining lifetime virgin budget V). Never touches scrap
// lots or the internal return-scrap pool. Always feasible (ppm=0 <= PPM_CAP,
// mass <= CAP, cumulative virgin <= V by construction). This is exactly the
// checker's internal baseline B, so it scores ratio ~= 0.1.
int main() {
    int T;
    cin >> T;
    double CAP, V, RF, PPM_CAP, P0, CV, BETA;
    int LAG;
    cin >> CAP >> V >> RF >> LAG >> PPM_CAP >> P0 >> CV >> BETA;

    double remV = V;
    cout << setprecision(9) << fixed;
    for (int t = 1; t <= T; t++) {
        int K;
        cin >> K;
        for (int j = 0; j < K; j++) {
            double a, p, pr;
            cin >> a >> p >> pr;
        }
        double use = min(CAP, remV);
        if (use < 0) use = 0;
        remV -= use;
        cout << use;
        for (int j = 0; j < K; j++) cout << ' ' << 0.0;
        cout << ' ' << 0.0 << '\n';
    }
    return 0;
}
