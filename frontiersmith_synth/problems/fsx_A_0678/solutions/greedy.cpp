// TIER: greedy
// The obvious first attempt: conventional single-beam steering. Find the ONE
// angle with the highest target power and phase every element so its
// contributions add constructively there:
//   phi_i = -2*pi*d*i*cos(theta*)   (mod 2*pi)
// This nails a single dominant lobe but has NO idea about secondary lobes,
// the rest of the pattern shape, or any hard-null constraint -- on multi-lobe
// or null-heavy tests it lands far from the target.
#include <bits/stdc++.h>
using namespace std;
static const double PI = 3.14159265358979323846;

int main(){
    int N, M, D1000;
    scanf("%d %d %d", &N, &M, &D1000);
    double d = D1000 / 1000.0;
    vector<double> a(N);
    for (int i = 0; i < N; i++){ int x; scanf("%d", &x); a[i] = x / 1000.0; }
    vector<int> ang(M);
    for (int m = 0; m < M; m++) scanf("%d", &ang[m]);
    vector<int> tgt(M);
    int bestM = 0, bestT = -1;
    for (int m = 0; m < M; m++){ scanf("%d", &tgt[m]); if (tgt[m] > bestT){ bestT = tgt[m]; bestM = m; } }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    int K; scanf("%d", &K);
    for (int i = 0; i < K; i++){ int x; scanf("%d", &x); }
    int thresh; scanf("%d", &thresh);

    double thetaStar = ang[bestM] * PI / 180000.0;
    vector<int> phi(N);
    for (int i = 0; i < N; i++){
        double needed = -2.0 * PI * d * i * cos(thetaStar);   // radians
        double deg = needed * 180.0 / PI;
        double units = deg * 10.0;                             // 0.1-degree steps
        long long p = (long long)llround(units);
        p %= 3600; if (p < 0) p += 3600;
        phi[i] = (int)p;
    }
    for (int i = 0; i < N; i++) printf("%d%c", phi[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
