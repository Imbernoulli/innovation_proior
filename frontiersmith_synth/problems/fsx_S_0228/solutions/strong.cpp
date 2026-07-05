// TIER: strong
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long T, M, S, D, P, h, H;
    scanf("%lld %lld %lld %lld %lld %lld %lld", &T, &M, &S, &D, &P, &h, &H);
    vector<long long> a(T);
    vector<int> av(T);
    for (long long i = 0; i < T; i++) scanf("%lld %d", &a[i], &av[i]);

    // next spot-available slot at or after index i
    vector<long long> nextSpot(T + 1, T); // T = "none"
    nextSpot[T] = T;
    for (long long i = T - 1; i >= 0; i--)
        nextSpot[i] = av[i] ? i : nextSpot[i + 1];

    vector<int> on(T, 0), sp(T, 0);
    long long q = 0;
    for (long long i = 0; i < T; i++) {
        q += a[i];
        // always use the cheap spot lift when available and there is a queue
        if (av[i] && q > 0) {
            sp[i] = 1;
            q -= min(q, S);
        }
        if (q > 0) {
            bool runon = false;
            if (i == T - 1) {
                runon = true; // must clear by the deadline
            } else {
                long long j = nextSpot[i + 1];
                long long wait = (j >= T) ? (T - i) : (j - i);
                // if holding this queue until the next cheap window costs more than a run, serve now
                if (q * h * wait > D) runon = true;
                // keep the queue bounded so a single final on-demand run can always clear it
                if (q > M / 2) runon = true;
                // no cheap window remains ahead: on-demand is the only way to drain
                if (j >= T) runon = true;
            }
            if (runon) {
                on[i] = 1;
                q -= min(q, M);
            }
            // else: pause and batch this demand into a later cheap window
        }
    }

    for (long long i = 0; i < T; i++) printf("%d %d\n", on[i], sp[i]);
    return 0;
}
