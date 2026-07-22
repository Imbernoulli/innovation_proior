// TIER: greedy
// Frequency-first load balancing: count each actor's total cue frequency, sort actors
// descending by frequency, then bin-pack into rooms by always adding the next actor to
// the room with the smallest running total frequency (classic LPT load balancer).
// This is a natural first idea -- "the busiest actors need their traffic spread evenly
// across rooms" -- but it never looks at WHEN actors are cued relative to each other, so
// it cannot tell apart "two huge but disjoint-in-time actors" (safe to share) from "two
// moderate actors that constantly alternate" (expensive to share).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, S, K, T;
    scanf("%d %d %d %d", &P, &S, &K, &T);
    vector<long long> freq(P + 1, 0);
    for (int t = 0; t < T; t++) {
        int x; scanf("%d", &x);
        freq[x]++;
    }

    vector<int> order(P);
    for (int i = 0; i < P; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (freq[a] != freq[b]) return freq[a] > freq[b];
        return a < b;
    });

    vector<long long> load(S + 1, 0);
    vector<int> color(P + 1, 0);
    for (int idx = 0; idx < P; idx++) {
        int a = order[idx];
        int best = 1;
        for (int r = 2; r <= S; r++) if (load[r] < load[best]) best = r;
        color[a] = best;
        load[best] += freq[a];
    }

    for (int i = 1; i <= P; i++) {
        printf("%d%c", color[i], (i < P) ? ' ' : '\n');
    }
    return 0;
}
