#include <bits/stdc++.h>
using namespace std;

int main() {
    long long m;
    int n;
    if (!(cin >> m >> n)) return 0;

    // Arc (s, L), 1 <= L <= m, covers circular markers s,...,s+L-1 (mod m).
    // Want minimum number of arcs whose union is the whole ring, else -1.
    //
    // MINIMUM CIRCULAR ARC COVER via greedy-exchange:
    //   Marker 0 must be covered by some chosen arc. We try, as the forced
    //   FIRST arc, each arc that covers marker 0. Forcing arc A fixes the
    //   linear window [A.start, A.start + m): we must cover that whole window.
    //   After A we have covered up to A.start + A.len. Then we repeatedly pick,
    //   among arcs whose linear start is <= current covered end, the one whose
    //   linear end reaches furthest, advancing the covered end. Each pick is one
    //   more arc. We stop once the covered end reaches A.start + m. The minimum
    //   over all choices of first arc is the answer.

    vector<long long> S(n), Ln(n);
    for (int i = 0; i < n; i++) {
        long long s, L;
        cin >> s >> L;
        s %= m; if (s < 0) s += m;
        S[i] = s; Ln[i] = L;
    }

    // A length-m arc covers the entire ring by itself.
    for (int i = 0; i < n; i++) if (Ln[i] >= m) { cout << 1 << "\n"; return 0; }

    long long best = LLONG_MAX;

    // Candidate first arcs = arcs covering marker 0: s == 0 OR s + L > m.
    for (int f = 0; f < n; f++) {
        bool coversZero = (S[f] == 0) || (S[f] + Ln[f] > m);
        if (!coversZero) continue;
        long long a0 = S[f];

        // Coordinates relative to a0: marker (s - a0 mod m). Cover [0, m).
        // Each arc j -> [rs, rs+L); if it wraps, also the prefix copy [rs-m, ...).
        vector<pair<long long,long long>> ivs; // (l, r)
        ivs.reserve(2 * n);
        for (int j = 0; j < n; j++) {
            long long rs = ((S[j] - a0) % m + m) % m;   // positive modulus!
            ivs.push_back({rs, rs + Ln[j]});
            if (rs + Ln[j] > m) ivs.push_back({rs - m, rs + Ln[j] - m});
        }
        sort(ivs.begin(), ivs.end());

        // Greedy minimum interval cover of [0, m). Forced first arc f maps to
        // rs = 0, seeding the sweep. Count arcs used.
        long long curEnd = 0, cnt = 0;
        size_t p = 0;
        bool ok = true;
        while (curEnd < m) {
            long long newEnd = curEnd;
            while (p < ivs.size() && ivs[p].first <= curEnd) {
                newEnd = max(newEnd, ivs[p].second);
                p++;
            }
            if (newEnd <= curEnd) { ok = false; break; }
            curEnd = newEnd;
            cnt++;
        }
        if (ok && curEnd >= m) best = min(best, cnt);
    }

    if (best == LLONG_MAX) cout << -1 << "\n";
    else cout << best << "\n";
    return 0;
}
