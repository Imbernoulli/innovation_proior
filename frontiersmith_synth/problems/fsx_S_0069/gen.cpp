#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder: harbor grows from a tiny yard to a full container port ----
    // testId 1 tiny (example scale); testId 10 large (~2500 radios).
    double f = (testId - 1) / 9.0;
    int N = (int)llround(6 + (2500 - 6) * (f * f));
    if (N < 4) N = 4;

    // channel budget varies to change how tight the assignment is
    int K = 4 + (testId % 4);               // 4..7

    // ---- lay the radios out over the port apron on a jittered grid ----
    int spacing = 10;
    int g = (int)ceil(sqrt((double)N));
    if (g < 1) g = 1;
    vector<int> px(N), py(N);
    for (int i = 0; i < N; i++) {
        int row = i / g, col = i % g;
        int jx = rnd.next(-3, 3), jy = rnd.next(-3, 3);
        px[i] = col * spacing + jx + 5;
        py[i] = row * spacing + jy + 5;
    }

    // ---- interference pairs: nearby radios interfere, closer => larger separation ----
    double R = 25.0;
    double R2 = R * R;
    double b1 = R / 3.0, b2 = 2.0 * R / 3.0;

    vector<array<int, 4>> edges; // u, v, s, p  (1-indexed nodes)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            long long dx = px[i] - px[j];
            long long dy = py[i] - py[j];
            long long d2 = dx * dx + dy * dy;
            if ((double)d2 > R2) continue;
            double d = sqrt((double)d2);
            int s;
            if (d < b1) s = 3;
            else if (d < b2) s = 2;
            else s = 1;
            if (s > K - 1) s = K - 1;
            if (s < 1) s = 1;
            int p = rnd.next(1, 10);
            edges.push_back({i + 1, j + 1, s, p});
        }
    }

    // safety net: guarantee at least one interference pair
    if (edges.empty()) {
        int s = min(2, K - 1);
        if (s < 1) s = 1;
        edges.push_back({1, min(2, N), s, rnd.next(1, 10)});
    }

    shuffle(edges.begin(), edges.end());

    int M = (int)edges.size();
    printf("%d %d %d\n", N, K, M);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
