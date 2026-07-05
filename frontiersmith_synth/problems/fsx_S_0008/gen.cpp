#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty / structure ladder
    // n (even), p (edge density), within-weight max, cross-weight [lo,hi]
    int n; double p; int wlo, clo, chi;
    switch (testId) {
        case 1:  n=6;   p=0.90; wlo=3;  clo=8;  chi=10; break;
        case 2:  n=20;  p=0.80; wlo=4;  clo=10; chi=20; break;
        case 3:  n=40;  p=0.60; wlo=5;  clo=20; chi=40; break;
        case 4:  n=60;  p=0.50; wlo=5;  clo=30; chi=60; break;
        case 5:  n=100; p=0.40; wlo=6;  clo=40; chi=80; break;
        case 6:  n=150; p=0.30; wlo=8;  clo=50; chi=100;break;
        case 7:  n=200; p=0.50; wlo=10; clo=50; chi=100;break;
        case 8:  n=300; p=0.40; wlo=5;  clo=60; chi=100;break;
        case 9:  n=450; p=0.30; wlo=4;  clo=70; chi=100;break;
        case 10: n=600; p=0.50; wlo=3;  clo=80; chi=100;break;
        default: n=100; p=0.40; wlo=6;  clo=40; chi=80; break;
    }

    // planted balanced partition, independent of vertex ids
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());
    vector<int> planted(n, 0);
    for (int i = 0; i < n / 2; i++) planted[perm[i]] = 1; // hidden group

    // build edges over id pairs; weight depends on planted membership
    vector<array<int,3>> edges;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            bool include = (rnd.next() < p);
            // guarantee a link crossing the reference (id) split so B > 0
            if (i == 0 && j == n - 1) include = true;
            if (!include) continue;
            int w;
            if (planted[i] != planted[j]) w = rnd.next(clo, chi);  // heavy cross link
            else                          w = rnd.next(1, wlo);    // light internal link
            edges.push_back({i + 1, j + 1, w});
        }
    }

    printf("%d %d\n", n, (int)edges.size());
    for (auto &e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
