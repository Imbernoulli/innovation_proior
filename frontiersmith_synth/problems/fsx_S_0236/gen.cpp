#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty / structure ladder:
    // n (even), p (wire density), noise (fraction of wires whose type is flipped
    // away from the planted-optimal type -> optimum < total, keeps it open-ended)
    int n; double p, noise;
    switch (testId) {
        case 1:  n=6;   p=0.90; noise=0.05; break;
        case 2:  n=20;  p=0.70; noise=0.08; break;
        case 3:  n=40;  p=0.55; noise=0.10; break;
        case 4:  n=60;  p=0.45; noise=0.12; break;
        case 5:  n=100; p=0.35; noise=0.10; break;
        case 6:  n=160; p=0.30; noise=0.08; break;
        case 7:  n=220; p=0.40; noise=0.15; break;
        case 8:  n=320; p=0.35; noise=0.12; break;
        case 9:  n=460; p=0.25; noise=0.10; break;
        case 10: n=600; p=0.35; noise=0.08; break;
        default: n=100; p=0.35; noise=0.10; break;
    }

    // hidden balanced partition P* over the module ids (random permutation, so it is
    // uncorrelated with the id-order reference split -> reference is a mediocre baseline).
    vector<int> planted(n, 0);
    {
        vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i;
        shuffle(perm.begin(), perm.end());
        for (int i = 0; i < n / 2; i++) planted[perm[i]] = 1;
    }

    // build wires over id pairs
    // type consistent with P*:  differ in P* -> coupling(0) (wants cut);
    //                           same  in P* -> bus(1)       (wants uncut).
    // with prob 'noise' the type is flipped, so P* no longer earns it.
    vector<array<int,4>> wires; // u v t w  (1-indexed)
    long long forcedU = 1, forcedV = n / 2 + 1; // 1-indexed, straddles the reference split
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // reserve the forced crossing wire; add it explicitly below (no duplicate)
            if (i == 0 && j == n / 2) continue;
            if (rnd.next() >= p) continue;
            int t = (planted[i] != planted[j]) ? 0 : 1;
            if (rnd.next() < noise) t ^= 1; // corrupt the type
            int w = rnd.next(1, 100);
            wires.push_back({i + 1, j + 1, t, w});
        }
    }
    // forced coupling wire (type 0) between module 1 and module n/2+1: the reference split
    // puts them in different cryostats, so it is cut and earns its value -> guarantees B > 0.
    wires.push_back({(int)forcedU, (int)forcedV, 0, rnd.next(50, 100)});

    // emit in shuffled order so ids do not leak the construction order
    shuffle(wires.begin(), wires.end());

    printf("%d %d\n", n, (int)wires.size());
    for (auto &e : wires) printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
