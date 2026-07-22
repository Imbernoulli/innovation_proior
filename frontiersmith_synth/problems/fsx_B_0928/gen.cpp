#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct Ship { long long r, v, arr, dep; };

int W, H, K;
vector<long long> T;
vector<Ship> ships;

// tide grid: T[0]=0, then K-1 increasing gaps in [gapLo,gapHi]
void makeTicks(int gapLo, int gapHi) {
    T.assign(K, 0);
    T[0] = 0;
    for (int i = 1; i < K; i++) T[i] = T[i - 1] + rnd.next(gapLo, gapHi);
}

// Add a ship with a GUARANTEED feasible window: pick a 0-based entry tick
// index in [entryLo, entryHi] (clamped to <= K-2), then a 0-based exit tick
// index >= entry + minSpan (clamped to <= K-1); arr = T[entry], dep = T[exit].
void addShip(long long r, long long v, int entryLo, int entryHi, int minSpan) {
    entryHi = min(entryHi, K - 2);
    entryLo = max(0, min(entryLo, entryHi));
    int entry = rnd.next(entryLo, entryHi);
    int exitLo = min(entry + max(1, minSpan), K - 1);
    int exit_ = rnd.next(exitLo, K - 1);
    ships.push_back({r, v, T[entry], T[exit_]});
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    ships.clear();

    if (t == 1) {
        // tiny hand-checkable sanity case (matches the statement's worked example)
        W = 10; H = 10; K = 3;
        T = {0, 5, 10};
        ships.push_back({2, 30, 0, 10});
        ships.push_back({4, 100, 4, 10});
    } else if (t == 2) {
        // small random mix
        W = 25; H = 25; K = 4;
        makeTicks(6, 12);
        int n = 10;
        for (int i = 0; i < n; i++) {
            long long r = rnd.next(1, 4);
            long long v = rnd.next(10, 120);
            addShip(r, v, 0, K - 2, rnd.next(1, 2));
        }
    } else if (t == 3) {
        // TRAP: one big high-value ship needs the exact centre for one specific
        // epoch; many small cheap ships are available the whole day. A greedy
        // that anchors ships full-span, arrival-first, at the basin centre
        // permanently squats the only spot the big ship can ever use.
        W = 30; H = 30; K = 4;
        makeTicks(8, 14);
        addShip(11, 500, 1, 1, 1);           // the one big ship, needs epoch index 1 only
        int n = 16;
        for (int i = 0; i < n; i++) {
            long long v = rnd.next(20, 40);
            addShip(2, v, 0, 0, K - 1);        // full horizon (entry=tick0, exit=last tick)
        }
    } else if (t == 4) {
        // mixed random, medium scale
        W = 60; H = 60; K = 5;
        makeTicks(10, 20);
        int n = 30;
        for (int i = 0; i < n; i++) {
            long long r = rnd.next(1, 15);
            long long v = rnd.next(10, 300);
            addShip(r, v, 0, K - 2, rnd.next(1, K - 1));
        }
    } else if (t == 5) {
        // TRAP + NEEDLE: a sea of full-span squatters and unrelated decoys hide
        // one very valuable large ship that needs one specific late epoch.
        W = 50; H = 50; K = 6;
        makeTicks(10, 18);
        int nSmall = 30;
        for (int i = 0; i < nSmall; i++) {
            long long v = rnd.next(15, 35);
            addShip(2, v, 0, 0, K - 1);
        }
        int nDecoy = 12;
        for (int i = 0; i < nDecoy; i++) {
            long long r = rnd.next(3, 6);
            long long v = rnd.next(20, 80);
            addShip(r, v, 0, K - 2, rnd.next(1, 2));
        }
        addShip(17, 900, K - 2, K - 2, 1);   // the needle: needs the second-to-last epoch
    } else if (t == 6) {
        // dense general adversarial mix; includes one radius that can never fit,
        // to stress-test rejection of physically infeasible ships
        W = 80; H = 80; K = 5;
        makeTicks(14, 24);
        int n = 60;
        for (int i = 0; i < n; i++) {
            long long r = rnd.next(1, 20);
            long long v = rnd.next(10, 500);
            addShip(r, v, 0, K - 2, rnd.next(1, K - 1));
        }
        ships.push_back({(long long) W, 999, 0, T[K - 1]}); // radius == W: never fits
    } else if (t == 7) {
        // TRAP, doubled: two big ships wanting the centre at two DIFFERENT
        // epochs (so they don't conflict with each other), buried under
        // full-span squatters that conflict with both if placed first.
        W = 40; H = 40; K = 6;
        makeTicks(9, 15);
        addShip(15, 600, 1, 1, 1);
        addShip(15, 650, K - 2, K - 2, 1);
        int n = 22;
        for (int i = 0; i < n; i++) {
            long long v = rnd.next(20, 45);
            addShip(2, v, 0, 0, K - 1);
        }
    } else if (t == 8) {
        // large-scale general random stress test (fills the constraint envelope)
        W = 200; H = 200; K = 7;
        makeTicks(20, 40);
        int n = 300;
        for (int i = 0; i < n; i++) {
            long long r = rnd.next(1, 60);
            long long v = rnd.next(1, 1000);
            addShip(r, v, 0, K - 2, rnd.next(1, K - 1));
        }
    } else if (t == 9) {
        // TRAP, tripled, more epochs: three big ships each need a distinct
        // epoch spread across the schedule; a large squatter population must
        // be routed around all three simultaneously.
        W = 45; H = 45; K = 8;
        makeTicks(8, 13);
        addShip(16, 700, 1, 1, 1);
        addShip(16, 750, 3, 3, 1);
        addShip(16, 800, 5, 5, 1);
        int n = 35;
        for (int i = 0; i < n; i++) {
            long long v = rnd.next(15, 40);
            addShip(2, v, 0, 0, K - 1);
        }
    } else {
        // t == 10: largest combined stress test -- trap structure, a needle,
        // and heavy random noise together, near the top of the size envelope.
        W = 300; H = 300; K = 9;
        makeTicks(30, 55);
        int nBig = 8;
        for (int i = 0; i < nBig; i++) {
            long long v = 700 + rnd.next(0, 300);
            addShip(90, v, i % (K - 1), i % (K - 1), 1);
        }
        int nSquat = 150;
        for (int i = 0; i < nSquat; i++) {
            long long v = rnd.next(15, 60);
            addShip(rnd.next(2, 5), v, 0, 0, K - 1);
        }
        int nNoise = 250;
        for (int i = 0; i < nNoise; i++) {
            long long r = rnd.next(1, 80);
            long long v = rnd.next(1, 1000);
            addShip(r, v, 0, K - 2, rnd.next(1, K - 1));
        }
    }

    printf("%d %d %d\n", W, H, K);
    for (int i = 0; i < K; i++) printf("%lld%c", T[i], i + 1 == K ? '\n' : ' ');
    printf("%d\n", (int) ships.size());
    for (auto &s : ships) printf("%lld %lld %lld %lld\n", s.r, s.v, s.arr, s.dep);
    return 0;
}
