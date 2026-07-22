#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker for "Seasonal Spectrum Assignment".
// Input : N C M  then M lines  u v w1..w7
// Output: N ints ch_1..ch_N, ch_i in [1,C]
// F = sum of TWO LARGEST per-season conflict totals (over same-channel pairs).
// B = same objective for the fixed round-robin baseline ch_i = ((i-1)%C)+1.
// ratio = min(1000, 100*B/max(1,F)) / 1000.

ll topTwoSum(ll S[8]) {
    ll a = -1, b = -1;
    for (int s = 1; s <= 7; s++) {
        if (S[s] > a) { b = a; a = S[s]; }
        else if (S[s] > b) { b = S[s]; }
    }
    if (b < 0) b = 0;
    return a + b;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int C = inf.readInt();
    int M = inf.readInt();

    vector<int> eu(M), ev(M);
    vector<array<int,8>> ew(M);
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt(1, N, "u");
        ev[i] = inf.readInt(1, N, "v");
        if (eu[i] == ev[i]) quitf(_fail, "generator produced a self-loop");
        array<int,8> w{};
        for (int s = 1; s <= 7; s++) w[s] = inf.readInt(0, 1000000, "w");
        ew[i] = w;
    }

    // participant output
    vector<int> ch(N + 1, 0);
    for (int i = 1; i <= N; i++) {
        ch[i] = ouf.readInt(1, C, "channel");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after N channel assignments");

    ll S[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < M; i++) {
        if (ch[eu[i]] == ch[ev[i]]) {
            for (int s = 1; s <= 7; s++) S[s] += ew[i][s];
        }
    }
    ll F = topTwoSum(S);

    // internal baseline: round-robin assignment, ignores all weights
    vector<int> chB(N + 1, 0);
    for (int i = 1; i <= N; i++) chB[i] = ((i - 1) % C) + 1;
    ll SB[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < M; i++) {
        if (chB[eu[i]] == chB[ev[i]]) {
            for (int s = 1; s <= 7; s++) SB[s] += ew[i][s];
        }
    }
    ll B = topTwoSum(SB);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
