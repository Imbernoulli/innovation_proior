// gen.cpp -- fsx_A_0996 : Balanced Word Maximal Factor Richness
#include "testlib.h"
#include <vector>
#include <string>

using namespace std;

// Per-test table: a (alphabet size), r[] (ratio out of 6 -- freq_c = r_c * L/6),
// L (word length, multiple of 6), K (max factor length window), tol_w (window
// balance tolerance), tol_g (global frequency tolerance), w_pal (richness weight).
struct T {
    int a;
    vector<long long> r;   // sums to 6
    long long L;
    int K;
    int tol_w, tol_g, w_pal;
};

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    vector<T> table = {
        /*1*/  {2, {3,3},     48,   8, 1, 2, 2},   // tiny, balanced -> example scale
        /*2*/  {2, {2,4},    120,  14, 1, 2, 3},
        /*3*/  {2, {3,3},    240,  20, 1, 2, 4},   // planted: exact 1:1 balance
        /*4*/  {2, {1,5},    180,  14, 1, 2, 3},   // needle: heavy skew, small scale
        /*5*/  {2, {2,4},    360,  20, 1, 2, 4},
        /*6*/  {2, {3,3},    798,  30, 1, 2, 5},   // trap: fixed-precision ratio drifts at scale
        /*7*/  {2, {2,4},    840,  28, 1, 2, 5},   // trap
        /*8*/  {2, {1,5},   1398,  36, 1, 2, 6},   // trap: skew + large scale
        /*9*/  {2, {1,5},   1596,  40, 1, 2, 7},   // trap
        /*10*/ {2, {3,3},   3000,  48, 1, 2, 8},   // largest, fills envelope
    };

    T t = table[testId - 1];
    long long block = t.L / 6;
    vector<long long> freq(t.a);
    long long sum = 0;
    for (int c = 0; c < t.a; c++) { freq[c] = t.r[c] * block; sum += freq[c]; }
    // sanity (should always hold by table construction)
    if (sum != t.L) freq[t.a - 1] += (t.L - sum);

    printf("%d %lld %d %d %d %d\n", t.a, t.L, t.K, t.tol_w, t.tol_g, t.w_pal);
    for (int c = 0; c < t.a; c++) printf("%lld%c", freq[c], c + 1 == t.a ? '\n' : ' ');

    return 0;
}
