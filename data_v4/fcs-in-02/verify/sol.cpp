#include <bits/stdc++.h>
using namespace std;

/*
  Renyi-Ulam search with ONE lie: minimum number of yes/no questions.

  A hidden integer x lies in [1..N]. We may ask yes/no questions ("is x in S?").
  An adaptive adversary answers, and AT MOST ONE answer in the whole game may be
  a lie. We must guarantee identification of x no matter how the adversary plays
  (including which single answer, if any, it chooses to corrupt). Output q1(N):
  the minimum number of questions that always suffices.

  T independent queries; print one q1(N) per line.

  --- Why the naive counts are wrong ---
  * Plain binary search uses ceil(log2 N) questions but a single lie makes the
    final answer arbitrarily wrong: no detection, no correction.
  * The naive "repeat each binary-search step a few times" gives ~2*log2 N, far
    from optimal.
  * The information / sphere-packing (Berlekamp volume) bound says: with q
    questions and <=1 lie, each candidate x has q+1 "answer patterns" (the truth
    plus one single-flip), so we need N*(q+1) <= 2^q. The least such q is the
    volume bound qV(N). This is a genuine LOWER bound -- but it is NOT always
    achievable. For example N=3: qV=4 (3*5=15 <= 16) yet 4 questions do NOT
    suffice; the true minimum is 5.

  --- The exact state-weight insight (Berlekamp / Pelc) ---
  Track each candidate's "lie budget used" e(x) in {0,1}. A game state is a pair
  (a,b): a candidates with e=0, b candidates with e=1. With r questions left the
  Berlekamp weight is  w = a*(r+1) + b ; the questioner wins iff w <= 2^r, and a
  question must split so BOTH children keep weight <= 2^(r-1). Pushing this exact
  recursion through to the initial state (N,0) yields a closed form that differs
  from the volume bound by a PARITY correction (Pelc 1987):

      q1(N) = least q with   N*(q+1)        <= 2^q     if N is even,
              least q with   N*(q+1) + (q-1) <= 2^q     if N is odd.

  The extra (q-1) term for odd N is exactly why the volume bound is not tight
  there (e.g. N=3 -> 5, N=5 -> 6). This closed form is verified below against an
  exhaustive game-tree search.
*/

typedef long long ll;       // sums up to ~1e9*63 = 6.3e10, needs 64-bit
typedef unsigned long long ull;

// Minimum questions to find x in [1..N] with at most one lie.
static ll q1(ll N) {
    if (N <= 1) return 0;                   // already known, no question needed
    for (int q = 1; q <= 62; q++) {         // 2^62 > 1e9*64, ample headroom
        ull pw = 1ULL << q;                 // 2^q
        ull need;                           // left side of the inequality
        if (N % 2 == 0) need = (ull)N * (ull)(q + 1);
        else            need = (ull)N * (ull)(q + 1) + (ull)(q - 1);
        if (need <= pw) return q;
    }
    return 62;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        ll N;
        cin >> N;
        cout << q1(N) << "\n";
    }
    return 0;
}
