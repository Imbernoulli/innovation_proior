#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Generator for "Warehouse Gantry Restacking".
//
// testId is a difficulty/structure ladder (1 tiny .. 10 largest/adversarial):
//   * PLANTED (t2,t4,t5): goal keeps most bottom prefixes of the start (many
//     crates already settled) -> a good hidden structure the naive teardown
//     wastes; a settle-aware solver captures it.
//   * TRAP    (t3,t6): tall towers whose goal is the reversal -> the obvious
//     "tear everything down" is near-worst-case; smart direct placement helps.
//   * NEEDLE  (t5): mostly-correct config with a few displaced crates.
//   * ADVERSARIAL / ENVELOPE (t7..t10): independent random goals; t10 fills the
//     n=40 envelope with very tall towers.
//
// Layout: crates 1..n live on shelf positions [0..n-1]; staging area [n..2n-1]
// (= [n..p-1]) is always empty.  Prints:  n p H  then p start lines then p goal
// lines, each "k c_1..c_k" bottom-to-top.
// -----------------------------------------------------------------------------

int n, p, H;

vector<int> shuffled(int nn){
    vector<int> v(nn);
    for (int i = 0; i < nn; i++) v[i] = i + 1;
    for (int i = nn - 1; i > 0; i--) swap(v[i], v[rnd.next(0, i)]);
    return v;
}

// Distribute the crates in `perm` into K nonempty shelf columns (chosen from
// [0..n-1]); returns a length-p vector of stacks (bottom-to-top).
vector<vector<int>> buildConfig(const vector<int>& perm, int K){
    int m = (int)perm.size();
    K = max(1, min(K, m));
    vector<vector<int>> cols(p);
    // pick K distinct shelf columns
    vector<int> sh(n);
    for (int i = 0; i < n; i++) sh[i] = i;
    for (int i = n - 1; i > 0; i--) swap(sh[i], sh[rnd.next(0, i)]);
    vector<int> chosen(sh.begin(), sh.begin() + K);
    sort(chosen.begin(), chosen.end());
    // partition [0..m) into K contiguous nonempty groups
    set<int> cs;
    while ((int)cs.size() < K - 1) cs.insert(rnd.next(1, m - 1));
    vector<int> cut(cs.begin(), cs.end());
    cut.insert(cut.begin(), 0);
    cut.push_back(m);
    for (int i = 0; i < K; i++)
        for (int j = cut[i]; j < cut[i + 1]; j++)
            cols[chosen[i]].push_back(perm[j]);
    return cols;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    n = 3 + (int)llround(f * 37.0);      // 3 .. 40
    p = 2 * n;
    H = 4 + (int)llround(f * 16.0);      // 4 .. 20

    // start density (# nonempty shelf columns)
    int Kstart;
    switch (testId){
        case 1:  Kstart = max(1, n / 2);  break;
        case 2:  Kstart = n;              break;   // spread
        case 3:  Kstart = max(1, n / 6);  break;   // towers
        case 4:  Kstart = max(1, n / 2);  break;
        case 5:  Kstart = max(1, (2*n)/3);break;
        case 6:  Kstart = max(1, n / 8);  break;   // tall towers
        case 7:  Kstart = max(1, n / 2);  break;
        case 8:  Kstart = max(1, n / 3);  break;
        case 9:  Kstart = max(1, n / 2);  break;
        default: Kstart = max(1, n / 10); break;   // t10 very tall towers
    }

    vector<vector<int>> start = buildConfig(shuffled(n), Kstart);
    vector<vector<int>> goal;

    bool planted  = (testId == 2 || testId == 4 || testId == 5);
    bool reversal = (testId == 3 || testId == 6);

    if (planted){
        goal = start;
        int t = (testId == 5) ? max(1, n / 8) : max(1, n / 4);  // needle t5 moves fewer
        vector<int> pool;
        for (int r = 0; r < t; r++){
            vector<int> nonempty;
            for (int c = 0; c < n; c++) if (!goal[c].empty()) nonempty.push_back(c);
            if (nonempty.empty()) break;
            int col = nonempty[rnd.next(0, (int)nonempty.size() - 1)];
            pool.push_back(goal[col].back());
            goal[col].pop_back();
        }
        for (int x : pool){ int col = rnd.next(0, n - 1); goal[col].push_back(x); }
    } else if (reversal){
        goal.assign(p, {});
        for (int c = 0; c < p; c++){ goal[c] = start[c]; reverse(goal[c].begin(), goal[c].end()); }
    } else {
        int Kgoal;
        switch (testId){
            case 1:  Kgoal = max(1, n / 2);  break;
            case 7:  Kgoal = max(1, n / 2);  break;
            case 8:  Kgoal = n;              break;   // spread goal
            case 9:  Kgoal = max(1, n / 4);  break;
            default: Kgoal = max(1, n / 10); break;   // tower goal
        }
        goal = buildConfig(shuffled(n), Kgoal);
    }

    // guarantee start != goal
    if (start == goal){
        int i = -1;
        for (int c = 0; c < p; c++) if (!goal[c].empty()){ i = c; break; }
        int j = (i + 1) % n;
        goal[j].push_back(goal[i].back());
        goal[i].pop_back();
    }

    printf("%d %d %d\n", n, p, H);
    for (int c = 0; c < p; c++){
        printf("%d", (int)start[c].size());
        for (int x : start[c]) printf(" %d", x);
        printf("\n");
    }
    for (int c = 0; c < p; c++){
        printf("%d", (int)goal[c].size());
        for (int x : goal[c]) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
