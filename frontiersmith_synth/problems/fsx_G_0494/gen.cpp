// Generator for "Cooperative Ribozyme Folding with Aptamer Pockets".
// testId 1..10 is a difficulty/structure ladder: small -> large, with PLANTED helices
// (a good structure exists), a TRAP (parenthesis-greedy grabs local noise pairs and
// misses the real helices), and a NEEDLE case (one huge-value pocket amid noise).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static char comp_of(char c) {
    switch (c) { case 'A': return 'U'; case 'U': return 'A';
                 case 'C': return 'G'; case 'G': return 'C'; }
    return 'A';
}
static char randBase() { const char* b = "ACGU"; return b[rnd.next(4)]; }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int Larr[11] = {0, 24, 40, 64, 100, 160, 240, 360, 520, 720, 1000};
    int L = Larr[t];

    // GC-content knob: some tests GC-rich (denser, stronger stacking), some AU-rich.
    // gcBias in [0,100]: probability (percent) a random background base is C or G.
    int gcBias = 50;
    if (t == 3 || t == 4) gcBias = 62;      // trap: complementary-rich noise
    if (t == 6) gcBias = 40;

    string seq(L, 'A');
    for (int i = 0; i < L; i++) {
        if (rnd.next(100) < gcBias) seq[i] = (rnd.next(2) ? 'C' : 'G');
        else seq[i] = (rnd.next(2) ? 'A' : 'U');
    }

    bool needle = (t == 7);

    // number of planted hairpins
    int H = max(1, L / 40);
    if (needle) H = 1;
    if (t == 3) H = max(1, L / 90);         // trap: few real helices, lots of noise

    vector<pair<int,int>> pockets;          // (pos0, val)

    int cursor = rnd.next(0, min(6, L / 8 + 1));
    int planted = 0;
    while (planted < H) {
        int s = rnd.next(3, 6);             // arm length (stacked rungs)
        int loop = rnd.next(4, 8);          // hairpin loop length
        int span = 2 * s + loop;
        if (cursor + span - 1 >= L) break;
        // build the two complementary arms (Watson-Crick -> weight 2 or 3, all stacked)
        for (int k = 0; k < s; k++) {
            char lb = randBase();
            seq[cursor + k]           = lb;
            seq[cursor + span - 1 - k] = comp_of(lb);
        }
        // loop bases stay as-is (random); place a pocket in the loop middle
        int lp = cursor + s + loop / 2;
        int val = needle ? rnd.next(38, 50) : rnd.next(5, 12);
        pockets.push_back({lp, val});
        cursor += span + rnd.next(2, 15);   // gap before the next planted hairpin
        planted++;
    }

    // noise pockets (needle/low-value distractors)
    int nn = max(1, L / 50);
    for (int k = 0; k < nn; k++) {
        int p = rnd.next(0, L - 1);
        int v = rnd.next(1, 3);
        pockets.push_back({p, v});
    }

    // dedupe pockets by position (keep the larger value)
    map<int,int> mp;
    for (auto& pk : pockets) {
        auto it = mp.find(pk.first);
        if (it == mp.end() || it->second < pk.second) mp[pk.first] = pk.second;
    }

    printf("%d\n", L);
    printf("%s\n", seq.c_str());
    printf("%d\n", (int)mp.size());
    for (auto& kv : mp) printf("%d %d\n", kv.first + 1, kv.second);
    return 0;
}
