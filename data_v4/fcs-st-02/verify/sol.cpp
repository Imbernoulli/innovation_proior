#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;          // empty input -> treat as n = 0
    vector<int> pi(n);
    for (int i = 0; i < n; i++) cin >> pi[i];

    // ---- Quick structural rejects (cheap, before construction) -------------
    // pi[i] is a length in [0, i]; pi[0] must be 0; a border cannot grow by more
    // than one per step (pi[i] <= pi[i-1] + 1). These are necessary but NOT
    // sufficient -- transitive border composition is the real test, handled by
    // the verify pass below.
    bool bad = false;
    for (int i = 0; i < n; i++) {
        if (pi[i] < 0 || pi[i] > i) { bad = true; break; }
        if (i > 0 && pi[i] > pi[i - 1] + 1) { bad = true; break; }
    }
    if (bad) { cout << -1 << "\n"; return 0; }

    // ---- Reconstruct a candidate string ------------------------------------
    // Letters are 0..25 mapped to 'a'..'z'. Rule:
    //   pi[i] > 0  : the border of length pi[i] forces s[i] == s[pi[i]-1].
    //   pi[i] == 0 : s[i] must NOT continue any border of s[0..i-1]. Collect the
    //                set of "next" characters that every border (lengths
    //                pi[i-1], pi[pi[i-1]-1], ...) would demand, and pick any
    //                letter outside that set. With an alphabet of 26 there is
    //                always a free letter (the chain has < n forbidden chars,
    //                and a 0 only forbids the distinct chain heads).
    string s(n, 'a');
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            s[0] = 'a';                 // position 0 is unconstrained (pi[0] == 0)
        } else if (pi[i] > 0) {
            s[i] = s[pi[i] - 1];        // border of length pi[i] forces this char
        } else {
            // forbidden = chars that would extend some border of s[0..i-1].
            // Walk the border chain pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0; each
            // border of length k could be extended only by s[k], so s[i] must
            // avoid every such s[k], else pi[i] would be positive.
            bool used[26] = {false};
            int k = pi[i - 1];
            while (true) {
                used[(int)(s[k] - 'a')] = true;
                if (k == 0) break;
                k = pi[k - 1];
            }
            int c = 0;
            while (c < 26 && used[c]) c++;
            if (c == 26) { cout << -1 << "\n"; return 0; } // unreachable for valid pi
            s[i] = char('a' + c);
        }
    }

    // ---- Verify: recompute the prefix function of s and compare ------------
    // This is what turns "construct" into a sound feasibility test: if the input
    // pi is not realizable, the recomputed prefix function will disagree.
    vector<int> chk(n, 0);
    for (int i = 1; i < n; i++) {
        int k = chk[i - 1];
        while (k > 0 && s[i] != s[k]) k = chk[k - 1];
        if (s[i] == s[k]) k++;
        chk[i] = k;
    }
    for (int i = 0; i < n; i++) {
        if (chk[i] != pi[i]) { cout << -1 << "\n"; return 0; }
    }

    cout << s << "\n";
    return 0;
}
