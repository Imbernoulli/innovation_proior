// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious first attempt: build one Hamiltonian chain by nearest-neighbor, always
// attaching the still-unused pattern with the largest overlap against either end of the
// string built so far -- but "overlap" here is computed with PLAIN LITERAL character
// equality, treating '?' as an ordinary character that only matches another '?'. This is
// exactly the classical shortest-common-superstring greedy-overlap heuristic applied
// naively to strings that contain wildcards: it never realizes that a '?' can stand in for
// any concrete letter, so on heavily wildcard-masked planted clusters it sees almost no
// usable overlap and the resulting S stays close to the do-nothing baseline.

static bool literalCompatible(char a, char b) { return a == b; }

// max k: last k of sbuf compatible with first k of q (q attaches on the right).
static int overlapRight(const vector<char> &sbuf, const string &q, bool (*compat)(char, char)) {
    int sn = (int)sbuf.size(), qn = (int)q.size();
    int maxK = min(sn, qn);
    for (int k = maxK; k >= 0; k--) {
        bool ok = true;
        int base = sn - k;
        for (int j = 0; j < k; j++)
            if (!compat(sbuf[base + j], q[j])) { ok = false; break; }
        if (ok) return k;
    }
    return 0;
}

// max k: last k of q compatible with first k of sbuf (q attaches on the left).
static int overlapLeft(const vector<char> &sbuf, const string &q, bool (*compat)(char, char)) {
    int sn = (int)sbuf.size(), qn = (int)q.size();
    int maxK = min(sn, qn);
    for (int k = maxK; k >= 0; k--) {
        bool ok = true;
        int qbase = qn - k;
        for (int j = 0; j < k; j++)
            if (!compat(q[qbase + j], sbuf[j])) { ok = false; break; }
        if (ok) return k;
    }
    return 0;
}

static string buildChain(const vector<string> &pat, int start, bool (*compat)(char, char)) {
    int n = (int)pat.size();
    vector<bool> used(n, false);
    vector<char> sbuf(pat[start].begin(), pat[start].end());
    used[start] = true;
    int remaining = n - 1;
    while (remaining > 0) {
        int bestJ = -1, bestK = -1;
        bool bestRight = true;
        for (int j = 0; j < n; j++) {
            if (used[j]) continue;
            int kr = overlapRight(sbuf, pat[j], compat);
            if (kr > bestK) { bestK = kr; bestJ = j; bestRight = true; }
            int kl = overlapLeft(sbuf, pat[j], compat);
            if (kl > bestK) { bestK = kl; bestJ = j; bestRight = false; }
        }
        const string &q = pat[bestJ];
        int k = bestK;
        if (bestRight) {
            int base = (int)sbuf.size() - k;
            for (int j = 0; j < k; j++)
                if (sbuf[base + j] == '?' && q[j] != '?') sbuf[base + j] = q[j];
            for (int j = k; j < (int)q.size(); j++) sbuf.push_back(q[j]);
        } else {
            int qn = (int)q.size();
            for (int j = 0; j < k; j++) {
                int idxQ = qn - k + j;
                if (sbuf[j] == '?' && q[idxQ] != '?') sbuf[j] = q[idxQ];
            }
            vector<char> pre(q.begin(), q.begin() + (qn - k));
            sbuf.insert(sbuf.begin(), pre.begin(), pre.end());
        }
        used[bestJ] = true;
        remaining--;
    }
    for (char &c : sbuf) if (c == '?') c = 'A';
    return string(sbuf.begin(), sbuf.end());
}

int main() {
    int n;
    scanf("%d", &n);
    vector<string> pat(n);
    char buf[64];
    for (int i = 0; i < n; i++) { scanf("%s", buf); pat[i] = buf; }

    string S = buildChain(pat, 0, literalCompatible);
    printf("%d\n%s\n", (int)S.size(), S.c_str());
    return 0;
}
