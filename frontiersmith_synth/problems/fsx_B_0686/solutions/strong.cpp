// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// The insight: redefine the similarity metric before running the same class of algorithm.
// Overlap between the string built so far and a candidate pattern is computed with
// WILDCARD-COMPATIBLE matching (position i compatible iff either side is '?' or they are
// literally equal), exposing overlaps that plain literal comparison cannot see at all. Each
// merge also resolves any still-open '?' slot in the overlap zone against the candidate's
// concrete letters (checked live against the CURRENT accumulated string, never a stale
// pairwise snapshot, so no resolution can ever be contradicted by a later merge). Patterns
// may attach on either end of the growing string (a candidate can extend the chain forward
// OR backward), and we try several different starting patterns, keeping the shortest result
// -- a cheap way to dodge a single bad early commitment.

static bool wildcardCompatible(char a, char b) { return a == '?' || b == '?' || a == b; }

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

    int numStarts = min(n, 6);
    string best;
    for (int s = 0; s < numStarts; s++) {
        int start = (int)((long long)s * n / numStarts); // spread starts across the list
        string cand = buildChain(pat, start, wildcardCompatible);
        if (best.empty() || cand.size() < best.size()) best = cand;
    }
    printf("%d\n%s\n", (int)best.size(), best.c_str());
    return 0;
}
