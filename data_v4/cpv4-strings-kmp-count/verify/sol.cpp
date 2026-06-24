#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;            // empty / no input -> nothing to print
    int n = (int)s.size();

    // KMP failure function: pi[i] = length of the longest proper prefix of
    // s[0..i] that is also a suffix of s[0..i].  pi[0] = 0.
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // cnt[L] = number of occurrences (overlaps allowed) of the length-L prefix
    // of s inside s, for L = 1..n.  We index cnt by length, cnt has size n+1.
    //
    // Step 1: every end position i contributes one occurrence of the prefix of
    // length pi[i] (the longest prefix-suffix ending at i).  Length 0 carries no
    // prefix, so we only seed positive lengths.
    vector<long long> cnt(n + 1, 0);
    for (int i = 0; i < n; i++)
        if (pi[i] > 0) cnt[pi[i]]++;

    // Step 2: if the length-L prefix occurs, then every shorter prefix that is a
    // border of it (length pi[L-1], then pi[pi[L-1]-1], ...) also occurs at each
    // of those positions.  Push counts down the failure chain.  Process lengths
    // from long to short so each cnt[L] is final before it is propagated.
    for (int L = n; L >= 1; L--) {
        int b = pi[L - 1];                // longest border length of prefix of length L
        if (b > 0) cnt[b] += cnt[L];
    }

    // Step 3: the length-L prefix also occurs once as the whole prefix itself
    // (the trivial occurrence at position 0), which the failure function never
    // counts.  Add it exactly once per length.
    for (int L = 1; L <= n; L++) cnt[L] += 1;

    for (int L = 1; L <= n; L++) {
        cout << cnt[L];
        cout << (L == n ? '\n' : ' ');
    }
    return 0;
}
