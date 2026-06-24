#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; } // empty input -> score 0
    int n = (int)s.size();

    // KMP prefix function: pi[i] = length of the longest proper border of s[0..i].
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // occ[len] = number of times the length-`len` prefix occurs as a substring of s.
    // Every border of length `pi[i]` ending at i is one such occurrence; chase the
    // border chain by propagating counts from longer borders to shorter ones, then
    // add 1 to every prefix length for its "own" occurrence at the start.
    vector<long long> occ(n + 1, 0);
    for (int i = 0; i < n; i++) occ[pi[i]]++;
    for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];
    for (int len = 1; len <= n; len++) occ[len]++;

    // Self-similarity score = total number of (prefix, occurrence) incidences.
    long long score = 0;
    for (int len = 1; len <= n; len++) score += occ[len];

    cout << score << "\n";
    return 0;
}
