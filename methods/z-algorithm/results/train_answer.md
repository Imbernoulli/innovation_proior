I want a single number at every starting position of a string $s$ of length $n$: how many characters from that position match the front of the string. Writing $z[i]$ for that count — the length of the longest substring starting at $i$ that is also a prefix of $s$, equivalently the longest common prefix of $s$ and the suffix $s[i:]$ — I want the whole array $z$ in linear time, and then a pattern search whose total work is proportional to $\mathrm{len}(\text{pattern}) + \mathrm{len}(\text{text})$. Since $z[0]$ would just measure the string against itself and carries no information, I set $z[0] = 0$ by convention. The literal computation is obvious: at each $i$, compare $s[0]$ with $s[i]$, then $s[1]$ with $s[i+1]$, and keep going until the characters differ or the suffix runs out. It is correct, but it is wasteful. On a string like $\texttt{"aaaaa"}$ it rescans almost the same run again and again — $z[1]$ scans four characters, $z[2]$ scans three, then two, then one — and the sum is quadratic. The waste is not subtle: each position discovers a long prefix match and then throws that knowledge away before moving on. The whole problem is to stop discarding what I just learned.

The method is the Z-algorithm. The idea is to carry, as I sweep left to right, the single most useful fact discovered so far: the copied prefix block that reaches farthest right. A nonzero $z[l]$ certifies that $s[l : l + z[l]]$ is a verbatim copy of the prefix, so if that copy reaches through index $r$ then $s[l..r]$ equals $s[0..r-l]$. Among all such certified copies I keep the one with the largest $r$, storing its endpoints as $l$ and $r$. The point $r$ is a frontier: everything up to $r$ is mirror-known, and beyond $r$ I have no information yet. When I come to compute $z[i]$, there are two regimes. If $i > r$, the frontier does not touch $i$, so there is nothing to reuse; I start at zero and compare forward directly. If instead $i \le r$, then $i$ sits inside the certified copy, and because $s[l..r]$ is a copy of the prefix, the characters from $i$ through $r$ equal the characters from $i-l$ through $r-l$ near the front of the string. The mirror index $i-l$ is strictly smaller than $i$, so $z[i-l]$ is already computed, and that mirrored value is exactly what should transfer to $i$.

The one subtlety — and the load-bearing design choice — is that the mirror is only trustworthy while I stay inside the certified region. If $z[i-l]$ is shorter than the number of characters still certified, $r - i + 1$, then the mismatch that ended the mirrored match happens strictly inside the copy, so the identical mismatch happens at $i$, and $z[i] = z[i-l]$ exactly, with no character comparison at all. But if $z[i-l]$ reaches the edge of the certified region or would run past it, the mirror can only promise a match up to $r$; past the frontier the copy says nothing, and the true match at $i$ might continue further. So I cannot simply copy $z[i-l]$ — I must clamp it. The seed is therefore

$$z[i] = \min\bigl(z[i-l],\; r - i + 1\bigr),$$

and then in every case I run the same forward scan starting from that seed,

```cpp
while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
```

so when $i > r$ the seed is $0$ and the scan starts at $i$, when the mirror is strictly inside the seed already equals the answer and the loop's first test fails immediately, and when the mirror reaches the edge the seed is $r - i + 1$ and the scan resumes exactly at the first unknown character past the frontier. Whichever branch ran, after the scan I check whether this match extends the frontier, and if $i + z[i] - 1 > r$ I update $l, r = i, i + z[i] - 1$. The reason to keep the farthest-reaching copy rather than, say, the most recent one is precisely that the frontier $r$ is what bounds the work: a copy that reaches farther right gives more positions a free mirror and pushes the no-information boundary further out.

That clamp-and-extend structure is what buys linear time. Every successful character comparison happens at a position beyond the current $r$: when $i > r$ the scan begins at $i$ itself; when $i \le r$ with $z[i-l] \ge r - i + 1$ the first unchecked comparison is at $r+1$; and when $z[i-l] < r-i+1$ no comparison runs at all, because the mismatch is already certified — were that mirrored position to actually match further, $z[i-l]$ would have been undercounted, a contradiction. So every comparison that succeeds advances the frontier $r$ rightward, and $r$ only ever increases and never exceeds $n-1$. The total number of successful comparisons across the entire run is therefore at most $n$; adding the one terminating (failing) comparison per position keeps the whole Z-array computation $O(n)$. The amortization is paid by a single monotone endpoint.

With the Z-array in hand, pattern matching is almost free. An occurrence of $\texttt{pattern}$ at text index $j$ means $\texttt{pattern}$ and $\texttt{text[j:]}$ share a common prefix of length exactly $\mathrm{len}(\texttt{pattern})$ — which is exactly what the Z-array reports if I make the pattern the prefix of a longer sequence. I build $\texttt{list(pattern)} + [\texttt{separator}] + \texttt{list(text)}$, where $\texttt{separator}$ is a fresh sentinel that cannot equal any text element. The cleanest choice here is a brand-new $\texttt{object()}$ rather than assuming some character is absent from the text: a fresh object compares unequal to everything in the input by identity, so any prefix match that starts in the text region must stop at the separator after at most $\mathrm{len}(\texttt{pattern})$ characters and can never run through it. With $m = \mathrm{len}(\texttt{pattern})$, the text position $j$ lands at combined index $j + m + 1$, and a Z-value equal to $m$ at that index is exactly an occurrence — no more, no less, because the value can equal $m$ only when the full pattern matches and cannot exceed $m$ because of the separator. This reports overlapping matches naturally and runs in $O(\mathrm{len}(\texttt{pattern}) + \mathrm{len}(\texttt{text}))$, dominated by the single Z-array pass over the combined sequence. The empty-pattern case is handled separately, returning every index from $0$ through $\mathrm{len}(\texttt{text})$ inclusive, since an empty pattern occurs at every position.

Packaged as a single self-contained program, it reads the text $s$ on the first line of stdin and the pattern $p$ on the second, prints the Z-array of $s$, then prints the start indices in $s$ where $p$ occurs. The one detail that does not transcribe verbatim is the sentinel: a C++ `char` cannot be a fresh identity-distinct `object()`, so the separator becomes a control byte `'\1'` assumed absent from the inputs (any value outside the input alphabet works), playing exactly the same role.

```cpp
// Reads two lines from stdin: the text s on the first line and the pattern p on
// the second. Prints the Z-array of s (space-separated, with z[0]=0), then the
// sorted start indices in s where p occurs (overlaps included; empty line if none).
#include <bits/stdc++.h>
using namespace std;

// Z-array of s: z[i] is the length of the longest prefix of s that also starts
// at position i. z[0] is 0 by convention. Runs in O(n).
vector<int> z_function(const string &s) {
    int n = (int)s.size();
    vector<int> z(n, 0);
    int l = 0, r = 0;
    for (int i = 1; i < n; ++i) {
        if (i <= r) z[i] = min(z[i - l], r - i + 1);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
        if (i + z[i] - 1 > r) { l = i; r = i + z[i] - 1; }
    }
    return z;
}

// Sorted start indices i in text where pattern occurs (overlaps included).
// Builds pattern + sentinel + text and reports positions whose Z-value equals
// the pattern length. Total work O(len(pattern) + len(text)).
vector<int> find_occurrences(const string &text, const string &pattern) {
    int m = (int)pattern.size();
    vector<int> occ;
    if (m == 0) {                       // empty pattern occurs at every index
        for (int i = 0; i <= (int)text.size(); ++i) occ.push_back(i);
        return occ;
    }
    // '\1' is a sentinel assumed absent from the inputs; pick any value outside
    // the alphabet of text and pattern if that assumption does not hold.
    string combined = pattern + '\1' + text;
    vector<int> z = z_function(combined);
    for (int j = 0; j < (int)text.size(); ++j)
        if (z[j + m + 1] == m) occ.push_back(j);
    return occ;
}

int main() {
    string s, p;
    getline(cin, s);
    getline(cin, p);

    vector<int> z = z_function(s);
    for (size_t i = 0; i < z.size(); ++i) cout << (i ? " " : "") << z[i];
    cout << "\n";

    vector<int> occ = find_occurrences(s, p);
    for (size_t i = 0; i < occ.size(); ++i) cout << (i ? " " : "") << occ[i];
    cout << "\n";
    return 0;
}
```
