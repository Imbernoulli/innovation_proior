A string $s$ arrives one character at a time, and at every moment I want the number of *distinct* non-empty palindromic substrings of the prefix read so far, updated the instant a new character lands. The naive approach is correct but unusable: there are $\Theta(n^2)$ substrings $s[l..r]$, and testing each against its reverse and dropping the palindromes into a set drifts toward $O(n^3)$ if I compare characters directly. Rolling hashes or precomputed palindrome radii cut the per-test cost, but I am still treating distinct palindromes as a quadratic candidate pool, and the online setting is even worse in spirit, because each appended character seems to demand a fresh scan over a large suffix-ending family. The way out is to stop counting substring *positions* and ask whether the distinct palindromic *strings* form a small family. The example $a^n$ is the hint: it is saturated with palindromic occurrences, yet its distinct palindromes are exactly $a, aa, aaa, \ldots, a^n$, only $n$ of them. That suggests the true bound is $n$, not $n^2$, and that a linear-size online object is possible.

The bound has to be proved before anything is built on it. Suppose I have processed $s'$ and append $c$ to form $s = s'c$. Any genuinely new palindrome must end at the new last position; list the left endpoints of the palindromes ending there as $l_1 < l_2 < \cdots < l_k$, so $s[l_1..|s|]$ is the longest such end-palindrome and the rest are shorter. For any $i \ge 2$, the palindrome $s[l_i..|s|]$ is a suffix of the longer palindrome $s[l_1..|s|]$; mirror symmetry inside that longer palindrome carries this suffix onto the equal-length prefix $s[l_1..\,l_1+|s|-l_i]$, and since $s[l_i..|s|]$ is itself a palindrome, reversing the mirrored copy reproduces the same string, so the two strings are equal. The right endpoint of that earlier occurrence is $l_1+|s|-l_i < |s|$, so every end-palindrome except possibly the longest already appeared wholly inside $s'$. Appending one character therefore creates at most one new distinct palindrome, and when it does, that palindrome is the longest palindromic suffix of the new string. The one-character base case then forces at most $n$ distinct palindromes in a length-$n$ string, and $a^n$ shows the bound is tight.

I propose the palindromic tree (an eertree) as the online structure that realizes this bound. The idea is to store exactly one node per distinct palindrome and to drive the whole computation off two relations between nodes. The first relation is structural: a palindrome of length at least $2$ is always $t = c\,u\,c$ for a single character $c$ and an inner palindrome $u$ obtained by stripping one character off each end. That makes $u$ the parent of $t$, with the wrapping character $c$ as the edge label, and from any node there is at most one outgoing edge per character, since wrapping a fixed inner palindrome with a fixed character yields one string. Even-length palindromes bottom out at the empty string, so I keep a length-$0$ root; odd-length palindromes bottom out at single characters, and to apply the same strip-two-ends rule uniformly I introduce an imaginary length-$-1$ root, so that wrapping a character around it produces a length-$1$ palindrome. With these two roots, every non-empty palindrome is reached by the single parent-plus-wrap rule, which is exactly why the $-1$ root is not a hack but the load-bearing trick that unifies odd and even lengths. I store these transitions as `go[x][c]`.

The second relation is a suffix link `fail[x]`, pointing from each node to its longest *proper* palindromic suffix. This is what makes appends efficient. Maintain `last`, the node for the current string's longest palindromic suffix. When I append $c$ at index $i$, any new palindromic suffix longer than one character has the form $c\,t\,c$, where $t$ is a palindromic suffix of the old string whose immediately preceding character is also $c$; among all valid $t$ the longest one yields the new longest palindromic suffix. Following `fail` repeatedly from `last` visits precisely the palindromic suffixes of the current string in strictly decreasing length, because every palindromic suffix of the whole string is also a palindromic suffix of its longest palindromic suffix; both roots fail to the length-$-1$ root, so the walk always terminates. The wrap test is purely local once a candidate length is known: a node $x$ representing a length-$\text{length}[x]$ suffix occupies old positions $i-\text{length}[x]$ through $i-1$, so it can be wrapped by the new character exactly when the character just before it, at $i-1-\text{length}[x]$, equals the new one at $i$. Hence the walk condition

$$ s[i - 1 - \text{length}[x]] \neq s[i] \;\Longrightarrow\; x \leftarrow \text{fail}[x]. $$

For the length-$-1$ root this reduces to $s[i] = s[i]$, always true, guaranteeing a single-character palindrome is always available; for the length-$0$ root it reduces to $s[i-1] = s[i]$, exactly the condition for a length-$2$ palindrome. A guard value placed before the first real character makes the boundary tests fail cleanly, so I store the processed characters with a private guard at index $0$.

When the walk lands at `cur`, the new longest palindromic suffix is the node `go[cur][c]`. If that transition already exists the palindrome was seen before, so I reuse the node and the distinct count is unchanged; if it is missing, this is the one new palindrome the bound permits, and I create a node of length $\text{length}[\text{cur}] + 2$, add the transition, and increment the count. A new node still needs its own suffix link, and this is where a second, separate search is required: the proper palindromic suffixes of $c\,t\,c$ come from wrapping proper palindromic suffixes of $t$, so I run the same wrap test but starting from `fail[cur]` rather than `cur`. If that search lands at $v$, the longest proper palindromic suffix of the new node is the existing $c$-child `go[v][c]`. The single-character case is special and must be handled explicitly: its longest proper palindromic suffix is the empty palindrome, so its `fail` is the length-$0$ root; without this guard the general transition lookup would point back at the very node being created, which is not a proper suffix. Finally `last` becomes `go[cur][c]`, and the answer for the current prefix is the running count of created nodes, equivalently the node total excluding the two roots.

What makes this linear rather than quadratic is a potential argument over the suffix-link walks. Let $\Phi_i$ be the length of `last` after $i$ characters, with $\Phi_0 = 0$. During the first walk for character $i$, every failed step strictly decreases the candidate length, and if the walk stops at length $L$ the new `last` has length $L+2$, so the number of failed steps is at most $\Phi_{i-1} - L = \Phi_{i-1} - \Phi_i + 2$. Summing over all appends gives at most $2n + \Phi_0 - \Phi_n = O(n)$ failed steps. The second walk, used only when a new node is born, is the same proper-suffix-chain search; each of its failed steps again spends a strict drop in candidate length and is charged by the same potential, adding only a constant multiple of the first walk's work. So the total number of suffix-link jumps is $O(n)$. There are at most $n$ non-root nodes plus the two roots, and each non-root node has exactly one incoming wrapping edge with one label, so the transitions are also $O(n)$ when stored sparsely. With array transitions over a fixed alphabet $\Sigma$ each lookup is constant and construction is $O(n)$ time with $O(n\Sigma)$ space; with balanced maps it is $O(n\log\Sigma)$ time and $O(n)$ space; with hash maps the expected transition factor is constant. Either way the online counter maintains the distinct-palindrome count after every prefix in amortized linear total work.

The program reads one string `s` from stdin; it prints the number of distinct non-empty palindromic substrings of `s`, then a second line with the running count after each prefix `s[0..i]`.

```cpp
// Reads one string s from stdin; prints the number of distinct non-empty
// palindromic substrings of s, then a second line with the running count after
// each prefix s[0..i]. Online via a palindromic tree (eertree).
#include <bits/stdc++.h>
using namespace std;

struct Eertree {
    enum { ODD = 0,   // imaginary root, len -1; seeds odd palindromes
           EVEN = 1 };// root, len 0; seeds even palindromes

    vector<int> s;              // processed characters; s[0] is a guard
    vector<int> len;            // node -> palindrome length
    vector<int> fail;           // node -> longest proper palindromic suffix
    vector<map<int,int>> go;    // node -> {char: child}
    int last;                   // node of the longest palindromic suffix so far
    long long num_distinct;     // distinct non-empty palindromes seen

    Eertree() {
        s.push_back(-1);        // guard
        len = {-1, 0};
        fail = {ODD, ODD};
        go.resize(2);
        last = EVEN;
        num_distinct = 0;
    }

    // climb suffix links until s can be wrapped: s[i-1-len] == s[i]
    int walk(int x) const {
        int i = (int)s.size() - 1;
        while (s[i - 1 - len[x]] != s[i]) x = fail[x];
        return x;
    }

    // append character c; returns true iff a new distinct palindrome was created
    bool add(int c) {
        s.push_back(c);
        int cur = walk(last);                // longest pal. suffix we can extend
        bool created = false;
        auto it = go[cur].find(c);
        if (it == go[cur].end()) {
            int now = (int)len.size();        // new node for c + (cur's pal.) + c
            len.push_back(len[cur] + 2);
            go.push_back(map<int,int>());
            if (len[now] == 1) {
                fail.push_back(EVEN);         // single char -> empty palindrome
            } else {
                int f = walk(fail[cur]);      // continue up fail[cur]
                fail.push_back(go[f][c]);
            }
            go[cur][c] = now;
            ++num_distinct;
            created = true;
        }
        last = go[cur][c];
        return created;
    }
};

int main() {
    string text;
    {
        // read the whole input as one token (first whitespace-delimited word)
        if (!(cin >> text)) text = "";
    }

    // map distinct characters to small integer codes (compact alphabet)
    map<char,int> code;
    for (char ch : text)
        if (!code.count(ch)) {
            int id = (int)code.size();
            code[ch] = id;
        }

    Eertree et;
    vector<long long> per_prefix;
    per_prefix.reserve(text.size());
    for (char ch : text) {
        et.add(code[ch]);
        per_prefix.push_back(et.num_distinct);
    }

    cout << et.num_distinct << "\n";
    for (size_t i = 0; i < per_prefix.size(); ++i)
        cout << per_prefix[i] << (i + 1 < per_prefix.size() ? ' ' : '\n');
    if (per_prefix.empty()) cout << "\n";
    return 0;
}
```
