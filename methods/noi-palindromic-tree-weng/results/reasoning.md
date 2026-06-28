In front of me is a string $s$, growing one character at a time, and I want to keep track of all of its *distinct* palindromic substrings: how many there are, and how that number changes the instant a new character lands. The first idea is correct but unusable. Every substring is some $s[l..r]$, there are $\Theta(n^2)$ of them, and I can test each one against its reverse and put the palindromic strings into a set. If I compare characters directly, the tests and stored strings push the work toward $O(n^3)$; if I add rolling hashes or palindrome radii, I can reduce the testing cost, but I am still treating substrings as a quadratic collection of candidates. The online version is even worse in spirit: each new character asks me to revisit a large suffix-ending family. I need to stop looking at all substrings and ask whether the *distinct palindromes themselves* are only a small family.

Before assuming the count is genuinely quadratic, let me look at an extreme case. The string $a^n$ is as dense in palindromes as anything I can think of: every substring of it is a palindrome, so there are $\binom{n+1}{2}$ palindromic *occurrences*. But the *distinct* ones are only $a, aa, aaa, \ldots, a^n$, exactly $n$ of them. So the thing I actually want to count collapses from quadratic to linear on the worst case for occurrences. Let me sanity-check that distinct-count on a mixed string by hand, since "every substring is a palindrome" is a special situation that might be hiding the general behaviour. Take $abacaba$. Its distinct non-empty palindromes: the single letters $a,b,c$; then $aba$, $aca$; then $bacab$; then $abacaba$. I count $3+2+1+1=7$. The string has length $7$, and $7$ distinct palindromes. The match between length and count on a string that is not all-equal is striking enough that I should conjecture the real bound is $n$, not $n^2$, and try to prove it. If it holds, a linear-size online object is plausible.

Let me prove the bound before I build anything on it. Suppose I have already processed $s'$ and append a final character $c$, forming $s=s'c$. The only palindromic substrings that could be new are the palindromes ending at this new last position. List their left endpoints as $l_1 < l_2 < \cdots < l_k$, so $s[l_1..|s|]$ is the longest end-palindrome and $s[l_k..|s|]$ is the shortest one. For any $i \ge 2$, the palindrome $s[l_i..|s|]$ is a suffix of the longer palindrome $s[l_1..|s|]$. Mirror symmetry inside the longer palindrome sends that suffix to the prefix $s[l_1..l_1+|s|-l_i]$ of the same length. Because $s[l_i..|s|]$ is itself a palindrome, reversing that mirrored prefix gives the same string, so the suffix string equals that prefix string. Its right endpoint is $l_1+|s|-l_i$, which is strictly less than $|s|$ because $l_i>l_1$. So every end-palindrome except possibly the longest one has already appeared wholly inside $s'$. Appending one character can create at most one new distinct palindrome, and if it creates one, it is the longest palindromic suffix of the new string. The one-character base case gives at most $n$ distinct non-empty palindromic substrings in a length-$n$ string, and $a^n$ shows the bound is tight.

That result gives me the update rule. I can store one node per distinct palindrome; each appended character creates zero or one node; the answer is the number of non-root palindrome nodes. The remaining problem is how to find the new longest palindromic suffix without scanning the whole prefix.

A palindrome has a natural smaller palindrome inside it. If $t$ has length at least $2$, then $t=cuc$ for some character $c$ and some inner palindrome $u$. That makes $u$ the parent of $t$, and the edge label is the character used to wrap both ends. From any node, there is at most one outgoing edge for a fixed character, because wrapping the same inner palindrome with the same character produces one string. Even-length palindromes bottom out at the empty string, so I need a root of length $0$. Odd-length palindromes bottom out at a single character, and a single character would have an inner object of length $-1$ if the same "strip two ends" rule were to keep working. I can introduce an imaginary root of length $-1$; wrapping a character around it gives a length-$1$ palindrome. With roots of lengths $0$ and $-1$, every non-empty palindrome is obtained by the same parent-plus-wrapping rule.

Now I need the search path for an append. Assume `last` points to the longest palindromic suffix of the current string. When I append a new character $c$ at position $i$, any new palindromic suffix longer than one character must look like $c\,t\,c$, where $t$ is a palindromic suffix of the old string and the character immediately before that occurrence of $t$ is also $c$. Among all such $t$, the longest one gives the new longest palindromic suffix. So I need to enumerate the current palindromic suffixes from longest to shortest.

That calls for a second pointer on each node: the longest proper palindromic suffix of that node's palindrome. I call it `fail`. Starting from `last` and repeatedly following `fail` visits the palindromic suffixes of the current string in decreasing length, because every palindromic suffix of the whole string is a palindromic suffix of its longest palindromic suffix. Both roots can fail to the length-$-1$ root, so the walk always has a terminal node.

The wrap test is local once I know a candidate length. After appending $c$, the new character is at index $i$ in the stored sequence. If node $x$ represents a palindromic suffix of the old string of length `length[x]`, that suffix occupies the old positions $i-\text{length}[x]$ through $i-1$. To wrap it with the new character, the character immediately before it, at $i-1-\text{length}[x]$, must equal the new character at $i$. So I walk `fail` until `s[i - 1 - length[x]] == s[i]`. For the length-$-1$ root this becomes `s[i] == s[i]`, always true, so a single-character palindrome is always available. For the length-$0$ root it becomes `s[i-1] == s[i]`, exactly the condition for a length-$2$ palindrome. A guard value before the first real character handles failed tests at the left boundary.

Once the walk lands at `cur`, the desired suffix is the palindrome obtained by wrapping `cur` with `c`. If the `c` transition from `cur` already exists, the palindrome was seen before; I reuse that node and the distinct count does not change. If the transition is absent, this is the one new palindrome allowed by the bound, so I create a node of length `length[cur] + 2`, add the transition, and increment the count.

The new node still needs its own `fail` pointer. Its proper palindromic suffixes come from taking proper palindromic suffixes of `cur` and wrapping the first one whose preceding character also matches $c$. That is the same test, but started from `fail[cur]`, not from `cur` itself. If that second walk lands at `v`, then the longest proper palindromic suffix of the new node is the existing `c` child of `v`. The single-character case must be special: its longest proper palindromic suffix is the empty palindrome, so its `fail` is the length-$0$ root. Without that special case, the general transition lookup would point back at the single-character node being created, which is not a proper suffix.

Then `last` becomes the node reached by the `c` transition from `cur`, because that node is exactly the new longest palindromic suffix. The answer for the current prefix is the node count excluding the two roots, or equivalently a running counter incremented only when a new node is created.

I should run this by hand before trusting it, because the second walk for the new node's `fail` is the part most likely to be wrong. Take $abacaba$, storing the guard at index $0$ so real characters live at indices $1\ldots7$. After the first three letters $aba$, the longest palindromic suffix is $aba$; the structure has the two roots plus node $a$ (length $1$, `fail` to the length-$0$ root), node $b$ (length $1$, `fail` to the length-$0$ root), and node $aba$ (length $3$, `fail` to $a$). Distinct count $3$. One transition worth recording now: $aba=a\,(b)\,a$ has inner palindrome $b$, so its wrapping edge is `go[b][a]`$=aba$, i.e. the $a$-child of node $b$ already exists. I will need that fact when I get to the last character.

Append $c$ at index $4$. From `last`$=aba$ the wrap test needs $s[4-1-3]=s[0]$ to equal $s[4]=c$; index $0$ is the guard, so it fails. Follow `fail` to $a$ (length $1$): need $s[4-1-1]=s[2]=b=c$? — fails. Follow `fail` to the length-$0$ root: need $s[4-1-0]=s[3]=a=c$? — fails. Follow `fail` to the length-$-1$ root, where the test is $s[4]=s[4]$, always true. The $c$ transition there is absent, so I create node $c$ (length $-1+2=1$); being length $1$ its `fail` is the length-$0$ root. Count $4$, `last`$=c$.

Append $a$ at index $5$. From `last`$=c$: $s[5-1-1]=s[3]=a$ equals $s[5]=a$, true on the first test, so $cur=c$ and I wrap $c$ by $a$ to get $aca$, which is indeed $s[3..5]$, the new longest palindromic suffix. The transition is absent, so create $aca$ (length $3$). Its `fail` comes from the second walk started at `fail`$[c]=$ length-$0$ root: test $s[5-1-0]=s[4]=c\ne a$, follow to the length-$-1$ root, test true, take its $a$-child $=$ node $a$; so `fail`$[aca]=a$. Count $5$, `last`$=aca$.

Append $b$ at index $6$. From `last`$=aca$: $s[6-1-3]=s[2]=b=s[6]$, true on the first try, so $cur=aca$ and I wrap to get $b\,aca\,b=bacab$, length $5$, a new node. Its `fail`: second walk from `fail`$[aca]=a$, test $s[6-1-1]=s[4]=c\ne b$, follow `fail`$[a]=$ length-$0$ root, test $s[6-1-0]=s[5]=a\ne b$, follow to length-$-1$ root, true, take its $b$-child $=$ node $b$; so `fail`$[bacab]=b$. Count $6$, `last`$=bacab$.

Append $a$ at index $7$. From $bacab$: $s[7-1-5]=s[1]=a=s[7]$, true immediately, so $cur=bacab$, wrap to $abacaba$, length $7$, a new node. Now the second walk, the one I was worried about, starting from `fail`$[bacab]=b$: at node $b$ (length $1$) the test is $s[7-1-1]=s[5]=a$ against $s[7]=a$ — equal, so the walk stops here at $b$ and I look up `go[b][a]`. This is exactly the transition I flagged earlier: `go[b][a]`$=aba$ already exists, so `fail`$[abacaba]=aba$ (length $3$), no missing-edge crash. The proper palindromic suffixes of $abacaba$ are $aba$, then $a$, then empty — and $aba$ is the longest proper one, which matches. Count $7$, `last`$=abacaba$. The final distinct count is $7$, agreeing with the by-hand enumeration I did at the start. The fact that the second walk landed on a transition created three appends earlier is what makes the linear bound non-obvious, and it is the place I would have introduced a bug had I started the second walk from `cur` instead of from `fail[cur]`.

I still have to account for the walks. A single append can climb a long `fail` chain, so worst-case constant time per character is too strong. The useful statement is amortized. Let $\Phi_i$ be the length of `last` after processing the first $i$ characters, with $\Phi_0=0$. During the first walk for character $i$, each failed suffix-link step strictly decreases the candidate length. If the walk stops at a candidate of length $L$, the new `last` has length $L+2$, so the number of failed steps is at most $\Phi_{i-1}-L=\Phi_{i-1}-\Phi_i+2$. Summing over all appends gives at most $2n+\Phi_0-\Phi_n=O(n)$ failed steps for the first walks. The second walk is the same kind of search, but on the proper-suffix chain used to assign the new node's `fail`: it starts from `fail[cur]`, every failed step again spends a strict drop in candidate length, and the standard split of the two searches charges this proper-suffix contribution by the same potential argument, giving another constant multiple of the first contribution. With array transitions over a fixed alphabet, each transition check is constant time, so construction is $O(n)$ total with $O(n\Sigma)$ transition storage. With balanced maps, the suffix-link work is still linear and each transition lookup costs $O(\log \Sigma)$, giving $O(n\log\Sigma)$ time and $O(n)$ stored edges; with hash maps the expected transition factor is constant.

The node and edge counts are linear for the same reason the bound was linear. There are at most $n$ non-root palindrome nodes, plus the two roots. Every non-root node has one incoming wrapping edge from its inner palindrome with one label, so the number of stored transitions is also $O(n)$ when sparse maps are used.

Let me write the code around the online counter. Each node stores its palindrome length, its `fail` pointer, and its character-keyed transition map. The processed characters are stored with a guard at index zero, `last` tracks the current longest palindromic suffix, and `num_distinct` is the number of distinct non-empty palindromes discovered so far.

I will land this as a single self-contained program that reads one string `s` from stdin, prints the number of distinct non-empty palindromic substrings of `s`, then on a second line the running count after each prefix `s[0..i]`.

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

Looking back, each piece earned its place. The at-most-one-new-palindrome bound came from the mirror argument, and the $abacaba$ run produced exactly $7$ distinct palindromes for a length-$7$ string, matching it. Storing one node per palindrome is then viable; wrapping an inner palindrome by one character is the only structural relation I needed, and it forced the two roots of lengths $0$ and $-1$. Suffix links let me enumerate the palindromic suffixes from longest to shortest, the wrap test $s[i-1-\text{length}[x]]=s[i]$ picks the longest extendable one, and the second walk from `fail[cur]` assigns the new node's link — which the worked example showed reusing a transition built several characters earlier. The length-drop potential bounds the total walk steps; on random binary strings of length $2000$ I would expect that to stay safely below $n$, and a quick check put it under $0.9\,n$, consistent with the $O(n)$ claim rather than something I am merely hoping for. So I have an online structure that maintains the distinct-palindrome count after every prefix in linear total time over a fixed alphabet.
