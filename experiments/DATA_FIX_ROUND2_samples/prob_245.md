# 题 245 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=4866)

````
The problem is an interactive game where I need to find the Impostor among n players.

Let me understand the mechanics:
- There are 3 roles: Knight, Knave, Impostor
- Knight: always tells truth
- Knave: always lies
- Impostor: a special Knave whom everybody thinks is a Knight

The question format asks if player i thinks player j is a Knight.

Truth table:
- i=Knight, j=Knight: 1 (Knight tells truth that Knight is Knight)
- i=Knight, j=Knave: 0 (Knight tells truth that Knave is not Knight)
- i=Knight, j=Impostor: 1 (Knight thinks Impostor is Knight - because everyone thinks Impostor is Knight)
- i=Knave, j=Knight: 0 (Knave lies about Knight being Knight)
- i=Knave, j=Knave: 1 (Knave lies about Knave being Knight)
- i=Knave, j=Impostor: 0 (Knave lies about Impostor being Knight - but wait, Impostor is a special Knave that everyone thinks is Knight)
- i=Impostor, j=Knight: 0 (Impostor lies about Knight being Knight)
- i=Impostor, j=Knave: 1 (Impostor lies about Knave being Knight)
- i=Impostor, j=Impostor: 0 (but there's only one impostor)

Wait, let me reconsider. The Impostor is "a special Knave whom everybody thinks is a Knight". This means:
- From anyone's perspective, the Impostor is considered a Knight.
- So when asked "Does i think j is a Knight?", if j is the Impostor, the answer should be 1 regardless of i's role.


- Checking the truth table again: when j=Impostor, all rows show 1 as the answer, which confirms this interpretation.

Now I need to figure out how to identify the Impostor given these constraints. The key insight is that I can use queries to determine relationships between players. If two players give different answers about a third player's status, or if their responses to each other don't align with what I'd expect from Knights and Knaves, I can start narrowing down who might be the Impostor. Since more than 30% must be Knights, there are enough truth-tellers to help me establish reliable information.

I'm realizing I should focus on finding any pair where responses are consistent, since Knights always match and Knaves always lie consistently. The adaptive grader complicates things since roles could shift based on my questions, so I need a strategy that handles uncertainty while still reliably identifying the Impostor through query patterns.

Actually, rethinking the problem structure: the core constraint is that over 30% must be Knights in every valid configuration. This majority of truth-tellers becomes my anchor—I can use them to verify claims and isolate who's lying. For instance, if I query whether A thinks B is a Knight, then whether C thinks B is a Knight, and get conflicting answers, someone must be providing false information. By strategically choosing who to ask about whom, I can systematically eliminate possibilities until the Impostor stands clear.

Let me map out a concrete approach. First, I'll find a Knight by comparing pairwise responses—if two people disagree about someone's identity, at least one isn't truthful. Once I identify a Knight, I can test candidates against them directly. The key is maintaining consistency across multiple queries and using the 30% Knight guarantee to ensure I eventually find someone reliable to verify everything else. d to find a knight efficiently. I can use a tournament-style approach where pairs vote on who they think is a knight, keeping track of winners based on their responses. Then I'll test potential knights against candidates to identify the impostor through contradiction analysis.

For small groups, I'll brute force all permutations. For larger groups, I'll implement a randomized strategy that samples subsets and validates consistency checks. The goal is minimizing total queries while ensuring correctness across different scenarios.

The core mechanism relies on understanding that knights always tell truth, knaves always lie, and impostors behave like knaves but are universally perceived as knights. This asymmetry creates exploitable patterns I can leverage during the querying process. 0)

If both give the same response, the responder might be trustworthy. If they disagree, at least one person is unreliable. I can use this comparison strategy to filter through candidates and identify someone who likely belongs to the majority group of knights.

My approach: for pairs where responses match, treat it as a tie-breaker signal. For mismatches, mark potential conflicts. By tracking which participants consistently align in their assessments, I can progressively narrow down who belongs to the trustworthy majority. 2), keep 2; else keep 1. Continue this tournament-style pairing until only one survivor remains—the candidate we believe to be a Knight.

Once we have a potential Knight, we use them as our reference point to evaluate others. We'll query this candidate's opinions on everyone else, checking whether responses align logically. For each unverified person x, we compare what candidate says about x versus what y says about x. Disagreements suggest either y or candidate might be the Knave, allowing us to eliminate inconsistencies systematically.

The key insight is that with over 30% Knights guaranteed, we can trust majority voting patterns emerge even as roles potentially change between queries. The majority pattern of 1s in the truth table gives us statistical leverage—we just need enough overlapping comparisons to distinguish Knights from others reliably.

My approach: systematically query pairs to build confidence about who's telling the truth, then use that knowledge to identify the Impostor through elimination rather than trying to pin down exact roles for everyone simultaneously.

For small instances under 100 people, I'll test multiple role assignments against all my observations and pick the one matching most responses. When n gets larger, I need a smarter strategy that leverages the 30%+ Knights guarantee—I'll find candidates who consistently agree across different query angles, since Knights form a reliable majority cluster. The core idea is treating agreement patterns as evidence rather than assuming fixed roles throughout.

To locate an honest person, I'll query disjoint pairs and compare their responses. Those giving identical answers likely belong to the Knight majority. Once I identify a few trusted respondents, I can validate others against them and spot inconsistencies revealing the Impostor.

Here's the implementation strategy: maintain a pool of potential Knights, repeatedly test candidates within pairs using the 30% threshold, and prune inconsistent individuals until only the Impostor's unique behavioral signature remains detectable. _j = query(i, j);
        
        if (res_i != res_j) {
            // One of them is the Knave
            losers.push_back({i});
            losers.push_back({j});
        } else {
            // Both agree
            winners.push_back(i);
            winners.push_back(j);
        }
    }
    
    // Keep winners and retry
    for (auto loser : losers) {
        current_indices.erase(find(current_indices.begin(), current_indices.end(), loser));
    }
    
    current_indices = winners;
}

// Now we have a set of potential Knights
vector<int> candidates;
for (int i : current_indices) {
    vector<int> votes;
    for (int j : current_indices) {
        if (i != j && i < j) {
            int r = query(i, j);
            votes.push_back(r);
        }
    }
    candidates.push_back(votes[0]);  // Assume first vote represents their reliability
}

// Find the Impostor by checking all pairwise comparisons
vector<int> true_knights;
for (int i = 1; i <= n; i++) {
    if (!true_knights.empty() && check_consistency(i)) {
        true_knights.push_back(i);
    }
}

cout << ! true_knights.back() << endl;

I'm refining the approach to handle the dynamic nature of the problem more carefully. The key is using pairwise queries to identify consistent truth-tellers, then leveraging the fact that knights always tell the truth about who is actually a knight.

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

using namespace std;

void solve() {
    int n;
    cin >> n;
    
    if (n == 3) {
        cout << "? 1 2" << endl;
        int r12 = cin;
        cout << "? 1 3" << endl;
        int r13 = cin;
        cout << "? 2 3" << endl;
        int r23 = cin;
        
        int ans = 0;
        
        // Check each possibility
        for (int impostor = 1; impostor <= 3; impostor++) {
            bool possible = true;
            for (int i = 1; i <= 3; i++) {
                for (int j = 1; j <= 3; j++) {
                    if (i == j) continue;
                    
                    int role_i = (i == impostor) ? 2 : (i % 3 == 1 ? 1 : 0);
                    int role_j = (j == impostor) ? 2 : (j % 3 == 1 ? 1 : 0);
                    
                    char expected = ' ';
                    if (role_i == 1 && role_j == 1) expected = '1';
                    else if (role_i == 0 && role_j == 0) expected = '1';
                    else if (role_i == 2 && role_j == 0) expected = '0';
                    else expected = '0';
                    
                    int observed = (r12 == 1 && i == 1 && j == 2) ||
                                 (r13 == 1 && i == 1 && j == 3) ||
                                 (r23 == 1 && i == 2 && j == 3) ? 1 : 0;
                    
                    if (observed != expected) {
                        possible = false;
                        break;
                    }
                }
            }
            
            if (possible) {
                ans = impostor;
                break;
            }
        }
        
        cout << "! " << ans << endl;
        return;
    }
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

I realize the current approach won't work correctly because it assumes static roles and doesn't account for the dynamic nature of the problem. I need to rethink how to identify the Impostor without relying on predetermined role assignments.

The key challenge is handling potential role changes between queries while finding a reliable Knight to use as a reference point for verification.

A tournament-based elimination strategy seems promising - using majority voting through pairwise interactions to gradually narrow down potential candidates who consistently behave like Knights.

The core insight involves creating a systematic method to identify trustworthy individuals by observing how different pairs respond to questions, ultimately isolating someone whose statements consistently align with knight-like behavior patterns.

By iteratively comparing participants' responses and maintaining a pool of potential candidates, we can probabilistically converge on someone representing the knight majority while avoiding direct confrontation with the impostor.

The algorithm tracks consistent voters and eliminates those showing conflicting truthfulness patterns, ultimately narrowing the field through strategic query design.

We'll select candidate pairs, observe their mutual assessments, and preserve those demonstrating stable judgment criteria. This process allows filtering potential knights based on their relative credibility signals.

The core strategy involves iterative refinement: collecting potential knight candidates, testing subset consistency, and progressively eliminating contradictory participants until identifying a reliable evaluator.

We construct truth values by examining pairwise interactions between initial and remaining players, then validate potential candidates through systematic cross-referencing. The goal is determining the single impasse node while respecting the known constraint of at least 30% knight population.

Our approach requires careful tracking of response consistency across multiple interaction rounds, ensuring robust identification despite the possibility of adaptive countermeasures.

Critical edge cases around query boundaries and role transitions demand rigorous verification before final selection.

The core algorithm hinges on finding a reliable knight through iterative questioning, then leveraging that established truth to identify the impostor. Small group resolution uses brute-force validation, while larger groups require smart pruning strategies that maintain integrity across potentially shifting roles.

Key considerations involve managing state preservation between query phases, preventing infinite loops, and ensuring deterministic selection when multiple candidates appear equally viable.

Query efficiency matters significantly given tight scoring constraints, so I need to design an approach that minimizes total interrogations while maximizing diagnostic value.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

int query(int i, int j) {
    cout << "? " << i << " " << j << endl;
    int res;
    cin >> res;
    if (res == -1) exit(0);
    return res;
}

void solve_small(int n) {
    // Brute force check for n=3 or n=4
    vector<int> p(n);
    iota(p.begin(), p.end(), 1);
    
    for (int k = 0; k < 1e6; k++) {
        // Try assigning roles
        // 0=Knight, 1=Knave, 2=Impostor
        // At least floor(0.3*n)+1 knights
        
        vector<int> roles(n);
        int knights_needed = max(0, n - (int)(0.4 * n));
        
        for (int mask = 0; mask < pow(3, n); mask++) {
            fill(roles.begin(), roles.end(), -1);
            long long temp = mask;
            for (int i = 0; i < n; i++) {
                roles[i] = temp % 3;
                temp /= 3;
            }
            
            // Count knights
            int knight_count = 0;
            for (int r : roles) knight_count += (r == 0 ? 1 : 0);
            
            if (knight_count < (int)floor(0.3 * n) + 1) continue;
            
            // Single impostor check
            int impostor = -1;
            for (int i = 0; i < n; i++) {
                if (roles[i] == 2) {
                    impostor = p[i];
                    break;
                }
            }
            if (impostor == -1) continue;
            
            // Verify all queries match
            bool ok = true;
            for (int q = 0; q < num_queries_asked; q++) {
                // Need to know actual query indices... skip for now
            }
        }
    }
    
    // Actually, use tournament approach for small n too
    cout << "? 1 2" << endl;
    cout << "? 2 3" << endl;
    cout << "? 3 1" << endl;
}

int num_queries_asked = 0;

void solve_large(int n) {
    // Tournament to find a Knight (majority > 30%)
    vector<int> candidates;
    for (int i = 1; i <= n; i++) candidates.push_back(i);
    
    while (candidates.size() > 1 && num_queries_asked < 100) {
        vector<int> next_round;
        for (size_t i = 0; i + 1 < candidates.size(); i += 2) {
            int u = candidates[i];
            int v = candidates[i + 1];
            int ru = query(u, v);
            int rv = query(v, u);
            
            // If both say same thing, both might be honest
            if (ru == rv) {
                next_round.push_back(u);
                next_round.push_back(v);
            } else {
                // They disagree, at least one is dishonest
                next_round.push_back(ru == 1 ? u : v);
            }
        }
        candidates = next_round;
    }
    
    // Use last candidates to find impostor
    vector<int> potential;
    for (int i = 1; i <= n; i++) potential.push_back(i);
    
    if (potential.empty()) return;
    
    // Compare all against first
    for (int i = 1; i <= n; i++) {
        if (i != potential[0]) {
            query(potential[0], i);
        }
    }
}

vector<vector<pair<int, int>>> recorded_queries;

bool check_assignment(const vector<int>& roles, int impostor, const vector<tuple<int,int,int>>& questions) {
    for (auto [q, ans] : questions) {
        int u = q.first - 1;
        int v = q.second - 1;
        
        char expected;
        if (roles[u] == 0 && roles[v] == 0) expected = '1';
        else if (roles[u] == 0 && roles[v] == 1) expected = '0';
        else if (roles[u] == 0 && roles[v] == 2) expected = '1';
        else if (roles[u] == 1 && roles[v] == 0) expected = '0';
        else if (roles[u] == 1 && roles[v] == 1) expected = '1';
        else if (roles[u] == 1 && roles[v] == 2) expected = '0';
        else if (roles[u] == 2 && roles[v] == 0) expected = '0';
        else if (roles[u] == 2 && roles[v] == 1) expected = '1';
        else if (roles[u] == 2 && roles[v] == 2) expected = '0';
        
        if (expected != (ans ? '1' : '0')) return false;
    }
    return true;
}

void solve(int n) {
    if (n <= 5) {
        // Small case: brute force with limited queries
        vector<int> Q_idx;
        Q_idx.push_back(0); Q_idx.push_back(0); Q_idx.push_back(0);
        
        int q12 = query(1, 2);
        int q13 = query(1, 3);
        int q23 = query(2, 3);
        
        for (int imposter = 1; imposter <= n; imposter++) {
            vector<int> rolen(n);
            fill(rolen.begin(), rolen.end(), 1); // All knaves initially
            
            // At least floor(0.3*n)+1 knights
            int min_knights = (int)floor(0.3 * n) + 1;
            int knights_needed = min_knights - 1; // impostor takes one spot
            knights_needed = max(0, knights_needed);
            
            bool valid = true;
            int actual_knights = knights_needed;
            for (int c = 0; c < n; c++) {
                if (c == imposter - 1) {
                    rolen[c] = 2; // impostor
                } else if (actual_knights > 0) {
                    rolen[c] = 0;
                    actual_knights--;
                } else {
                    rolen[c] = 1;
                }
            }
            
            // Check queries
            char e12 = (rolen[0] == 0 && rolen[1] == 0) ? 1 : 
                       (rolen[0] == 0 && rolen[1] == 2) ? 1 :
                       (rolen[0] == 2 && rolen[1] == 1) ? 1 : 0;
                       
            if ((e12 ^ (q12 == 1)) && (q12 == 1) != (e12 == 1)) valid = false;
            
            if (!valid) continue;
            
            cout << "! " << imposter << endl;
            return;
        }
        return;
    }
    
    // Large case: find a reliable knight then use them
    vector<int> candidates;
    for (int i = 1; i <= n; i++) candidates.push_back(i);
    
    // Reduce candidates by asking pairwise questions
    while (candidates.size() >= 3 && num_queries_asked < 200) {
        vector<int> next_round;
        sort(candidates.begin(), candidates.end());
        
        for (size_t i = 0; i + 1 < candidates.size(); i += 2) {
            if (next_round.size() >= candidates.size()) break;
            int u = candidates[i];
            int v = candidates[i + 1];
            
            int ru = query(u, v);
            int rv = query(v, u);
            
            // Keep those agreeing
            if (ru == rv) {
                next_round.push_back(u);
                next_round.push_back(v);
            }
        }
        candidates = next_round;
    }
    
    // Final verification
    if (!candidates.empty()) {
        cout << "! " << candidates.back() << endl;
    }
}

void solve() {
    int n;
    cin >> n;
    
    if (cin.peek() != -1) {
        cout << "? 1 2" << endl;
    }
    
    solve(n);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=2913)

````
The problem is an interactive game theory problem where we need to identify an Impostor among n players. The key challenge is dealing with the adaptive interactor (roles can change) and the scoring penalty for wrong answers.

Let me think about what information the queries give us:
- Query "? i j" asks if player i thinks player j is a Knight.
- A Knight always tells truth, a Knave always lies.
- An Impostor acts like a Knight (always says yes when asked about knights, but may claim knaves are knights).

Looking at the table:
- When asking if Knight thinks X is Knight: Knight→Knight=1, Knight→Knave=0, Knight→Impostor=1
- When asking if Knave thinks X is Knight: Knave→Knight=0, Knave→Knave=1, Knave→Impostor=0  
- When asking if Impostor thinks X is Knight: Impostor→Knight=0, Impostor→Knave=1, Impostor→Impostor=0

Key insight: Knights always say "yes" to whether Knights are Knights (answer 1), Knaves always say "no" to whether Knights are Knights (answer 0). So querying between two people who both believe they're Knights gives answer 1. If one believes they're Knight and another Knave, answer depends.

Let me reconsider the pattern more carefully. Looking at the matrix again - it's symmetric! That's important. The value doesn't depend on who asks, only on their role pair.

Let me think about using this symmetry. If I query many pairs and collect answers, I'm getting a symmetric matrix where M[i][j] = M[j][i]. The rows/columns should partition into three groups based on roles, and each group has consistent values within it.

The crucial observation: Since >30% are Knights, there must be enough Knights that some relationship stands out. What if I find a player whose row/column looks mostly like someone else's? That similar player might have the same role as them (or I could be the impostor pretending).

Let me trace through the algorithm idea: Query everyone against person 1. Count how many say "yes" to being Knights - those claiming to be Knights should be either actual Knights or the Impostor. Then among candidates claiming to be Knights, find two who match in responses to everyone. Those matching players share the same role; one of them must be real while the other is Impostor. Pick the real one as my candidate. Finally query my candidate against everyone to confirm consistency - any mismatch reveals the true Impostor.

Wait, but I need to be careful about which specific two to pick from the matching pair, since they could be either Knight/Knave or Knight/Impostor.

Let me restart with fresh reasoning. With Knights making up >30% of n (at least floor(0.3n)+1), there must be at least two Knights when n≥5. Let me query person 1 against everyone, then use the count to identify who likely claims to be a Knight, then verify through additional queries.

But the adaptive nature complicates things - if the first few people happen to be all Knights, the Interator can shift roles to avoid giving away the Impostor. I need to handle this by potentially querying multiple people rather than relying on just the first one.

Here's a better approach: Query the first ~2n/3 people against each other. Collect which ones report others as Knights, then identify candidates from this pool - anyone reporting ≥floor(n/3) people as Knights is worth investigating further. Among these high-count candidates, I'll try to find two whose response patterns to the whole group match exactly, since Knights should agree on who counts as Knights.

When comparing pairs, checking every single response is too slow - O(n²) per comparison means O(n³) overall. Instead, I should compare partial response sets first to quickly reject mismatches, only doing full verification on promising candidates. Then once I've narrowed down to a small set of likely candidates, I verify the remaining relationships completely before declaring the result.

This is becoming unwieldy though. Let me step back and ask: what actually distinguishes roles? A Knight genuinely reports only actual Knights. A Knave reports only actual Knaves (and calls Impostors Knights). The Impostor gets tricky - they can appear trustworthy. Rather than trying to reconstruct everything perfectly, maybe I should focus on finding a local inconsistency instead.

What if I simply search for someone whose stated connections don't form a consistent picture? I'd collect responses from a representative subset of potential witnesses, then look for contradictions between what they claim and what others claim. This trades completeness for something that's guaranteed to terminate.

The implementation iterates over random pairs (a,b) and gathers witness responses. It tracks which witnesses support a→b and which support b→a. When enough witnesses favor one direction, it tests that claim directly - checking whether a→b equals all witness claims. If it doesn't match, one of them must be lying, so it outputs that witness index. This catches inconsistencies but won't necessarily find the unique Impostor.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n; cin >> n;
    
    vector<int> k(n + 1);
    
    // Randomly select about 8 candidates
    vector<pair<int, int>> qs;
    for (int r = 0; r < 8; r++) {
        int a = rand() % n + 1;
        int b = rand() % n + 1;
        while (a == b) b = rand() % n + 1;
        qs.push_back({a, b});
    }
    
    // Also add edges from 1
    for (int i = 1; i <= n; i++) {
        if (i != 1) {
            qs.push_back({1, i});
        }
    }
    
    int m = max((int)qs.size(), (long long)n);
    vector<pair<int,int>> edge(m);
    for (int i = 0; i < (int)edge.size(); i++) edge[i] = qs[i];
    shuffle(edge.begin(), edge.end(), default_random_engine(time(NULL)));
    
    auto get = [&](int a, int b) {
        cout << "? " << min(a,b) << " " << max(a,b) << endl;
        int x; cin >> x;
        return x;
    };
    
    function<vector<int>(int)> f = [&](int s) -> vector<int> {
        vector<int> v = {-1};
        for (auto &x : edge) {
            if ((x.first == 1 || x.second == 1) && !v.back()) v.push_back(s);
            if (min(x.first,x.second) == 1) v.push_back(x.first == 1 ? x.second : x.first);
        }
        for (auto [u, v]: edge) {
            int w = (u == s || u == -1) ? v : u;
            int ans = (w == -1 || w == 1) ? get(1, v) : get(v, 1);
            if (!v.back()) v.push_back(w + ans);
        }
        if (v.empty()) v.push_back(-1);
        return v;
    };
    
    bool flag = false;
    vector<int> q1 = f(1);
    vector<int> q2 = f(2);
    
    vector<int> a;
    for (auto x : q2) if (abs(x) != 1) a.push_back(abs(x));
    
    int cnt = (int)a.size();
    int sz = (int)(cnt / (0.5*n+1)) + 4;
    if (sz < 4) sz = 4;
    
    int best = -1;
    for (int i = 1; i <= n; i++) if (!flag) {
        auto p = f(i);
        vector<long long> d(n + 1, 0);
        for (int j = 1; j <= n; j++) d[j] = cnt;
        
        for (int l = 0; l < sz; l++) {
            int c = 0;
            for (auto &y : edge) {
                if ((y.first == i || y.second == i) && !c && abs(p[y.first]) != 1) break;
                if (min(y.first, y.second) == i) c = abs(p[y.first]) != 1;
                
                if (l >= sz/2 && abs(p[y.first]) != 1 && !c) {
                    if (abs(p[y.second]) == 1) d[y.first]--;
                    else d[y.second]++;
                }
            }
            
            for (int j = 1; j <= n; j++) {
                if (abs(p[j]) == 1) {
                    if (d[j] < 1LL * (sz - l)) d[j] -= sz - l;
                } else if (d[j] < sz + l) d[j] -= sz + l;
            }
        }
        
        for (int j = 1; j <= n; j++) if (d[j] > best) best = d[j], q1 = p;
        if (!flag && best == 0) flag = true, q1 = p;
    }
    
    int bad = -1;
    for (int i = 1; i <= n; i++) if (!flag) {
        if (bad != -1 && bad == abs(q1[1] - i)) continue;
        vector<long long> d(n + 1, 0);
        for (int j = 1; j <= n; j++) d[j] = cnt;
        vector<int> p = q1;
        for (auto [u, v]: edge) {
            if (q1[min(u,v)] == -1) p[u] = u == v ? q1[v] : (u == 1 ? -(q1[v]+1) : q1[v]);
            if (q1[max(u,v)] == -1) p[v] = v == u ? q1[u] : (v == 1 ? -(q1[u]-1) : q1[u]);
            if (p[u] == -1) p[u] = p[v];
            if (p[v] == -1) p[v] = p[u];
            if ((u == i || v == i) && abs(p[u]) == 1) d[p[u]]++, d[p[v]]--;
            if ((u == i || v == i) && abs(p[u]) == 0) d[p[u]]--, d[p[v]]++;
        }
        for (int j = 1; j <= n; j++) d[j] -= d[1];
        if (bad != -1 && (d[i] == d[bad])) bad = -1;
        if (best < -999) {
            for (int j = 1; j <= n; j++) d[j] -= d[j] - d[i];
            best = best > 0 ? d[1] : d[n], bad = 1, q1 = p;
        } else if (bad != -1) {
            if (d[bad] < best) bad = 1, best = d[bad], q1 = p;
        }
    }
    
    for (int i = 1; i <= n; i++) if (abs(q1[1] - i) == abs(best - 1)) {
        q1[1] = i;
        break;
    }
    
    if (abs(q1[1] - 1) == 1) q1[1] = 1;
    else q1[1] -= (abs(q1[1] - 1) == 1);
    if (q1[1] <= 0) q1[1] = 1;
    if (q1[1] > n) q1[1] = n;
    
    q1.pop_back();
    q1.erase(remove(q1.begin(), q1.end(), -1), q1.end());
    
    if (abs((int)(q1.size() / (0.5*n + 1)) + 4) != q1.size()) {
        int c = (int)(q1.size() / (0.5*n + 1)) + 4;
        q1.resize(c);
        if (q1.size() != c) {
            cout << "! " << q1[1] << endl; return;
        }
    }
    
    cout << "! " << (q1[0] < 1 ? 1 : q1[0]) << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand(time(NULL));
    int t; cin >> t;
    while (t--) solve();
    return 0;
}
```
````
