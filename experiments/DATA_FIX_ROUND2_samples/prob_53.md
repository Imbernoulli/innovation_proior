# 题 53 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 0.0,completion_tokens=2587)

````
The solution requires implementing an interactive problem. Let me understand the problem:

1. There's a hidden permutation p of length n (4 ≤ n ≤ 100)
2. We need to choose a constant k (1 ≤ k ≤ n) and then make at most 10*n queries
3. Each query involves sending a permutation q, and we get back the count of pairs (i, j) where i < j, p[q_i] = q_j, and i ≠ k
4. We need to recover p

Key observations:
- The condition p[q_i] = q_j means that position q_i in permutation p has value equal to q_j
- When q is [1, 2, ..., n], q_i = i and q_j = j, so we're counting pairs where p[i] = j with i < j and i ≠ k
- This gives us information about the structure of permutation p

Strategy for hard version:
Since we have at most 10*n queries for n up to 100, and sum of n² ≤ 10^4, we can be somewhat generous with our approach.

A key insight from similar problems:
1. First, pick a good k value - maybe k = 1 or some fixed value
2. Use different permutations to gather information about positions
3. Try to determine each position one by one, or use constraints to deduce the permutation

Let me think about a systematic approach:

When we query with q as identity [1, 2, ..., n]:
- We get count of pairs (i, j) where i < j, i ≠ k, and p[i] = j
- This tells us about values in p at positions where index < position_value


- For each position i, if p[i] = v > i, it contributes to the count

Actually, I should reconsider what information each query provides. When q = [1, 2, ..., n], we're counting pairs (i, j) where i < j, i ≠ k, and p[i] = j. Since p[i] = j means the value at position i equals position j, this reveals specific mappings. By querying with different arrangements of q, I can collect constraints that help me reconstruct the entire permutation systematically.

For instance, if I send q = [n, 1, 2, ..., n-1], the indices shift but the underlying relationship p[q[i]] = q[j] stays meaningful. A more practical strategy would be to first select k (maybe k=1), then try swapping elements systematically. If I swap two adjacent values like q[i+1] and q[i+2], comparing the results before and after could reveal which positions contain those swapped values. This differential approach lets me pinpoint element locations without needing exhaustive queries.

Given the constraint of 10n total queries and sum of n² ≤ 10⁴, I can afford O(n²) complexity overall. I should fix k at 1, query the base permutation [1, 2, ..., n] to establish a baseline count, then systematically swap adjacent elements to identify each position's value through the resulting differences.

When q[i] changes, the contribution to our answer shifts based on whether p[q[i]] still satisfies the conditions p[q[i]] = q[j]. By measuring how counts change when we swap elements, I can trace which original values occupy which positions.

A more systematic approach: fix k = 1, start with q = [1, 2, ..., n], record the initial answer, then perform targeted swaps and compare responses to deduce p incrementally.

To build p completely, I'll use multiple carefully constructed queries that create distinguishable patterns—perhaps cycling through different arrangements or leveraging the fact that distinct permutations yield distinct answers. Given the 10n query budget and the constraint that sum of n² ≤ 10⁴, an O(n²) solution should work well.

I'll start with k = 1 and systematically query permutations to map out p. By testing different structures—like identity ordering or modified sequences—I can observe how answers change and reverse-engineer which element maps to which position.

One useful observation: if I know all of p except one position, I might be able to complete it by checking which remaining value produces the expected answer pattern. I need to reconsider what `cnt` actually represents in my query. When I ask `? q_1 ... q_n`, I'm getting the count of pairs (i, j) where i < j and p[q[i]] equals q[j]. Testing with concrete examples helps clarify this relationship between the permutation I query and what gets counted.

Rather than continuing to analyze the formula abstractly, I should implement a working solution using a strategic approach: keep k=1 consistent, construct specific test permutations carefully, and track the answers to piece together the full permutation through elimination and deduction. d to implement a working solution. My approach will involve:
1. Selecting k=1 as our reference point
2. Querying the identity permutation [1, 2, ..., n] to establish a baseline count
3. Systematically determining each position in p by making targeted queries and observing how the count changes

For implementation, I'll need to flush output after each query to maintain proper interaction protocol, verify the interactor responds correctly, and handle edge cases where the interactor might signal termination with -1.

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    cin >> n;
    
    // Choose k = 1
    cout << 1 << endl;
    
    // Query identity permutation
    vector<int> q(n);
    iota(q.begin(), q.end(), 1);
    
    vector<int> ans(n + 1, 0);
    
    for (int i = 0; i <= n; ++i) {
        cout << "? ";
        for (int j = 0; j < n; ++j) {
            cout << q[j] << (j == n - 1 ? "" : " ");
        }
        cout << endl;
        cin >> ans[i];
    }
    
    for (int pval : p) {
        cout << ! pval << (pval != p[n-1] ? " " : "");
    }
    cout << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

Wait, I need to reconsider my approach. The example shows that when p=[3,1,4,2] and k=1, query[1,2,3,4] returns only pair(3,4). With i=3, j=4, I check i≠k (3≠1 ✓) and p[q_3]=q_4, which gives p[3]=4.

Since p[3]=4 matches q[4]=4, this contributes to the answer. So the count tracks valid pairs where both the index inequality and the position-value equality hold simultaneously.

Now I'm considering a systematic approach: query multiple configurations of q to extract information about which indices map to which values in p. Using binary search across different q permutations could efficiently narrow down the possibilities within the 10*n query limit.

Actually, a more straightforward strategy emerges. Fix k=1, start with q=[1,2,...,n], then progressively swap elements in q and observe how the answer changes. Comparing responses before and after each swap reveals specific relationships between p's values and indices.

But there's a complication: simply counting changes doesn't directly tell me which exact value occupies which position. I need a cleaner method.

What if I construct queries strategically? Consider this: if I know p[i] for certain positions, I can use that knowledge to deduce remaining positions. Starting with a few known positions (found by examining basic patterns), I can fill in the rest through careful queries and logical deduction.

Alternatively, I could leverage cycle decomposition since any derangement forms disjoint cycles. Though tracking cycles during interaction feels complicated, maybe there's a simpler path.

Let me reconsider with a direct construction approach. I'll try building p by making queries where I set most of q to fixed values and vary just a couple of positions, observing how the answer changes to identify unknown elements.

Actually, given the constraints and query limit, I should focus on efficiency. A reasonable strategy:

1. Set k=1 consistently
2. Query identity permutation first to establish a baseline
3. Use additional queries to isolate individual positions
4. Reconstruct p step by step

Let me implement this systematically:

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    cin >> n;
    
    cout << 1 << endl;
    
    vector<int> q(n);
    iota(q.begin(), q.end(), 1);
    
    int base_ans = 0;
    cout << "?";
    for (int x : q) cout << " " << x;
    cout << endl;
    cin >> base_ans;
```

I notice the current approach needs adjustment. Rather than trying to extract information from one query, I should consider multiple targeted queries that reveal specific relationships between positions and values. I need to finish the thought about the interaction model, but I cannot complete this because the next thinking is incomplete. The partial text cuts off mid-example, so I'm unable to provide a coherent rewrite that maintains the original meaning.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    cin >> n;
    cout.flush();
    
    cout << 1 << endl;
    cout.flush();
    
    vector<int> q(n);
    for (int i = 0; i < n; i++) q[i] = i + 1;
    
    int cnt_base = 0;
    cout << "?";
    for (int x : q) cout << " " << x;
    cout << endl;
    cin >> cnt_base;
    cout.flush();
    
    vector<int> p(n + 1);
    bool done[n + 1];
    for (int i = 1; i <= n; i++) done[i] = false;
    
    for (int i = 1; i <= n; i++) {
        if (done[i]) continue;
        
        cout << "?";
        for (int j = 1; j <= n; j++) {
            cout << " " << q[j - 1];
        }
        cout << endl;
        cin >> cnt_base;
        cout.flush();
        
        for (int j = i; j <= n && !done[j]; j++) {
            if (done[j]) break;
            
            int old_p_val = p[j];
            int new_q_pos = n - j + i;
            
            q[new_q_pos - 1] = j;
            int curr_cnt = 0;
            
            cout << "?";
            for (int x : q) cout << " " << x;
            cout << endl;
            cin >> cnt_base;
            cout.flush();
            
            if (cnt_base != old_p_val) {
                p[j] = n - j + i;
                done[j] = true;
            }
        }
    }
    
    cout << "!";
    for (int x = 1; x <= n; x++) cout << " " << p[x];
    cout << endl;
    cout.flush();
}

int main() {
    ios_base::sync_with_stdio(false);
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

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=729)

````
The solution strategy uses the property that for any two indices i and j, there exists some permutation q where querying (i,j) as the only matching pair reveals information about p_i. The key insight is that by constructing q = [n-i+2, n-j+1], we can determine whether p has a fixed point of 1 in certain positions through binary search.

Let me implement this:
```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    cin >> n;
    cout << n - 1;
    for (int i = 1; i <= n; i++) cout << " ";
    
    vector<int> a(n + 1);
    bool ans[1005];
    
    for (int i = 1; i < n; i++) {
        cout << "?";
        for (int j = 0; j < n; j++) cout << " " << n - i + 1 + j % (n - i + 1);
        cin >> x;
        int cnt = 0;
        
        for (int j = 1; j <= n; j++) {
            if (j < n - i + 1 && p[n - i + 1] == j) cnt++;
            if (j >= n - i + 1 && p[n - j + 1] == j) cnt += __builtin_popcount(j >> i);
        }
        cout << "!";
        for (int j = 1; j <= n; j++) cout << " " << p[j];
        cout << "\n\n";
    }
}

int main() {
    cin.tie(NULL);
    ios_base::sync_with_stdio(false);
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
}
```
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        cout << n - 1;
        for (int i = 1; i <= n; i++) cout << " ";
        
        vector<int> a(n + 1);
        bool ans[1005];
        
        for (int i = 1; i < n; i++) {
            cout << "?";
            for (int j = 0; j < n; j++) cout << " " << n - i + 1 + j % (n - i + 1);
            int x;
            cin >> x;
            int cnt = 0;
            
            for (int j = 1; j <= n; j++) {
                if (j < n - i + 1 && p[n - i + 1] == j) cnt++;
                if (j >= n - i + 1 && p[n - j + 1] == j) cnt += __builtin_popcount(j >> i);
            }
            cout << "!";
            for (int j = 1; j <= n; j++) cout << " " << p[j];
            cout << "\n\n";
        }
    }
}
```
````
