# 题 104 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=1458)

````
The problem is an interactive problem where we need to find exactly one absent student among n students. We can make queries about ranges [l, r] and get back the number of students who raised their hands (x), where x equals either the actual number of present students in that range, or off by 1.

Key constraints:
- Only 1 student is actually absent
- Students can be dishonest following specific patterns
- The pattern constraints: never answer honestly 3 times in a row, never answer dishonestly 3 times in a row
- We can mark at most 2 students as absent
- Query limit: 2*⌈log_{1.116}n⌉ which is roughly 4*log_2(n) for practical purposes
- Time limit: 2s
- Memory limit: 256 MB

Let me understand the query responses:
- If we query [l, r], let p = r - l + 1 be the number of students in range
- If the absent student is NOT in [l, r]: all p students are present
  - True Positive: p hands (honest)
  - False Negative: p-1 hands (dishonest)
- If the absent student IS in [l, r]: only p-1 students are present
  - True Negative: p-1 hands (honest)
  - False Positive: p hands (dishonest)

So if we get response x = p, it means either:
- All present (absent not in [l,r]) AND honest
- Present count is p-1 (absent IN [l,r]) AND dishonest (False Positive)

If we get response x = p-1, it means either:
- All present (absent not in [l,r]) AND dishonest (False Negative)
- Present count is p-1 (absent IN [l,r]) AND honest (True Negative)


So we have some information but it's ambiguous due to potential dishonesty.

The crucial insight is tracking honesty across queries. We need to monitor whether responses are honest or dishonest based on whether we observe p or p-1 hands, combined with what we know about the absent student's location. This sequential logic should help narrow down candidates efficiently within the query budget.

I'll start by querying a large range like [1, n]. If I get n hands, the absent person must be in that range and we're seeing a False Positive (dishonest response). If I get fewer than n hands, they're outside and we might see a False Negative (also dishonest) or just honest reporting.

By strategically dividing ranges and watching how many hands appear versus the expected count, I can infer both where the absent student likely sits and whether previous answers were truthful. The constraint about no three consecutive honest or dishonest responses gives me enough signal to detect patterns.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (!(cin >> t)) return 0;
    
    while (t--) {
        int n;
        cin >> n;
        
        vector<pair<int, int>> honest_dishonest(1000000, {0, 0});
        int query_count = 0;
        int absent_candidates[3];
        int candidate_count = 0;
        
        // First, try binary search approach with range splitting
        auto ask = [&](int l, int r) -> int {
            cout << "? " << l << " " << r << "\n";
            cout.flush();
            int mid = l + (r - l) / 2;
            int len = r - l + 1;
            
            int x;
            cin >> x;
            if (x >= l && x <= r && (len == x || len - 1 == x)) {
                return x;
            } else {
                exit(0);
            }
        };
        
        // Start with first query on [1, n]
        int res = 1000000000;
        cout << "? " << 1 << " " << n << "\n";
        cout.flush();
        res = cin.peek() == EOF ? 0 : cin.ignore(8192, '\n') > 0 ? 0 : 1;
        string s;
        getline(cin, s);
        stringstream ss(s);
        ss >> res;
        
        // We track potential absent students through binary search
        // Use binary search with careful verification
        int low = 1, high = n;
        bool found = false;
        int last_honest = -10;
        int last_dishonest = -10;
        
        // Binary search with multiple checks
        int L = 1, R = n;
        while (L <= R) {
            if (query_count++ >= 2 * ceil(log((double)n) / log(1.116))) break;
            
            int M = L + (R - L) / 2;
            cout << "? " << L << " " << M << "\n";
            cout.flush();
            
            string line;
            getline(cin, line);
            stringstream ss(line);
            int r;
            ss >> r;
            
            int len = M - L + 1;
            bool abs_in_range = false;
            
            if (r == len) {
                // Could be honest with all present (absent not in range)
                // Or dishonest with all present when absent IS in range (False Positive)
                abs_in_range = true;
            } else {
                // r should be len-1 if absent in range and honest
                // Or len-1 if absent not in range and dishonest (False Negative)
                abs_in_range = true;
            }
            
            int next_L = L, next_R = R;
            if (abs_in_range) {
                R = M - 1;
            } else {
                L = M + 1;
            }
            
            // Keep last two queries to track pattern
            if (query_count == 1) {
                last_honest = 0;
                last_dishonest = 0;
            }
        }
        
        // Mark top 2 candidates
        cout << "! " << L << "\n";
        cout.flush();
        int y1;
        cin >> y1;
        
        cout << "#\n";
        cout.flush();
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=1195)

````
Let's work through the logic. We have 4 possible cases for each query, and we need to track them across queries while ensuring:
1. Never 3 honest answers in a row
2. Never 3 dishonest answers in a row
3. The absent student is within our final marked range (we mark at most 2 students)

Let me define key variables:
- `honest_count`: number of consecutive honest answers
- `dishonest_count`: number of consecutive dishonest answers  
- `honest_queries`: total honest queries so far
- `dishonest_queries`: total dishonest queries so far
- `absent_candidates`: students potentially absent based on honest queries

The strategy:

**When we see False Negative (dn):** Student might be absent or answering dishonestly. Since they should answer honestly if present, this suggests a specific subset might be absent. Marking candidates here narrows down who could be absent.

**When we see True Positive (hp):** Everyone raised hands, no one can be absent in this range. Honest answers increase our confidence in the honest strategy being correct, but we risk 3 honest in a row, so we must ask a follow-up to get dishonest and reset counters.

**When we see True Negative (ht):** Range confirms everyone present (or absent student not in range). This helps us eliminate candidates but risks 3 honest in a row, so we add a forced dp question.

**When we see False Positive (fp):** Range had missing student but count was still good - indicates some outside the range are answering dishonestly. Risk 3 dishonest in a row, so add fp_q.

For the actual query generation I need to balance narrowing down the candidate set while maintaining the consecutive constraint. The pattern of alternating query types (dp, hn, hp, fn, ht, fp_q) with strategic overlaps ensures we stay under the ~8 query limit even when n=10^5.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cout << flush;
    
    int t; cin >> t;
    while(t--) {
        int n; cin >> n;
        cout.flush();
        
        int honest_count = 0, dishonest_count = 0;
        int honest_queries = 0, dishonest_queries = 0;
        vector<int> absent_candidates; // initially all candidates
        
        auto get_hn_query = [&](vector<int>& cand, vector<int>& left, int idx) {
            left.push_back(cand[idx]);
            return "? " + to_string(idx * n / cand.size()) + " " + to_string((idx + 1) * n / cand.size());
        };
        
        auto get_dp_query = [&](int& i) {
            vector<int> qn; 
            for(int j = 0; j < abs(i/2); j++) qn.push_back(abs(i/2 - j));
            for(int j = 0; j < abs(i/2); j++) qn.push_back(n - 1 - j);
            if(i >= 0 && i < n) qn.push_back(i + 1);
            if(i <= 0 && i > -n) qn.push_back(i + n);
            string qs;
            for(auto c : qn) qs += "? " + to_string(c) + "\n";
            cout << qs;
            return "";
        };
        
        auto query_fn = [&](int i, bool force = false) -> int {
            int l = max(1, abs(i/2)), r = min((i+1)*n/(i==0), n);
            if(force && r < n) { r = l + ((r - l) * 2 / (r - l + 1)) + 1; }
            cout << "? " << l << " " << r << endl; cout.flush();
            int x; cin >> x;
            return !(x == r - l + 1);
        };
        
        auto process = [&](int i) {
            if(!i) i--; else i++;
            if(i < n/2 || i >= 0 && i < n) { i = max(0LL, min((long long)i, (long long)n - 1)); }
            if(query_fn(i)) { i = min(max(0LL, min((long long)i, (long long)n - 1LL)), (long long)(n - i)); }
            else { i = max(0LL, min((long long)i, (long long)(n - 1LL))); }
            return i;
        };
        
        int i = 0, k = 0, j = 0;
        int hn = 0, dn = 0;
        int l = abs(i/2), r = n - abs(i/2);
        vector<long long> lq, rq;
        
        long long cur = 0;
        int cnt = 0;
        for(int _ = 1; _ <= n; _++, cnt++) {
            cout << "? " << cnt << " " << n << endl; cout.flush();
            int val; cin >> val;
            cout << "#\n" << endl << flush;
            cerr << endl;
        }
        cout.flush();
        cout << "! 1\n! 2\n#\n";
    }
    return 0;
}
```
````
