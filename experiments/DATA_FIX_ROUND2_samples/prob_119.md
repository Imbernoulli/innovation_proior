# 题 119 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 0.0,completion_tokens=2632)

````
The problem is about recovering the sequence of operators (addition or multiplication) between n+1 integers. We have n operators, and we can query the program by providing a sequence of n+1 numbers and getting the result.

The key insight here is that addition and multiplication behave differently when applied to specific numbers:
- Addition with 0 gives the same number: x + 0 = x
- Multiplication with 0 gives 0: x × 0 = 0
- Addition with 1 gives x + 1
- Multiplication with 1 gives x

So we can use strategic queries to determine which operator is at each position.

Let me think about this more carefully. We have operators op₁, op₂, ..., opₙ between n+1 numbers.

Query format: ? a₀ a₁ ... aₙ
Response: ((...((a₀ op₁ a₁) op₂ a₂) ... opₙ aₙ)) mod (10⁹ + 7)

Strategy idea:
1. Use queries with all 1s to detect multiplication (since adding 1 repeatedly vs multiplying by 1 give different results)
2. Use queries with 0s to detect multiplication vs addition

Actually, let me think more systematically.

If I query with all zeros except one position:
- For addition chain: if there's any addition before that zero, it propagates through
- For multiplication chain: multiplying by zero makes everything zero

Let me try a different approach:

Since we want to recover each operator, we can query strategically.

One good strategy is to use powers of some value, since multiplication distributes over exponentiation but addition doesn't interact with powers nicely.

Actually, let's use a simpler approach:

For each position i from 1 to n:


I'll construct a series of targeted queries to isolate each operator's behavior. By carefully choosing input values, I can distinguish between addition and multiplication at each step.

The core challenge is designing queries that reveal unique characteristics of each operator while staying within the limited query count of 600.

Key observation: using 1s and 0s creates distinctive patterns in the computation that help identify operator types. When an operator encounters a zero, multiplication eliminates all subsequent values, while addition preserves them.

This means I can create specific test cases that will definitively show whether an operator is addition or multiplication based on how it handles zero or one inputs.

The strategy involves constructing queries where only one specific operator sees a non-zero value, allowing me to trace its exact effect through the entire calculation sequence.

By carefully selecting input values and observing computational results, I can determine each operator's type without explicitly tracking intermediate values. The key is creating controlled scenarios that expose operator behaviors through final computed outputs.

A systematic approach would involve probing each operator's position with carefully chosen test inputs that reveal its mathematical operation characteristic.

The most promising technique uses sequential binary-like input values, where each query tests a specific range of operator positions by strategically placing zeros and ones.

By comparing the resulting output across different queries, we can infer whether an operator represents addition or multiplication based on how it transforms the input sequence.

The critical observation is that multiplication will propagate zeros differently than addition, creating distinct computational signatures for each operator type.

A strategic binary encoding of the input sequence allows us to uniquely identify operator behaviors through careful comparison of calculated results.

We construct two specialized queries: one with all zeros and one with all ones, revealing critical information about each operator's fundamental transformation mechanism. By examining how these simple sequences change, we can deduce whether each operator fundamentally multiplies or adds.

The key insight emerges from checking a single additional test point: specifically, querying an intermediate value like 2 alongside the base operations helps confirm our initial hypothesis about each operator's nature. This minimal additional query dramatically constrains our uncertainty, typically confirming our classification with high probability.

The computational complexity remains remarkably low - just O(n) queries, well within the problem's 40-query limit for full scoring.

Careful verification confirms our operator identification process maintains consistent mathematical properties across different input configurations.

The strategy involves three targeted queries: first with all zeros, then all ones, and finally mixed values. Each query reveals critical information about operator behavior. By comparing output patterns, we can systematically distinguish between addition and multiplication operators.

Key observations emerge from analyzing zero and one sequences. Multiplication operators propagate zeros through the calculation, while addition operators incrementally increase values. Mixed sequences provide nuanced insights into operator interactions.

The algorithm efficiently narrows potential operator types through strategic query selection and comparative analysis. By leveraging modular arithmetic properties, we achieve precise operator identification with minimal computational overhead. 1, aᵢ = 1, aₚ₊₂ = 2, others = 1

Result P should differ from Q because opₚ now receives input 2 instead of 1. If opₚ is addition: Q = (result_p + 1) mod MOD, P = (result_p * 1) mod MOD. Since multiplication by 1 keeps the same value while addition adds 1, we can distinguish these cases.

The key insight is that addition changes values differently than multiplication when we vary the input at position p.

But this requires knowing the exact intermediate result after opₚ, which I don't have directly. So I need a different approach.

Let me consider using all-zeros and all-ones queries as baseline:
- Query Z: ? 0 0 0 ... 0 → all multiplications make this 0, additions keep accumulating 0
- Query O: ? 1 1 1 ... 1 → all additions accumulate, all multiplications keep 1

Wait, actually:
- For addition chains: a + b where both are 0 or 1 accumulates
- For multiplication chains: multiplying by 1 or 0 behaves differently

Let me think more simply.

At each position i, I want to determine if opᵢ is "+" or "×".

Idea: If I set a₀ = 1, and all other aⱼ = 1 except aᵢ = 0:
- With "+": the 0 gets added, doesn't become 0
- With "×": the 0 gets multiplied, becomes 0

But the issue is previous computations might already be different values.

Let me try yet another approach using a clever trick:

Set a₀ = 1, and for each position i:
- Make a₁ through aₙ equal to 0 except aᵢ = 1
- Observe the difference between queries

Actually, I think the cleanest solution is:

For each i from 1 to n:
1. Query with a₀=1, a₁=...=a_{i-1}=0, a_i=1, a_{i+1}=...=a_n=0
2. Query with a₀=0, a₁=...=a_{i-1}=0, a_i=0, a_{i+1}=...=a_n=1
3. Compare results

Hmm, this still seems complex.

Let me go back to basics and use the property that I can distinguish + and × with two values:

Given any previous value v, applying +x gives v+x, applying ×x gives v×x.

For a given position i, I want to know what happens when we apply opᵢ with certain inputs.

Here's my plan:

**Phase 1**: Query with all 0s and all 1s
- Z = query(0, 0, ..., 0)
- O = query(1, 1, ..., 1)

If ALL operators are "+", then:
- Z = 0
- O = sum of all (i.e., n+1 or something depending on start)

If ALL operators are "×", then:
- Z = 0  
- O = 1

With actual operators, Z is always 0 (because 0 times anything or plus 0 stays 0).

O depends on the mix of + and ×:
- Each multiplication by 1 doesn't change the value
- Each addition by 1 increases the value

So O tells us how many additions occur overall? Not quite, because additions happen at different points.

Actually, let's trace through an example:
If ops are +, ×, and inputs are all 1s:
- Step 1: 1 + 1 = 2
- Step 2: 2 × 1 = 2
Final: 2

If ops are ×, +, and inputs are all 1s:
- Step 1: 1 × 1 = 1
- Step 2: 1 + 1 = 2
Final: 2

Both give 2! So all-ones query alone doesn't distinguish well.

But what if we query with a specific pattern at each position?

I'll focus on determining each operator individually by isolating its effect at position i.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1e9 + 7;
int n;
vector<int> a(n + 1);

int query() {
    string line;
    getline(cin, line);
    if (!line.empty() && line[0] == 'E') return 0;
    
    stringstream ss(line);
    int val;
    ss >> val;
    return val;
}

void print_query(const vector<int>& seq) {
    cout << "? ";
    for (int i = 0; i < seq.size(); ++i) {
        cout << seq[i] << (i == seq.size() - 1 ? "" : " ");
    }
    cout << endl;
    fflush(stdout);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    string line;
    getline(cin, line);
    n = stoi(line);
    
    vector<int> op(n, -1); // -1 unknown, 0 = +, 1 = x
    
    for (int i = 1; i <= n; ++i) {
        vector<int> base(n + 1, 0);
        
        // First query: mostly 1s except a[i-1] = 0
        fill(base.begin(), base.end(), 1);
        base[i - 1] = 0;
        print_query(base);
        int q1 = query();
        
        // Second query: same pattern except set a[i-1] = 2
        fill(base.begin(), base.end(), 1);
        base[i - 1] = 2;
        print_query(base);
        int q2 = query();
        
        // If multiplication: (result from all 1s) * 0 = 0, vs (result from all 1s) * 2 = 2*prev
        
        // Third query: use a pattern where a[i-1] = 1, rest 0 before, rest 1 after
        vector<int> test1(n + 1, 0);
        test1[0] = 1;
        for (int k = i; k <= n; ++k) test1[k] = 1;
        print_query(test1);
        int res_zero = query();
        
        vector<int> test2(n + 1, 0);
        test2[0] = 1;
        for (int k = 1; k < i; ++k) test2[k] = 1;
        test2[i - 1] = 1; // Position i-1 has 1
        for (int k = i; k <= n; ++k) test2[k] = 1;
        print_query(test2);
        int res_one = query();
        
        if (res_zero == res_one) {
            op[i - 1] = 0; // Addition keeps value same pattern
        } else {
            op[i - 1] = 1; // Multiplication changes
        }
    }
    
    print_result(op);
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=1317)

````
I need to solve this interactive problem where I need to recover n operators (each being + or ×) by querying the program's evaluation result on input sequences.

Key constraints:
- At most Q_max = 600 queries
- If queries ≤ 40, get full score

This is a classic binary search/decision problem. Each query gives us information about all operators at once!

The insight: Let me consider what happens if I use input sequence of powers of some large base. But there's a simpler approach - let me think about single operations first.

If I have two known results R₁ and R₂, can I determine an operator op?

Suppose I know that with input (a₀, a₁):
- If op is +: a₀ + a₁ ≡ R₁ (mod M)
- If op is ×: a₀ × a₁ ≡ R₁ (mod M)

So I can check which case holds by comparing (a₀ + a₁) mod M with R₁, or checking if (a₀ × a₁) mod M equals R₁.

But I want to query multiple positions at once! So let me think differently...

What if I process the operators from position n down to 1? At step i (going backwards), I know the operators op_{i+1}, …, op_n already, so I can compute what value each possible choice of op_i would produce as input to the final computation.

Let me define for each position j > i:
- v_plus_j = value computed from a₀,…,aᵢ,opᵢ₊₁,…,opⱼ when using + at position i
- v_mult_j = same but using × at position i

These can be computed recursively going backwards from n. With these values in hand, I should be able to query each unknown position one at a time rather than all at once—this way a single query resolves exactly one unknown. The trick is choosing input values carefully so that addition and multiplication give distinguishable results. Specifically, I can set a₀ equal to (v_plus_i * inv_a_i - 1) mod P and a_k appropriately for later positions. Then computing the actual answer tells me whether adding or multiplying at position i yields the matching output—if they differed, I'd learn op_i immediately. The catch is the modular arithmetic might not cleanly separate the two cases after inversion. A better approach: work backwards through positions 2 to n. When I reach position k with k ≥ 3, I can query both candidate outputs (one assuming op_k = "+", another assuming op_k = "×") using the same inputs. Since only one is valid, whichever produces the interactor's answer reveals op_k. For early positions where I lack enough prior information, I still resolve each individually—position 1 by trying its two possibilities, then position 2 against whatever value the first operation produced. This processes each position independently, giving O(n) total queries well under the limit.
</think>

```cpp
#include <iostream>
#include <vector>

#define MOD 1000000007LL

using namespace std;

long long power(long long a, long long b) {
    long long res = 1;
    while (b > 0) {
        if (b % 2 == 1) res = (res * a) % MOD;
        a = (a * a) % MOD;
        b /= 2;
    }
    return res;
}

long long power(long long a, int neg) {
    return power(a, neg != 0 ? neg : 1);
}

long long inverse(long long a) {
    return power(a, MOD - 2);
}

int main() {
    cout << flush;
    cin >> n;
    
    vector<long long> dp_plus(n + 1, 0);
    vector<long long> dp_mult(n + 1, 0);
    
    long long dp_val[n + 1], ans;
    
    for (int i = 0; i <= n; i++) {
        dp_plus[i] = 0;
        dp_mult[i] = 0;
    }
    
    dp_val[1] = 0;
    dp_val[2] = 0;
    dp_val[3] = 0;
    // dp_val[4] onwards filled as needed
    
    ans = 0;
    
    for (int i = 2; i < n; i++) {
        cin >> ans;
        
        long long a = dp_val[1];
        long long b = dp_val[2];
        long long c = dp_val[3];
        
        if (ans == ((a + b) % MOD)) {
            dp_val[4] = 0;
            dp_val[5] = 0;
            dp_val[6] = 0;
            dp_plus[i] = (dp_val[1] + dp_val[2]) % MOD;
            dp_mult[i] = (dp_val[1] * dp_val[2]) % MOD;
        } else {
            dp_val[4] = (dp_val[1] * dp_val[2]) % MOD;
            dp_val[5] = (dp_val[1] * dp_val[2]) % MOD;
            dp_val[6] = (dp_val[1] * dp_val[2]) % MOD;
            dp_plus[i] = (dp_val[1] + dp_val[2]) % MOD;
            dp_mult[i] = (dp_val[1] * dp_val[2]) % MOD;
        }
    }
    
    cout << "!";
    for (int i = 0; i < n; i++) {
        cout << (dp_plus[i] == (dp_val[1] + dp_val[2]) % MOD ? "+" : "*");
        cout << " ";
    }
    cout << endl;
    cout.flush();
    return 0;
}
```
````
