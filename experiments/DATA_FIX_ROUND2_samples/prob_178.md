# 题 178 — base均分 55.3 vs 直接SFT均分 0.0(掉点 55.3)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 92.8,completion_tokens=27372)

````
The user wants me to solve a 3-SAT problem, but with a specific twist: I need to maximize the number of satisfied clauses, not necessarily satisfy all of them (since 3-SAT is NP-complete). However, given the constraints $n \le 50$ and $m \le 1000$, this suggests we might be able to use some optimization or heuristics rather than exact exponential algorithms.

Wait, actually, looking at the scoring and notes: "Any assignment is accepted and scored". This means I don't need to output the optimal assignment, just *any* valid assignment that maximizes the score as much as possible. Since it's an "Easy" version where exact 3-SAT is hard, but with small $n$, maybe randomized local search or simulated annealing works? Or even a simple greedy approach?

Given $N \le 50$, trying all $2^{50}$ assignments is impossible.
However, since we want to maximize satisfied clauses, this is the MAX-3-SAT problem.
For small $N$, maybe we can try some random restarts with local search.
Or, since $N$ is up to 50, maybe we can iterate through all possibilities if we prune? No, $2^{50}$ is still too large.

Let's look at the constraints again. $N \le 50$.
Usually, for Max-SAT problems in competitive programming with these constraints, randomized local search (like Simulated Annealing or simply hill climbing) works well enough.
A common strategy is:
1. Assign random values.
2. Iterate through variables. If flipping a variable improves the score (or worsens less), flip it.
3. Repeat multiple times with different seeds.

Actually, since $N$ is small enough for some backtracking with pruning, but too large for full search, and time limit isn't specified but usually is around 1-2 seconds.
Let's consider a simple heuristic: Randomized Local Search.
For each clause, count how many literals are false. If flipping a variable fixes more unsatisfied clauses than it breaks, do it. We can repeat this process from different random initial states to get a good solution.

Algorithm idea:
1. Initialize assignment randomly.
2. Calculate current satisfied count.
3. For each iteration (say, 100 iterations):
    a. Try to find a variable to flip that increases satisfied count.
    b. If found, flip it and update counts.
    c. If no such variable, pick one that decreases least (steepest ascent/greedy).
4. Output best found assignment.

Since we want to maximize, let's implement a simple hill-climbing with random restarts.

Actually, there is an even simpler observation: if $N$ is small, maybe we can use a SAT solver library? But external libraries aren't allowed.
So we write our own solver.
With $N=50$, pure random + greedy improvement is standard.

Let's refine the logic:
Variables $x_1 \dots x_n$.
Clauses $(a \lor b \lor c)$.
Each clause has 3 literals.
We want to maximize $\sum \text{IsSatisfied}(clause_i)$.

Initialization:
Generate random assignment.
Evaluate cost (unassigned or satisfied).
Improve iteratively:
Try flipping each variable $i$ from $0 \to 1$ or $1 \to 0$.
Check impact on total satisfied clauses.
If flipping $i$ increases satisfied count, flip it.
Repeat until stable (no single flip helps).
Do this multiple times (restarts).
Keep the best overall result.

Complexity:
Per iteration: $O(N \times M)$ to check all variables against all clauses (updating clause evaluation). Actually, we can optimize by maintaining clause satisfaction status. But with $M=1000, N=50$, $O(N \cdot M) = 50,000$ operations per step. With say 100 restarts, that's fine.
Even better: just re-evaluate affected clauses. Each variable appears in some clauses. Sum of occurrences of variables across all clauses is roughly $3M$. So checking impact is $O(M)$.
Total complexity roughly $K \times \text{restarts} \times (\text{convergence steps}) \times N/M$. Wait, convergence takes maybe $O(M)$ flips? So total $K \times M \times N$. With $N=50, M=1000$, that's $50,000$. Times maybe 10 restarts = 500,000 ops. Very fast.

Let's double check if I should output exactly n integers. Yes.

Code structure:
Read n, m.
Store clauses. Each clause: list of literals {u, sign}. u is index 1..n. sign is +1/-1.
Function `evaluate(assignment)`:
  returns number of satisfied clauses.
Function `get_clause_eval_changes(var_index, current_val)`:
  returns change in satisfied count if `var_index` flips.

Optimization:
Precompute for each variable the list of clauses it belongs to.
For each clause, track its current contribution (0 or 1).
When we evaluate flipping var `v`, we only care about clauses containing `v`.
There are at most $3M$ such literal appearances.
In each step, iterate over all variables. For each variable, check which clauses involve it. Count how many become satisfied/unsatisfied. If net change > 0, apply flip.

Wait, finding the maximum might require more sophisticated methods like WalkSAT if hill climbing gets stuck.
Hill climbing is prone to local optima.
WalkSAT: pick an unsatisfied clause, pick one variable in it at random and flip it with probability p based on success rate... but here any assignment is fine, so we just need a reasonable one. Hill climbing often finds very high scores for random CNF instances.
Also, we can add randomness: if no improving move is possible, pick a random variable and flip it to escape local optimum, accepting lower score temporarily. This is Metropolis / Simulated Annealing style.

Let's stick to Randomized Local Search with Simulated Annealing behavior or simple hill climbing with random restarts. Given it's a competitive programming context and limits are loose for this problem type (implied by N=50 for Max-3SAT which is generally PSPACE-hard but solvable heuristically), simple repeated greedy search is likely sufficient.
However, since the problem asks to "assign truth values... to satisfy as many clauses as possible", maximizing is key. A greedy hill climber works well on average.

Let's code a robust solution.

Input parsing:
Lines with `a b c`. Convert to indices 0..n-1. Signs are handled by negation.
Note: Input uses 1-based indexing [-n, n]. 0 is invalid according to description.
If input is positive, variable index is `val-1`. If negative, variable is `-(val-1)` conceptually, meaning literal is `!variable`.
Let's store clause as `{id1, id2, id3}` where ids are -n to -1 or 0 to n-1? Easier: store absolute variable index and a flag whether it's negated.
Example: Clause `(1 v 2 v -3)`.
Literals:
1 -> var 0, neg=false
2 -> var 1, neg=false
-3 -> var 2, neg=true

Evaluation function:
Iterate all clauses.
Check if any literal evaluates to true.
If true, count++.

Optimization for updates:
We can cache the status of each clause.
When changing variable $v$:
Check all clauses containing $v$.
Update their status.
Recalculate total score.
Repeat this $N \cdot K$ times?
Better loop:
While improved:
  Found best flip = None, gain = 0.
  For each var $i \in [0, n-1]$:
    temp_gain = 0
    Check clauses involving $i$:
      If currently contributes to satisfaction but would stop after flip: temp_gain--
      If currently does NOT contribute but would start satisfying after flip: temp_gain++
    If temp_gain > 0 and temp_gain > max_gain:
       best_flip = i
       max_gain = temp_gain
  
  If max_gain > 0:
     Flip `best_flip`.
  Else:
     Break (local optimum).
  
After convergence, save state. Reset and run again from new random seed. Keep global best.

This looks correct and safe.

Constraints:
Time: Typically 1-2s.
Memory: 256MB+.
My approach:
Memory: storing clauses ($O(M)$). Minimal overhead.
Time: $10 \text{ restarts} \times (\text{steps}) \times N \times (\text{avg occurrences})$. 
Avg occurrences per variable is $3M/N$.
Total steps per restart: potentially bounded by $M$.
So complexity per restart $\approx M^2 / N$. $1000^2 / 50 = 20,000$. Very low.
100 restarts -> $2,000,000$ ops. Well within limits.

Corner cases:
$m=0$: Score 1. Any assignment.
$n=1$.
Negative literals handled correctly.
Multiple optimal solutions allowed.

I will implement this.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct Clause {
    int var_idx[3]; // 0-based index of variable
    bool neg[3];    // true if literal is negated (e.g., !x)
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    vector<int> var_in_clauses(n); // To quickly find clauses involving a variable
    // Since iterating clauses is O(m), we can just filter clauses. 
    // Optimization: build adjacency list? 
    // Given constraints, direct iteration over clauses inside the inner loop is O(M) per variable check.
    // Checking all variables per step is O(N*M). 
    // Total complexity O(Steps * N * M). 
    // Steps approx M. N*M*M = 50 * 1000 * 1000 = 5e7. Acceptable for 1 sec C++.
    
    // Let's build lookup to speed up finding clauses for a variable.
    // adj[var] = list of clause indices.
    vector<vector<int>> adj(n);

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        
        auto convert = [&](int val, int& idx, bool& neg) {
            idx = abs(val) - 1;
            neg = (val < 0);
        };
        
        Clause cl;
        convert(a, cl.var_idx[0], cl.neg[0]);
        convert(b, cl.var_idx[1], cl.neg[1]);
        convert(c, cl.var_idx[2], cl.neg[2]);
        
        clauses[i] = cl;
        adj[cl.var_idx[0]].push_back(i);
        adj[cl.var_idx[1]].push_back(i);
        adj[cl.var_idx[2]].push_back(i);
    }

    // Assignment: 1 = True, 0 = False
    vector<int> assignment(n, 0); // default 0
    
    // Helper to calculate satisfied count
    auto count_satisfied = [&]() {
        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            bool sat = false;
            for (int j = 0; j < 3; ++j) {
                int v = clauses[i].var_idx[j];
                bool lit_val = (clauses[i].neg[j] ? !assignment[v] : assignment[v]);
                if (lit_val) sat = true;
            }
            if (sat) cnt++;
        }
        return cnt;
    };

    // Better to maintain clause status dynamically to avoid O(M) scan
    // Status: true if satisfied
    vector<bool> clause_status(m, false);
    auto init_and_count = [&]() -> int {
        int cnt = 0;
        for(int i=0; i<m; ++i) {
            bool sat = false;
            for(int j=0; j<3; ++j) {
                int v = clauses[i].var_idx[j];
                bool val = (clauses[i].neg[j] ? !assignment[v] : assignment[v]);
                if(val) sat = true;
            }
            clause_status[i] = sat;
            if(sat) cnt++;
        }
        return cnt;
    };

    // Main solving loop: Random Restarts + Greedy Improvement
    int best_score = 0;
    vector<int> best_assignment = assignment;

    mt19937 rng(time(0)); // Use time seed

    // Run multiple restarts
    for (int r = 0; r < 20; ++r) {
        // Randomize assignment
        iota(assignment.begin(), assignment.end(), 0); // dummy init
        uniform_int_distribution<> dist(0, 1);
        fill(assignment.begin(), assignment.end(), 0);
        for(int i=0; i<n; ++i) assignment[i] = dist(rng);

        // Re-init status
        init_and_count();
        int current_score = init_and_count(); // Just return value
        
        // Actually init_and_count updates global clause_status
        // So we call it once, then update manually or re-call
        
        // Wait, init_and_count sets clause_status and returns score.
        // Let's make sure we use the returned score.
        
        // Optimization: Hill Climbing
        while (true) {
            int max_improvement = 0;
            int candidate_var = -1;

            // Find variable to flip
            // We check every variable
            for (int v = 0; v < n; ++v) {
                int improvement = 0;
                // Check clauses involving v
                for (int cid : adj[v]) {
                    // Currently, if clause_status[cid] is true, it must be because THIS literal is satisfied
                    // OR because another literal is satisfied.
                    // If clause_status[cid] was already true, it stays true unless ALL satisfied literals 
                    // were due to this one being True.
                    
                    // This requires knowing specifically which literal made the clause true.
                    // That complicates tracking.
                    // Alternative: Recalculate impact locally?
                    // For 3-SAT, max 3 literals.
                    // If we flip v:
                    // Old value: assignment[v]
                    // New value: 1 - assignment[v]
                    
                    // Check if clause becomes satisfied.
                    // If already satisfied, will it become unsatisfied?
                    // Only if exactly 1 literal is true and it is the one we flipped.
                    
                    // Simple logic:
                    // Evaluate clause with OLD value of v. Check is true.
                    // Evaluate clause with NEW value of v. Check is true.
                    // Difference = count(new) - count(old).
                    
                    // We only need to know if status changes.
                    // Since M is small, evaluating clause takes constant time (3 checks).
                    
                    bool old_sat = false;
                    bool new_sat = false;
                    
                    // Temporarily swap to check? No, just math.
                    
                    // Current assignment v value
                    int current_v_val = assignment[v];
                    int new_v_val = 1 - current_v_val;
                    
                    // Evaluate clause with current_v_val
                    bool sat_old = false;
                    for(int j=0; j<3; ++j) {
                        int u = clauses[cid].var_idx[j];
                        bool val = (clauses[cid].neg[j] ? !assignment[u] : (u==v ? current_v_val : assignment[u]));
                        if(val) sat_old = true;
                    }
                    
                    // Evaluate clause with new_v_val
                    bool sat_new = false;
                    for(int j=0; j<3; ++j) {
                        int u = clauses[cid].var_idx[j];
                        bool val = (clauses[cid].neg[j] ? !assignment[u] : (u==v ? new_v_val : assignment[u]));
                        if(val) sat_new = true;
                    }
                    
                    if (sat_new && !sat_old) improvement++;
                    else if (!sat_new && sat_old) improvement--;
                    
                    if (improvement > max_improvement) {
                        max_improvement = improvement;
                        candidate_var = v;
                    }
                }
                
                // Small optimization: if max_improvement == 0, we cannot improve further?
                // Not strictly true, we could have improvements later, but if max_improvement <= 0 globally, we are stuck.
                // Wait, we need to check ALL variables. If any gives > 0 improvement, we continue.
            }
            
            if (max_improvement > 0) {
                assignment[candidate_var] = 1 - assignment[candidate_var];
                // Update status for efficiency?
                // Re-checking status takes O(M) worst case? No, we only changed one var.
                // But inside loop, if we decide to break, we might want to keep partial info.
                // But simplicity favors recalculating full score occasionally or keeping manual update.
                // Manual update is faster.
                // Let's just track status array 'clause_status'
                // When flipping v, update status of clauses in adj[v].
                
                // Update clause_status efficiently
                for(int cid : adj[candidate_var]) {
                    bool sat = false;
                    for(int j=0; j<3; ++j) {
                        int u = clauses[cid].var_idx[j];
                        bool val = (clauses[cid].neg[j] ? !assignment[u] : assignment[u]);
                        if(val) sat = true;
                    }
                    clause_status[cid] = sat;
                }
                break; // Restart inner loop if needed? Usually one greedy step or iterative.
                     // Standard hill climbing repeats until no improvement possible.
                     // My 'max_improvement' check above was flawed because I used 'break' inside 'for(v)' loop?
                     // No, I didn't put break there. I just checked.
                     // I need to collect ALL improvements first to find best move.
            } else {
                // No single flip improves the score.
                // Escape local minimum?
                // If we strictly follow "satisfy as many", sometimes accepting slight loss helps reach global.
                // But for this constraint size, pure hill climb might suffice.
                // Let's try a few passes of pure hill climb with restarts.
                // Or just simulate annealing logic: pick random variable and flip if 50% chance?
                // Let's stick to hill climb. If stuck, break and try new seed.
                break;
            }
            
            // After flip, recalc score to verify?
            // Let's integrate the loop properly.
            
            // Wait, my logic inside the `while(true)` needs to be cleaner.
            // The previous snippet had a flaw: inside the for(v) loop, I was doing full clause evaluations.
            // If max_improvement > 0, I perform the flip. Then I re-run the check.
            // This is fine.
        }
    }
    
    // Wait, the snippet above is messy. Let's rewrite cleanly.
    
    // ...
}
```

Refined Logic Plan:
1. Read Input. Store clauses. Build Adjacency List `adj[var_id]` -> `vector<clause_indices>`.
2. `assignment` vector.
3. Function `update_clause_status(var_id)`:
   Iterates `adj[var_id]`. For each clause, re-evaluate satisfaction based on current `assignment` and update boolean vector `is_satisfied`. Return delta score.
4. Main loop:
   For `restarts` = 1 to `MAX_RESTARTS`:
     Init `assignment` randomly.
     Calc initial score and `is_satisfied`.
     Loop `improved`:
       Find `v` that maximizes `gain` (gain > 0 preferred).
       If max_gain > 0:
         Apply flip.
         Update `is_satisfied` for affected clauses.
         Continue loop.
       Else:
         // Stuck. Maybe try stochastic step or just restart.
         // Given constraints, pure greedy is risky on hard instances. 
         // Add a probabilistic step: pick random v, flip with prob p, accept if score >= prev_best or sometimes accept worse.
         // Actually, for MaxSAT, simulated annealing is good.
         // However, usually, with random restarts, simple hill climbing gets close to optimal for these sizes.
         // Let's add a "random walk" phase when stuck.
         Break if stuck without progress after X attempts.
     
     Compare with best global.

Wait, the problem says "Score is defined as s/m". If m=0, score=1.
Maximize s.
Since N is small, we can perhaps try to verify if we can satisfy all?
No, just maximize.
Let's just run the greedy local search multiple times. It works extremely well for Max-3SAT on small N.
Also, to avoid infinite loops in hill climbing (though impossible here since score is integer bounded), ensure termination.

Implementation details:
`is_satisfied` array size M.
Score = sum(is_satisfied).
Flip variable:
  Old Score contribution of clauses in `adj[v]`:
    If `is_satisfied[c]` is true.
    It remains true if other literals satisfy it.
    It becomes false only if THIS literal was the ONLY satisfier.
    How to track "which literal satisfied the clause"? Hard.
    So re-evaluating clauses in `adj[v]` is safest and fast enough. `adj` ensures we visit each clause twice per variable flip (once before? no).
    Wait, `adj[v]` gives all clauses. When we flip `v`, ONLY clauses containing `v` change status potentially. Clauses NOT containing `v` don't change.
    So we loop `c in adj[v]`, recalculate `sat(c)`. Update score diff.
    
    Complexity: $O(M)$ to re-evaluate (each clause visited at most once per var flip? No, once per flip).
    So per iteration: $N \times M$ operations?
    No. Inside one iteration (checking one potential move):
      We check all vars $v=0..N-1$.
      For each $v$, we sum impact of clauses in $adj[v]$.
      Cost: $\sum |adj[v]| = 3M$.
      So checking ALL moves costs $O(N \cdot M)$ if we iterate vars, or better $O(M)$ if we compute impacts incrementally?
      Actually $\sum |adj[v]|$ is total number of literal occurrences, which is $3M$.
      So for one pass of checking all vars, it takes $O(M)$ work?
      Wait, we need to find the BEST var to flip.
      We can iterate $v$ from $0$ to $n-1$. For each $v$, calculate gain.
      Gain calculation involves summing changes over $adj[v]$.
      Total cost for one full pass of all vars: $\sum_{v=0}^{n-1} \sum_{c \in adj[v]} 3$ (check literals) $= 3M$.
      So one pass is $O(M)$.
      Number of passes per restart: roughly $O(M)$ in worst case (each flip improves by 1, score goes from 0 to M).
      Total per restart: $O(M^2)$.
      With $M=1000$, $M^2 = 1,000,000$.
      100 restarts -> $10^8$. Might be tight for 1s?
      C++ is fast. $10^8$ simple ops is ~0.1s - 0.5s.
      Should be fine.

One optimization:
Instead of full scan, check if ANY var provides gain > 0.
Stop scanning if `max_gain` found so far is 0? No.
Just run full scan.

Wait, input `a b c` are non-zero, between `-n` and `n`.
Output format: `x1 ... xn`. 0 or 1.

One edge case: `n` could be smaller than max literal index? No, `n` is number of variables. Lits are in range `[-n, n]`.

Also include `<cstdio>`? `cin` is fine.

Final check on scoring logic:
"Any assignment is accepted". So finding a high score is the goal.
Greedy with multiple restarts is the standard way to solve this.

Let's clean up the code.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

using namespace std;

// Structure to hold clause information
struct Literal {
    int var_idx; // 0-based index (0 to n-1)
    bool negated;
};

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    // We need to map 1..n to 0..n-1
    // Clauses are stored
    struct Clause {
        vector<Literal> lits;
    };
    
    vector<Clause> clauses(m);
    // Adjacency list: which clauses contain variable i?
    vector<vector<int>> adj(n);
    
    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        Clause cl;
        cl.lits.reserve(3);
        
        auto push_lit = [&](int x, int& idx, bool& neg) {
            if (x == 0) {
                 idx = -1; neg = false; // Should not happen per problem
            } else if (x > 0) {
                 idx = x - 1;
                 neg = false;
            } else {
                 idx = -x - 1;
                 neg = true;
            }
        };
        
        push_lit(u, cl.lits[0].var_idx, cl.lits[0].negated);
        push_lit(v, cl.lits[1].var_idx, cl.lits[1].negated);
        push_lit(w, cl.lits[2].var_idx, cl.lits[2].negated);
        
        clauses[i] = cl;
        
        for(const auto& lit : cl.lits) {
            adj[lit.var_idx].push_back(i);
        }
    }
    
    // State
    vector<bool> assignment(n, false);
    vector<bool> clause_satisfied(m, false);
    
    // Function to evaluate clause status
    auto evaluate_clause = [&](int id, const vector<bool>& assign) -> bool {
        for (const auto& lit : clauses[id].lits) {
            bool val = lit.negated ? !assign[lit.var_idx] : assign[lit.var_idx];
            if (val) return true;
        }
        return false;
    };
    
    // Initialize score and status
    auto init = [&]() {
        for(int i=0; i<m; ++i) clause_satisfied[i] = evaluate_clause(i, assignment);
    };

    mt19937 rng(1337); // Fixed seed for reproducibility or time()? Time is better for variety
    // Actually for CP, deterministic is safer for debugging, but variety is better for search.
    // I'll use current time seed if available, else fixed.
    // To be safe and consistent with "best effort", random seeds are good.
    // Using std::chrono
    mt19937 g_seed(0); 
    g_seed.seed((unsigned)clock()); 

    vector<int> best_assignment = assignment;
    int max_score = -1;

    // Run multiple random restarts
    // For N=50, maybe 50 restarts? 100 is plenty.
    // Limit restarts to keep runtime low
    int num_restarts = 50; 
    
    for (int r = 0; r < num_restarts; ++r) {
        // Random initialization
        for(int i=0; i<n; ++i) assignment[i] = (rng() % 2);
        init();
        
        int current_score = 0;
        for(bool s : clause_satisfied) current_score += s;
        
        if (current_score > max_score) {
            max_score = current_score;
            best_assignment = assignment;
        }

        // Hill Climbing
        while (true) {
            int best_gain = 0;
            int cand = -1;
            
            // Find variable to flip that maximizes gain
            // Iterate all variables
            for (int v = 0; v < n; ++v) {
                int gain = 0;
                
                // For each clause containing v
                for (int cid : adj[v]) {
                    // Determine effect of flipping v on clause cid
                    
                    // 1. Is it currently satisfied?
                    bool currently_sat = false;
                    bool will_be_sat_after_flip = false;
                    
                    // Check CURRENTLY
                    // Need to check if OTHER literals satisfy it or if v itself satisfies it.
                    // We can compute this directly.
                    
                    // Before flip
                    for (size_t k = 0; k < 3; ++k) {
                        int u = clauses[cid].lits[k].var_idx;
                        bool neg = clauses[cid].lits[k].negated;
                        bool val = neg ? !assignment[u] : assignment[u];
                        if(val) currently_sat = true; 
                    }
                    
                    // Wait, the above loop re-evaluates everything every time.
                    // Since N*M is small enough, it's fine.
                    
                    // To optimize slightly, we rely on the fact that only v changed.
                    // However, since we are re-calculating `currently_sat` inside the loop anyway:
                    
                    // Let's rewrite cleanly:
                    // We need to know if clause is sat WITHOUT v and WITH v (flipped).
                    // Actually we know v's current value. Let's denote it curr.
                    // New value is !curr.
                    
                    // Check if clause satisfied assuming v=curr
                    bool sat_curr = false;
                    for (size_t k = 0; k < 3; ++k) {
                         int u = clauses[cid].lits[k].var_idx;
                         bool neg = clauses[cid].lits[k].negated;
                         bool val = neg ? !assignment[u] : assignment[u];
                         if(val) sat_curr = true;
                    }
                    
                    // Check if clause satisfied assuming v=!curr
                    bool sat_flip = false;
                    assignment[v] ^= 1; // Toggle v
                    for (size_t k = 0; k < 3; ++k) {
                         int u = clauses[cid].lits[k].var_idx;
                         bool neg = clauses[cid].lits[k].negated;
                         bool val = neg ? !assignment[u] : assignment[u];
                         if(val) sat_flip = true;
                    }
                    
                    // Restore v
                    assignment[v] ^= 1;
                    
                    if (sat_curr && !sat_flip) gain--;
                    else if (!sat_curr && sat_flip) gain++;
                    // else no change
                
                    if (gain > best_gain) {
                        best_gain = gain;
                        cand = v;
                        // Optimization: if best_gain > 0, we found a beneficial move.
                        // Can we stop? We want the MAX gain to pick the best one for steepest ascent.
                        // If we just pick *any* positive gain, it works too, but max gain is standard.
                        // Also if best_gain > 1, definitely good.
                    }
                }
                
                if (best_gain > 0) {
                    // We found a move that improves score.
                    // Note: We must apply this move immediately and break out to restart search?
                    // Or apply best? Applying best is safer for local optima.
                    break; 
                }
            }
            
            // We broke out from 'for(v)'. Now check what happened.
            // Wait, the break is inside 'if(best_gain > 0)'.
            // This means we picked the FIRST v that gave a gain > 0.
            // This might suboptimal compared to picking the one with highest gain?
            // Does it matter? Usually not for local optima.
            // But for consistency, let's fix the logic:
            // Scan ALL vars, store best_gain and cand.
            // If best_gain > 0, flip and break outer loop.
            // Else break outer loop (stuck).
            
            // Correct logic inside 'while(true)':
            // Reset best_gain and cand
            // Loop v 0..n-1
            //   calc gain
            //   update best_gain/cand if better
            // If best_gain > 0:
            //    apply flip
            //    update clause statuses
            //    continue while loop (keep searching for better)
            // Else:
            //    break while loop
            
            // Re-implementing the loop block properly below.
        }
    }
}
```
Wait, the inner loop logic in the code draft is slightly broken regarding the flow.
Let's rewrite the loop part clearly.

```cpp
        // ... Initialization ...
        
        while (true) {
            int max_gain = 0;
            int best_v = -1;
            
            for (int v = 0; v < n; ++v) {
                int gain = 0;
                for (int cid : adj[v]) {
                    // Evaluate with current assignment
                    bool sat_before = false;
                    bool sat_after = false;
                    
                    // Check BEFORE flip (assignment[v] is as is)
                    for (auto& lit : clauses[cid].lits) {
                        bool val = lit.negated ? !assignment[lit.var_idx] : assignment[lit.var_idx];
                        if (val) sat_before = true;
                    }
                    
                    // Check AFTER flip (flip v temporarily or logically)
                    // Logical: new val is !assignment[v]
                    // Recompute using temporary swap
                    assignment[v] = 1 - assignment[v];
                    for (auto& lit : clauses[cid].lits) {
                        bool val = lit.negated ? !assignment[lit.var_idx] : assignment[lit.var_idx];
                        if (val) sat_after = true;
                    }
                    assignment[v] = 1 - assignment[v]; // restore
                    
                    if (!sat_before && sat_after) gain++;
                    else if (sat_before && !sat_after) gain--;
                }
                
                if (gain > max_gain) {
                    max_gain = gain;
                    best_v = v;
                }
            }
            
            if (max_gain > 0) {
                // Perform the flip of best_v
                assignment[best_v] = 1 - assignment[best_v];
                // Update satisfied status of clauses involving best_v
                for (int cid : adj[best_v]) {
                    clause_satisfied[cid] = evaluate_clause(cid, assignment);
                }
            } else {
                // Local Optimum reached
                // Sometimes we can accept a worse move to escape?
                // But with multiple restarts, usually pure greedy is enough.
                // Adding some random walk might help.
                // Let's stick to pure greedy to keep code simple and fast.
                // If stuck, we just stop this trajectory.
                break;
            }
        }
        // End of inner loop (hill climbing)
        
        // Check if we got a better global score
        int current_total_score = 0;
        for(bool s : clause_satisfied) current_total_score += s;
        if(current_total_score > max_score) {
             max_score = current_total_score;
             best_assignment = assignment;
        }
    }
```

Wait, `evaluate_clause` is called inside the loop every time for clauses involving `best_v`.
The `clause_satisfied` array keeps track of the global state.
But inside the `while(true)` loop, when checking gains, we need to know if flipping `v` improves the state.
My logic `gain++` or `gain--` depends on comparing `sat_before` and `sat_after`.
`clause_satisfied` is maintained in sync with `assignment`.
So `sat_before` corresponds to `clause_satisfied[cid]`?
NO! `clause_satisfied[cid]` tells us if clause is satisfied.
But flipping `v` might turn a satisfied clause into unsatisfied if `v` was the sole satisfier.
OR an unsatisfied clause might become satisfied.
To determine this correctly, we MUST re-evaluate the specific clauses involving `v`.
Just reading `clause_satisfied[cid]` is insufficient because we don't know *why* it is satisfied.
So yes, inside the loop we must re-evaluate the clauses.
This adds a factor of 3. Still fine.

Wait, optimization:
We can maintain the "count of true literals" for each clause.
Then, flipping `v` in clause `cid` changes the count.
If count > 0, it's satisfied. If count drops to 0, it becomes unsatisfied.
Maintaining literal counts:
Array `literal_counts[m]`.
Init: `count[cid] = 0`. Increment for true literals.
When flipping `v`:
For each `cid` in `adj[v]`:
  Decrement count if literal `v` contributed to count.
  Increment count if literal `v` now contributes (was false, became true).
  Update `clause_satisfied` based on count > 0.
This avoids recomputing `evaluate_clause` (loop of 3) for each check.
It makes it purely `O(deg(v))` per check.
Sum of degrees is `3M`.
Checking all `v` takes `O(3M)`.
This is better.

Let's refine the implementation to use literal counts.
`cnt[i]` = number of true literals in clause `i`.
Initially computed.
When flipping `v`:
  For each clause `i` in `adj[v]`:
    Lit value before: `old_val`
    Lit value after: `new_val`
    If `old_val` was true and `new_val` is false: `cnt[i]--`
    If `old_val` was false and `new_val` is true: `cnt[i]++`
    Update `sat[i] = (cnt[i] > 0)`
    Update `total_satisfied` accordingly.
    
    Wait, `cnt[i]` tracks literals. We can't easily access "is this specific literal true?" without looking at `assignment` and `lit`.
    We need `val` before and after.
    So for `cid` in `adj[v]`:
      Look up literal index of `v` in `clauses[cid].lits`. It appears at some position `pos`.
      Get `old_val` from `assignment` (based on `clauses[cid].lits[pos].negated`).
      Set `new_val` = !`old_val`.
      If `old_val` is true: `cnt[cid]--`; `sat[cid]` becomes `(cnt[cid]>0)`
      If `new_val` is true: `cnt[cid]++`; `sat[cid]` becomes `(cnt[cid]>0)`
      Wait, if `old_val` true -> `cnt` decreases. `sat` might stay true if other literals true.
      If `old_val` false -> `cnt` doesn't change yet. We check `new_val`.
      Actually simpler:
      Compute contribution of `v` to `cnt[cid]`.
      Remove it. Compute `cnt` without `v`. Check if `> 0`. If so, it was satisfied without `v` too?
      No, easier to just use the counts logic:
      
      Current `cnt[cid]` includes `v`'s contribution.
      `bool v_contributes = (neg ? !assign[v] : assign[v])`.
      `cnt[cid] -= (v_contributes ? 1 : 0)`.
      `bool will_contribute = (neg ? !(1-assign[v]) : (1-assign[v]))`.
      `cnt[cid] += (will_contribute ? 1 : 0)`.
      `sat[cid] = (cnt[cid] > 0)`.
      
      This logic is perfectly sound and efficient.
      
      Complexity:
      Precompute: $O(M)$.
      Per Flip Step:
        Iterate all `v` (N times).
        For each `v`, iterate `adj[v]` (size deg).
        Operations inside: constant.
        Total ops per step: $\sum_{v} deg(v) = 3M$.
        So finding best flip takes $O(M)$.
        Number of steps: At most $M$ (score increases by at least 1).
        Total per restart: $O(M^2)$.
        50 restarts -> $50 \times 10^6 = 5 \times 10^7$.
        Very fast.

One edge case: `cnt[cid]` initialized.
Variable mapping:
Input: `a` in `[-n, n]`.
Map `a` to index `abs(a)-1`. Flag `sign = (a < 0)`.
`lit_val` is true if `(sign != assignment[idx])`. i.e., `if(sign) !assign[idx] else assign[idx]`.

Let's double check memory/time.
Memory: minimal.
Time: 50 million ops is well within 1 second limit (usually ~100M ops/sec).

One tiny detail:
When calculating `max_gain`:
We check gain for *each* variable.
Gain = `(new_sat_count) - (current_sat_count)`.
Change in total score.
Since `current_sat_count` is fixed during the check loop (unless we modify `sat` array? No, `sat` should reflect current assignment).
Wait, if I modify `cnt` and `sat` inside the check for `v1`, it messes up checking `v2`.
So:
Inner check loop (scan all vars):
  Iterate `v`.
  For `cid` in `adj[v]`:
    Temporarily modify `cnt[cid]` and check `sat` vs previous `sat[cid]`.
    Accumulate delta.
    IMPORTANT: Do NOT permanently update `cnt[sat]` inside the check loop. Revert or copy?
    Reverting is easy: undo the count modification.
    Or simply:
    `delta = 0`.
    For `cid` in `adj[v]`:
      `int old_sat_contrib = 1 if clause satisfied by this literal? else 0`.
      Wait, tracking "contributed by this literal" is tricky if clause satisfied by others.
      BUT:
      If clause is satisfied (`sat[cid]==true`) AND count drops to 0 -> -1 to score.
      If clause is NOT satisfied (`sat[cid]==false`) AND count rises to >0 -> +1 to score.
      Otherwise 0.
      So, calculate `cnt` excluding `v`. Let it be `cnt_minus_v`.
      `sat_minus_v = (cnt_minus_v > 0)`.
      Current `sat_curr = (cnt > 0)`. (where `cnt` is actual count including `v`).
      Change `sat_curr` -> `sat_minus_v` (removing `v`) -> `sat_plus_new` (adding flipped `v`).
      Wait.
      `cnt` currently includes contribution of `v` (let's call it `v_contrib`).
      `cnt_minus_v = cnt - v_contrib`.
      If we flip `v`, new `v_new_contrib = 1 - v_contrib`.
      `cnt_new = cnt_minus_v + v_new_contrib`.
      We compare `(cnt_minus_v > 0)` vs `(cnt > 0)`? No.
      We compare `sat_curr` vs `sat_new`.
      Wait. `sat_curr` is true if `cnt > 0`.
      We don't need `cnt_minus_v` explicitly.
      We can just compute `sat_curr` (using `cnt` and knowing it matches array) and `sat_new`.
      But to do this efficiently, we need to know `cnt` values.
      We can't just change `cnt` array.
      But wait, we have `cnt` array already computed.
      `cnt` represents current state.
      `sat` array is derived from `cnt`.
      For a clause `cid`:
        Is it satisfied? `sat[cid]`.
        Is `v` contributing? `bool contrib = (neg ? !assignment[v] : assignment[v])`.
        If `contrib`:
           `temp_cnt = cnt[cid] - 1`.
           `is_sat_now = temp_cnt > 0`.
        Else (`not contrib`):
           `temp_cnt = cnt[cid]`.
           `is_sat_now = false`? No, `is_sat_now` refers to state WITHOUT `v`.
           If `v` was not contributing, removing it leaves `cnt` same. `sat` remains same.
           So if `v` was not contributing, flipping `v` (making it contribute) implies:
           `new_cnt = cnt[cid] + 1`.
           `sat_new = (new_cnt > 0)`.
           If `v` was contributing, flipping `v` (making it not contribute) implies:
           `new_cnt = cnt[cid] - 1`.
           `sat_new = (new_cnt > 0)`.
      So logic:
      `bool was_contributing = (neg ? !assignment[v] : assignment[v]);`
      `int current_cnt = cnt[cid];`
      `int future_cnt = current_cnt - (was_contributing ? 1 : 0) + (1 - was_contributing ? 1 : 0);` // 1 - contributing? No, flipping means contribution becomes `1 - was`.
      Wait.
      If `was_contributing` is true (val=1), flip -> val=0 (0 contrib). Change: -1.
      If `was_contributing` is false (val=0), flip -> val=1 (1 contrib). Change: +1.
      So `future_cnt = current_cnt + (was_contributing ? -1 : 1);`
      `sat_curr = (current_cnt > 0);`
      `sat_new = (future_cnt > 0);`
      `gain += (sat_new ? 1 : 0) - (sat_curr ? 1 : 0);`
      
      This allows checking gain for `v` in `O(deg(v))` without modifying arrays.
      Wait, `cnt` array IS modified when we perform a flip.
      When searching for the BEST flip among all `v`, we must treat `cnt` as read-only (snapshot of current state).
      Since we are just simulating gains, we don't touch `cnt` inside the check loop.
      So we just need `cnt` and `assignment` which are current.
      Perfect.
      
      Wait, `cnt[cid]` stores the count of TRUE literals.
      If `cnt[cid] > 0`, it's satisfied.
      We check hypothetical flip.
      This seems solid.

      Implementation details:
      Global variables or passed by ref.
      
      Wait, what if `cnt[cid]` is already 0?
      Then `was_contributing` must be 0? No. If count is 0, nothing contributes. So `was_contributing` must be 0.
      If `was_contributing` was 0, future_cnt = 1. Satisfied becomes true. Gain +1.
      
      What if `cnt[cid]` is 1?
      If `was_contributing` is 1. Future_cnt = 0. Satisfied becomes false. Gain -1.
      If `was_contributing` is 0. Future_cnt = 1. Satisfied stays true (still 1). Gain 0.
      
      Logic holds.

      Data structures:
      `vector<int> cnt(m, 0)`
      `vector<bool> sat(m)` // redundant if using cnt, but useful for final score
      Actually `sat[i] = (cnt[i] > 0)`.
      
      One catch: `cnt` needs to be updated when a flip happens (committing the step).
      Yes, `cnt` is stateful between steps of outer loop.
      During inner scan (finding next best flip), we assume `cnt` is static.
      So when we find best `v`, we commit to it:
      `assignment[v] = !assignment[v]`
      `update cnt` for clauses in `adj[v]`
      
      Wait, I said earlier "update status" -> `cnt` is the status.
      So:
      Loop:
         Find best `v` maximizing gain (computed via `cnt` and `assignment` without changing them).
         If best_gain > 0:
            Update `assignment[v]`
            Update `cnt` for all `c` in `adj[v]`
            (Recompute `sat` implicitly)
            `score_diff += best_gain`
         Else:
            Break (stuck)
      
      This looks perfect.

Double check the inputs.
Line 1: n m.
Next m lines: a b c.
a, b, c in [-n, n].
Non-zero.
Positive x -> x. Negative -x -> !x.

Example:
1 1
1 1 -1
Wait, a b c distinct? Problem doesn't say distinct. Just "three literals".
Can be same variable. e.g. `x v x v x`. This simplifies to `x`.
Logic handles it naturally (multiple increments).

Scoring:
If m > 0: score = s / m.
Output n integers.

Okay, writing the code.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct Literal {
    int var_idx;
    bool negated;
};

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    struct Clause {
        vector<Literal> lits;
    };
    vector<Clause> clauses(m);
    vector<vector<int>> adj(n);
    vector<int> clause_cnt(m, 0); // Number of true literals in clause

    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        Clause cl;
        cl.lits.reserve(3);
        
        auto parse = [&](int x, int& idx, bool& neg) {
            if (x > 0) { idx = x - 1; neg = false; }
            else          { idx = -x - 1; neg = true; }
        };
        
        int lu, lv, lw;
        bool nu, nv, nw;
        parse(u, lu, nu);
        parse(v, lv, nv);
        parse(w, lw, nw);
        
        cl.lits.push_back({lu, nu});
        cl.lits.push_back({lv, nv});
        cl.lits.push_back({lw, nw});
        
        clauses[i] = cl;
        
        // Update counts
        // Wait, counts need assignment. Initially assign false (0)? 
        // Initial counts should be 0 if assignment is all 0 (False).
        // Assuming assignment starts all False?
        // Problem: "assign truth values". Usually default to something.
        // I will iterate over assignment which is initially initialized.
        // Let's init assignment to all False.
        
        // Actually, I will init assignment later.
        // Just populate adjacency list first.
        adj[lu].push_back(i);
        adj[lv].push_back(i);
        adj[lw].push_back(i);
    }
    
    // Initial Assignment
    vector<bool> assignment(n, false); // Default to false
    
    // Initialize counts based on this assignment
    for(int i=0; i<m; ++i) {
        bool lit_vals[] = { /* ... */ };
        // Just a loop
        for(size_t k=0; k<clauses[i].lits.size(); ++k) {
            const auto& l = clauses[i].lits[k];
            bool val = l.negated ? !assignment[l.var_idx] : assignment[l.var_idx];
            if(val) clause_cnt[i]++;
        }
    }
    
    // Helper to check score
    auto get_score = [&]() {
        int score = 0;
        for(int i=0; i<m; ++i) if(clause_cnt[i] > 0) score++;
        return score;
    };
    
    // Helper to flip and update
    auto update_state = [&](int v) {
        assignment[v] = !assignment[v];
        // Update adjacent clauses
        for(int cid : adj[v]) {
            // Identify which literal is v in this clause
            // There is at most one.
            // Find it.
            for(size_t k=0; k<clauses[cid].lits.size(); ++k) {
                if(clauses[cid].lits[k].var_idx == v) {
                    // Calculate contribution change
                    // Before: assignment[v] was old
                    // After: assignment[v] is new
                    // Contribution was: (neg ? !old : old)
                    // New contribution: (neg ? !new : new)
                    // Old val: assignment[v] (before flip) -> NO, we flipped it.
                    // So we need to know what the contribution WAS.
                    // It is determined by OLD assignment.
                    // But wait, I set assignment[v] = !assignment[v] just now.
                    // So current assignment[v] is the NEW value.
                    // We need to revert or compute carefully.
                    
                    // Let's do it differently:
                    // Determine if it was contributing before flip.
                    // It was contributing if: !assignment[v] (if neg) OR assignment[v] (if pos).
                    // Wait, assignment[v] is NOW new.
                    // So we need to reverse logic.
                    
                    // Let's just recompute contribution of v based on CURRENT assignment?
                    // No, that's new.
                    // Let's use the fact that `val` toggled.
                    
                    bool was_contrib = false;
                    // We can infer:
                    // If literal is negated: old val was assignment[v]_old.
                    // New val is !assignment[v]_old.
                    // contribution was (neg ? !assignment[v]_old : assignment[v]_old)
                    // contribution now is (neg ? !assignment[v] : assignment[v])
                    
                    // Instead, let's look at `clause_cnt[cid]` directly.
                    // `clause_cnt` holds the count including the OLD value.
                    // If the literal was true (contributing), count decreases by 1.
                    // If the literal was false (not contributing), count increases by 1 (assuming it becomes true).
                    // Is it true that if it was false, it becomes true? Yes, because we flip boolean.
                    
                    // Check if literal was true:
                    bool is_true_now = (clauses[cid].lits[k].negated ? !assignment[v] : assignment[v]);
                    // Wait, this is the NEW state.
                    // If `is_true_now` is true:
                    //   If it wasn't true before: cnt increases.
                    //   If it was true before: cnt decreases.
                    // If `is_true_now` is false:
                    //   If it wasn't true before: cnt unchanged.
                    //   If it was true before: cnt decreases.
                    // Basically:
                    // cnt -= (was true ? 1 : 0);
                    // cnt += (is true now ? 1 : 0);
                    
                    // We need to know `was true`.
                    // Since `assignment[v]` is already updated, `was true` is `!is_true_now`.
                    
                    bool was_true = !is_true_now;
                    
                    if (was_true) clause_cnt[cid]--;
                    if (is_true_now) clause_cnt[cid]++;
                    
                    // Wait, is this correct?
                    // Example: Lit = v (positive).
                    // Old assignment[v] = 0. cnt does NOT count this.
                    // Flip. New assignment[v] = 1. cnt SHOULD count this.
                    // Logic: is_true_now = 1. was_true = 0.
                    // Decrease by 0. Increase by 1. Net +1. Correct.
                    
                    // Example: Lit = -v (negative).
                    // Old assignment[v] = 0. Literal value = !0 = 1. cnt counts this.
                    // Flip. New assignment[v] = 1. Literal value = !1 = 0.
                    // Logic: is_true_now = 0. was_true = 1.
                    // Decrease by 1. Increase by 0. Net -1. Correct.
                    
                    // Example: Lit = v. Old 1. New 0.
                    // is_true_now = 0. was_true = 1.
                    // Decrease 1. Increase 0. Net -1. Correct.
                    
                    // Example: Lit = -v. Old 1. New 0.
                    // Old assignment[v]=0. Wait, old value was 1.
                    // Logic: assignment[v] flipped.
                    // If I use `is_true_now` on NEW assignment.
                    // And `was_true` is inferred as `!is_true_now`.
                    // This assumes binary toggle.
                    // Yes, boolean variables flip.
                    // So `was_true` is strictly `!is_true_now` relative to the flip logic ONLY IF we assume the literal's truth value also flips.
                    // Which it does: if literal is `x`, and `x` flips, truth value flips.
                    // If literal is `!x`, and `x` flips, `!x` flips.
                    // So truth value always flips for the literal.
                    // So `was_true` is indeed `!is_true_now`.
                    
                    // WAIT. `clause_cnt` stores the count of TRUTHFUL literals.
                    // Does `clause_cnt` depend on OTHER variables? No.
                    // So yes, if I know `cnt` reflects the state with `assignment[v]` (OLD),
                    // and I update `cnt` to reflect `assignment[v]` (NEW),
                    // then the change is `+1` if literal becomes true, `-1` if literal becomes false.
                    // Becoming true means `is_true_now` is true.
                    // Becoming false means `is_true_now` is false (implies was true).
                    // So change is `is_true_now ? +1 : -1`?
                    // No. If `is_true_now` is true, it means it became true (either from false->true or true->true?).
                    // Since it flips, it MUST have come from false.
                    // Wait. If it was true, it becomes false.
                    // If it was false, it becomes true.
                    // It can't go True -> True or False -> False.
                    // So change is ALWAYS `is_true_now ? +1 : -1`?
                    // Let's verify.
                    // Case 1: Lit was false, becomes true. (change +1). `is_true_now` is true. Matches.
                    // Case 2: Lit was true, becomes false. (change -1). `is_true_now` is false. Matches.
                    // YES! It is simply `+1` if `is_true_now`, else `-1`.
                    // Because `cnt` was tracking truthfulness. The literal's contribution flips.
                    // If it's true now, it contributed +1 to NEW count. Previously contributed 0. Diff +1.
                    // If it's false now, it contributed 0 to NEW count. Previously contributed +1. Diff -1.
                    
                    // Wait. This logic holds if `cnt` was updated ONLY by this variable?
                    // Yes, we are updating `cnt` for the specific clauses affected.
                    // So yes, `clause_cnt[cid] += (is_true_now ? 1 : -1)`.
                    
                    clause_cnt[cid] += (is_true_now ? 1 : -1);
                }
            }
        }
    };
    
    // Wait, the update logic needs to be careful.
    // If a clause contains `v` TWICE (e.g. x v x y), then `adj[v]` contains the clause index twice?
    // My adjacency construction: `adj[lu].push_back(i)` etc.
    // If `lu` == `lv`, `adj` will have duplicate indices.
    // Loop over `adj[v]` will iterate `cid` twice.
    // If I update `cnt[cid]` twice for same clause?
    // That would be WRONG.
    // If `cid` appears twice in `adj[v]`, I must handle it.
    // Ideally, `adj` should store unique clause indices.
    // Or use a map/set for adj.
    // Given N=50, M=1000, duplicates are possible.
    // I should store unique IDs in `adj`.
    // Or check `cid` equality before processing.
    // Let's use `std::sort` + `unique` or `set` to avoid duplicates?
    // Building `adj` with `vector<vector<int>>` is fast.
    // After building, sort and unique.
    // Wait, `lits` order matters?
    // `lit_idx` is derived.
    // Duplicates in `adj` mean `clause_cnt` is updated multiple times for the SAME clause.
    // If clause has `v` twice, `cnt` change should account for both literals.
    // If I iterate `adj` twice, I apply update twice. That is CORRECT.
    // Because if a clause has two copies of `v`, flipping `v` affects BOTH copies.
    // Both copies flip from T->F or F->T.
    // So `cnt` changes by +1/+1 or -1/-1.
    // So duplicates in `adj` are fine and actually desirable for logic.
    // BUT, I need to identify WHICH literal index `k` corresponds to `v` in the CLAUSE definition.
    // My update loop does:
    // `for(size_t k=0; k<clauses[cid].lits.size(); ++k) if(lit.var_idx == v)`
    // If `v` appears twice in clause, this loop runs twice.
    // So I update `cnt` twice. Correct.
    // So duplicates in `adj` don't matter?
    // Wait, `adj` contains `cid`. If I iterate `adj[v]`, I visit `cid` for each occurrence of `v` in the clause.
    // Example: Clause `(v, v, u)`.
    // `adj[v]` has `cid` twice.
    // Outer loop iterates `cid` twice.
    // First time: finds first `v` in clause. Updates `cnt[cid]`.
    // Second time: finds second `v` in clause. Updates `cnt[cid]`.
    // Total change for clause: applied twice. Correct.
    // So `adj` duplicates are handled correctly by logic.
    
    // However, sorting `adj` helps performance if we want to avoid duplicate scans?
    // No, duplicates are needed to count correctly if `adj` was constructed naively.
    // Actually, constructing `adj` naively is simplest.
    
    // But wait, I implemented `update_state` inside a loop checking `clauses[cid].lits`.
    // If `adj[v]` has duplicates, I re-scan the whole clause for each duplicate?
    // `clauses[cid].lits` has 3 items. Scanning is cheap.
    // So yes, naive construction works.
    
    // Wait, I should ensure `cnt` update logic is correct.
    // The loop `for(size_t k... if(...) clause_cnt[cid] += ...`
    // This finds all occurrences of `v` in clause.
    // If there are 2 occurrences, it applies the correction 2 times.
    // This is correct because `cnt` sums up contributions of ALL literals.
    // Flipping `v` flips the contribution of each occurrence.
    
    // Wait, `adj` construction pushes `i` multiple times.
    // If I just use a simple loop, it works.
    
    // Final check on `update_state`:
    // `is_true_now` is computed based on NEW `assignment[v]`.
    // If `lit` was `v`: `!new` (false). `!new` is false -> `is_true_now` = false.
    // Wait, logic:
    // `bool val = (clauses[cid].lits[k].negated ? !assignment[v] : assignment[v]);`
    // If neg=false (just v): val = assignment[v].
    // If neg=true (-v): val = !assignment[v].
    // This computes current truth.
    // Logic `cnt += (val ? 1 : -1)`.
    // Is this correct?
    // `cnt` currently holds value assuming `OLD` assignment.
    // The literal's contribution flips.
    // If it was contributing (True), it is now contributing (False). Change -1.
    // If it was not contributing (False), it is now contributing (True). Change +1.
    // Wait.
    // If `val` is computed from `NEW` assignment.
    // If `val` is True:
    //   It means literal is True NOW.
    //   Did it contribute BEFORE?
    //   Since it FLIPS truth value:
    //   If it is True NOW, it must have been FALSE BEFORE.
    //   So it went 0 -> 1. Change +1.
    // If `val` is False:
    //   It means literal is False NOW.
    //   So it must have been TRUE BEFORE.
    //   So it went 1 -> 0. Change -1.
    // So `cnt += (val ? 1 : -1)` is indeed correct.
    
    // Wait. Is it guaranteed that truth value flips?
    // Yes. Variable flips.
    // Literal `v` flips `v`.
    // Literal `-v` flips `-v`.
    // Both flip.
    // So yes.
    
    // One optimization: `clause_cnt` update is fast.
    // Finding `k` in `lits` is fast (size 3).
    
    // Algorithm Flow:
    // 1. Build data structures.
    // 2. Init assignment (randomly).
    // 3. Init `cnt` based on assignment.
    // 4. Best score tracker.
    // 5. Repeated attempts (restarts).
    //    In each attempt:
    //      While (found improving flip):
    //        Scan all v to find best gain.
    //        If none improving, break.
    //        Pick best v.
    //        Flip v (update assignment and cnt).
    //      Save best assignment.
    
    // Note: I need to initialize `cnt` *after* picking a random assignment.
    // In my previous thought, I init `cnt` assuming all false.
    // I need to change that.
    
    // I will wrap everything in a clean function.
    // Use `mt19937` for randomness.
    
    // Corner case: m=0.
    // Loop won't execute anything. Score 1.
    // Code should handle m=0 gracefully (loops over m will skip).
    // But `cin >> n >> m` works.
    
    // Output: `x1 ... xn`.
    // If `best_assignment` is not saved globally, I need to save it.
    // I'll maintain `max_score` and `best_assign` vector.
    
    // Include headers.
    // Use fast I/O.
}
```
Looks good.

One small logic error check:
`is_true_now` calculation uses `assignment[v]`.
Wait, I need to calculate `is_true_now` for EACH `k` in `lits`.
But inside `adj[v]` loop, I'm updating `cnt` multiple times.
Wait, if I loop `k` to find `v`, I should update `cnt` for that instance.
Correct.

Wait, `adj` construction:
`adj[lu].push_back(i);`
If `lu` and `lv` are different variables, they are added separately.
If `lu` == `lv`, `adj` has `i` twice.
Inside `update_state(v)`:
Loop `cid` in `adj[v]`.
Inside, loop `k` over `lits`.
If `lits[k]` corresponds to `v`.
Update `cnt`.
Since `adj` has duplicates, `cid` is visited multiple times.
So `cnt` updated multiple times. Correct.
If `adj` had NO duplicates, `cnt` would only be updated once.
BUT, if clause is `(v, v, u)`, `cnt` tracks sum of trues.
If `v` is true, both literals contribute +1. Total +2.
Flipping `v`: both become false. Total -2.
If `adj` has `cid` once:
  Loop `k`:
    Finds first `v`. `cnt` updates (+/-1).
    Loop continues? No, if `adj` has `cid` once, we loop `k` 0..2.
    We find `v` at `k=0` -> update.
    We find `v` at `k=1` -> update.
    So we update twice regardless of `adj` content.
    So duplicates in `adj` are redundant if we scan `lits`.
    Wait, if I scan `lits` completely, I find ALL `v`s.
    So `adj` shouldn't have duplicates to avoid re-scanning the clause multiple times for the same `cid`.
    Scanning `lits` takes 3 iterations.
    Scanning `adj` takes degree.
    Sum of degrees is 3M.
    If I remove duplicates from `adj`, degree sum is 3M (assuming distinct vars in clause, or rather 1 entry per var-instance in clause? No).
    Wait. If clause has `(v, u, w)`. `adj[v]` has `cid`.
    Loop `k`: finds `v` at 0. Updates `cnt`.
    Finds `u` at 1. Skips.
    Finds `w` at 2. Skips.
    Total 1 update. Correct.
    If I kept duplicate `cid` in `adj[v]` (because `adj` was built by pushing `i` for each var match):
    Wait, I pushed `i` for `lu`, `lv`, `lw`.
    If `lu=v` and `lv=u` and `lw=w` (distinct), `adj[v]` has 1 entry `cid`.
    Wait, my loop `for(const auto& lit : cl.lits) adj[lit.var_idx].push_back(i);`
    So `adj` entries correspond to occurrences.
    If `lu=v` and `lv=v`, `adj[v]` gets `cid` twice.
    Inside `update_state`:
    We loop `adj[v]` -> encounter `cid` twice.
    Encounter 1: Loop `k`. Finds `v` at 0. `cnt` updated.
    Encounter 2: Loop `k`. Finds `v` at 1 (same clause). `cnt` updated.
    Total: `cnt` updated twice. Correct.
    Wait, scanning `lits` takes 3 ops.
    Doing this 2 times takes 6 ops.
    Ideally we do 6 ops once? No, we need to update twice.
    So duplicates in `adj` just cause duplicated work.
    Since work is small, leaving duplicates is fine.
    However, removing duplicates from `adj` (by keeping track if processed) is cleaner?
    No, simpler to leave it, 3M is small.
    Wait, actually `adj` duplicates `cid` is fine.
    
    Wait, inside `adj` loop, I scan `lits`.
    If I have duplicates in `adj`, I scan `lits` fully.
    If I scan `lits`, I find `v` again.
    So `cnt` updated again.
    It is correct.
    But `adj` having duplicates means `cnt` update happens multiple times for SAME clause instance?
    Wait. `cnt` is an accumulator.
    `cnt` should reflect sum of `true` literals.
    If clause has `v` twice, `cnt` starts with +2 if `v` true.
    Flipping `v`: `cnt` becomes +0.
    Difference -2.
    My logic updates -1 for each occurrence.
    So -1 + -1 = -2.
    So it is correct.
    
    BUT, if I remove duplicates from `adj`, then inside the loop over `adj[v]`:
    `adj[v]` has `cid` once.
    Scan `lits`:
    Find `v` at 0. Update `cnt`.
    Find `v` at 1. Update `cnt`.
    Result: -2.
    Same result.
    So duplicates in `adj` are redundant for correctness but increase overhead.
    Overhead is trivial. I'll stick to simple construction.
    Actually, `adj` duplicates are fine, but note:
    If I use `adj[v]` to identify clauses, I am essentially asking "which clauses contain v?".
    If `v` is in `cid` twice, `cid` appears twice in `adj[v]`.
    So I visit `cid` twice.
    So redundancy is real.
    I will just accept it.
    
    Wait, one issue:
    `update_state` modifies `cnt` and `assignment`.
    The search loop (scanning `v`) reads `cnt` and `assignment`.
    If I perform updates INSIDE the search loop (bad), it messes up.
    I only perform updates AFTER choosing the best `v`.
    So the state is consistent during search.
    
    Okay, code structure:
    - Init `assignment`
    - Init `cnt`
    - Best assignment tracker
    
    Inside main loop:
    - While(true)
      - Compute best `v` by iterating all `v` (N times).
        - For each `v`, compute gain by iterating `adj[v]` (degree).
          - For each `cid` in `adj[v]`:
            - Scan `lits` to find occurrences of `v`.
            - For each occurrence:
               - Calc `val_now`.
               - `delta += (val_now ? 1 : -1)`.
            - Note: `cnt[cid]` must be used for reference? No, `gain` depends on `sat` change.
            - I need `sat` change.
            - `sat_curr` depends on `cnt[cid]`.
            - `sat_new` depends on `cnt[cid] + delta`.
            - Wait, `cnt[cid]` is static during the check.
            - `sat_curr = (cnt[cid] > 0)`.
            - `sat_new = (cnt[cid] + delta > 0)`.
            - `gain += (sat_new ? 1 : 0) - (sat_curr ? 1 : 0)`.
      - If max_gain > 0:
          - Flip `v`. Update `cnt` and `assignment` for `adj[v]`.
          - Continue.
      - Else:
          - Break.
    - Update global best.
    
    Wait, calculating `delta`:
    The loop `for(size_t k...)` finds all occurrences of `v`.
    For each occurrence, `cnt` changes by `+1` or `-1`.
    `delta` accumulates these changes.
    Correct.
    
    Wait, I should check `gain` for the *whole* flip of variable `v`.
    If `v` appears in multiple clauses, total gain is sum of individual clause impacts.
    My loop over `adj[v]` does exactly that.
    
    Wait, I missed one thing: `cnt` might be 0 or >0.
    `sat_curr` is `(cnt[cid] > 0)`.
    If `cnt[cid]` is 1. `sat_curr` = true.
    `delta` calculated.
    `sat_new` = `(cnt[cid] + delta > 0)`.
    Correct.
    
    Seems complete.

```cpp
/*
Solution for Max-3SAT with N <= 50, M <= 1000.
Uses Randomized Local Search (Hill Climbing with random restarts).
*/
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct Clause {
    int lits_var[3];
    bool lits_neg[3];
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    // adj[v] contains list of clause indices where variable v appears
    // Duplicates are allowed (if v appears twice in a clause)
    vector<vector<int>> adj(n);

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        
        auto parse = [&](int x) -> pair<int, bool> {
            int idx = abs(x) - 1;
            bool neg = (x < 0);
            return {idx, neg};
        };
        
        pair<int, bool> la = parse(a);
        pair<int, bool> lb = parse(b);
        pair<int, bool> lc = parse(c);
        
        clauses[i].lits_var[0] = la.first; clauses[i].lits_neg[0] = la.second;
        clauses[i].lits_var[1] = lb.first; clauses[i].lits_neg[1] = lb.second;
        clauses[i].lits_var[2] = lc.first; clauses[i].lits_neg[2] = lc.second;
        
        adj[la.first].push_back(i);
        adj[lb.first].push_back(i);
        adj[lc.first].push_back(i);
    }

    // Variables: 1 means True, 0 means False
    // Using vector<bool> for assignment
    // We will use random restarts, so initial state is set outside main loop
    
    vector<int> best_assignment(n, 0);
    int max_satisfied = 0;
    
    // Adjacent to speed up checks: pre-calculate literal value logic?
    // Just inline logic.
    
    mt19937 rng(time(0));
    
    for (int r = 0; r < 50; ++r) { // 50 restarts
        // Random initialization
        vector<int> assignment(n);
        for (int i = 0; i < n; ++i) assignment[i] = (rng() % 2);

        // Initialize clause counts (number of true literals)
        vector<int> cnt(m, 0);
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < 3; ++k) {
                int u = clauses[i].lits_var[k];
                bool neg = clauses[i].lits_neg[k];
                bool val = neg ? !assignment[u] : assignment[u];
                if (val) cnt[i]++;
            }
        }

        // Hill Climbing
        while (true) {
            int max_gain = 0;
            int best_v = -1;
            
            // Check every variable to find the best flip
            for (int v = 0; v < n; ++v) {
                int gain = 0;
                // Iterate over clauses containing v
                for (int cid : adj[v]) {
                    // Calculate impact of flipping v on clause cid
                    int delta = 0;
                    // Count changes in this clause due to v
                    for (int k = 0; k < 3; ++k) {
                        int u = clauses[cid].lits_var[k];
                        bool neg = clauses[cid].lits_neg[k];
                        if (u == v) {
                            // Contribution flips
                            // We need to know current contribution to see direction
                            // Current assignment[v] is 'assign_val'
                            // But wait, assignment[v] is NOT flipped yet in this check loop
                            // We need to know if literal was TRUE before flip.
                            // Truth value depends on assignment[v] (before flip)
                            // If assigned[v] was true:
                            //   Literals true: val=True -> becomes False. Delta -1.
                            //   Literals neg: val=False -> becomes True. Delta +1.
                            // Generally: Delta = (NewVal - OldVal) * 1
                            // OldVal = (neg ? !assignment[v] : assignment[v])
                            // NewVal = (neg ? !(1-assignment[v]) : (1-assignment[v])) = !OldVal
                            // Delta = NewVal - OldVal (if viewed as integers 0/1)
                            // Wait. NewVal = !OldVal.
                            // So if OldVal=1, NewVal=0. Delta = -1.
                            // If OldVal=0, NewVal=1. Delta = +1.
                            // So Delta = (OldVal == 0 ? 1 : -1).
                            // OldVal = (neg ? !assignment[v] : assignment[v])
                            
                            bool old_val = (neg ? !assignment[v] : assignment[v]);
                            if (old_val) delta--; else delta++;
                        }
                    }
                    
                    // Now check impact on satisfaction
                    // Current satisfaction
                    bool sat_curr = (cnt[cid] > 0);
                    // Predicted satisfaction
                    // New count = cnt[cid] + delta
                    bool sat_new = (cnt[cid] + delta > 0);
                    
                    if (sat_new && !sat_curr) gain++;
                    else if (!sat_new && sat_curr) gain--;
                }
                
                if (gain > max_gain) {
                    max_gain = gain;
                    best_v = v;
                }
            }
            
            if (max_gain > 0) {
                // Apply flip
                // We need to apply it permanently
                // Update assignment
                assignment[best_v] = 1 - assignment[best_v];
                // Update cnt for affected clauses
                // We iterate adj[best_v] again
                // But wait, duplicates in adj? 
                // If duplicates, we must ensure we update cnt correctly for ALL occurrences.
                // As analyzed, updating cnt multiple times for a clause with multiple literals of v is correct.
                for (int cid : adj[best_v]) {
                    int change = 0;
                    for (int k = 0; k < 3; ++k) {
                        int u = clauses[cid].lits_var[k];
                        bool neg = clauses[cid].lits_neg[k];
                        if (u == best_v) {
                            // Now v is flipped. 
                            // Check contribution in NEW state (or deduce from logic)
                            // Wait, we already flipped assignment[best_v].
                            // So 'assignment[best_v]' is now the NEW value.
                            // We need to add/subtract difference.
                            // If it's true now, we ADD 1.
                            // If it's false now, we SUBTRACT 1? No.
                            // Logic derived: delta = (OldVal == 0 ? 1 : -1).
                            // But we need to update cnt.
                            // cnt stores old sum.
                            // We need new sum.
                            // New sum = Old sum + sum(delta over occurrences).
                            // Occurrences are independent.
                            // So we need to compute delta for each occurrence and add to cnt.
                            // delta_i = (OldVal_i == 0 ? 1 : -1).
                            // To find OldVal_i: we need to remember assignment before flip OR deduce from NEW state.
                            // Deduction:
                            // If neg: OldVal = !(assign_new).
                            // If neg: NewVal = !(assign_new).
                            // Wait. assign_new is current.
                            // OldVal was opposite.
                            // If neg: NewVal = !assign_new. OldVal = assign_new.
                            // If pos: NewVal = assign_new. OldVal = !assign_new.
                            // So OldVal = (neg ? assign_new : !assign_new).
                            // If OldVal is 1 -> delta = -1.
                            // If OldVal is 0 -> delta = +1.
                            // So cnt += ( (neg ? assignment[v] : !assignment[v]) ? -1 : 1 ).
                            
                            bool old_val = (neg ? assignment[best_v] : !assignment[best_v]);
                            if (old_val) change--; else change++;
                            
                            // But wait, duplicates in adj.
                            // If clause has v twice. adj has cid twice.
                            // Loop visits cid twice.
                            // First time: calculates delta. Adds to cnt.
                            // Second time: calculates delta. Adds to cnt.
                            // Correct.
                            cnt[cid] += change;
                        }
                    }
                }
            } else {
                // Local optimum
                break;
            }
        }
        
        // Check global best
        int current_score = 0;
        for (int c : cnt) if (c > 0) current_score++;
        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = assignment;
        }
    }
    
    // Output
    for (int i = 0; i < n; ++i) cout << (best_assignment[i] ? 1 : 0) << (i == n - 1 ? "" : " ");
    cout << endl;

    return 0;
}
```
One logic bug:
In `update` phase (the inner loop `if (max_gain > 0)` block), I iterate `adj[best_v]`.
Since `adj` contains duplicates (if var appears multiple times), I recompute `change` for each occurrence.
And `cnt` is accumulated.
Wait.
Example: `adj[best_v]` has `cid` twice.
First iteration: finds `k` corresponding to `v`. Calculates `change`. Adds to `cnt`.
Second iteration: finds `k` corresponding to `v`. Calculates `change`. Adds to `cnt`.
This implies `cnt` accumulates changes.
Is it correct?
Yes. If a clause has 2 `v`s, and we flip `v`, both literals flip.
If `v` was contributing to clause (count included), both contribute 1 to count.
Flip: count decreases by 2.
My logic: `change` calculated twice. `cnt += change` twice. Total decrease 2.
So duplicates in `adj` work perfectly for updating `cnt`.
BUT wait. `adj` stores `cid`.
If I iterate `adj` twice, I enter `cnt[cid] += change` twice.
Is `change` calculated correctly?
In the first pass, `cnt[cid]` is old.
In the second pass, `cnt[cid]` is partially updated?
Wait. `cnt` is shared.
If I update `cnt[cid]` in first visit, the second visit sees the updated `cnt`.
Does this affect correctness?
No, we are not using `cnt` to *calculate* change (we calculate change based on `assignment` only).
We use `cnt` to accumulate result.
So sequential addition is fine.
Wait, do I need to calculate `change` based on `assignment`?
Yes.
Does `cnt` usage inside `cnt[cid] += change` interfere? No.
So this logic is safe.

One minor issue:
If `adj` has duplicates, `cnt` is updated multiple times for SAME clause.
This is required if clause has multiple `v`s.
However, `adj` might contain `cid` even if `v` appears only ONCE in clause?
Yes, `adj[lu].push_back(i)` adds `i` for each occurrence.
If `lu=v`, `adj[v]` gets `i`.
If `lv!=v`, `adj[v]` doesn't get `i`.
So `adj[v]` contains `i` exactly equal to the number of times `v` appears in clause `i`.
So if `adj[v]` contains `cid` once, it means `v` appears once.
If `adj[v]` contains `cid` twice, `v` appears twice.
So iterating `adj[v]` and adding `change` for each `cid` is EXACTLY correct for updating `cnt` correctly.
Because `cnt` tracks total true literals. If 2 `v`s exist, we subtract 2 (if `v` was true).
If `v` appeared once, we subtract 1.
So this is correct.

Wait. The logic `cnt[cid] += change` relies on `change` being computed for that *specific* occurrence.
But if I iterate `adj[v]`, I visit `cid` for each occurrence.
Is `change` calculation dependent on which `k` we are at?
My loop `for (int k = 0; k < 3; ++k) if (u == best_v) ...`
This iterates all literals in clause.
If `v` appears twice, `k` matches twice.
So `change` is computed twice (once for each match).
AND `cnt[cid]` is updated twice (once per match).
Wait. If I iterate `k` for EACH visit of `cid` in `adj`.
Suppose `v` appears twice. `adj[v]` has `cid` twice.
Visit 1: `k` loop runs. Finds `v` at index 0. Computes `change`. Updates `cnt`.
Find `v` at index 1. Computes `change`. Updates `cnt`.
Visit 2: `k` loop runs. Finds `v` at index 0. Computes `change`. Updates `cnt`.
Find `v` at index 1. Computes `change`. Updates `cnt`.
Total updates: 4.
Total expected: 2 (since `v` appears twice, total contribution to `cnt` drops by 2).
So we OVERCOUNTED.
We should NOT iterate `k` if we rely on `adj` having one entry per `v`-occurrence?
Actually, `adj[v]` has 2 entries for `cid`.
If we want to simulate iterating occurrences of `v` in clause.
We should iterate `k` and check `u == best_v`.
If `k` loop finds `v`, it means "this is an occurrence".
So we update `cnt` ONCE for this occurrence.
BUT, if `adj` has 2 entries for `cid`, we enter `cid` loop 2 times.
In each entry, we scan `k` and update `cnt` for all `v` occurrences found in clause.
So if `v` is in clause twice, we update `cnt` TWICE for EACH occurrence.
Wait. If I find `v` at `k=0`. I update `cnt`.
If I find `v` at `k=1`. I update `cnt`.
This happens in Visit 1.
Then in Visit 2, I find `v` at `k=0`. I update `cnt`.
At `k=1`. I update `cnt`.
This is BAD. I update 4 times instead of 2.
I need to ensure that for each `cid`, I calculate total delta for ALL `v` occurrences and add ONCE?
No, `cnt` needs to be adjusted for EACH `v` occurrence.
If clause has `v, v`.
Total change is `delta(v) + delta(v)`.
If I visit `cid` twice, I do:
Visit 1: `delta(v) + delta(v)`.
Visit 2: `delta(v) + delta(v)`.
Total `2 * (delta + delta)`. Incorrect.
So `adj[v]` should NOT contain duplicates for `cid`.
It should contain `cid` ONCE.
BUT how to know number of `v`s in clause?
I can iterate `k` loop and count `v` occurrences?
Or just store unique `cid` in `adj`.
If I store unique `cid`, then inside `adj` loop:
  Iterate `k` loop.
  If `u == v`:
     Compute `delta`. Add to `total_delta_for_this_visit`.
  After `k` loop: `cnt[cid] += total_delta`.
This ensures we only update `cnt` once per clause.
So, I MUST REMOVE DUPLICATES from `adj`.
Easiest way: Use a `vector` and `sort` + `unique`, or check existence before adding.
Given N is small (50), simple linear check is fine?
Or just `vector<vector<int>> adj(n)` and `adj[lu].push_back(i)`... then `sort(unique)`.
Wait, `adj` needs to preserve multiplicity of `cid` in terms of "how many ways to visit"? No.
The adjacency list should represent "Which clauses contain this variable?".
Usually, we need to know for a clause `cid`, how many times `v` appears to update `cnt`.
So `adj` should map `v` -> `list of (cid, count_of_v_in_cid)`.
Or simply `adj` maps `v` -> `list of cid`.
And inside the update loop, we calculate the total delta for `cid` considering all occurrences of `v` in `cid`.
So `adj` should have UNIQUE `cid`.
So I will ensure uniqueness.
If I do this, then I calculate delta by scanning `k` and summing `change`.
Correct.

Wait, if I have duplicates in `adj` (meaning I push `cid` multiple times), I can just ignore duplicates?
No, that's bad.
So: Build `adj`, then for each row, sort and unique.
This costs negligible time.

Wait, if `v` appears twice in `cid`, `adj[v]` will have `cid` ONCE.
Inside loop:
  Iterate `cid` (once).
    Iterate `k` (0..2).
      Find `v` at `k1`. `delta1`.
      Find `v` at `k2`. `delta2`.
      Total `delta = delta1 + delta2`.
      `cnt[cid] += delta`.
This is correct.

So:
Add logic to make `adj` rows unique.
Wait, `adj` rows are `vector<int>`.
Constructing them:
```cpp
    // ...
    for(const auto& lit : cl.lits) {
        adj[lit.var_idx].push_back(i);
    }
    // ...
    // Remove duplicates
    for(int i=0; i<n; ++i) {
        auto &vec = adj[i];
        sort(vec.begin(), vec.end());
        vec.erase(unique(vec.begin(), vec.end()), vec.end());
    }
```
This ensures each clause index appears at most once per variable.

Wait, if `v` appears twice, I want to calculate the delta correctly (which is sum of deltas for each occurrence).
Does `k` loop cover both occurrences?
Yes. `clauses[cid].lits` contains all literals.
So iterating `k` and checking `u == v` covers ALL occurrences.
So summing `change` for all `k` works.
So `adj` needing unique `cid` is sufficient.

Okay, ready to code.

One detail:
`best_assignment` update.
The problem asks for output `n` integers.
I'll use `vector<int>` or `vector<bool>` then print.

One edge case: `max_satisfied` initialization.
If `m=0`, score 1. Code outputs assignment (all 0).
If `m>0`, `max_satisfied` starts at 0.
Is it possible `max_satisfied` remains 0? Yes.
Is it possible score is 0? Yes.
Correct.

Also, `mt19937 seed`.
Using `clock()` might give same seed if runs instantly.
I'll use `time(0)`.

Final check: Constraints.
`n` up to 50.
`m` up to 1000.
Execution time.
My solution:
`50` restarts.
Each restart:
  Hill climb:
    Inner loop: check all `v` (50).
    For each `v`: check `adj[v]`. Sum of `deg` is `3M` (since unique `adj`, sum is number of pairs `(v, cid)` such that `v` in `cid`).
    Actually, sum of `deg` is `3M`.
    So checking ALL `v` takes `3M` operations.
    Wait. `3M = 3000`.
    So one pass is 3000 ops.
    How many steps in hill climb?
    Score goes from 0 to M. Each step increases by at least 1. Max M steps.
    Total per restart: `M * (3M) = M^2`.
    `1000^2 = 1,000,000`.
    50 restarts -> `50,000,000` ops.
    Fast enough.

Memory:
`clauses`: `1000 * 3 * (4+1)` bytes ~ 15KB.
`adj`: `50 * 1000` ints ~ 200KB.
Tiny.

Looks good.

One detail: `unique` requires sorted vector.
Sorting `vector<int>` of size `M` (worst case if `v` in all clauses).
`3M/N` avg. Size is small. Sort is fast.

Final Check: `adj` update logic.
Inside `cnt` update loop:
`int change = 0;`
`for (int k=0; k<3; ++k) if (...) { change += (old_val ? -1 : 1); }`
`cnt[cid] += change;`
Wait. `change` is computed based on `assignment[best_v]` which is ALREADY UPDATED in the "Apply flip" block.
Wait.
In my code logic:
```cpp
            if (max_gain > 0) {
                assignment[best_v] = 1 - assignment[best_v];
                // update cnt
                for (int cid : adj[best_v]) {
                     // compute change
                     // ...
                     cnt[cid] += change;
                }
            }
```
Yes, `assignment` is updated.
Inside computation of change:
`bool old_val = (neg ? assignment[best_v] : !assignment[best_v]);`
Here `assignment[best_v]` is the NEW value.
If neg is false (`v`):
  `old_val` = `!new_value`.
  If `new_value` was True -> `old_val` was False.
  If `new_value` was False -> `old_val` was True.
  Wait.
  If `new_value` is 1 (True). `old_val` is 0 (False).
  Change should be: `cnt` increased by 1 (from old to new).
  Formula `old_val ? -1 : 1`.
  If `old_val` is 0, `1`.
  So `cnt += 1`.
  Correct.
  If `new_value` is 0 (False). `old_val` is 1 (True).
  Change should be: `cnt` decreased by 1.
  Formula: `old_val` is 1. `-1`.
  Correct.
  So logic holds.

Double check: `adj` construction duplicates.
If I use `unique`, I must ensure that `k` loop finds ALL occurrences.
Yes, `k` iterates `clauses[cid].lits`. It doesn't care about `adj` duplicates.
So `unique` is strictly necessary to prevent re-summing.
And `unique` doesn't affect `k` loop completeness.
Perfect.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct Clause {
    int lits_var[3];
    bool lits_neg[3];
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    vector<vector<int>> adj(n);

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        
        auto parse = [&](int x) -> pair<int, bool> {
            int idx = abs(x) - 1;
            bool neg = (x < 0);
            return {idx, neg};
        };
        
        pair<int, bool> pa = parse(a);
        pair<int, bool> pb = parse(b);
        pair<int, bool> pc = parse(c);
        
        clauses[i].lits_var[0] = pa.first; clauses[i].lits_neg[0] = pa.second;
        clauses[i].lits_var[1] = pb.first; clauses[i].lits_neg[1] = pb.second;
        clauses[i].lits_var[2] = pc.first; clauses[i].lits_neg[2] = pc.second;
        
        adj[pa.first].push_back(i);
        adj[pb.first].push_back(i);
        adj[pc.first].push_back(i);
    }
    
    // Remove duplicates in adjacency lists
    for(int i=0; i<n; ++i) {
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }

    vector<int> best_assignment(n, 0);
    int max_satisfied = 0;
    
    mt19937 rng(time(0));
    
    // Try multiple random restarts
    // For small N, hill climbing is effective.
    // N=50 suggests we can afford more work.
    int restarts = 100;
    
    for (int r = 0; r < restarts; ++r) {
        // Random initialization
        vector<int> assignment(n);
        for (int i = 0; i < n; ++i) assignment[i] = (rng() % 2);

        // Initialize clause counts
        vector<int> cnt(m, 0);
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < 3; ++k) {
                int u = clauses[i].lits_var[k];
                bool neg = clauses[i].lits_neg[k];
                bool val = neg ? !assignment[u] : assignment[u];
                if (val) cnt[i]++;
            }
        }

        // Hill Climbing
        while (true) {
            int max_gain = 0;
            int best_v = -1;
            
            // Try flipping every variable
            for (int v = 0; v < n; ++v) {
                int gain = 0;
                
                // Calculate total impact on score for flipping v
                // We iterate unique clauses containing v
                for (int cid : adj[v]) {
                    int change = 0;
                    // Sum changes for all occurrences of v in this clause
                    for (int k = 0; k < 3; ++k) {
                        if (clauses[cid].lits_var[k] == v) {
                            // If current value of literal was 1, it becomes 0 (cnt -1)
                            // If current value was 0, it becomes 1 (cnt +1)
                            // Literal value depends on current assignment[v] (BEFORE flip)
                            // But wait, inside search loop, assignment[v] is still OLD.
                            bool old_val = (clauses[cid].lits_neg[k] ? !assignment[v] : assignment[v]);
                            if (old_val) change--; else change++;
                        }
                    }
                    
                    // Impact on satisfaction
                    bool sat_curr = (cnt[cid] > 0);
                    bool sat_new = ((cnt[cid] + change) > 0);
                    
                    if (sat_new && !sat_curr) gain++;
                    else if (!sat_new && sat_curr) gain--;
                }
                
                if (gain > max_gain) {
                    max_gain = gain;
                    best_v = v;
                }
            }
            
            if (max_gain > 0) {
                // Apply flip
                assignment[best_v] = 1 - assignment[best_v];
                // Update counts
                for (int cid : adj[best_v]) {
                    int total_change = 0;
                    for (int k = 0; k < 3; ++k) {
                        if (clauses[cid].lits_var[k] == best_v) {
                            // Now assignment[best_v] is NEW.
                            // Check contribution of v in NEW state.
                            // If new contribution is 1 -> cnt increases by 1 relative to contribution of 0?
                            // Wait, cnt tracks total true literals.
                            // We just need to adjust cnt to reflect change.
                            // Previous cnt included "old contribution".
                            // We want "new contribution".
                            // New contribution is either 0 or 1.
                            // If new contribution is 1, we add 1 to cnt (relative to 0).
                            // If new contribution is 0, we subtract 1 from cnt (relative to 1).
                            // So change = (new_val ? 1 : -1).
                            // new_val = (neg ? !assignment[v] : assignment[v]).
                            bool new_val = (clauses[cid].lits_neg[k] ? !assignment[best_v] : assignment[best_v]);
                            if (new_val) total_change++; else total_change--;
                        }
                    }
                    cnt[cid] += total_change;
                }
            } else {
                // No improvement possible
                break;
            }
        }
        
        // Check best
        int current_score = 0;
        for(int c : cnt) if (c > 0) current_score++;
        if(current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = assignment;
        }
    }
    
    for (int i = 0; i < n; ++i) cout << (best_assignment[i] ? 1 : 0) << (i == n - 1 ? "" : " ");
    cout << "\n";

    return 0;
}
```
Wait, one subtle point in "Applying flip" block.
I calculate `total_change`.
If I iterate `k` for EACH `cid` in `adj` (which is unique `cid` per `v`), then I sum up `change` for all `k` matching `v`.
Example: `v` appears twice in clause. `adj` has `cid` once.
Loop `k` finds two matches. `total_change` sums both.
Then `cnt[cid] += total_change`.
This is CORRECT. `cnt` accumulates total delta for all occurrences.
Previously I worried about `adj` duplicates. With unique `adj` and summing `k` loop, it works.
Wait, previously I had `cnt[cid] += change` inside `k` loop.
If I put it inside `k` loop, and `v` appears twice, I add `change` twice.
Wait. If `v` appears twice.
Occurrences are independent?
Yes.
If `v` at `k=0` contributes 1, and at `k=1` contributes 1.
Flip: both become 0. `cnt` decreases by 2.
My code:
`if (matches) change = +/-1`.
Then `cnt[cid] += change` is executed TWICE (once for `k=0`, once for `k=1`).
So `cnt` changes by -1 then -1. Total -2.
So putting `cnt[cid] += change` INSIDE `k` loop is correct provided I don't accidentally double count `k`s?
Wait, `k` loop iterates 0..2. It checks `if (var == v)`.
So it executes body 2 times.
So adding `cnt` inside loop is correct.
BUT, `change` is computed relative to `assignment` state.
Is `cnt` state preserved?
Yes, `cnt` is updated sequentially.
Wait.
First `k` match (say `k=0`). `old_val` computed based on `old` assignment. `cnt` updated.
Second `k` match (say `k=1`). `old_val` computed based on `old` assignment. `cnt` updated.
Both correct.
What if `adj` contained duplicates of `cid`?
Then `adj` loop runs twice for same `cid`.
Then `k` loop runs again.
Then we double update `cnt`.
That would be wrong.
So `adj` must be unique.
With `adj` unique, and `cnt` update inside `k` loop, logic is correct.
Wait.
If `cnt` is updated inside `k` loop.
Iteration `k=0`: update `cnt`.
Iteration `k=1`: update `cnt`.
`cnt` accumulates correctly.
However, I used `cnt[cid] + change` in `max_gain` calculation.
Here `change` is `sum(changes)`.
So `max_gain` logic:
`change = 0`. Loop `k`: if match, `change += (val?1:-1)`.
`sat_new` check: `cnt[cid] + change`.
This assumes `change` is the TOTAL delta for clause.
This is consistent.
Wait, inside `apply flip`:
I loop `cid`. Inside loop `k`. If match, update `cnt`.
Here I can just use the SAME logic:
`cnt[cid] += (new_val ? 1 : -1)`.
If I do this inside `k` loop, it updates `cnt` for each occurrence.
But wait.
If I use `cnt[cid] += change` once per `cid` (where `change` is sum over `k`), it's equivalent to summing updates inside `k`.
Is it safer to sum `change` inside `k` loop and add once?
Actually, if I sum inside `k` loop and add, I'm effectively applying update per occurrence.
If I sum `change` in a temp variable and add at end, it's the same.
I'll use the temp variable approach for clarity/safety.
So:
```cpp
                for (int cid : adj[best_v]) {
                    int total_change = 0;
                    for (int k = 0; k < 3; ++k) {
                        if (clauses[cid].lits_var[k] == best_v) {
                             bool new_val = (clauses[cid].lits_neg[k] ? !assignment[best_v] : assignment[best_v]);
                             if (new_val) total_change++; else total_change--;
                        }
                    }
                    cnt[cid] += total_change;
                }
```
This is cleaner and avoids confusion about loop nesting.
Wait. `adj` unique ensures `cid` visited once.
Inside `cid` loop, I find all `v` occurrences and sum changes.
`total_change` is sum of changes.
`cnt` updated by `total_change`.
This is mathematically equivalent to updating `cnt` for each occurrence and works correctly because `cnt` is additive.
So this logic is sound.

One last thing: `best_v` selection.
I select `v` based on `gain`.
`gain` depends on `cnt` and `change`.
`cnt` is correct.
`change` calculation in check loop:
```cpp
                            if (clauses[cid].lits_var[k] == v) {
                                bool old_val = (clauses[cid].lits_neg[k] ? !assignment[v] : assignment[v]);
                                if (old_val) change--; else change++;
                            }
```
Here `old_val` is correct because `assignment[v]` is OLD.
This is consistent.

Okay, code looks solid.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

// Structure to represent a clause literal
struct Literal {
    int var_idx; // 0-based index of variable (0 to n-1)
    bool negated; // true if literal is negated (e.g., !x)
};

// Structure to represent a clause
struct Clause {
    Literal lits[3];
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    // adj[v] contains a list of clause indices where variable v appears.
    // We ensure no duplicate clause indices for the same variable to simplify updates.
    vector<vector<int>> adj(n);

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        
        auto parse = [&](int x) -> pair<int, bool> {
            int idx = abs(x) - 1;
            bool neg = (x < 0);
            return {idx, neg};
        };
        
        pair<int, bool> la = parse(a);
        pair<int, bool> lb = parse(b);
        pair<int, bool> lc = parse(c);
        
        clauses[i].lits[0] = {la.first, la.second};
        clauses[i].lits[1] = {lb.first, lb.second};
        clauses[i].lits[2] = {lc.first, lc.second};
        
        adj[la.first].push_back(i);
        adj[lb.first].push_back(i);
        adj[lc.first].push_back(i);
    }
    
    // Remove duplicate clause indices from adjacency lists
    for(int i = 0; i < n; ++i) {
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }

    vector<int> best_assignment(n, 0);
    int max_satisfied = 0;
    
    // Random generator
    mt19937 rng(time(0));
    
    // Number of restarts
    // For N=50, M=1000, 100 restarts is sufficient to find a high quality solution.
    int restarts = 100;
    
    for (int r = 0; r < restarts; ++r) {
        // 1. Random initialization of assignment
        vector<int> assignment(n);
        for (int i = 0; i < n; ++i) assignment[i] = (rng() % 2); // 1=TRUE, 0=FALSE

        // 2. Initialize clause counts (number of TRUE literals in each clause)
        // cnt[i] > 0 implies clause i is satisfied
        vector<int> cnt(m, 0);
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < 3; ++k) {
                int u = clauses[i].lits[k].var_idx;
                bool neg = clauses[i].lits[k].negated;
                bool val = neg ? !assignment[u] : assignment[u];
                if (val) cnt[i]++;
            }
        }

        // 3. Hill Climbing / Local Search
        while (true) {
            int max_gain = 0;
            int best_v = -1;
            
            // Iterate over all variables to find the best one to flip
            for (int v = 0; v < n; ++v) {
                int gain = 0;
                
                // Calculate impact of flipping v on total satisfied clauses
                for (int cid : adj[v]) {
                    int change = 0;
                    // Find all occurrences of v in this clause and calculate total contribution change
                    for (int k = 0; k < 3; ++k) {
                        if (clauses[cid].lits[k].var_idx == v) {
                            // Contribution before flip (based on current assignment[v])
                            // If literal was TRUE, it contributes 1. If we flip, it becomes FALSE, contribution loses 1.
                            // If literal was FALSE, it contributes 0. If we flip, it becomes TRUE, contribution gains 1.
                            bool old_val = (clauses[cid].lits[k].negated ? !assignment[v] : assignment[v]);
                            if (old_val) change--; else change++;
                        }
                    }
                    
                    // Check if clause satisfaction status changes
                    bool sat_curr = (cnt[cid] > 0);
                    bool sat_new = ((cnt[cid] + change) > 0);
                    
                    if (sat_new && !sat_curr) gain++;
                    else if (!sat_new && sat_curr) gain--;
                }
                
                if (gain > max_gain) {
                    max_gain = gain;
                    best_v = v;
                }
            }
            
            if (max_gain > 0) {
                // Apply the best flip
                assignment[best_v] = 1 - assignment[best_v];
                
                // Update clause counts for clauses affected by the flip
                for (int cid : adj[best_v]) {
                    int total_change = 0;
                    for (int k = 0; k < 3; ++k) {
                        if (clauses[cid].lits[k].var_idx == best_v) {
                            // After flip, check if the literal is true
                            bool new_val = (clauses[cid].lits[k].negated ? !assignment[best_v] : assignment[best_v]);
                            if (new_val) total_change++; else total_change--;
                        }
                    }
                    cnt[cid] += total_change;
                }
            } else {
                // Local optimum reached (no single flip improves score)
                break;
            }
        }
        
        // 4. Update global best
        int current_score = 0;
        for (int c : cnt) if (c > 0) current_score++;
        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = assignment;
        }
    }
    
    // Output the result
    for (int i = 0; i < n; ++i) {
        cout << (best_assignment[i] ? 1 : 0);
        if (i < n - 1) cout << " ";
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=936)

````
I need to solve the Max-3-SAT problem. This is NP-hard, but with n ≤ 50 and m ≤ 1000, I can use a randomized approach (saturation algorithm / greedy local search) which typically works well for such small instances.

The approach:
1. Assign random truth values initially
2. Greedily flip variables that increase satisfied clauses (or keep same count when stuck in local optimum)
3. Keep improving until we reach an optimal assignment

This should efficiently find satisfying assignments or high-satisfying fraction solutions.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int n, m;
vector<vector<int>> clauses;

void add_clause(int a, int b, int c) {
    clauses.push_back({a, b, c});
}

bool satisfies(int x[], int literal) {
    return literal > 0 ? x[literal - 1] : !x[-literal - 1];
}

int main() {
    cin >> n >> m;
    clauses.assign(m, vector<int>(3));

    for (int i = 0; i < m; ++i) {
        cin >> clauses[i][0] >> clauses[i][1] >> clauses[i][2];
    }

    vector<bool> x(n, false);
    
    auto count_sat = [&]() {
        int s = 0;
        for (int i = 0; i < m; ++i) {
            if (satisfies(x.data(), clauses[i][0]) || 
                satisfies(x.data(), clauses[i][1]) || 
                satisfies(x.data(), clauses[i][2])) {
                s++;
            }
        }
        return s;
    };

    int current_satisfied = count_sat();

    for (int iter = 0; iter < 3000 && current_satisfied < m; ++iter) {
        vector<int> delta(n + 1, 0);

        for (const auto& cl : clauses) {
            int unsat = 0;
            for (int v : cl) {
                if (!satisfies(x.data(), v)) unsat++;
            }

            if (unsat == 3) {
                for (int v : cl) {
                    int flipped = v > 0 ? v : -v;
                    delta[flipped] += (x[flipped - 1] ? 1 : 0);
                    delta[flipped] -= (x[flipped - 1] ? -1 : 1);
                }
            } else if (unsat == 2) {
                int sat_var_idx = 0, unsat_lit_val = 0;
                for (int v : cl) {
                    if (satisfies(x.data(), v)) {
                        sat_var_idx++;
                    } else {
                        unsat_lit_val = (v > 0) ? v : -v;
                    }
                }
                int sat_var = sat_var_idx > 0 ? cl[sat_var_idx - 1] : cl[0];
                delta[sat_var] -= (x[sat_var - 1] ? -1 : 1);
                delta[unsat_lit_val] += (x[unsat_lit_val - 1] ? 1 : 0);
            }
        }

        int best_flip = 0;
        int max_gain = 0;
        for (int i = 1; i <= n; ++i) {
            if (delta[i] > max_gain) {
                max_gain = delta[i];
                best_flip = i;
            }
        }
        
        if (max_gain <= 0) break;

        if (x[best_flip - 1]) x[best_flip - 1] = false;
        else x[best_flip - 1] = true;

        current_satisfied = count_sat();
    }

    for (int i = 0; i < n; ++i) cout << x[i] << (i == n - 1 ? "" : " ");
    cout << endl;

    return 0;
}
```
````
