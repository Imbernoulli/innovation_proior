# 题 156 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=3258)

````
This problem asks us to find the optimal rotation of 8 types of tiles in a 30x30 grid to create two large loop lines. We need to output the number of 90-degree counterclockwise rotations for each tile (0-3).

The key observation is that this is an optimization problem where we want to maximize L1 * L2, where L1 and L2 are the lengths of the two longest loop lines.

Let's understand the tile types and their transitions:
- Types 0-3: Curved (one connection)
- Types 4-5: Two curved lines (two connections)
- Types 6-7: Straight line (two connections)

Wait, looking at the description again:
- Tiles 0 to 3 contain one curved line (these have endpoints, so they can only pass through once - dead ends unless matched properly)
- Tiles 4 and 5 contain two curved lines (these connect two opposite corners)
- Tiles 6 and 7 contain one straight line (these connect two opposite edges)

Actually, re-reading more carefully with the pseudo-code hint about `to[tiles[i][j]][d]`, I can see this defines the transition matrix for each tile type when a train enters from direction d.

The transition table given is:
```
to = [
	[1, 0, -1, -1],   // Type 0
	[3, -1, -1, 0],   // Type 1
	[-1, -1, 3, 2],   // Type 2
	[-1, 2, 1, -1],   // Type 3
	[1, 0, 3, 2],     // Type 4
	[3, 2, 1, 0],     // Type 5
	[2, -1, 0, -1],   // Type 6
	[-1, 3, -1, 1],   // Type 7
];
```

Where directions are: 0=left, 1=up, 2=right, 3=down
- di = [0, -1, 0, 1]
- dj = [-1, 0, 1, 0]

For the scoring:
- Find all cycles in the grid
- Get L1 (longest) and L2 (second longest)
- Score = L1 * L2

We have 900 tiles, each with 4 possible rotations. That's 4^900 possibilities, which is impossible to brute force.


There might be constraints or patterns that make this tractable. Since we need exactly two disjoint loops that cover most tiles, I should focus on finding valid configurations rather than exhaustive search.

Looking at the problem differently: if each edge connects two vertices and we're forming paths/cycles, then for a cycle to exist every vertex must have degree 2. But we need two separate cycles, which means we'd need even more careful coordination of how tiles connect.

Given this is from a competitive programming contest (AtCoder AHC 2010), it's likely there's either:
1. A heuristic approach that works well
2. A mathematical property I'm missing
3. The test cases are designed to be solved with reasonable computation

Without knowing more about the constraints or specific properties of the input, I'll implement a solution that uses backtracking with pruning to find good configurations. The key insight from the problem statement is that the score equals L1 × L2. So we want two reasonably long cycles rather than trying to cover everything in one big loop.

I'll model this as placing tiles on a grid where each placement constrains how adjacent tiles can connect. For each cell, I can try all 4 rotations and check if it maintains connectivity to form potential cycles. To keep this tractable, I could use a randomized greedy algorithm that builds up valid connections while tracking multiple paths simultaneously. Alternatively, I could use backtracking with strong pruning based on how many incomplete connections remain and what rotation choices are still available.

Given the complexity constraints, a local search seems practical—start with random rotations, then iteratively flip tile rotations to improve the two-longest-cycle product until no better moves exist. I'll encode the tile types and their directional transitions, then iterate through the grid computing which rotations create complete cycles versus dead ends. i >= N || j >= M) return false;
    }
};
    
auto get_to = [&](int t, int d) {
    return to[t][d];
};

// Compute cycles from current configuration
auto compute_cycles = [&](vector<vector<char>>& tiles) {
    vector<long long> vis(900);
    vector<long long> entry_dir(900);
    fill(vis.begin(), vis.end(), 0);
    
    vector<long long> cycles;
    vector<char> visited_cell(900, 0);
    fill(visited_cell.begin(), visited_cell.end(), 0);
    
    for (int start = 0; start < N * M; start++) {
        if (vis[start]) continue;
        
        int si = start / M;
        int sj = start % M;
        
        for (int sdir = 0; sdir < 4; sdir++) {
            vector<pair<int,int>> path;
            int ti = si;
            int tj = sj;
            int td = sdir;
            int steps = 0;
            bool found_cycle = false;
            
            while (steps < 900) {
                if (vis[(ti * M + tj) * 4 + td]) break;
                
                vis[(ti * M + tj) * 4 + td] = 1;
                path.push_back({ti * M + tj, td});
                
                auto next_d = get_to(tiles[ti][tj], td);
                if (next_d == -1) break;
                
                ti += dx[next_d];
                tj += dy[next_d];
                
                if (ti < 0 || ti >= N || tj < 0 || tj >= M) break;
                
                if ((ti * M + tj) * 4 + td == (si * M + sj) * 4 + sdir) {
                    found_cycle = true;
                    break;
                }
                
                td = next_d;
                steps++;
            }
            
            if (found_cycle && path.size() > 0) {
                long long cycle_len = path.size();
                cycles.push_back(cycle_len);
                for (auto& p : path) {
                    int pos = p.first;
                    int dir = p.second;
                    visited_cell[pos * 4 + dir] = 1;
                }
            }
        }
    }
    
    sort(cycles.rbegin(), cycles.rend());
    return cycles;
};

I cannot complete this thought as the next thinking section appears to be cut off mid-function definition. = max(L1, L2));
}
```

The constraint here is actually quite manageable - with 900 tiles and 4 possible rotations each, I need to explore 4^900 states, which is astronomical. However, the problem likely has structure I'm not exploiting fully. Since this is AtCoder AHC2010, it probably expects a randomized local search or some clever observation rather than pure backtracking. Let me refine my approach with a more robust local search strategy that considers tile interactions more carefully.

I should also verify whether the cycle detection needs to account for tiles being part of multiple cycles simultaneously. Given that trains follow single paths without branching, each tile can belong to at most one cycle at any direction. This means I need to ensure my cycle detection properly tracks visited states (position + incoming direction) to avoid counting the same path segment multiple times.

For optimization, I'll implement a hill-climbing algorithm that starts from random configurations and iteratively tries flipping individual tile rotations. Each iteration evaluates the resulting configuration by detecting cycles and calculating L1 × L2. If a move improves the score, I accept it. After enough attempts without improvement, I restart from a new random seed to escape local optima.

However, there's a subtle issue: when a tile flips rotation, it may break existing cycles or create new ones, so simply checking immediate gains isn't enough—I need to recalculate the entire cycle structure. Given the tight constraints, this simulation-based approach with frequent restarts and smart move selection should work, though I should optimize by tracking which tiles affect cycles to reduce redundant computations. I need to reconsider the fundamental structure. Each position with 4 possible rotations creates dependencies between adjacent cells, making the state space enormous at 4^900 states. Without stronger pruning, full exploration is computationally impossible.

I'll implement a simulated annealing approach instead, starting from random configurations and using temperature-scaled acceptance criteria to escape local minima. This trades guaranteed optimality for feasible computation within typical time limits.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>

using namespace std;

const int N = 30;
const int M = 30;

int dx[] = {0, -1, 0, 1};
int dy[] = {-1, 0, 1, 0};

int to_table[8][4] = {
    {1, 0, -1, -1},
    {3, -1, -1, 0},
    {2, -1, 3, -1},
    {-1, 2, 1, -1},
    {1, 0, 3, 2},
    {3, 2, 1, 0},
    {2, -1, 0, -1},
    {-1, 3, -1, 1}
};

struct Config {
    char tiles[900];
    char rotations[900];
};

Config init_config(const string* rows) {
    Config c;
    for (int i = 0; i < 900; i++) {
        c.tiles[i] = rows[i/M][i%M] - '0';
        c.rotations[i] = '0';
    }
    return c;
}

void rotate_tile(Config& config, int idx, char new_rot) {
    for (int i = 0; i < 4; i++) {
        if (config.rotations[idx] == i + '0') {
            config.rotations[idx] = new_rot;
            config.tiles[idx] = config.tiles[idx] - (i + '0');
            config.tiles[idx] += (new_rot - '0');
            config.tiles[idx] += (i - (new_rot - '0')) % 4;
            if (config.tiles[idx] < 0) config.tiles[idx] += 8;
            if (config.tiles[idx] >= 8) config.tiles[idx] -= 8;
            break;
        }
    }
    config.tiles[idx] = '0' + to_table[(config.tiles[idx]-'0')+((config.rotations[idx]-'0')%4)][3-(config.rotations[idx]-'0')/4];
}

long long compute_score(const Config& config) {
    vector<bool> visited_900(900, false);
    vector<bool> visited_edge(900 * 4, false);
    set<long long> cycles;
    
    for (int i = 0; i < 900; i++) {
        if (visited_900[i]) continue;
        for (int d = 0; d < 4; d++) {
            if (visited_edge[i * 4 + d]) continue;
            
            int xi = i / M;
            int xj = i % M;
            int length = 0;
            int yi = xi;
            int yj = xj;
            int yd = d;
            int cur_i = xi * M + xj;
            int cur_d = d;
            
            visited_edge[cur_i * 4 + cur_d] = true;
            while (true) {
                int t = config.tiles[cur_i * M + cur_j];
                int next_d = to_table[t][cur_d];
                if (next_d == -1) break;
                
                int ni = cur_i + dx[cur_d] * (cur_d == 1 ? -1 : (cur_d == 3 ? 1 : 0));
                int nj = cur_j + dy[cur_d] * (cur_d == 2 ? 1 : (cur_d == 3 ? 0 : 0));
                
                if (ni < 0 || ni >= M || nj < 0 || nj >= N) break;
                int nxt_i = ni * M + nj;
                int nxt_d = (cur_d + 2) % 4;
                
                if (nxt_i == cur_i && nxt_j == cur_j && nxt_d == cur_d) break;
                
                cur_i = nxt_i;
                cur_j = nxt_j;
                cur_d = nxt_d;
                length++;
                
                if (length > 900) break;
            }
        }
    }
    
    return cycles.empty() ? 0 : *cycles.rbegin() * *cycles.rbegin();
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    vector<string> grid(30);
    for (int i = 0; i < 30; i++) {
        cin >> grid[i];
    }
    
    mt19937 rng(1337);
    uniform_int_distribution<> dist(0, 3);
    
    Config best_config = init_config(grid.c_str());
    long long best_score = 0;
    
    for (int iter = 0; iter < 1000000; iter++) {
        Config new_config = best_config;
        int idx = rng() % 900;
        char old_rot = new_config.rotations[idx];
        new_config.rotations[idx] = dist(rng) + '0';
        
        long long new_score = compute_score(new_config);
        
        if (new_score > best_score || (best_score == 0 && new_score > 0)) {
            best_config = new_config;
            best_score = new_score;
        } else {
            if (rand() % 100 < 5) {
                best_config = new_config;
            }
        }
    }
    
    cout << best_config.rotations << endl;
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1192)

````
This is a competitive programming problem from AtCoder's AHC010 (AtCoder Heuristic Contest 2020, problem set for the 6th Problem Setter Competition).

Let me analyze the problem:
- We have a 30x30 grid of railroad tiles (900 tiles total)
- Each tile can be in one of 8 states (4 rotations × 2 types of basic shapes)
- We want to maximize the product of the lengths of two disjoint loop lines
- We need to choose which of 4 rotation states each tile takes

Key observations:
1. This is an optimization problem with approximately 4^900 possible configurations - completely unsolvable by brute force
2. We need a heuristic approach that can find good solutions quickly
3. The state space is huge but there are important structural properties we can exploit

Solution strategy:

**Key insight**: Use a genetic algorithm or simulated annealing, but with special structure-aware operations.

Actually, looking at this more carefully as a setter perspective:

The best approaches for this problem were:

1. **Simulated Annealing / Local Search** (most popular):
   - Start with a random configuration
   - Randomly pick a few tiles and try all 4 rotations each
   - Accept improvements, sometimes accept worse with decreasing probability
   - Continue until time runs out

2. **Iterative Improvement with Structure Awareness**:
   - The constraint is finding exactly 2 disjoint cycles
   - Once you build loops, you should keep them intact when rotating other tiles
   - Don't let optimizing single tiles break existing good structures

3. **Greedy/Buildup Approach**:
   - First build one large cycle greedily
   - Then work on a second cycle separately
   - Leave some tiles unused if it helps the objective

The winning strategies shared these characteristics:
- Simulated annealing framework
- Structure-aware modifications (protect existing cycles)


- Efficient evaluation rather than recomputing everything from scratch

For actual implementation, I'd use simulated annealing where I start with a random layout, then repeatedly select individual tiles and test their four possible orientations. When a rotation improves the cycle product score, I always apply it; occasionally I accept a worse configuration based on the current temperature. As the algorithm progresses, I gradually lower the temperature so it becomes more selective about what changes it makes.

Rather than recalculating everything after each modification, I compute the new score only once per iteration using fast path-tracing through the board. The critical part is designing a move operator that respects the existing structure—I should avoid moves that would fragment long cycles, preferring instead to rotate isolated segments away from established connections.

A hybrid method combines several elements: initially explore many small tile mutations randomly, then focus on larger groupings like adjacent blocks, and finally preserve whatever strong cycle structures emerged. This balances exploration early on with exploitation later in the run.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Solution {
    int tiles[30][30];
    
    int evaluate() {
        vector<int> vis(900, -1);
        int n = 0, m = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                if (vis[i * 30 + j] == -1) {
                    long long len = trace(i, j, n % 4, vis);
                    if (len <= 0) continue;
                    if (n > 0 && len != n) return 0;
                    n += len;
                    m++;
                }
            }
        }
        return (m >= 2) ? (long long)n / m : 0;
    }
};

const int di[] = {0, -1, 0, 1};
const int dj[] = {-1, 0, 1, 0};
const int to[] = {1, 0, -1, -1, 3, -1, -1, 0, -1, -1, 3, 2, -1, 2, 1, -1, 1, 0, 3, 2, 3, 2, 1, 0, 2, -1, 0, -1, -1, 3, -1, 1};

int trace(int x, int y, int d, const vector<int>& vis) {
    int i = x, j = y;
    long long len = 0;
    do {
        int idx = i * 30 + j;
        if (to[d * 8 + tiles[idx]] == -1 || i < 0 || i >= 30 || j < 0 || j >= 30) return 0;
        i += di[to[d * 8 + tiles[idx]]];
        j += dj[to[d * 8 + tiles[idx]]];
        if (i < 0 || i >= 30 || j < 0 || j >= 30) return 0;
        d = (to[d * 8 + tiles[idx]] + 2) % 4;
        ++len;
        if (i == x && j == y && d == n % 4) return len;
    } while (!(n * 30 + d));
    return 0;
}
```
````
