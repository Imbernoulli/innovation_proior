# 题 137 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=980)

````
The problem asks me to generate a map (grid) that satisfies certain conditions and is designed to resist agents gathering together. Let's break down the requirements:

1. Grid Size: n × m where 1 ≤ n, m ≤ 20
2. Cells are either empty (1) or walls (0)
3. At least two empty cells exist
4. Empty cells form a connected subgraph
5. Empty cells form an acyclic structure (a tree)
6. Each empty cell initially has exactly one kangaroo
7. All kangaroos move simultaneously on commands U, D, L, R
8. They move if the adjacent cell in that direction exists AND is empty; otherwise they stay
9. The goal is to minimize the probability of all kangaroos ending up in one single cell after many random moves
10. We need to maximize success rate over 500 test cases where each test case has 50,000 random moves

Key observations:
- Since we want to prevent all kangaroos from gathering, we should design a map where this is difficult
- The movement rule says kangaroos only move if the target cell is empty (and exists)
- With 500 test cases and 50,000 moves each, we need a strategy that works well statistically
- A tree structure means no cycles - this is important for preventing certain trapping patterns
- The simultaneous movement creates interesting dynamics since collisions cause staying in place

Strategy considerations:
1. Make the tree very long/narrow (line graph) - this could help because in a line, agents tend to drift apart unless they hit boundaries
2. However, with simultaneous movement, having a line might actually cause convergence as agents hit ends and bounce back
3. Maybe a binary tree or other branching structure would help keep agents separated
4. Since we want to avoid ALL agents gathering in ONE cell, we should make it hard for them to synchronize

Actually, thinking about this more carefully:
- With 50,000 moves and uniform random directions, after so many moves, agents will tend to explore their space
- In a line, agents at the ends will get stuck bouncing, while middle agents continue moving


- I'll create a linear arrangement to maximize separation potential
- Careful consideration of edge cases like end point movements is crucial
- Designing a map that prevents complete synchronization seems most promising

I'll focus on creating a linear tree structure that maintains agent separation through strategic placement and limited movement options. The goal is to minimize the likelihood of all kangaroos converging to a single cell within the given move constraints. I see a tree structure emerges from connecting nodes sequentially. Starting with node 1, then adding nodes 2, 3, 4 to create a simple chain. This allows maximum spread across grid dimensions. By using a thin vertical or horizontal line of empty cells, I can potentially maximize distance between kangaroos while maintaining connectivity.

I'll modify the code to implement a straightforward linear configuration, ensuring the map meets the specified requirements for kangaroo distribution and movement constraints.
</think>

```cpp
#include <iostream>

int main() {
    // Create a map with empty cells forming a linear tree structure
    // This helps maintain separation between kangaroos as they move
    
    const int n = 10;
    const int m = 3;
    
    // Initialize grid with walls
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << "0";
        }
        std::cout << "\n";
    }
    
    // Create a vertical line of empty cells (tree structure)
    // Using column 1 to avoid boundary issues
    for (int i = 0; i < n; ++i) {
        std::cout << "1" << "\n";
    }
    
    // Actually let me fix the output format properly
    // Print dimensions first
    std::cout << n << " " << m << "\n";
    
    // Clear output and print correctly
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (j == 1) {
                std::cout << "1";
            } else {
                std::cout << "0";
            }
        }
        std::cout << "\n";
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=303)

````
The goal is to print a grid such that at least 125 out of 500 random control sequences fail to gather all agents into a single cell. The map must satisfy: size ≤ 20×20, connected empty cells forming an acyclic structure (a tree), and at least two empty cells.

The key observation is that if the number of empty cells equals the product of n and m, there is no wall to trap any kangaroo, which allows for easy design of an untrappable configuration. A good strategy is to place walls in positions with small numbers of valid moves, specifically the top-left corner, as these locations are less likely to be part of a converging path for the agents. By removing one well-chosen corner, the remaining open space forms a shape where some initial agent placements make convergence impossible even after many random moves. This approach directly targets maximizing the failure rate of agent gathering without violating the structural requirements.
</think>

```cpp
int main() {
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    int n = 16, m = 19;
    vector<string> g(n, string(m, '1'));
    g[0][0] = '0';
    cout << n << " " << m << "\n";
    for (auto& s : g) cout << s << "\n";
}
```
````
