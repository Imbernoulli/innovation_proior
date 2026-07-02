# 题 85 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=1214)

````
The problem is about finding the exit in an infinitely deep binary tree where each node has three edges of different colors (red, yellow, blue). We start at some node with a known distance to the exit.

Key observations:
1. This is a binary tree structure rooted at the exit (Level 0)
2. Each node has 3 edges leading to it from its parent and one more edge going deeper
3. Actually, looking at the diagram description, it's a binary tree where each node branches into 2 children plus connects back to parent via 3rd color
4. Wait, let me re-read... "one becomes lost as soon as they cannot determine the absolute position"
5. The key is we need to figure out which node we're at by making moves and querying distances

Strategy:
- We know initial distance D to exit
- When we make a move, distance can change by ±1 or stay same depending on direction
- Query tells us current distance to exit
- We need to find the shortest path back to exit (distance 0)

Since we have limited queries (≤ 10^5), we should be efficient.

Key insight from similar problems: This is essentially searching in a tree where we don't know our position but know distance to root. We can use BFS/dijkstra-like approach or try to narrow down possible positions.

Actually, thinking about this differently:
- Each node at depth d from root has exactly one path to root
- If we track our path history and query distances, we can potentially identify our position
- Since it's a binary tree with 3 colored edges, from any node there are 3 choices

Better approach:
- Use BFS to explore and track distance changes
- When query returns specific distance, update possible positions
- Eventually reach distance 0 (exit)

Since time limit is generous and we have memory, we could maintain all possible current positions that are consistent with all distance queries.


- Initial: set of all nodes at distance `initialDeep`
- After each move: filter positions based on new distance
- Continue until distance = 0

However, this could grow exponentially. Better to use bidirectional search or A* algorithm focusing on nodes within reasonable distance bounds.

Let me reconsider the structure: it's a binary tree with each node having 3 edges. From any node, 2 edges lead away from root and 1 leads toward root (parent). But actually, since each node has 3 different colored edges, and we're in a tree, this means from the perspective of any node, one edge goes to parent and two go to children (or vice versa depending on orientation).

I'm tracking distance changes carefully. When moving, the distance to exit can increase or decrease by 1 based on whether I'm moving away from or toward the root. By querying my current distance after each move, I can narrow down potential positions.

The core strategy involves making strategic moves while monitoring how distance changes. I'll use bidirectional search techniques to efficiently locate the exit, maintaining a queue of potential paths and pruning those that violate distance constraints.

My approach focuses on exploring the tree systematically, using query results to eliminate impossible paths and guide my movement toward the root node.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <queue>
#include <algorithm>
#include <iomanip>

using namespace std;

int main() {
    int initialDeep;
    cin >> initialDeep;
    
    // Try direct upward moves to exit if possible
    for (int i = 0; i <= initialDeep && i <= 100; i++) {
        // Make a random valid move (we don't know which way is up)
        // Strategy: alternate between different directions to gather info
        int lastDir = -1;
        
        for (int j = 0; j < initialDeep + 1; j++) {
            // Move in direction 0, 1, or 2
            int dir = (j % 3);
            if (lastDir == dir) dir = (dir + 1) % 3;
            lastDir = dir;
            
            cout << "move " << dir << endl;
            fflush(stdout);
            
            int distAfterMove;
            cin >> distAfterMove;
            
            if (distAfterMove == 0) {
                return 0; // Reached exit
            }
        }
    }
    
    // Systematic exploration to find the exit
    // Maintain possible distances and try to converge
    
    int currentDist = initialDeep;
    string pathHistory = "";
    
    // Greedy approach: keep trying moves and observe distance changes
    vector<int> attempts;
    attempts.reserve(1000);
    
    for (int moveNum = 0; moveNum < 10000; moveNum++) {
        // Alternate through the 3 colors
        int color = moveNum % 3;
        
        cout << "move " << color << endl;
        fflush(stdout);
        
        int nextDist;
        cin >> nextDist;
        
        if (nextDist == 0) {
            break; // Exit found
        }
        
        if (nextDist > currentDist) {
            // Moved away, switch strategy
            color = (color + 1) % 3;
        } else if (nextDist == currentDist) {
            // Stayed same distance, might not be optimal
            color = (color + 2) % 3;
        }
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=735)

````
I need to find the exit in an infinite binary tree (SCP-167) where I can only see relative paths (edges colored red/yellow/blue). Starting from some unknown position with known distance `initialDeep`, I need to return to root.

Key insight: At each level, there are exactly 2 nodes at that depth - one on left branch and one on right branch of a specific "level". So if I know which level I'm on, I just need to figure out whether I'm on the "left" or "right" node.

Strategy: Start by moving until I reach the same distance as initialDeep but via different path. This means I've completed one full cycle around - I'm now back on the same level but opposite node! Then systematically work through both positions.
</think>

```cpp
#include <iostream>

int main() {
    int d;
    std::cin >> d;
    
    // We'll track both the current node and the mirrored node we want to meet at the same level
    char cur = '0';
    char target = '0';
    int deep = d;
    int step = 3;
    
    // Move along the same color repeatedly until we complete a cycle around
    while (deep != 0) {
        cur += char((cur - '0' + step % 3) % 3);
        int newDeep;
        std::cin >> std::flush;
        std::cin >> newDeep;
        
        if (newDeep == d && newDeep != deep) {
            // Completed a cycle - reached mirror position at same level
            break;
        }
        
        deep = newDeep;
        step++;
    }
    
    target += char((target - '0' + step % 3) % 3);
    int tDeep = d;
    int sStep = step + 1;
    
    // Now search for the exit knowing both the current node and its mirror
    while (d > 0) {
        for (char i = '0'; i <= target; i++) {
            if (i != cur) continue;
            
            for (int j = sStep - 1; j >= 0; j--) {
                char col = char((sStep - j + ((cur - '0') * 3 + 1)) % 3);
                
                if (i == cur && (j == 0 || (cur == target && j == 1))) {
                    std::cout << "move " << col << "\n";
                    std::cout.flush();
                    
                    int ret;
                    std::cin >> std::flush;
                    std::cin >> ret;
                    
                    if (ret == 1) break;
                } else {
                    i += char((col + 1) % 3);
                }
            }
        }
        cur = target;
        target += char((step % 3) + ((cur == target ? 1 : 0)));
        d -= 2;
        sStep++;
    }
    
    std::cout << "query\n";
    std::cout.flush();
    int res;
    std::cin >> std::flush;
    std::cin >> res;
    std::cin >> std::flush;
    
    return 0;
}
```
````
