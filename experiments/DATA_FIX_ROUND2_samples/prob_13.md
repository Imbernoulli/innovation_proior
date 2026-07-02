# 题 13 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=1027)

````
The problem is an interactive game where we need to trap a robot moving on a 2D grid. Let's understand the mechanics:

1. The robot starts at (sx, sy) where 1 <= sx, sy <= 20
2. We can mark cells black in T = 3000 turns maximum
3. Each turn:
   - We pick (xm, ym) and mark it black
   - Robot moves from current position to adjacent cell (including diagonals)
   - Robot must stay in x > 0, y > 0
   - If robot lands on a black cell, it explodes

Key observations:
- The robot knows our black cells and will try to avoid them
- The robot can move in any of 8 directions (excluding staying in place)
- We have 3000 turns to make the robot explode
- Initial position is within [1, 20] for both coordinates

Strategy:
The key insight is that we need to "herd" or "squeeze" the robot into a corner where we've already marked multiple cells black. Since the robot moves up to distance 1 in each direction, we need to surround it.

One effective approach is to work towards creating a "wall" of black cells that the robot cannot escape through. Since the robot wants to avoid black cells, if we create a pattern that leaves no safe white cells reachable, the robot will eventually be forced to step on a black cell.

A simpler approach would be to mark cells along the diagonal or in a specific pattern that forces the robot into a corner.

Let me think about a systematic approach:

1. First, I should try to narrow down the robot's location by marking cells around potential positions
2. Once I know roughly where the robot is, I can build a wall around it
3. Since the initial position is small (≤20), I can use the available space effectively

Here's my strategy:
1. Mark some cells to limit the robot's movement options
2. Systematically fill in areas to squeeze the robot


3. Create walls along boundaries or diagonals to force the robot into corners

I'll implement a solution using this systematic approach.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int sx, sy;
    if (!(cin >> sx >> sy)) return 0;
    
    // Set to track all marked black cells
    set<pair<int,int>> black;
    pair<int,int> current = {sx, sy};
    
    for (int t = 1; t <= 3000; t++) {
        // Strategy: Mark cells to gradually squeeze robot toward origin
        // First phase: mark cells near origin to limit robot's escape routes
        // Then systematically cover more area
        
        int xm, ym;
        
        // Phase 1: Try to trap robot by marking a shrinking rectangle
        // Mark cells progressively closer to potential robot locations
        
        if (t % 5 == 1) {
            // Fill rows first
            xm = (current.second / 5 + 2) * 5 - 2;
            ym = max(1, min(T, current.first));
        } else if (t % 5 == 2) {
            // Mark above/below strategically
            xm = current.first + (current.second % 2 == 0 ? 1 : -1);
            ym = max(1, min(T, current.second));
        } else if (t % 5 == 3) {
            xm = current.first;
            ym = min(current.second + 2, T);
        } else if (t % 5 == 4) {
            xm = max(1, current.first - 1);
            ym = max(1, current.second);
        } else {
            // Final marks
            xm = 1;
            ym = 1 + ((current.second - 1) / 2);
        }
        
        xm = max(1, min(xm, 3000));
        ym = max(1, min(ym, 3000));
        
        cout << xm << " " << ym << "\n";
        fflush(stdout);
        
        black.insert({xm, ym});
        
        pair<int,int> next_pos;
        cin >> next_pos.first >> next_pos.second;
        
        current = next_pos;
        if (next_pos.first == 0 && next_pos.second == 0) {
            break;
        }
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=934)

````
The problem is an interactive game where we need to trap a robot moving on a 2D grid. Let me understand the mechanics:

1. The robot moves from (sx, sy) and can move to any of the 8 adjacent cells
2. We mark a cell black, then the robot moves to one of its neighbors
3. If the robot lands on a black cell, it explodes
4. We have T = 3000 moves maximum

Key insight: We're in a "cat and mouse" situation where:
- We want to place black cells so the robot eventually lands on one
- The robot chooses which neighbor to move to, so we can't directly control where it goes

Strategy idea - Surrounding corners:
The most effective strategy would be to surround the robot with black cells in all 8 directions it could possibly move to. Since the robot must move within the first quadrant (x > 0, y > 0), it's confined to positive coordinates.

Let me think about this more carefully. If we build up a configuration where there are black cells that create a "trap":

For example, if we put black cells around certain regions, the robot might have no safe white neighboring cells.

Let me reconsider the optimal approach:

**Corner-surround strategy:**
Since x > 0 and y > 0, the smallest possible coordinate is 1. 

Let me mark cells such that they surround the corner region:
- (1,1), (1,2), ..., (T,1), (T,T), ..., (T, T/2), etc.

Actually, let me trace through the example more carefully to understand what pattern works:

Robot starts at (5,5)
Turn 1: We mark (6,1). Robot moves to (5,4) - safe (not black)
Turn 2: We mark (4,1). Robot moves to (5,3) - safe


Turn 3: We mark (6,2). Robot moves to (5,2) - safe
Turn 4: We mark (4,2). Robot moves to (5,1) - safe? Wait, but it should explode...

Ah wait, looking at the interactor output again: after turning off step 1, the robot is at position (5, 1). But the next turn shows us marking (5, 1) itself! So now the robot has nowhere safe to go and gets trapped on a black cell, causing it to explode.

The strategy builds a wall of black cells that forces the robot into a corner where it must step onto a black tile. I should verify this works by tracing different starting positions against various trap configurations. Looking back at the example solution though, I notice the moves don't follow my expected pattern—marking (6,1) then (4,1) instead of building diagonally. This suggests either the interactor uses a specific deterministic movement rule or the optimal strategy differs from what I initially derived. I need to reconsider my approach entirely rather than trying to guess the exact sequence.

What if I construct concentric layers of blocked squares? If I fill the entire T×T grid layer by layer—starting with innermost squares, then expanding outward—the robot becomes completely confined once it reaches the boundary. Once surrounded on all sides, every reachable adjacent cell is already blocked, forcing it to explode when it tries to move.

I'll implement this by marking cells in spiral order: start at (1, 1), progress right along the bottom edge, then up the right edge, left across the top edge, and down the left edge, continuing until all T×T cells are marked black.
</think>

```cpp
#include <iostream>
using namespace std;

int main() {
    int sx, sy;
    cin >> sx >> sy;

    // First, block the entire row below the starting position
    for (int i = sy - 1; i >= 1; --i) {
        cout << sx << " " << i << endl;
    }

    // Block each column to one side of the current robot position
    for (int x = sx - 1; x >= 1; --x) {
        cout << x << " " << sy << endl;
        cout << x + 1 << " " << sy << endl;
    }

    return 0;
}
```
````
