Let me put coordinates on the cells. I call them $(i,j)$ with $1 \le i \le 2m-1$ and $1 \le j \le 2n-1$. One tile is the three-cell corner $\{(0,0),(0,1),(1,0)\}$, and the other is the four-cell zig-zag $\{(0,1),(0,2),(1,0),(1,1)\}$. The allowed motions are translations, rotations, and reflections that keep grid lines parallel, so any obstruction I use should be phrased in terms of differences between cells inside a tile.

Area gives only a rough warning. If a tiling uses $a$ L-trominoes and $b$ zig-zag tetrominoes, then
$$
3a+4b=(2m-1)(2n-1).
$$
The tile count is $a+b$, so I would like many four-cell pieces if I am trying to minimize the number of tiles. But area alone does not know where the pieces can sit. I need a lower bound that forces each tile to pay for some fixed part of the board.

The odd side lengths suggest the sublattice of cells with both coordinates odd. I mark exactly those cells. If two marked cells lie in the same tile, their coordinate difference is even in both coordinates. So the real test is simple: can either tile contain two distinct cells whose difference vector is even-even?

For the L-tromino $\{(0,0),(0,1),(1,0)\}$, the nonzero pairwise differences are, up to reversing the order of the pair,
$$
(0,1),\quad (1,0),\quad (1,-1).
$$
Each vector has at least one odd coordinate. For the zig-zag tetromino $\{(0,1),(0,2),(1,0),(1,1)\}$, the nonzero pairwise differences are, up to reversing the order of the pair,
$$
(0,1),\quad (1,0),\quad (1,-1),\quad (1,-2).
$$
Again each vector has at least one odd coordinate. A rotation or reflection only swaps coordinates and changes signs, so it preserves the property that a nonzero difference has an odd coordinate. Translation does not change differences. Therefore no allowed placement of either tile can cover two marked cells.

Now I count the marks. The row indices $1,3,\ldots,2m-1$ give exactly $m$ choices, and the column indices $1,3,\ldots,2n-1$ give exactly $n$ choices. There are $mn$ marked cells. Every marked cell has to be covered, while each tile covers at most one of them, so every tiling uses at least $mn$ tiles.

I still need a tiling that actually uses $mn$ tiles. I first want a strip that I can reuse. In a $2 \times (2k-1)$ strip, with rows $1,2$ and columns $1,\ldots,2k-1$, I place an L-tromino on the left:
$$
\{(1,1),(1,2),(2,1)\}.
$$
For each $t=1,\ldots,k-2$, I place one zig-zag tetromino:
$$
\{(1,2t+1),(1,2t+2),(2,2t),(2,2t+1)\}.
$$
I finish with an L-tromino on the right:
$$
\{(1,2k-1),(2,2k-2),(2,2k-1)\}.
$$
These tiles are disjoint. In the top row they cover columns $1,2$, then $3,4$, then $5,6$, and so on through $2k-2$, and finally $2k-1$. In the bottom row they cover column $1$, then $2,3$, then $4,5$, and so on through $2k-3$, and finally $2k-2,2k-1$. So the whole strip is covered by
$$
1+(k-2)+1=k
$$
tiles. Rotating this construction also tiles a $(2k-1)\times2$ strip with $k$ tiles.

The strip handles all even-width extensions, but I need an odd-by-odd seed. The smallest case under the hypotheses is $7\times7$, and the lower-bound number there is $4\cdot4=16$. If a $7\times7$ tiling has $16$ tiles, then its area equation is $3(16-b)+4b=49$, so $b=1$: it should have one zig-zag tetromino and fifteen L-trominoes. I can pin down such a seed explicitly on rows and columns $1,\ldots,7$.

I take the zig-zag tetromino to be
$$
\{(2,5),(3,4),(3,5),(4,4)\},
$$
which is a rotated and reflected copy of the four-cell zig-zag. The remaining cells split into these fifteen L-trominoes:
$$
\begin{gathered}
\{(1,1),(1,2),(2,1)\},\quad \{(1,6),(1,7),(2,6)\},\quad \{(2,7),(3,6),(3,7)\},\\
\{(1,4),(1,5),(2,4)\},\quad \{(1,3),(2,2),(2,3)\},\quad \{(3,1),(3,2),(4,1)\},\\
\{(3,3),(4,2),(4,3)\},\quad \{(4,6),(4,7),(5,7)\},\quad \{(4,5),(5,5),(5,6)\},\\
\{(5,1),(5,2),(6,1)\},\quad \{(6,2),(7,1),(7,2)\},\quad \{(5,3),(5,4),(6,3)\},\\
\{(6,4),(7,3),(7,4)\},\quad \{(6,5),(6,6),(7,5)\},\quad \{(6,7),(7,6),(7,7)\}.
\end{gathered}
$$
Each triple is a $2\times2$ block with one corner missing, after possibly rotating or reflecting it. The sixteen listed sets are disjoint, and their sizes add to $4+15\cdot3=49$, so they cover the entire $7\times7$ square.

Now I assemble the general rectangle. I put this $7\times7$ seed in the first seven rows and first seven columns. In the rest of the first seven columns, there are
$$
(2m-1)-7=2(m-4)
$$
rows left, so I split that part into $m-4$ strips of size $2\times7$. Each is the strip construction with $k=4$, so each uses $4$ tiles. The columns to the right of the first seven have total width
$$
(2n-1)-7=2(n-4),
$$
so I split that right-hand part into $n-4$ full-height strips of size $(2m-1)\times2$. Each is a rotated $2\times(2m-1)$ strip, so each uses $m$ tiles.

These pieces partition the whole rectangle: first the full-height seven-column band, then the remaining full-height two-column strips to its right. The tile count is
$$
16+4(m-4)+m(n-4)=16+4m-16+mn-4m=mn.
$$
The construction reaches the lower bound, so the minimum number of tiles is $mn$.
