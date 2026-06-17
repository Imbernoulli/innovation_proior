# Minimum Number of Tiles

The minimum number of tiles is
$$\boxed{mn}.$$

## Lower Bound

Index the cells by $(i,j)$ with $1 \le i \le 2m-1$ and $1 \le j \le 2n-1$. Mark the cells for which both $i$ and $j$ are odd. There are exactly $m$ odd row indices and $n$ odd column indices, so there are $mn$ marked cells.

No tile can cover two marked cells. Two marked cells differ by an even number in both coordinates. For the L-tromino with offsets $\{(0,0),(0,1),(1,0)\}$, the nonzero differences, up to reversing the order of the pair, are
$$
(0,1),\quad (1,0),\quad (1,-1).
$$
For the zig-zag tetromino with offsets $\{(0,1),(0,2),(1,0),(1,1)\}$, the nonzero differences, up to reversing the order of the pair, are
$$
(0,1),\quad (1,0),\quad (1,-1),\quad (1,-2).
$$
Every listed vector has at least one odd coordinate. Rotations and reflections only swap coordinates and change signs, so this parity property remains true in every allowed orientation. Hence each tile covers at most one marked cell.

Since every marked cell must be covered, every tiling uses at least $mn$ tiles.

## Construction

For every $k \ge 3$, a $2 \times (2k-1)$ strip can be tiled with $k$ tiles. In rows $1,2$ and columns $1,\ldots,2k-1$, use the left L-tromino
$$
\{(1,1),(1,2),(2,1)\},
$$
then, for $t=1,\ldots,k-2$, use the zig-zag tetromino
$$
\{(1,2t+1),(1,2t+2),(2,2t),(2,2t+1)\},
$$
and finish with the right L-tromino
$$
\{(1,2k-1),(2,2k-2),(2,2k-1)\}.
$$
The top row is covered in consecutive blocks $1,2$, then $3,4$, and so on, ending with $2k-1$; the bottom row is covered in consecutive blocks $1$, then $2,3$, then $4,5$, and so on, ending with $2k-2,2k-1$. Thus these $k$ tiles are disjoint and cover the strip. By rotation, a $(2k-1)\times2$ strip also uses $k$ tiles.

The following labeled $7\times7$ grid gives a seed tiling with $16$ tiles. Equal letters are one tile; $F$ is the single zig-zag tetromino, and every other letter is an L-tromino.

$$
\begin{array}{ccccccc}
A&A&E&D&D&B&B\\
A&E&E&D&F&B&C\\
G&G&H&F&F&C&C\\
G&H&H&F&J&I&I\\
K&K&M&M&J&J&I\\
K&L&M&N&O&O&P\\
L&L&N&N&O&P&P
\end{array}
$$

The $F$ cells form the rotated/reflected zig-zag pattern
$$
\{(2,5),(3,4),(3,5),(4,4)\},
$$
and each other letter occupies three cells in a $2\times2$ block with one corner removed. Hence the grid is a valid tiling of the $7\times7$ square by $15$ L-trominoes and $1$ zig-zag tetromino, for a total of $16$ tiles.

Now tile the full $(2m-1)\times(2n-1)$ rectangle:

- Put the $7\times7$ seed in the first seven rows and first seven columns, using $16$ tiles.
- In the rest of the first seven columns, split the remaining $(2m-1)-7=2(m-4)$ rows into $m-4$ strips of size $2\times7$, using $4(m-4)$ tiles.
- Split the remaining $(2n-1)-7=2(n-4)$ columns into $n-4$ full-height strips of size $(2m-1)\times2$, using $m(n-4)$ tiles.

These regions partition the rectangle. The total number of tiles is
$$
16+4(m-4)+m(n-4)=mn.
$$

Thus a tiling with $mn$ tiles exists, and the lower bound proves that no tiling can use fewer. Therefore the minimum is $\boxed{mn}$.
