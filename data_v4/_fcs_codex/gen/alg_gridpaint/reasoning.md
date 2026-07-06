The input has at most 50 rows and 50 columns, so there are at most 100 paintable lines. The grid has up to 2500 cells. I need something around polynomial time in these sizes. I immediately reject searching over stroke orders or subsets: even deciding an order over rows and columns is exponential, and allowing colors per stroke would make brute force hopeless. Dynamic programming over all row/column states would also TLE/MLE.

I first reduce the problem to the last useful stroke on each row or column. If a row is painted multiple times, only its last row-stroke can be the final cause of any cell in that row; all earlier row-strokes are overwritten everywhere in that row either by the later row-stroke or by later column-strokes. Such earlier strokes can be removed without changing the final grid. The same applies to columns. Therefore an optimal solution uses each row and each column at most once.

So a solution is: choose some rows and columns to paint once, assign each chosen line one color, and order the chosen lines. For any cell `(r,c)`, its final color is determined by whichever of row `r` and column `c` is painted later. If only one is painted, that one determines the cell. If neither is painted, the cell remains blank, impossible.

For a fixed set of painted lines and their colors, every cell imposes constraints:
- If row `r` is painted color `x` and column `c` is not painted, then target `(r,c)` must be `x`.
- If column `c` is painted color `y` and row `r` is not painted, then target `(r,c)` must be `y`.
- If both are painted:
  - If row color equals the target and column color does not, then row must be after column.
  - If column color equals the target and row color does not, then column must be after row.
  - If both equal the target, either order is fine.
  - If neither equals the target, impossible.

The difficulty is minimizing the number of chosen lines and colors. A row painted once must have one color. If it is used to cover any cells, that color is forced by those cells. The same for columns.

A more useful view is to decide, for each final cell, whether its final color is supplied by its row or by its column. If a row supplies at least one cell, all cells it supplies must have the same color, which becomes that row's paint color. If a column supplies at least one cell, all cells it supplies must have the same color. After that, if both the row and column are painted and one side does not match a cell, we get an ordering constraint.

This still looks like a huge assignment problem, but the small number of colors helps. I can enumerate the color assigned to every row and every column, with an extra value `0` meaning the line is not painted. Naively this is `(K+1)^(R+C)`, impossible. Instead, I use the fact that `R,C <= 50` but `K <= 10`, and solve the feasibility/minimum problem with min-cut for each possible line color choice? That also seems too large if all line colors are free.

I need a cleaner structural observation. Consider the reverse process. The last stroke in any valid sequence paints a full row or column whose target cells are all the stroke color at that moment. Since it is last, in the final grid that entire row or column must already be one uniform color. Remove that line, and repeat on the remaining subgrid. Therefore a grid is paintable iff we can repeatedly delete a remaining row or column that is monochromatic in the still-remaining cells. Each deletion corresponds to one stroke in reverse. This is the standard peeling characterization.

This gives feasibility and also suggests a stroke count. If I delete a row, that row has one stroke. If I delete a column, that column has one stroke. Since each stroke paints a whole row or column and each useful line is painted at most once, the number of strokes equals the number of deleted painted lines. I do not necessarily need to delete every row and every column: once all cells are covered, every remaining line may be unnecessary. In the reverse deletion model, deleting a row covers all currently remaining cells in that row; deleting a column covers all currently remaining cells in that column. I need cover every cell while respecting that each deleted line is monochromatic among cells not already covered by later strokes.

Now I need minimize deletions. Since there are at most 100 lines, a greedy peeling is not obviously minimum. Example choices can affect how many total rows/columns are needed. I need an exact method.

Think forward again, but with last-stroke peeling. A final stroke can be any currently uniform full line. If I choose it as last, I pay 1 and remove that row/column. This is like finding a minimum-size elimination set that covers all cells. Because `R,C <= 50`, maybe the answer has a simple complement form: if I leave some rows and columns undeleted, then any cell at their intersection would never be painted, so I cannot leave both a row and a column while their intersection remains. Thus after covering all cells, the undeleted lines can only be all rows or all columns, not a mixture with cells between them. In a full rectangular grid, if at least one row and one column are both undeleted, their intersection is uncovered. Therefore a complete cover deletes all rows, or deletes all columns, or both. More generally, the number of strokes is `R + C - max(lines left)`, and the lines left must be only rows or only columns.

If I leave rows undeleted, then every column must be deleted. Rows left are never painted. For a cell in a left row, its color must be supplied by its column stroke. A column stroke has one color, so every left row must have the same color in that column. Also, in reverse, after all columns are removed, the left rows have no remaining cells, so they need not be monochromatic. The deleted rows/columns must admit a valid peeling order ending with all columns removed and some rows left.

Symmetrically, I can leave columns undeleted.

This turns the optimization into maximizing how many rows can be left while all columns and the other rows can be peeled, or maximizing how many columns can be left while all rows and the other columns can be peeled. The answer is:
`min(R + C - bestLeftRows, R + C - bestLeftCols)`.

Now I need compute the largest set of rows that can be left. Let `S` be rows left. Since every column is painted, for each column `c`, all target colors among rows in `S` must be identical; otherwise no single column color can satisfy those unpainted rows. The other rows may be painted or overwritten.

Additionally, the remaining deletions must be peelable. Starting with all rows and columns active, I may delete rows not in `S` when they are monochromatic on active columns, and delete columns when they are monochromatic on active rows. Rows in `S` are protected and cannot be deleted. I need eventually delete every column and every row outside `S`.

Can I search over `S`? There are `2^R`, impossible.

Maybe there is a monotonicity property. If a protected set `S` is feasible, any subset of `S` is feasible: leaving fewer rows only allows more row deletions and relaxes column color consistency. So I need maximum feasible protected rows in a downward-closed family. Still not trivial.

Because `K <= 10`, rows can be grouped by their full color vector across columns. If two protected rows differ in a column, they cannot both be protected. In fact protected rows must have identical full row vectors: for every column, protected rows must share the same color. Therefore all left rows must be exact duplicate rows. That is a big simplification.

So for leaving rows, I only need consider each distinct row pattern `P`; the candidate protected set can be any subset of rows equal to `P`. Since feasibility is monotone and equal rows impose identical constraints, if I can leave one row of pattern `P`, I can leave all rows of pattern `P`: protecting duplicate rows does not make any column less monochromatic than protecting one, because they contribute the same color in every column. It also only reduces row-deletion burden. Thus the best protected row set is one whole duplicate class.

Similarly, protected columns must be duplicate column vectors.

For each candidate duplicate row class `S`, I test peelability while forbidding deletion of rows in `S`. The peeling test is straightforward:
- Maintain active rows and columns.
- A deletable unprotected row is active and all active cells in it have one color.
- A deletable column is active and all active cells in it have one color.
- Repeatedly delete any such line.
- Success if all active cells are gone, equivalently no active columns remain and all unprotected rows are gone; protected rows may remain.

If this succeeds, `|S|` rows can be left. I also include `S = empty`, which corresponds to deleting every row and column as needed, but the final formula with `0` left gives at most `R+C`.

For correctness of the greedy peeling test, if a protected set has any valid reverse deletion order, the first deletion in that order is currently monochromatic and allowed. Deleting any currently allowed line cannot destroy the existence of some completion: removing cells only makes other lines easier to become monochromatic, not harder. Therefore exhaustive greedy peeling succeeds exactly when a valid order exists.

I check a concrete example:
```
2 2
1 2
1 2
```
Rows are duplicate class `{0,1}` with pattern `[1,2]`. Protect both rows. Initially each column is monochromatic among active rows: column 0 is all `1`, column 1 is all `2`. Delete both columns. No active columns remain, success. Best left rows is 2, so strokes are `R+C-2 = 2`, matching “paint column 0 color 1, column 1 color 2”.

For
```
2 2
1 2
2 1
```
No duplicate row class larger than 1. Protect row 0: column 0 has colors `1,2`, not monochromatic; column 1 has `2,1`, not monochromatic; row 1 is `[2,1]`, not monochromatic, so stuck. Protect row 1 similarly fails. Protecting no rows also cannot peel any line because no row or column is monochromatic. Same for columns. Output `-1`, which matches brute force intuition: two rows and two columns with crossed colors cannot be made by full-line overwrites.

I also sanity-check against brute force for tiny grids conceptually: enumerate all sequences using each line at most once, compare reachable targets, and compare with the peeling criterion. The last stroke of any brute-force sequence is necessarily a monochromatic line in the final remaining grid, so recursively removing it gives a peel. Conversely, reversing a peel gives a valid painting sequence. The protected-line optimization is just choosing which rows or columns never appear in the sequence, and in a complete grid all unpainted lines must lie on one side only. This matches the formula.

The algorithm is efficient. There are at most `R` row classes and `C` column classes. Each peel simulation scans up to 100 lines, each scan costs at most 2500 cells, so even the simple implementation is far below the budget. Memory is just the grid and active flags. This is general: it never relies on special case constants beyond the stated input bounds.