Let me look at the operation before trying to guess an answer. Five integers sit around a pentagon, their total is positive, and when three consecutive entries are $x,y,z$ with $y<0$, I may replace them by $x+y,-y,z+y$. The middle entry becomes positive, but the same negative amount is pushed into both neighbors. So merely counting negative entries cannot work: one negative entry can disappear while one or two neighbors become negative. I need something less local, something integer-valued and bounded below that strictly drops every time I make a legal move.

The total sum is the first thing to check. The old contribution of the three changed entries is $x+y+z$, and the new contribution is $(x+y)+(-y)+(z+y)=x+y+z$, so the total
$$s=x_1+x_2+x_3+x_4+x_5$$
is invariant. Since it starts positive, $s>0$ forever. That is useful control, but it does not measure progress because it never changes. I need to use this fixed positive number inside some other quantity.

The most obvious size measures are too crude. The sum of absolute values can go up: from $(-5,-5,1,5,5)$, whose sum is $1$, operating at the second entry gives $(-10,5,-4,5,5)$, so the absolute-value sum rises from $21$ to $29$. The ordinary sum of squares is not safe either. If I operate at a value $y<0$ with neighbors $a$ and $c$, then only those three entries change, and
$$
\Delta\sum_i x_i^2=(a+y)^2-a^2+(-y)^2-y^2+(c+y)^2-c^2
=2y(a+c+y).
$$
The sign depends on $a+c+y$, not just on $y<0$ or on the positive total. For instance, in $(-4,1,5,5,-5)$, operating at the last entry has $y=-5$ and neighbors $5$ and $-4$, so $a+c+y=-4$ and the square sum increases by $40$. Coordinate sizes are too violent; the operation can make an individual entry much larger while still being part of a terminating process.

So I should stop staring at single coordinates and watch what happens to consecutive sums. Around a local triple $a,y,c$, take the running sums along a chosen direction. If the running sum just before $a$ is $p$, then before the move I see
$$
p,\quad p+a,\quad p+a+y,\quad p+a+y+c.
$$
After replacing $a,y,c$ by $a+y,-y,c+y$, the same four boundary sums are
$$
p,\quad p+a+y,\quad p+a,\quad p+a+y+c.
$$
The two middle partial sums have simply swapped. Because $y<0$, this swap moves the smaller one before the larger one. The operation is chaotic on entries but very orderly on partial sums, so the monovariant should be made out of all consecutive block sums.

Let
$$
M=\sum_{j=1}^{5}\sum_{L=1}^{4}\left|x_j+x_{j+1}+\cdots+x_{j+L-1}\right|,
$$
with indices read modulo $5$. This is the total absolute value of every proper consecutive arc sum. It is a nonnegative integer, because all entries are integers and I am summing absolute values. Now I have to check the hard part: every legal move must strictly decrease it.

Write the five entries, in cyclic order, as $(a,b,c,d,e)$, and suppose I operate at the middle value $b<0$. The new pentagon is
$$
(a+b,-b,c+b,d,e).
$$
I will compare all proper consecutive sums before and after the move. Before the move they are
$$
\begin{aligned}
&a,b,c,d,e,\\
&a+b,b+c,c+d,d+e,e+a,\\
&a+b+c,b+c+d,c+d+e,d+e+a,e+a+b,\\
&a+b+c+d,b+c+d+e,c+d+e+a,d+e+a+b,e+a+b+c.
\end{aligned}
$$
After the move they are
$$
\begin{aligned}
&a+b,-b,c+b,d,e,\\
&a,c,c+b+d,d+e,e+a+b,\\
&a+b+c,c+d,b+c+d+e,d+e+a+b,e+a,\\
&a+b+c+d,c+d+e,a+c+d+e+2b,d+e+a,e+a+b+c.
\end{aligned}
$$
Now the cancellation is visible. The term $|-b|$ matches $|b|$. The terms $|a|,|c|,|d|,|e|$ all appear on both sides, as do $|a+b|$, $|b+c|$, $|c+d|$, $|d+e|$, $|e+a|$, $|a+b+c|$, $|b+c+d|$, $|c+d+e|$, $|d+e+a|$, $|e+a+b|$, $|a+b+c+d|$, $|b+c+d+e|$, $|d+e+a+b|$, and $|e+a+b+c|$. Exactly one old absolute value is missing from the new list, namely
$$
|c+d+e+a|=|s-b|,
$$
the four-vertex arc that omits the operated entry. Exactly one new absolute value has no old partner, namely
$$
|a+c+d+e+2b|=|s+b|.
$$
So the whole change in the potential is not a vague decrease; it is exactly
$$
M_{\text{new}}-M_{\text{old}}=|s+b|-|s-b|.
$$

Now I only have to check the sign, and this is where the positive invariant sum enters. Since $b<0$, put $t=|b|=-b>0$. Then
$$
|s+b|-|s-b|=|s-t|-|s+t|.
$$
The second term is $s+t$, because $s>0$ and $t>0$. If $s\ge t$, the first term is $s-t$, so the change is $(s-t)-(s+t)=-2t$. If $s<t$, the first term is $t-s$, so the change is $(t-s)-(s+t)=-2s$. In both cases
$$
M_{\text{new}}-M_{\text{old}}=-2\min(s,|b|).
$$
The numbers $s$ and $|b|$ are positive integers, so this is at most $-2$. Thus every legal operation strictly decreases the nonnegative integer $M$.

That is enough. A nonnegative integer cannot strictly decrease forever, and here it even drops by at least $2$ at each move. Therefore only finitely many legal operations can occur. When the process stops, there is no negative entry left, because any negative entry would still be a legal middle entry with its two neighbors. So the repeated operation necessarily comes to an end after finitely many steps.
