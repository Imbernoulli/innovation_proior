# The Partial-Sum Monovariant

## The result

Yes. Starting from any integer assignment on the pentagon with positive total, the repeated operation must terminate after finitely many steps.

## The monovariant

Let the current entries be $x_1,\dots,x_5$ in cyclic order, and let
$$
s=x_1+x_2+x_3+x_4+x_5>0 .
$$
The operation preserves $s$, since
$$
(x+y)+(-y)+(z+y)=x+y+z .
$$

Define
$$
M=\sum_{j=1}^{5}\sum_{L=1}^{4}\left|x_j+x_{j+1}+\cdots+x_{j+L-1}\right|,
$$
where indices are modulo $5$. This is a nonnegative integer: it is the sum of the absolute values of all proper consecutive-arc sums.

## Exact decrease

Suppose the operation is applied to consecutive entries $(a,b,c)$ with $b<0$, and write the full cyclic order as $(a,b,c,d,e)$. After the operation the entries are
$$
(a+b,-b,c+b,d,e).
$$
The old proper consecutive-arc sums are
$$
\begin{aligned}
&a,b,c,d,e,\\
&a+b,b+c,c+d,d+e,e+a,\\
&a+b+c,b+c+d,c+d+e,d+e+a,e+a+b,\\
&a+b+c+d,b+c+d+e,c+d+e+a,d+e+a+b,e+a+b+c .
\end{aligned}
$$
The new proper consecutive-arc sums are
$$
\begin{aligned}
&a+b,-b,c+b,d,e,\\
&a,c,c+b+d,d+e,e+a+b,\\
&a+b+c,c+d,b+c+d+e,d+e+a+b,e+a,\\
&a+b+c+d,c+d+e,a+c+d+e+2b,d+e+a,e+a+b+c .
\end{aligned}
$$
Taking absolute values, all terms pair with equal terms except that the old term
$$
|a+c+d+e|=|s-b|
$$
is replaced by the new term
$$
|a+c+d+e+2b|=|s+b|.
$$
Therefore
$$
\Delta M=M_{\text{new}}-M_{\text{old}}=|s+b|-|s-b|.
$$
Put $t=|b|=-b>0$. Then
$$
\Delta M=|s-t|-|s+t|.
$$
If $s\ge t$, this is $(s-t)-(s+t)=-2t$. If $s<t$, this is $(t-s)-(s+t)=-2s$. Hence
$$
\Delta M=-2\min(s,|b|)\le -2<0,
$$
because $s$ and $|b|$ are positive integers.

## Conclusion

Each legal operation strictly decreases the nonnegative integer $M$. Such a quantity cannot strictly decrease indefinitely, so only finitely many operations are possible. Once no operation is possible, no entry is negative. Thus the procedure necessarily comes to an end after a finite number of operations.
