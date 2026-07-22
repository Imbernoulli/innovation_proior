Let me strip the statement down to the arithmetic. The numbers $1,2,\ldots,1978$ are split among six countries, and I need to force one country to contain a relation $x+y=z$, with the two summands allowed to be the same. The doubling clause is not a separate phenomenon; it is just the case $x=y$, so $z=2x$.

I want to understand what it would mean for the conclusion to fail. If one country $C$ contains $x,y,z$ with $x+y=z$, then $C$ also contains two elements whose positive difference is a third element of $C$: for instance $z-x=y$. Conversely, if $C$ contains $u<v$ and also contains $v-u$, then $u+(v-u)=v$ is exactly the required within-country sum. So avoiding the desired conclusion is equivalent to saying that in every country, no positive difference of two members of that country is again a member of that same country. This is the right form to use, because sums can run past $1978$, but positive differences of two listed numbers always stay inside $1,\ldots,1977$.

Now I assume, for contradiction, that every country avoids such an internal difference. The first count is forced: if every country had at most $329$ members, the total would be at most $6\cdot329=1974$, but there are $1978$ members. So one country, call it $C_1$, has at least $330$ members. I only need $330$ of them; write them as
$$a_1<a_2<\cdots<a_{330}.$$
Form the $329$ differences
$$a_2-a_1,\ a_3-a_1,\ \ldots,\ a_{330}-a_1.$$
They are distinct positive integers, each at most $1977$, so each is one of the numbered members and therefore belongs to some country. If any one of them belongs to $C_1$, then $a_1+(a_i-a_1)=a_i$ is already a forbidden within-country sum in $C_1$. Under the contradiction assumption, none of them can be in $C_1$, so all $329$ differences must lie in the other five countries.

The $329$ differences now have only five countries available. Since $5\cdot65=325<329$, one of those five countries contains at least $66$ of the differences. Call it $C_2$, and write those $66$ numbers in increasing order as
$$b_1<b_2<\cdots<b_{66}.$$
Each $b_j$ is some $a_{p(j)}-a_1$. Now take the differences from the smallest $b_1$:
$$b_2-b_1,\ b_3-b_1,\ \ldots,\ b_{66}-b_1.$$
There are $65$ of them. If one lands in $C_2$, then $b_1+(b_j-b_1)=b_j$ gives the required within-country sum in $C_2$. But each one is also a difference of two original $C_1$ members, because the common base cancels:
$$b_j-b_1=(a_{p(j)}-a_1)-(a_{p(1)}-a_1)=a_{p(j)}-a_{p(1)}.$$
So if one of these new differences lands in $C_1$, it also gives the required sum, this time as $a_{p(1)}+(b_j-b_1)=a_{p(j)}$. Therefore, as long as the contradiction assumption is still alive, all $65$ of these second-level differences must avoid both $C_1$ and $C_2$ and must lie in the four untouched countries.

This cancellation is the mechanism I need to preserve. At any stage I have a list inside a new country. When I subtract the smallest element of that list from the others, each new number is a difference of two members of the current country. At the same time, every element of the current list was itself made as a difference over a common base at the previous level, so the base cancels and the same new number is also a difference of two members of the previous country. Repeating that unwinds through all earlier levels. Thus every fresh difference is dangerous for every country already touched: if it lands in any of them, it becomes the missing third member of an internal difference relation there, hence a within-country sum.

So I can keep descending. The $65$ second-level differences are in four countries, so because $4\cdot16=64<65$, some new country $C_3$ contains at least $17$ of them. Write those as
$$c_1<c_2<\cdots<c_{17}.$$
The $16$ differences $c_i-c_1$ are differences of two members of $C_3$ directly. By the same base-cancellation, they are also differences of two members of $C_2$ and of two members of $C_1$. If any lands in $C_1$, $C_2$, or $C_3$, the desired sum appears in that country. If not, all $16$ must lie in the three countries not yet touched.

The arithmetic still stays just above the next threshold. With $16$ differences spread over three untouched countries, $3\cdot5=15<16$, so some country $C_4$ contains at least $6$ of them. Taking differences from the smallest of those $6$ gives $5$ new differences. Each is a difference of two members of $C_4$ and, by telescoping backward, also of two members of $C_3,C_2,C_1$. If any lands in one of those four countries, it is dangerous for that country and I am done; otherwise the $5$ differences must lie in the two remaining countries.

Now $2\cdot2=4<5$, so one of those two remaining countries, call it $C_5$, contains at least $3$ of the $5$ differences. Taking differences from the smallest of those $3$ gives $2$ positive differences. Again, either one lands in one of $C_1,\ldots,C_5$ and completes a within-country sum there, or both must lie in the one country not yet used. Call that last country $C_6$, and call the two differences it contains
$$e_1<e_2.$$

There is no room left to hide. The number $e_2-e_1$ is a positive integer at most $1977$, so it belongs to some country $C_k$ among the six. If $k=6$, then $e_1,e_2,e_2-e_1$ are all in $C_6$, and $e_1+(e_2-e_1)=e_2$ is the required sum. If $k$ is one of the earlier countries, the same telescoping identity says that $e_2-e_1$ is a difference of two members of that earlier country too. Concretely, $e_1$ and $e_2$ were made by subtracting a common base at the $C_5$ stage, so $e_2-e_1$ is a difference of two $C_5$ members; those two $C_5$ members were themselves made over a common base at the $C_4$ stage, so the same number is a difference of two $C_4$ members; continuing backward gives the same conclusion for $C_3,C_2,C_1$. Thus in the country $C_k$ that actually contains $e_2-e_1$, there are two members $u<v$ of $C_k$ with
$$v-u=e_2-e_1\in C_k.$$
Then $u+(e_2-e_1)=v$ is a within-country sum, contradicting the assumption that no country contains one.

The doubling case is also covered. The sum I produce has the form $u+d=v$, where $d=v-u$ is itself a member of the same country. It may happen that $u=d$; then $v=2u$, so the member numbered $v$ is twice the number of a member from his own country. If $u\ne d$, it is the sum of two distinct members from that country. The argument never needs the two summands to be distinct, which is exactly what the statement allows.

This contradicts the assumption that all six countries avoid internal differences. Therefore at least one country contains numbers $x,y,z$, with $x$ and $y$ possibly equal, such that $x+y=z$.
