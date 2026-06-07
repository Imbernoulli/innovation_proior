
ATAPformats

r = 11; n = 1000;
x = linspace(0,1,n+1)';

% Construct Vandermonde matrix on [0,1]
Vde = zeros(n+1,r);
for i = 1:r
    Vde(:,i) = x.^(i-1);
end

% approximate Fekete points
[Q,R,pivot] = qr(Vde','vector');
fekete = pivot(1:r);

% equispaced points
equi = 1:n/(r-1):(n+1);


pts = fekete; color = 'b';
%pts = equi; color = 'r';

% build interpolant
f = abs(x.^2-0.5);
coefs = Vde(pts,:)\f(pts);
pstar = Vde*coefs;


hold on,
plot(x,f,'k-');
plot(x,pstar,color);
plot(x(pts),pstar(pts),[color '.']);
box on

%export_fig('qr_fekete','-pdf');

%%

[Q,R,pivot] = qr(Psi*Psi','vector');
sensors = pivot(1:p);
