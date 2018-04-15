
% 0 mean
% cov (p*p)
% eigen cov
% sort eigen
% E keep q largest (q*p)
% multiply dataset by E
% multiply by E'
% error

q=10;
p=50;
n=500;
ex = randn(p,n);
meanE = mean(mean(ex));
ex_Omean = ex -meanE;
cov_ex = cov(ex_Omean');
[v,d] = eigs(cov_ex,50);
reduced = v'*ex_Omean;
expanded = v*reduced;
restored = expanded + meanE;
error = sqrt(mean(mean((ex-restored).^2)))

[x, PS_std] = mapstd(ex);
[Y,PS] = processpca(x,0.1);
x_restored = processpca('reverse',Y,PS);
x_restored = mapstd('reverse',x_restored,PS_std);
error = sqrt(mean(mean((ex-x_restored).^2)))