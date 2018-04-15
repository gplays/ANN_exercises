load threes -ASCII

n_sample = 1;
q=4;
plot_opt = 0;
range=50;

k_errors = 1:50;


for k=1:50
    
    [threes_std, PS_std] = mapstd(threes);
    cov_ex = cov(threes');
    [v, ~] = eigs(cov_ex,k);
    % plot(diag(d))
    reduced = v'*threes_std;
    expanded = v*reduced;
    restored = mapstd('reverse',expanded,PS_std);
    error = sqrt(mean(mean((threes-restored).^2)));
    k_errors(k)=error;    
    
end

[threes_std, PS_std] = mapstd(threes);
cov_ex = cov(threes');
[~,d] = eigs(cov_ex,256);
eig_val = diag(d);
cum_eigen = ones(256,1).*sum(eig_val)-cumsum(eig_val)

figure;
subplot(1,3,1), plot(1:50,k_errors)
subplot(1,3,2), plot(1:50,cum_eigen(1:50))
subplot(1,3,3), plot(1:50,k_errors'./cum_eigen(1:50))

function [] = plot_img(n_sample,threes,restored)
sample = randi([1 500],1,n_sample);
    for i = sample
        figure;
        colormap('gray');
        subplot(1,2,1), imagesc(reshape(threes(i,:),16,16),[0,1])
        subplot(1,2,2), imagesc(reshape(restored(i,:),16,16),[0,1])
    end
end