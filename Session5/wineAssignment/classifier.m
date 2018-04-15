function [perfs] = classifier(maxFail, n_units, q, numNN,seed,keepBest,trainAlg,perfFcn,transferFcn)
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
%% Parameters
rng(133*seed);
reg =0;
nEpochs = 500;
% n_units = 20;
divideFcn = 'dividerand';
trainRatio = 0.6;
valRatio = 0.2;
testRatio = 0.2;
maxTime = 30;
% numNN = 20;
% maxFail = 10;
%% Loading data
% data = load_wine_table();
% data = table2array(data(data.quality == 5 | data.quality == 6,:));
% features= data(:,1:11)';
% target = data(:,12)';
% save('wine_data','features','target')
data = load('wine_data');
features = data.features;
target = data.target;
target = target == 5;
data_size = size(target,2);

%% Preparing data for net
[~,~,~,trainInd,valInd,testInd] = feval(divideFcn,1:data_size,trainRatio,valRatio,testRatio);
data_y_seq = con2seq(target);
y = {target(:,trainInd),target(:,valInd),target(:,testInd)};


%% Parametring net
net = patternnet(n_units,trainAlg);
for i = 1:size(net.layers,1)
    net.layers{i}.transferFcn=transferFcn;

net.trainParam.showWindow = 0;
net.divideFcn = 'divideind';
struct = net.divideParam(1);
struct.trainInd = trainInd;
struct.valInd = valInd;
struct.testInd = testInd;
net.divideParam = struct;
net.divideMode = 'time';
net.trainParam.max_fail= maxFail;
net.trainParam.time=maxTime;
net.performFcn = perfFcn; % sae sse mse mae crossentropy
% net.trainParam.mu_dec=0.01;
% net.trainParam.mu=0.00001;
if reg ~= 0
    net.performParam.regularization=10.0^-reg;
end
net.trainParam.epochs=nEpochs;  % set the number of epochs for the training 

%% Net training
% numNN = 10;
% perfs = zeros(1, numNN);
% yTotal = zeros(1,size(val_y,2));
% for i = 1:numNN
%   neti = train(net, data_x_seq, data_y_seq);
%   y = cell2mat(neti(val_x_seq));
%   perfs(i) = 100*sum((val_y - round(y)).^2)/ size(val_y,2);
%   yTotal = yTotal + y;
% end
% yAverageOutput = yTotal / numNN;
% ccrMean = sum((val_y - round(yAverageOutput)).^2)*100/ size(val_y,2)
% meanccr = sum(perfs)/10

%% PCA
% [x, PS_std] = mapstd(features);
% [Y,PS] = processpca(x,0.05)
% x_restored = processpca('reverse',Y,PS);
% x_restored = mapstd('reverse',x_restored,PS_std);
% error = sqrt(mean(mean((features-x_restored).^2)))


[x, PS_std] = mapstd(features);
cov_ex = cov(x');
[v,d] = eigs(cov_ex,q);
eig_val = diag(d)';
% plot(eig_val)
features_pca = v'*x;
data_x_seq = con2seq(features_pca);
train_x_seq = con2seq(features_pca(:,trainInd));
val_x_seq = con2seq(features_pca(:,valInd));
test_x_seq = con2seq(features_pca(:,testInd));
x_seq = {train_x_seq, val_x_seq, test_x_seq};

perfs = {zeros(1, numNN),zeros(1, numNN),zeros(1, numNN)};
y_all = {zeros(numNN,size(trainInd,2)),zeros(numNN,size(valInd,2)),zeros(numNN,size(testInd,2))};
maxPerf = [0,0,0];
for i = 1:numNN
    neti = train(net, data_x_seq, data_y_seq);
    for j = 1:3
        y_inferred = cell2mat(neti(x_seq{j}));
        perfs{j}(i) = 100*sum(y{j} == round(y_inferred))/ size(y_inferred,2);
        maxPerf(j) = max(perfs{j}(i), maxPerf(j));
        y_all{j}(i,:) = y_inferred;
    end
end

[~,idx] = sort(perfs{2},'descend');



for j = 1:3
    y_best = y_all{j}(idx(1:keepBest),:);
    ccrMean(j) = sum(y{j} == round(mean(y_all{j})))*100/ size(y_all{j},2);
    ccrMean_best(j) = sum(y{j} == round(mean(y_best)))*100/ size(y_best,2);
    meanccr(j) = mean(perfs{j});
end

perfs = [meanccr;ccrMean;ccrMean_best;maxPerf]
perfs = [meanccr;ccrMean;ccrMean_best];


% expanded = v*reduced;
% error = sqrt(mean(mean((x-expanded).^2)))
% x_restored = mapstd('reverse',expanded,PS_std);


end

