function [perfMean, meanPerf] = my_nn(divideFcn, nEpochs, maxTime, reg, n_units)

%% Parameters
% n_units = [50 2 2];
trainAlg = 'trainlm';
% nEpochs = 1000;
% maxTime = 30;

% divideFcn = 'dividerand'; %'divideint' &'dividerand'
trainRatio = 0.333;
valRatio= 0.333;
testRatio = 1-trainRatio-valRatio;

%% Data creation
data = get_datasets(divideFcn);
data_size = size(data,1);

%% Data preparation
data_x = [data(:,1)';data(:,2)'];
data_x_seq = con2seq(data_x);
data_y = data(:,3)';
data_y_seq = con2seq(data_y);

[~,~,~,trainInd,valInd,testInd] = feval(divideFcn,1:data_size,trainRatio,valRatio,testRatio);
test_x_seq = con2seq(data_x(:,testInd));
test_y = data_y(:,testInd);

%% Net parametrization
net=feedforwardnet(n_units,trainAlg);
net.trainParam.showWindow = 1;
net.divideFcn = 'divideind';
struct = net.divideParam(1);
struct.trainInd = trainInd;
struct.valInd = valInd;
struct.testInd = testInd;
net.divideParam = struct;
net.divideMode = 'time';
net.trainParam.max_fail=10;
net.trainParam.time=maxTime;
% net.trainParam.mu_dec=0.01;
% net.trainParam.mu=0.00001;
if reg ~= 0
    net.performParam.regularization=10.0^-reg;
end
net.trainParam.epochs=nEpochs;  % set the number of epochs for the training 

%% Net training
numNN = 10;
perfs = zeros(1, numNN);
yTotal = zeros(1,size(test_y,2));
for i = 1:numNN
  [neti,tr] = train(net, data_x_seq, data_y_seq);
  perfs(i) = tr.best_tperf;
  y = neti(test_x_seq);
  yTotal = yTotal + cell2mat(y);
end
yAverageOutput = yTotal / numNN;
perfMean = sum((test_y - yAverageOutput).^2)/ size(test_y,2);
meanPerf = sum(perfs)/10;

end

