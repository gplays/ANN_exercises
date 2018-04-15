%% Parameters
rng(1); %for reproducibility
n_units = [10 10];
trainAlg = 'trainlm';
nEpochs = 1000;
maxTime = 30;
numNN = 10;


divideFcn = 'dividerand'; %'divideint' &'dividerand'
trainRatio = 0.333;
valRatio= 0.333;
testRatio = 1-trainRatio-valRatio;

%% Data creation
data_x = rand(2,1000);
myfun = @(x,y) x+y*y +sin(x+y);
data_y = arrayfun(myfun,data_x(1,:),data_x(2,:));

data_size = size(data_y,2);

%% Data preparation
data_x_seq = con2seq(data_x);
data_y_seq = con2seq(data_y);

% There you can compute your train/val/test split, it returns data and
% indices
[~,~,~,trainInd,valInd,testInd] = dividerand(1:data_size,trainRatio,valRatio,testRatio);
test_x_seq = con2seq(data_x(:,testInd));
test_y = data_y(:,testInd);

%% Net parametrization
net=feedforwardnet(n_units,trainAlg);
% Let's get rid of this horrible window popping
net.trainParam.showWindow = 0;
% Following is a trick to provide specific indices 
% Function used for dividinf sets
net.divideFcn = 'divideind'; % One of divideRand divideint divideind
% Usable only with divideind other wise provide .trainRatio ...
net.divideParam(1).trainInd = trainInd;
net.divideParam(1).valInd = valInd;
net.divideParam(1).testInd = testInd;
% THE Trick to use for sequences (= cells)
net.divideMode = 'time';
% Number of epoch allowed without an increase on the validation set
net.trainParam.max_fail=10;
% Number of seconds before early stopping = upper bound on time spent on train
net.trainParam.time=maxTime;
% To change objective function
net.performFcn = 'mse'; % sae sse mse mae crossentropy

% To change transfer functions = Activation layers
% transferFcn = 'tansig'; % logsig tansig purelin
% for i = 1:size(net.layers,1)
%     net.layers{i}.transferFcn=transferFcn;
% For parametrization till the end, used to slow down convergence
% net.trainParam.mu_dec=0.01;
% net.trainParam.mu=0.00001;
% Regularization isn't natively present except on bayesian networks
if reg ~= 0
    net.performParam.regularization=10.0^-reg;
end
% Set the number of epochs for the training 
net.trainParam.epochs=nEpochs; 

%% Net training
% Running multiple time to average
% NB: All the nets will be train using the same train split

perfs = zeros(1, numNN);
yTotal = zeros(1,size(test_y,2));
for i = 1:numNN
  [neti,tr] = train(net, data_x_seq, data_y_seq);
  perfs(i) = tr.best_tperf;
%   Nice graph!
  plotperform(tr)
end
meanPerf = sum(perfs)/10;
