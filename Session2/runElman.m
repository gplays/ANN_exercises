function [train_mse,test_mse] = runElman(fh,fnc_idx, n_tr, n_neurons, ne )
%RUNELMAN Summary of this function goes here

%%   Detailed explanation goes here

% Set the parameters of the run
%n_tr;             % Number of training points (this includes training and validation).
perc_test = 0.6;        % Number between 0 and 1. The test set will be n_tr * perc_test.
n_te = round(n_tr*perc_test); 
%n_neurons;         % Number of neurons
n = 1000;               % Total number of samples
%ne;               % Number of epochs
perc_training = 0.7;    % Number between 0 and 1. The validation set will be 1-perc_training.

n = n_tr+n_te;




%% Create the samples
% Allocate memory
u = zeros(1, n);
x = zeros(1, n);
y = zeros(1, n);

% Initialize u, x and y
u(1)=randn;
[x(1), y(1)]=feval(fh{fnc_idx},0,u(1));

% Calculate the samples
for i=2:n
    u(i)=randn;
    [x(i), y(i)]=feval(fh{fnc_idx},x(i-1),u(i));
end

%% Create the datasets
% Training set
X=num2cell(u(1:n_tr)); 
T=num2cell(y(1:n_tr));

% Test set
T_test=num2cell(y(n_tr+1:n));
X_test=num2cell(u(n_tr+1:n));

%% Train and simulate the network
% Create the net and apply the selected parameters
net = newelm(X,T,n_neurons);        % Create network
net.trainParam.epochs = ne;         % Number of epochs
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 1-perc_training;
net.divideParam.trainRatio = perc_training;
net.trainParam.showWindow = 0; 


[net,tr] = train(net,X,T);               % Train network
plotperform(tr)
T_test_sim = sim(net,X_test);       % Test the network

%%
R = corrcoef(cell2mat(T_test),cell2mat(T_test_sim));
R = R(1,2);

train_mse = tr.best_perf;
test_mse = mse(cell2mat(T_test)-cell2mat(T_test_sim));

end

