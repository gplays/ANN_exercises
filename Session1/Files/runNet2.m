function [ train_mse,test_mse2,test_mse] = runNet2(trainAlg,n_data_points, noiseLevel, numUnits,batch)
%RUNNET Summary of this function goes here
%   Detailed explanation goes here


%generation of training data
step = 3*pi/n_data_points;
x=repmat(0.02:step:3*pi,batch);
x= reshape(x,1,[]);
y=sin(x.^2);

if noiseLevel>0
    y=awgn(y,noiseLevel);
end

p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%generation of test data
x_test=x+step/2;
p_test=con2seq(x_test);
y_test=sin(x_test.^2);

%creation of networks
net=feedforwardnet(numUnits,trainAlg);
net.trainParam.showWindow = 1;
net.divideMode = 'time';
net.trainParam.max_fail=25;
net.trainParam.time=100;

[net,tr]= train(net,p,t);
% plotperform(tr)
train_mse = tr.best_perf;
test_mse2 = tr.best_tperf;
test_mse = mean((cell2mat(sim(net,p_test))-y_test).^2);
end

