function [ r1, r15, r1000, o1, o15, o1000 ] = runNet(trainAlg,step, noiseLevel, numUnits,trainData)
%RUNNET Summary of this function goes here
%   Detailed explanation goes here


%generation of training data
x=repmat(0.02:step:3*pi,trainData);
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
net1=feedforwardnet(numUnits,trainAlg);
net1.trainParam.showWindow = 0;

%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training 
net1=train(net1,p,t);   % train the networks
a11=sim(net1,p);  % simulate the networks with the input vector p
a11_t=sim(net1,p_test);

net1.trainParam.epochs=14;
net1=train(net1,p,t);
a12=sim(net1,p);
a12_t=sim(net1,p_test);

net1.trainParam.epochs=985;
net1=train(net1,p,t);
a13=sim(net1,p);
a13_t=sim(net1,p_test);

[~,~,o1] = postregm(cell2mat(a11),y);
[~,~,o15] = postregm(cell2mat(a12),y);
[~,~,o1000] = postregm(cell2mat(a13),y);

[~,~,r1] = postregm(cell2mat(a11_t),y_test);
[~,~,r15] = postregm(cell2mat(a12_t),y_test);
[~,~,r1000] = postregm(cell2mat(a13_t),y_test);

o1 = o1-r1;
o15 = o15-r15;
o1000 = o1000-r1000; 
end

