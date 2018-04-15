net = feedforwardnet(5,'traingd');
P=[2 1 -2 -1; 2 -2 2 1];
T=[0 1 0 1];
net = train(net,P,T);

a=sim(net,P);
%Y=net(P);
[m,b,r]=postreg(a,T)