
x=0:0.05:3*pi; y=sin(x.^2);
p=con2seq(x); t=con2seq(y);

trainAlgs = {'traingd', 'traingda', 'traincgf','traincgp','trainbfg','trainlm','trainbr'};

i=1;
trs = {};
for trainAlg = trainAlgs
    net = feedforwardnet(50,trainAlg{1});
    net.divideMode='time';
    [neti,tr] = train(net,p,t);
    trs{i} = tr;
    i=i+1;
    
end   

for i = 1:7
    plotperform(trs{i})
    pause
end