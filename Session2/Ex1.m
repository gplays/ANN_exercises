T = [1 1; -1 -1; 1 -1]';
net=newhop(T);
% a= [1 0]
% a= [0 0]
% a= [0 -1]
% a= [-1 1]
a = [-1 1]'
Y = net([ ],[ ], a)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)
Y = net([ ],[ ], Y)

%typically takes 4-5 iterations