%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [ 1 1 1; -1 -1 1;1 -1 -1]';
net = newhop(T);
plot3(T(1,:),T(2,:),T(3,:),'gO');  
n=100;
for i=1:n
    b= 2*rand()-1
    a={[b b b]'};                         % generate an initial point                   
    [y,Pf,Af] = net({50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');
