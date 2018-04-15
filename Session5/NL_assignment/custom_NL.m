function [dataset] = custom_NL()
%Create a custom non-linear function using my private r-number
%My r-number is r-0685320
%The last 5 digits are 85320 which are conveniantly already sorted
myVars = {'X1','X2','T1','T2','T3','T4','T5'};
V = load('Data_Problem1_regression(1).mat',myVars{:});
d1=8;
d2=5;
d3=3;
d4=2;
d5=0;
Tnew = (d1*V.T1 + d2*V.T2 + d3*V.T3 + d4*V.T4 + d5*V.T5);
Tnew = Tnew / (d1 + d2 + d3 + d4 + d5);

dataset = [V.X1,V.X2,Tnew];
size_dataset = size(dataset);

end

