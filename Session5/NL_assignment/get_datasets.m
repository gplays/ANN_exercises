function [dataSample] = get_datasets(method)
%GET_DATASETS Summary of this function goes here
%   Detailed explanation goes here
plot=0;
k=1000;
population = custom_NL();

if nargin<1
    method='dividerand';
end

if strcmp(method,'divideint')
    pop_size=size(population,1);
    step_size = floor(pop_size / (k+1));
    offset = randsample(step_size,3);
    train = population(offset(1):step_size:k,:);
    sample=randsample([offset(1):step_size:pop_size, ...
                       offset(2):step_size:pop_size, ...
                       offset(3):step_size:pop_size],3*k);
    sample = sort(sample);
else
    sample = randsample(size(population,1),3*k);
    train = population(sample(1:k,:),:);
        
end
dataSample = population(sample,:);
size_dataSample = size(dataSample);

if plot == 1
    myplot(train(:,1),train(:,2),train(:,3))
end
end

function [] = myplot(X1,X2,Tnew)
 f = scatteredInterpolant(X1,X2,Tnew);
 xlin=linspace(0,1,1000);
 ylin=linspace(0,1,1000);
 [x,y] = meshgrid(xlin,ylin);
 z=f(x,y);
 mesh(x,y,z)

end

