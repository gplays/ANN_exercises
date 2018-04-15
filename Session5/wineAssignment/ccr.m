function perf = ccr(net,varargin)
%MSE Mean squared error performance function.
%
% ccr(net,targets,outputs,errorWeights,...parameters...) calculates a
% network performance given targets, outputs, error weights and parameters
% as the correct classification ratio. The CCR is defined as: CCR = Number 
% of Correctly classified data x 100 / Total number of data


% Arguments
param = nn_modular_fcn.parameter_defaults(mfilename);
[args,param,nargs] = nnparam.extract_param(varargin,param);
if (nargs < 2), error(message('nnet:Args:NotEnough')); end
t = args{1};
y = args{2};
if nargs < 3, ew = {1}; else ew = varargin{3}; end
net.performParam = param;
net.performFcn = mfilename;

% Apply
perf = nncalc.perform(net,t,y,ew,param);
end