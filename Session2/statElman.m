
fh = localfunctions;
fncs = {'linSin', 'sqrtSin', 'affLinSin', 'expLinSin'};
numRep = 5;
logPath = '~/dev/ANN_exercises/.log/log2-1.txt';
fileID = fopen(logPath,'a');
formatSpec = '%10s %8s %6s %6s %8s %8s %8s\n';
fprintf(fileID,formatSpec,'func','#train','width','epoch', 'tr_mse', 'tst_mse','cputime');
fclose(fileID);

for fnc_idx = 1:4
    for n_tr = [100 300 500]
        for n_neurons = [25 50 500]
            for ne = [5 50 500 1000]
                res = [0,0];
                t = cputime;
                for k = 1:numRep
                    [train_perf, test_perf] = runElman(fh,fnc_idx, n_tr, n_neurons, ne);
                    res=res+[train_perf, test_perf];
                end
                res=res/numRep;
                e = cputime-t;
                logResults(logPath,func2str(fncs{fnc_idx}), n_tr, n_neurons, ne, res, e);

            end
        end
    end
end


function [x_i,y_i] = linSin(x_prec, u)
x_i = .8*x_prec + sin(u);
y_i = x_i;
end

function [x_i,y_i] = sqrtSin(x_prec, u)
x_i = exp(-x_prec*x_prec) + sin(u);
y_i = x_i;
end

function [x_i,y_i] = affLinSin(x_prec, u)
x_i = .8*x_prec + sin(u);
y_i = .5*x_i+1;
end

function [x_i,y_i] = expLinSin(x_prec, u)
x_i = .8*x_prec + sin(u);
y_i = exp(-x_i*x_i);
end


function[ ] = logResults(logPath,fnc, n_tr, n_neurons, ne, res, e)
% log results

fileID = fopen(logPath,'a');
formatSpec = '%10s %8.0f %6.0f %6.0f %8.4f %8.4f %8.2f\n';
fprintf(fileID,formatSpec,fnc, n_tr, n_neurons, ne, res, e);
fclose(fileID);

end 
