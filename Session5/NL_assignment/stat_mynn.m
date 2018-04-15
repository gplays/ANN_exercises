

logPath='~/Documents/Exercises Session/ANN/Session5/log1-2.txt';
fileID = fopen(logPath,'w');
formatSpec = '%12s %8s %8s %12s %10s %10s %8s\n';
fprintf(fileID,formatSpec,'divideFcn','maxTime','reg','n_units','perfMean','meanPerf','e');
fclose(fileID);
nEpochs = 1000;
for divide = {'dividerand'}
    divideFcn = divide{1};
    for maxTime = [30 100]
        for reg = [0 5 10]
            for n_units = {10 50 100 [10 10] [20 5 20] [20 20] [10 10 10]}
                t = cputime;
                [perfMean, meanPerf] = my_nn(divideFcn, nEpochs, maxTime, reg, n_units{1});
                e= cputime-t;
                logResults(logPath, divideFcn, maxTime, reg, n_units{1},perfMean, meanPerf,e);
            end
        end
    end
end


function[ ] = logResults(logPath, divideFcn, maxTime, reg, n_units,perfMean, meanPerf,e)
% log results

fileID = fopen(logPath,'a');
formatSpec = '%12s %8i %8i %12s %10.2e %10.2e %8.2f\n';
fprintf(fileID,formatSpec,divideFcn,maxTime,reg,mat2str(n_units),perfMean,meanPerf,e);
fclose(fileID);

end
