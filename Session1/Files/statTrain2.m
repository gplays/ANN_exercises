
trainAlgs = {'traingd', 'traingda', 'traincgf','traincgp','trainbfg','trainlm','trainbr'};
numRep = 5;
logPath='~/Documents/Exercises Session/ANN/Session1/log.txt';
fileID = fopen(logPath,'w');
formatSpec = '%10s %8s %6s %6s %6s %8s %8s %8s %8s\n';
fprintf(fileID,formatSpec,'trainAlg','batch_s','#data','noise', 'units', 'perfwgn','tperfwgn','tperf','cputime');
fclose(fileID);
for batch = [1 2 10]
    for numUnits = {10,20,100,200,[10 10], [20 20]}
        for noiseLevel = [0 1 10 20]
            for n_data = [50 100 1000]
                for trainAlg = trainAlgs
                    perfs = [0,0,0];
                    t = cputime;
                    for i=1:numRep
                        [train_perf,test_perf,r_test_perf] = runNet2(trainAlg{1},n_data, noiseLevel, numUnits{1},batch);
                        perfs=perfs+[train_perf,test_perf,r_test_perf];
                    end
                    perfs=perfs/numRep;
                    e= cputime-t;
                    logResults(logPath, trainAlg{1},batch, n_data, noiseLevel, numUnits{1}, perfs,e)
                    
                end
            end
        end
    end
end


function[ ] = logResults(logPath, trainAlg,trainData,step, noiseLevel, numUnits, perfs,e)
% log results

fileID = fopen(logPath,'a');
formatSpec = '%10s %8i %6i %6i %6s %8.4f %8.4f %8.4f %8.2f\n';
fprintf(fileID,formatSpec,trainAlg,trainData,step, noiseLevel, mat2str(numUnits), perfs,e);
fclose(fileID);

end
