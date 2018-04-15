
trainAlgs = {'traingd', 'traingda', 'traincgf','traincgp','trainbfg','trainlm','trainbr'};
numRep = 5;
logPath='~/Documents/Exercises Session/ANN/Session 1/log.txt';
fileID = fopen(logPath,'w');
formatSpec = '%10s %8s %6s %6s %6s %8s %8s %8s %8s %8s %8s %8s\n';
fprintf(fileID,formatSpec,'trainAlg','#series','step','noise', 'units', 'r1','r15','r1000', 'o1','o15','o1000','cputime');
fclose(fileID);
for trainData = [1 2 10]
    for numUnits = [50 500]
        for noiseLevel = [0 1 10 20]
            for step = [0.05 0.01]
                for trainAlg = trainAlgs
                    if ~strcmp(trainAlg{1},'trainbfg') || numUnits~=500
                        r = [0,0,0,0,0,0];
                        t = cputime;
                        for i=1:numRep
                            [r1,r15,r1000,o1,o15,o1000] = runNet(trainAlg{1},step, noiseLevel, numUnits,trainData);
                            r=r+[r1,r15,r1000,o1,o15,o1000];
                        end
                        r=r/numRep;
                        e= cputime-t;
                        logResults(logPath, trainAlg{1},trainData, step, noiseLevel, numUnits, r,e)
                    end
                end
            end
        end
    end
end


function[ ] = logResults(logPath, trainAlg,trainData,step, noiseLevel, numUnits, r,e)
% log results

fileID = fopen(logPath,'a');
formatSpec = '%10s %8.0f %6.4f %6.0f %6.0f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n';
fprintf(fileID,formatSpec,trainAlg,trainData,step, noiseLevel, numUnits, r,e);
fclose(fileID);

end
