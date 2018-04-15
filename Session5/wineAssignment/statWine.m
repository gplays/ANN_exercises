

logPath='~/dev/ANN_exercises/.log/log5-2.txt';
fileID = fopen(logPath,'w');
formatSpec = '%8s %12s %8s %8s %10s %12s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n';
fprintf(fileID,formatSpec,'maxFail','n_units','n_feat','time','trainAlg','perfFcn','transferFcn','tr','tr_e','tr_e_b','v', 'v_e','v_e_b','t','t_e','t_e_b');
fclose(fileID);
numNN = 50;
for maxFail = [10 100]
    for n_feat = [1 2 3 5 8 11]
        for n_units = {10 50 100 [10 10] [20 5 20] [20 20] [10 10 10]}
            t = cputime;
            ccr= ones(5,3,3);
            for keepBest = [5 10 50]
                for trainAlg = {'trainlm', 'trainscg', 'trainrp'}
                    for perfFcn = {'sae', 'sse', 'mse', 'mae', 'crossentropy'}
                        if ~strcmp(trainAlg{1},'trainlm')||(strcmp(perfFcn{1},'sse')||strcmp(perfFcn{1},'mse'))
                            for transferFcn = {'tansig','logsig'}
                                for seed = 1:5
                                    ccr(seed,:,:) = classifier(maxFail, n_units{1}, n_feat, numNN,seed,keepBest,trainAlg{1},perfFcn{1},transferFcn{1});
                                end
                                ccr_mean= mean(ccr);
                                e= cputime-t;

                                tr = ccr_mean(1);
                                tr_e = ccr_mean(2);
                                tr_e_b = ccr_mean(3);
                                v = ccr_mean(4);
                                v_e = ccr_mean(5);
                                v_e_b = ccr_mean(6);
                                t = ccr_mean(7);
                                t_e = ccr_mean(8);
                                t_e_b = ccr_mean(9);

                                logResults(logPath, maxFail, n_units{1}, n_feat, e,trainAlg{1},perfFcn{1},transferFcn{1},tr, tr_e, tr_e_b, v, v_e, v_e_b, t, t_e, t_e_b);
                            end
                        end
                    end
                end
            end
        end
    end
end



function[ ] = logResults(logPath, maxFail, n_units, q, e,trainAlg,perfFcn,transferFcn, tr, tr_e, tr_e_b, v, v_e, v_e_b, t, t_e, t_e_b)
% log results

fileID = fopen(logPath,'a');
formatSpec = '%8i %12s %8i %8.2f %10s %12s %8s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n';
fprintf(fileID,formatSpec,maxFail, mat2str(n_units), q, e,trainAlg,perfFcn,transferFcn, tr, tr_e, tr_e_b, v, v_e, v_e_b, t, t_e, t_e_b);
fclose(fileID);

end
