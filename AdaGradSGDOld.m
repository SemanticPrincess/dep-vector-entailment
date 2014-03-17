% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta ] = AdaGradSGD(theta, options, thetaDecoder, trainingData, ...
    hyperParams, testDatasets)
% Home-baked implementation of SGD with AdaGrad.

N = length(trainingData);
prevCost = intmax;
bestTestErr = 1;
lr = options.lr;

for pass = 0:options.numPasses - 1
    numBatches = ceil(N/options.miniBatchSize);
    sumSqGrad = zeros(size(theta));
    randomOrder = randperm(N);
   
    for batchNo = 0:(numBatches-1)
        beginMiniBatch = (batchNo * options.miniBatchSize+1);
        endMiniBatch = min((batchNo+1) * options.miniBatchSize,N);
        batchInd = randomOrder(beginMiniBatch:endMiniBatch);
        batch = trainingData(batchInd);
        [ ~, grad ] = ComputeFullCostAndGrad(theta, thetaDecoder, batch, hyperParams);
        sumSqGrad = sumSqGrad + grad.^2;
        
        % Do AdaGrad update
        adaEps = 0.001;
        theta = theta - lr * (grad ./ (sqrt(sumSqGrad) + adaEps));
    end

    % Do mid-run testing
    if mod(pass, options.testFreq) == 0
        
        % Test on training data
        if mod(pass, options.examplesFreq) == 0 && pass > 0
            hyperParams.showExamples = true;
            disp('Training data:')
        else
            hyperParams.showExamples = false;
        end
        [cost, ~, acc] = ComputeFullCostAndGrad(theta, thetaDecoder, trainingData, hyperParams);
        
        % Test on test data
        if nargin > 5
            if mod(pass, options.confusionFreq) == 0 && pass > 0
                hyperParams.showConfusions = true;
            else
                hyperParams.showConfusions = false;
            end

            if mod(pass, options.examplesFreq) == 0 || mod(pass, options.confusionFreq) == 0
                disp('Test data:')
            end
            testErr = TestModel(theta, thetaDecoder, testDatasets, hyperParams);
            if bestTestErr > testErr
                bestTestErr = testErr;
            end
        else
            testErr = -1;
        end
        if testErr ~= -1
            disp(['pass ', num2str(pass), ' train PER: ', num2str(acc), ...
                  ' test PER: ', num2str(testErr), ' (best: ', ...
                  num2str(bestTestErr), ')']);
        else
            disp(['pass ', num2str(pass), ' PER: ', num2str(acc)]);
        end
    else
       cost = ComputeFullCostAndGrad(theta, thetaDecoder, batch, hyperParams);
    end
    if mod(pass, options.checkpointFreq) == 0
        save([options.name, '/', 'theta-', options.runName, '@', ...
            num2str(pass)] , 'theta', 'thetaDecoder');
    end
        
    disp(['pass ', num2str(pass), ' cost: ', num2str(cost)]);
    if abs(prevCost - cost(1)) < 10e-7
        disp('Stopped improving.');
        break;
    end
    prevCost = cost(1);
    
end

end
