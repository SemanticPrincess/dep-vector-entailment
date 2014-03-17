% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder ] = InitializeDepModel(hyperParams, wordFeatures, depMap)
% Initialize the learned parameters of the model. 

DIM = hyperParams.dim;
WORDDIM = hyperParams.wordDim;
PENULT = hyperParams.penultDim;
TOPD = hyperParams.topDepth;
NUMCOMP = depMap.length;

% Randomly initialize softmax layer
if hyperParams.maxPool
    classifierParameters = rand(hyperParams.numRelations, PENULT + 1 + (hyperParams.poolSize ^ 2)) .* .02 - .01;
else
    classifierParameters = rand(hyperParams.numRelations, PENULT + 1) .* .02 - .01;
end

% Randomly initialize tensor parameters
classifierMatrices = rand(DIM , DIM, PENULT) .* .02 - .01;
classifierMatrix = rand(PENULT, DIM * 2) .* .02 - .01;
classifierBias = rand(PENULT, 1) .* .02 - .01;

linearLayer = rand(DIM, WORDDIM + 1) .* .02 - .01;
linearLayer = linearLayer + eye(DIM, WORDDIM + 1);

if hyperParams.diagonalComposition 
    compositionMatrix = rand(DIM, 2, NUMCOMP) .* .02 - .01;
    compositionMatrix(:,1,:) = compositionMatrix(:,1,:) + 1;
else
    compositionMatrix = rand(DIM, DIM + 1, NUMCOMP) .* .02 - .01;
    for i = 1:NUMCOMP
        compositionMatrix(:,:,i) = compositionMatrix(:,:,i) + eye(DIM, DIM + 1);
    end
end
    

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* .02 - .01;
classifierExtraBias = rand(PENULT, TOPD - 1) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, ...
    linearLayer, compositionMatrix, classifierExtraMatrix, ...
    classifierExtraBias);

end

