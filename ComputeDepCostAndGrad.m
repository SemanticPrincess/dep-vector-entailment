% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, pred ] = ComputeDepCostAndGrad( theta, decoder, dataPoint, hyperParams )
% Compute cost, gradient, and predicted label for one example.

% Unpack theta
[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters, wordFeatures, ...
    linearLayer, compositionMatrix, classifierExtraMatrix, ...
    classifierExtraBias] ...
    = stack2param(theta, decoder);

% Unpack hyperparams
DIM = hyperParams.dim;
WORDDIM = hyperParams.wordDim;
NUMCOMP = hyperParams.numDepTypes; % First is the self relation

leftTree = dataPoint.leftTree;
rightTree = dataPoint.rightTree;
trueRelation = dataPoint.relation;

% Make sure word features are current
leftTree.updateFeatures(wordFeatures, linearLayer, compositionMatrix, hyperParams.compNL);
rightTree.updateFeatures(wordFeatures, linearLayer, compositionMatrix, hyperParams.compNL);

leftFeatures = leftTree.getFeatures();
rightFeatures = rightTree.getFeatures();
    
% Compute classification tensor layer:
tensorInnerOutput = ComputeInnerTensorLayer(leftFeatures, ...
    rightFeatures, classifierMatrices, classifierMatrix, classifierBias);
classTensorOutput = hyperParams.classNL(tensorInnerOutput);


% Run extra layers forward
extraInputs = zeros(hyperParams.penultDim, hyperParams.topDepth);
extraInnerOutputs = zeros(hyperParams.penultDim, hyperParams.topDepth - 1);
extraInputs(:,1) = classTensorOutput;
for layer = 1:(hyperParams.topDepth - 1) 
    extraInnerOutputs(:,layer) = (classifierExtraMatrix(:,:,layer) ...
                                    * extraInputs(:,layer)) + ...
                                    classifierExtraBias(:,layer);
    extraInputs(:,layer + 1) = hyperParams.classNL(extraInnerOutputs(:,layer));
end

% Create max pool if needed
if hyperParams.maxPool
    pool = getPool(leftTree.headFirst(), rightTree.headFirst(), hyperParams.poolSize);
    if hyperParams.sortCols
        col_max = max(pool);
        [~,indices] = sort(col_max);
        pool = pool(:,indices);
    elseif hyperParams.sortRows
        row_max = max(pool, [], 2);
        [~,indices] = sort(row_max);
        pool = pool(indices,:);
    end
    pool = mypool(pool, hyperParams.poolSize, 2); 
    classifierFeatures = [extraInputs(:,hyperParams.topDepth); pool(:)];
else
    classifierFeatures = extraInputs(:,hyperParams.topDepth); 
end

relationProbs = ComputeSoftmaxProbabilities( ...
                    classifierFeatures, classifierParameters);

% Compute cost
cost = Objective(trueRelation, relationProbs);

% Produce gradient
if nargout > 1    
    % Initialize the gradients
    localWordFeatureGradients = sparse([], [], [], ...
        size(wordFeatures, 1), size(wordFeatures, 2), 10);
    
    localLinearLayerGradients = zeros(DIM, WORDDIM + 1);
    if hyperParams.diagonalComposition
        localCompositionMatrixGradients = zeros(DIM, 2, NUMCOMP);
    else
    	localCompositionMatrixGradients = zeros(DIM, DIM + 1, NUMCOMP);
    end
    [localSoftmaxGradient, softmaxDelta] = ...
        ComputeSoftmaxGradient (hyperParams, classifierParameters, ...
                                relationProbs, trueRelation,...
                                classifierFeatures);
    
    % Compute gradients for extra top layers
    [localExtraMatrixGradients, ...
          localExtraBiasGradients, extraDelta] = ...
          ComputeExtraClassifierGradients(hyperParams, ...
          classifierExtraMatrix, softmaxDelta, extraInputs, ...
          extraInnerOutputs);

    % Compute gradients for classification tensor layer
    [localClassificationMatricesGradients, ...
        localClassificationMatrixGradients, ...
        localClassificationBiasGradients, classifierDeltaLeft, ...
        classifierDeltaRight] = ...
      ComputeTensorLayerGradients(leftFeatures, rightFeatures, ...
          classifierMatrices, classifierMatrix, classifierBias, ...
          extraDelta, hyperParams.classNLDeriv, tensorInnerOutput);
     
    [ upwardWordGradients, ...
      upwardCompositionMatrixGradients, upwardLinearLayerGradients ] = ...
       leftTree.getGradient(classifierDeltaLeft, wordFeatures, ...
                            linearLayer, compositionMatrix, ...
                            hyperParams.compNLDeriv);
                      
    localWordFeatureGradients = localWordFeatureGradients ...
        + upwardWordGradients;
    localLinearLayerGradients = localLinearLayerGradients ...
        + upwardLinearLayerGradients;
    localCompositionMatrixGradients = localCompositionMatrixGradients...
        + upwardCompositionMatrixGradients;
                         
    [ upwardWordGradients, ...
      upwardCompositionMatrixGradients, upwardLinearLayerGradients  ] = ...
       rightTree.getGradient(classifierDeltaRight, wordFeatures, ...
                            linearLayer, compositionMatrix, ...
                            hyperParams.compNLDeriv);
    localWordFeatureGradients = localWordFeatureGradients ...
        + upwardWordGradients;
    localLinearLayerGradients = localLinearLayerGradients ...
        + upwardLinearLayerGradients;
    localCompositionMatrixGradients = localCompositionMatrixGradients...
        + upwardCompositionMatrixGradients;
    
    % Pack up gradients.
    grad = param2stack(localClassificationMatricesGradients, ...
        localClassificationMatrixGradients, ...
        localClassificationBiasGradients, localSoftmaxGradient, ...
        localWordFeatureGradients, localLinearLayerGradients, ...
        localCompositionMatrixGradients, ...
        localExtraMatrixGradients, localExtraBiasGradients);
end

% Compute prediction
if nargout > 2
    [~, pred] = max(relationProbs);
end

end

