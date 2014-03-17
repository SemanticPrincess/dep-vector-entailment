% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [softmaxGradient, softmaxDelta] = ...
    ComputeSoftmaxGradient (hyperParams, classifierParameters, ...
                            relationProbs, trueRelation, features)
% Compute the gradient for the softmax layer parameters, and the deltas to
% pass down.
                        
softmaxGradient = zeros(size(classifierParameters, 1), ...
    size(classifierParameters, 2));

% Compute node softmax error
targetRelationProbs = zeros(length(relationProbs), 1);
targetRelationProbs(trueRelation) = 1;
softmaxDeltaFirstHalf = classifierParameters' * ...
                        (relationProbs - targetRelationProbs);
                    
% Compute nonlinearity and append intercept
softmaxDeltaSecondHalf = hyperParams.classNLDeriv([1; features]);
softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);

for relEval = 1:hyperParams.numRelations
    % Del from UFLDL Wiki on softmax
    softmaxGradient(relEval, :) = -([1; features] .* ...
        ((trueRelation == relEval) - relationProbs(relEval)))';
end

softmaxDelta = softmaxDelta(2:hyperParams.penultDim+1); 
% Don't create a delta for the max pool

end