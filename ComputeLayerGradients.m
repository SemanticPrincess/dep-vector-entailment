% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, delta] = ...
      ComputeLayerGradients(x, rawMatrix, delta, ...
                            nonlinearityDeriv)
% Compute the gradients and deltas for an DTreeRNN layer for a given example.


innerOutput = rawMatrix * [x; 1];

if nargin > 3
    innerDeriv = nonlinearityDeriv(innerOutput);
else
    innerDeriv = (innerOutput .* 0) + 1;
end
    
[outDim, inDim] = size(rawMatrix);

matrixGradients = zeros(outDim, inDim);

for i = 1:outDim
    matrixGradients(i, :) = (innerDeriv(i) * delta(i)) .* [x; 1];
end
matrixGradients(:, inDim) = (innerDeriv .* delta);

delta = (rawMatrix(:, 1:inDim - 1)' * (matrixGradients(:, inDim) .* innerDeriv));

end