% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, delta] = ...
      ComputeDiagonalLayerGradients(x, rawMatrix, delta, ...
                            nonlinearityDeriv)
% Compute the gradients and deltas for an DTreeRNN layer for a given example.


innerOutput = (rawMatrix(:, 1) .* x) + rawMatrix(:, 2);

if nargin > 3
    innerDeriv = nonlinearityDeriv(innerOutput);
else
    innerDeriv = (innerOutput .* 0) + 1;
end
    
outDim = size(rawMatrix, 1);

matrixGradients = zeros(outDim, 2);

for i = 1:outDim
    matrixGradients(i, 1) = innerDeriv(i) * delta(i) * x(i);
end

matrixGradients(:, 2) = (innerDeriv .* delta);

delta = (rawMatrix(:, 1) .* (matrixGradients(:, 2) .* innerDeriv));


% delta = (rawMatrix(:, 1:inDim - 1)' * (matrixGradients(:, inDim) .* innerDeriv));


end