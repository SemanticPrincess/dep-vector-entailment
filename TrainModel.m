% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(expName, minf, minib)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

if nargin > 1
    mkdir(expName); 
else
    expName = '.';
end

[wordMap, relationMap, relations] = ...
    InitializeMaps('sick_train/training_wordlist.txt');

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:

% The dimensionality of the word/phrase vectors.
hyperParams.dim = 25;

% The number of relations.
hyperParams.numRelations = 3; 

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 45;

% Regularization coefficient.
hyperParams.lambda = 0.0001; % 0.0001

% A vector of text relation labels.
hyperParams.relations = relations;

% Turn off to pretrain on a word pair dataset.
hyperParams.noPretraining = true;

% Turn on to use pretrained word vectors from disk.
hyperParams.loadWords = true;

% Regularize words towards the initial loaded vectors rather than zero.
hyperParams.anchorWords = 1;

% Use minFunc instead of SGD. Must be separately downloaded.
hyperParams.minFunc = minf; 

% Ignore. Modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

% L1 v. L2 regularization
hyperParams.norm = 2;

% Use untied composition layer params.
hyperParams.untied = false; 

% Nonlinearities.
hyperParams.compNL = @LReLU;
hyperParams.compNLDeriv = @LReLUDeriv; 
nl = 'M';
if strcmp(nl, 'S')
    hyperParams.classNL = @Sigmoid;
    hyperParams.classNLDeriv = @SigmoidDeriv;
elseif strcmp(nl, 'M')
    hyperParams.classNL = @LReLU;
    hyperParams.classNLDeriv = @LReLUDeriv;
end

disp(hyperParams)

% Randomly initialize.
[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams, wordMap);

% minfunc options
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 10000;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% AdaGradSGD learning options

% Rarely does anything interesting happen past 
% ~iteration ~200.
options.numPasses = 1000;
options.miniBatchSize = minib;

% LR for AdaGrad.
options.lr = 0.01;

% How often to reset the AdaGrad sum
options.resetSumSqFreq = 5000;

% AdaGradSGD display options

% How often (in full iterations) to run on test data.
options.testFreq = 1;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 32;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 32; 

% How often to save parameters to disk.
options.checkpointFreq = 0;

% The name assigned to the current full run. Used in checkpoint naming.
options.name = expName; 

% The name assigned to the current call to AdaGradSGD. Used to contrast ...
% pretraining and training in checkpoint naming.
options.runName = 'pre'; 

disp(options)

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);

if hyperParams.anchorWords
    oldTheta = theta;
    [classifierMatrices, classifierMatrix, classifierBias, ...
     classifierParameters, wordFeatures, compositionMatrices,...
     compositionMatrix, compositionBias, classifierExtraMatrix, ...
     classifierExtraBias] ...
    = stack2param(theta, thetaDecoder);
    wordFeatures = wordFeatures + wordFeatures;
    anchor = param2stack(classifierMatrices, classifierMatrix, classifierBias, ...
                   classifierParameters, wordFeatures, compositionMatrices,...
                   compositionMatrix, compositionBias, classifierExtraMatrix, ...
                   classifierExtraBias);
anchor = anchor - oldTheta;
    hyperParams.anchor = anchor;
end

% Choose which files to load in each category.
listing = dir('sick_train/*parsed.txt');
splitFilenames = {};
trainFilenames = {listing.name};
testFilenames = {'SICK_trial.txt'};

trainFilenames = setdiff(trainFilenames, testFilenames);

% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, ...
    testFilenames, wordMap, relationMap);

% Train
disp('Training')
options.MaxFunEvals = 10000;
options.DerivativeCheck = 'off';
options.runName = 'tr';

if hyperParams.minFunc
    % Set up minfunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget metadata and repeat?
else
    theta = AdaGradSGD(theta, options, thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end

end
