% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainDepModel(expName, dataFile, loadTheta, thetaFile, sortcols, sortrows, poolsize)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

if nargin > 1
    mkdir(['~/', expName]);
    % diary(['~/', expName, '.txt']);
else
    expName = '.';
end

TRAINING_WORDLIST = 'sick_train/training_wordlist.txt';

[wordMap, relationMap, relations, depMap] = ...
    InitializeDepMaps(TRAINING_WORDLIST, 'sick_train/train_deptypes_uniq.txt');
    
% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:

% The dimensionality of the phrase vectors.
hyperParams.wordDim = 25;%UNDO

% The dimensionality of the word vectors.
hyperParams.dim = 25;%UNDO

% The number of relations.
hyperParams.numRelations = 3; 

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 75; % undo

% Regularization coefficient.
hyperParams.lambda = 0.0500; %undo

% A vector of text relation labels.
hyperParams.relations = relations;

% Turn on to use pretrained word vectors from disk.
hyperParams.loadWords = true; %UNDO

% Regularize words towards the initial loaded vectors rather than zero.
hyperParams.useAnchor = true;
hyperParams.anchorToIdent = true;

% Use minFunc instead of SGD. Must be separately downloaded.
hyperParams.minFunc = false; %undo 

% How many dependency types are there.
hyperParams.numDepTypes = depMap.length;

% Ignore. Modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

% Load data from text?
hyperParams.loadFiles = true;

% L1 v. L2 regularization
hyperParams.norm = 2;

% Use untied composition layer params.
hyperParams.untied = false; 

% Use diagonal composition matrices.
hyperParams.diagonalComposition = true;

% Load parameters from file.
hyperParams.loadTheta = loadTheta;
hyperParams.paramsFile = thetaFile;

% Use max pooling layer.
hyperParams.maxPool = true;
hyperParams.poolSize = poolsize;
hyperParams.sortRows = sortrows;
hyperParams.sortCols = sortcols;

hyperParams.noBackprop = false; %...

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

% Initialize vocabulary
if hyperParams.loadWords
    [ wordFeatures, fullVocab, fullWordmap ] = InitializeVocabFromFile(wordMap);
else
    % Randomly initialize the words
    wordFeatures = rand(length(wordMap), hyperParams.wordDim) .* .02 - .01;
end

if hyperParams.loadTheta
    a = load(thetaFile); % sickdata
    theta = a.theta;
    thetaDecoder = a.thetaDecoder;
else
    % Randomly initialize other parameters.
    [ theta, thetaDecoder ] = InitializeDepModel(hyperParams, wordFeatures, depMap);    
end
    
    
% minfunc options
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 10000;
options.DerivativeCheck = 'on';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% AdaGradSGD learning options

% Rarely does anything interesting happen past 
% ~iteration ~200.
options.numPasses = 5000;
options.miniBatchSize = 256;

% LR for AdaGrad.
options.lr = 0.01;

% How often to reset the AdaGrad sum
options.resetSumSqFreq = 150;

% AdaGradSGD display options

% How often (in full iterations) to run on test data.
options.testFreq = 1;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 10;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 10; 

% How often to save parameters to disk.
options.checkpointFreq = 10;

% The name assigned to the current full run. Used in checkpoint naming.
options.name = expName; 

% Decay the presence of id=0 values.
options.trimIdZeroData = false;
options.zeroDecay = 1;

disp(options)

if hyperParams.useAnchor
   oldTheta = theta;
    [classifierMatrices, classifierMatrix, classifierBias, ...
     classifierParameters, wordFeatures, ...
     linearLayer, compositionMatrix, classifierExtraMatrix, ...
     classifierExtraBias] ...
    = stack2param(theta, thetaDecoder);
    wordFeatures = wordFeatures + wordFeatures;
    
    linearLayer = linearLayer + eye(size(linearLayer, 1),...
        size(linearLayer, 2)); % Needed? 
    if hyperParams.diagonalComposition && hyperParams.anchorToIdent
        compositionMatrix(:,1,:) = compositionMatrix(:,1,:) + 1;
    elseif hyperParams.anchorToIdent
        DIM = size(compositionMatrix, 1);
        for i = 1:size(compositionMatrix, 3)
            compositionMatrix(:,:,i) = compositionMatrix(:,:,i) + ...
                eye(DIM, DIM + 1);
        end
    end

    anchor = param2stack(classifierMatrices, classifierMatrix, classifierBias, ...
                   classifierParameters, wordFeatures, ...
                   linearLayer, compositionMatrix, classifierExtraMatrix, ...
                   classifierExtraBias);
    anchor = anchor - oldTheta;
    hyperParams.anchor = anchor;
end

if hyperParams.loadFiles 
    % Choose which files to load in each category.
    listing = dir('sick_train/sick_train_dep.txt');
    splitFilenames = {};
    trainFilenames = {listing.name};
    testFilenames = {'sick_trial_dep.txt'};

    trainFilenames = setdiff(trainFilenames, testFilenames);

    if hyperParams.loadWords
        [trainDataset, testDatasets] = ...
            LoadDepDatasets(trainFilenames, splitFilenames, ...
            testFilenames, wordMap, relationMap, depMap, fullVocab, fullWordmap);
    else
        [trainDataset, testDatasets] = ...
        LoadDepDatasets(trainFilenames, splitFilenames, ...
        testFilenames, wordMap, relationMap, depMap);
    end
    
    save(dataFile, 'trainDataset', 'testDatasets');
else
    a = load(dataFile); % sickdata
    testDatasets = a.testDatasets;
    trainDataset = a.trainDataset;
end

% trainDataset = trainDataset(1:25); %

% trainDataset = trainDataset(1:2); %UNDO
   
% Train
disp('Training')
options.MaxFunEvals = 10000;
options.DerivativeCheck = 'on';
options.runName = 'tr';

if hyperParams.minFunc
    % Set up minfunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    theta = minFunc(@ComputeFullDepCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget gradients and repeat?
else
    theta = AdaGradSGD(theta, options, thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end

end
