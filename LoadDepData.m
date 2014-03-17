% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadDepData(filename, wordMap, relationMap, depMap, varargin)
% Load one file of deptree-pair data.

% Append directory if we don't have a full path:
if isempty(strfind(filename, '/'))
    filename = ['sick_train/', filename];
end
disp(filename)
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load dependency ID map

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'leftDepText', '', 'rightText', ...
    '', 'rightDepText', '', 'id', 0, 'score', 0), length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
% maxLine = 4; % undo
for line = 2:maxLine % Skip header
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        if ~(splitLine{1} == '%')
            rawData(itemNo).id = splitLine{1};
            rawData(itemNo).leftText = splitLine{2};
            rawData(itemNo).leftDepText = splitLine{3};
            rawData(itemNo).rightText = splitLine{4};
            rawData(itemNo).rightDepText = splitLine{5};
            rawData(itemNo).score = splitLine{6};
            rawData(itemNo).relation = relationMap(splitLine{7});
            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

% Build the dataset
data = repmat(struct('relation', 0, 'leftTree', DepTree(), 'rightTree', ...
    DepTree(), 'id', 0, 'score', 0), length(rawData), 1);

% Build Trees
parfor dataInd = 1:length(rawData)
    data(dataInd).leftTree = DepTree.makeTree(rawData(dataInd).leftText, rawData(dataInd).leftDepText, wordMap, depMap, varargin{:});
    data(dataInd).rightTree = DepTree.makeTree(rawData(dataInd).rightText, rawData(dataInd).rightDepText, wordMap, depMap, varargin{:});
    data(dataInd).relation = rawData(dataInd).relation;
    data(dataInd).score = rawData(dataInd).score;
    data(dataInd).id = rawData(dataInd).id;
    if mod(dataInd, 100) == 0
        disp(['Loading ', num2str(dataInd), '/', num2str(length(rawData))]);
    end
end

end

