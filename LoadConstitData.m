% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadConstitData(filename, wordMap, relationMap)
% Load one file of constituent-pair data.

% Append data-4/ if we don't have a full path:
if isempty(strfind(filename, '/'))
    filename = ['sick_train/', filename];
end
filename
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', Tree(), 'rightText', ...
    Tree(), 'id', 0, 'score', 0), length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
for line = 2:maxLine % Skip header
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).id = splitLine{1};
            rawData(itemNo).leftText = splitLine{2};
            rawData(itemNo).rightText = splitLine{3};
            rawData(itemNo).score = splitLine{4};
            rawData(itemNo).relation = relationMap(splitLine{5});
            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

% Build the dataset
data = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', ...
    Tree(), 'id', 0, 'score', 0), length(rawData), 1);

% Build Trees
for dataInd = 1:length(rawData)
    data(dataInd).leftTree = Tree.makeTree(rawData(dataInd).leftText, wordMap);
    data(dataInd).rightTree = Tree.makeTree(rawData(dataInd).rightText, wordMap);
    data(dataInd).relation = rawData(dataInd).relation;
    data(dataInd).score = rawData(dataInd).score;
    data(dataInd).id = rawData(dataInd).id;
end

end

