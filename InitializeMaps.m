% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations ] = ...
    InitializeMaps(filename)
% Load word-word pair data for pretraining and to generate a word map.

% For some experiments, this is only used to initialize the words and
% relations, and the data itself is not used.

% Establish (manually specified) relations
relations = {'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'};
relationMap = containers.Map(relations,1:length(relations));

% Load the file
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the word list
vocabulary = C{1};

% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

end

