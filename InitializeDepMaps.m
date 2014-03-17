% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations, depMap ] = ...
    InitializeDepMaps(wordFilename, depFilename)
% Load word-word pair data for pretraining and to generate a word map.

% For some experiments, this is only used to initialize the words and
% relations, and the data itself is not used.

% Establish (manually specified) relations
relations = {'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'};
relationMap = containers.Map(relations,1:length(relations));

% Load the file
fid = fopen(wordFilename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the word list
vocabulary = C{1};

% Build word map
%if length(vocabulary) > 1
    wordMap = containers.Map(vocabulary,1:length(vocabulary));
%else
%    wordMap = containers.Map({''}, [1]);
%end
    
% Load the file
fid = fopen(depFilename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the word list
depTypes = C{1};

% Build word map
depMap = containers.Map(depTypes,1:length(depTypes));

end

