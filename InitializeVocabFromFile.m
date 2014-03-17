function [ vocab, fullVocab, fullWordmap ] = InitializeVocabFromFile(wordMap, loc)

DIM = 25;
loadFromMat = false;
disp('Loading pretrained word vectors.');
wordlist = wordMap.keys();

if loadFromMat
    DIM = 100;
    v = load('sick_train/vars.normalized.100.mat');
    words = v.words;
    fullVocab = v.We2';
else
    fid = fopen('words_25d.txt');
    words = textscan(fid,'%s','Delimiter','\n');
    words = words{1};
    fclose(fid);
    fullVocab = dlmread('/john2/scr1/jpennin/word_vectors/vectors_25d.txt', ' ', 0, 1);
end

fullWordmap = containers.Map(words,1:length(words));
x = size(wordlist, 2);
vocab = zeros(x, DIM) .* .02 - .01;
for wordlistIndex = 1:length(wordlist)
    if fullWordmap.isKey(wordlist{wordlistIndex})
        loadedIndex = fullWordmap(wordlist{wordlistIndex});
    elseif strcmp(wordlist{wordlistIndex}, 'n''t')
        loadedIndex = fullWordmap('not');
        disp('Mapped not.');
    else
        loadedIndex = fullWordmap(',');
        disp(['Word could not be loaded: ', wordlist{wordlistIndex}]);
    end
    
    vocab(wordMap(wordlist{wordlistIndex}), :) = fullVocab(loadedIndex, :);
end

end