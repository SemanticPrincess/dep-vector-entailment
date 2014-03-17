% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef DepTree < handle
    
    % Represents a single binary branching syntactic DepTree with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the DepTree can be displayed.
    % - The features at the node.
    
    properties (Hidden) %TODO: Make all private.
        subtrees = []; % list of DepTrees
        text = 'NULL';
        word = 'NULL';
        features = []; % DIM x 1 vector
        wordIndex = -1; % -1 => Not initialized.
        fixedFeatures = [];
        type = 'root';
        typeIndex = 0;
        headIndex = 0;
    end
    
    methods(Static)

        function t = makeTree(rawText, depText, wordMap, depMap, varargin)
            t = DepTree.makeSubtree(depText, wordMap, depMap, '0', varargin{:});
            t = t.subtrees(1);  
            t.text = rawText;
        end
        
        function t = makeSubtree(iText, wordMap, depMap, headIndexString, varargin)
            C = strsplit(iText, '(\), )|\[|\)\]', 'DelimiterType','RegularExpression');
            t = DepTree();
            t.headIndex = str2double(headIndexString);

            for i = 1:length(C)
                if ~isempty(C{i}) 
                    Dep = strsplit(C{i}, ', |\(|\-', 'DelimiterType','RegularExpression');
                    if strcmp(Dep{5}, headIndexString) % If we found the head
                        t.type = Dep{1};
                        t.word = Dep{4};
                        
                        if wordMap.isKey(lower(t.word)) % TODO: Maybe don't ignore case?
                            t.wordIndex = wordMap(lower(t.word));
                        elseif length(varargin) > 1 && nargin > 4 && varargin{2}.isKey(lower(t.word));
                            t.wordIndex = -2; % Fixed
                            t.fixedFeatures = varargin{1}(varargin{2}(lower(t.word)), :)';
                        else
                            disp(['Failed to map word ', lower(t.word), ' from ', iText]);
                            t.wordIndex = wordMap('*UNK*');
                        end

                        if depMap.isKey(t.type) % TODO: Maybe don't ignore case?
                            t.typeIndex = depMap(t.type);
                        else
                            disp(['Failed to map dependency type ' t.type]);
                            t.typeIndex = 1; % TODO: Set up dummy relation.
                        end
                    elseif strcmp(Dep{3}, headIndexString) % If we found a dependent
                        t.subtrees = [t.subtrees; DepTree.makeSubtree(iText, wordMap, depMap, Dep{5}, varargin{:})];
                    end
                end
            end
        end
    end
    methods
        function featureVectors = headFirst(obj)
            featureVectors = zeros(length(obj.features), 256);
            len = 1;
            featureVectors(:, 1) = [ obj.features ];
            for i = 1:length(obj.subtrees)
                childVectors = obj.subtrees(i).headFirst();
                featureVectors(:, len + 1:len + size(childVectors, 2)) ...
                    = childVectors;
                len = len + size(childVectors, 2);
            end
            featureVectors = featureVectors(:, 1:len);
        end
        
        function string = print(obj)
            string = ['[' obj.type ':' obj.word];
            for i = 1:size(obj.subtrees, 1)
                subtree = obj.subtrees(i);
                string = [string ' ' subtree.print()];
            end
            string = [string ']'];
        end

        function resp = isLeaf(obj)
            resp = (isempty(obj.subtrees)); % TODO: Fill in for undefined.
        end
        
        function ld = getSubtrees(obj)
            st = obj.subtrees;
        end
        
        function t = getText(obj)
            t = obj.text;
        end
        
        function f = getFeatures(obj)
            f = obj.features;
        end
        
        function type = getType(obj)
            type = obj.type;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end
    
        function updateFeatures(obj, wordFeatures, linearLayer, compMatrix, compNL)

            % Initialize features from local word
            if isempty(obj.fixedFeatures)
                obj.features = linearLayer * [wordFeatures(obj.wordIndex, :)'; 1];
            else 
                obj.features = linearLayer * [obj.fixedFeatures; 1];
            end

            for i = 1:size(obj.subtrees, 1)
                subtree = obj.subtrees(i);

                subtree.updateFeatures(...
                    wordFeatures, linearLayer, compMatrix, ...
                    compNL);
                
                if size(compMatrix, 2) == 2
                    obj.features = obj.features + compNL( ...
                        compMatrix(:,1,subtree.typeIndex) .* subtree.getFeatures() +...
                        compMatrix(:,2,subtree.typeIndex));
                else
                	obj.features = obj.features + compNL( compMatrix(:,:,subtree.typeIndex) * [subtree.features; 1] );
                end
            end
        end
        
        function [ upwardWordGradients, ...
                   upwardCompositionMatrixGradients, ...
                   upwardLinearLayerGradients ] = ...
            getGradient(obj, delta, wordFeatures, ...
                        linearLayer, compMatrix, compNLDeriv) % Delta should be a column vector.
            
            DIM = size(compMatrix, 1); 
            WORDDIM = size(linearLayer, 2) - 1;
            NUMCOMP = size(compMatrix, 3);

            upwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            
            upwardLinearLayerGradients = zeros(DIM, WORDDIM + 1);
            if size(compMatrix, 2) == 2
                upwardCompositionMatrixGradients = zeros(DIM, 2, NUMCOMP);
            else
                upwardCompositionMatrixGradients = zeros(DIM, DIM + 1, NUMCOMP);
            end
            for i = 1:length(obj.subtrees)
                subtree = obj.subtrees(i);

                if size(compMatrix, 2) == 2
                    [tempCompositionMatrixGradients, compDelta ] = ...
                      ComputeDiagonalLayerGradients(subtree.features, ...
                          compMatrix(:,:, subtree.typeIndex), delta, ...
                          compNLDeriv);
                else
                    [tempCompositionMatrixGradients, compDelta ] = ...
                      ComputeLayerGradients(subtree.features, ...
                          compMatrix(:,:, subtree.typeIndex), delta, ...
                          compNLDeriv);                    
                end

                upwardCompositionMatrixGradients(:,:,subtree.typeIndex) = ...
                    upwardCompositionMatrixGradients(:,:,subtree.typeIndex) + ...
                    tempCompositionMatrixGradients;

                % Take gradients from below. FOR EACH SUBTREE
                [ incomingWordGradients, ...
                  incomingCompositionMatrixGradients, ...
                  incomingLinearLayerGradients ] = ...
                  subtree.getGradient( ...
                                compDelta, wordFeatures, linearLayer, ...
                                compMatrix, compNLDeriv);
                upwardWordGradients = upwardWordGradients + ...
                                      incomingWordGradients;
                upwardCompositionMatrixGradients = ...
                    upwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                upwardLinearLayerGradients = ...
                    upwardLinearLayerGradients + ...
                    incomingLinearLayerGradients;
            end

            % Compute word feature gradients here if there is a word to update.
            if isempty(obj.fixedFeatures)
                [tempLinearLayerGradients, compDelta] = ...
                  ComputeLayerGradients(wordFeatures(obj.wordIndex, :)', ...
                                        linearLayer, delta);
                                    
               upwardWordGradients(obj.wordIndex, :) = ...
                   upwardWordGradients(obj.wordIndex, :) + compDelta'; 
            else
                [tempLinearLayerGradients, ~] = ...
                  ComputeLayerGradients(obj.fixedFeatures, ...
                                        linearLayer, delta);
            end
                
            upwardLinearLayerGradients = ...
                upwardLinearLayerGradients + ...
                tempLinearLayerGradients;
              
        end
    end
end