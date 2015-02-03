function [results, c, sums]= miniBatchClusteringSupervision(data,dataClusters,iterations,batchSize,clusterSeeds,cellOfSuggestedDocs,factor,sumOfDistances)
%
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% Axel Soto, 2013
% Dalhousie University

%

%%The step 10 in the original mini-batch k-means algorithm by Sculley
%%http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf overwrites the
%%current centers (which shouldn't). Another variable should be used (here
%%indexCenter)

%%OUTPUT: results: cluster assignment, c: centroids
%%INPUT: 
% % data: data to be clustered
% % dataClusters: number of clusters
% % iterations: number of iterations for the fast clustering method
% % batchSize: batch size for the fast clustering method
% % clusterSeeds: may or may not contain the initial centroids
% % cellOfSuggestedDocs: A list of lists. One list for each word suggested by the user and within each component there is a list with the document indices
% % factor: variable selected by the user to affect the clustering results
% % sumOfDistances: indicates if we want the sum of distances of all points to their centroid

%Parameters for the supervision: MAXFACTOR and MINFACTOR are determined by
%the user, while penalties is a linear transformation for algorithm' specific influence. For the algorithm penalties are how much the distribution is modified: the
%sum of all distribution values (for a single data point) is equal to
% 1/penalty. The higher the penalty the less effect it has on the data
%similarities

tic;
MAXFACTOR = 10;MINFACTOR=1;MAXPENALTY=4;MINPENALTY=1;
slope = (MAXPENALTY-MINPENALTY)/(MINFACTOR-MAXFACTOR);
Y0 = MAXPENALTY - MINFACTOR*slope;
mY0 = MINPENALTY + MINFACTOR*slope;


%%Initialize centers
if ~exist('clusterSeeds', 'var') || isempty(clusterSeeds)
    c = zeros(dataClusters,size(data,2));                   %Cluster centers
    randPermutations=randperm(size(data,1));
    c(1:dataClusters,:)=data(randPermutations(1:dataClusters),:);
else
    c = clusterSeeds;
end

if ~exist('sumOfDistances', 'var') || isempty(sumOfDistances)
    sumOfDistances = false;
end

%Initialize distributions and total counts, if supervision is provided 
if ~isempty(cellOfSuggestedDocs)
    distributions = cell(1,length(cellOfSuggestedDocs));
    totalCounts = cell(1,length(cellOfSuggestedDocs));
    for indexWord=1:length(cellOfSuggestedDocs)
        distributions{indexWord}= zeros(1,dataClusters)+1/dataClusters;
        totalCounts{indexWord} = zeros(1,dataClusters);
    end
end

%Main loop of the fast clustering method
for i=1:iterations
    v = zeros(dataClusters,size(data,2));                   %Per-center counts    
    
    %Take a new batch
    randPermutations=randperm(size(data,1));
    M = data(randPermutations(1:batchSize),:);
    
    %Get boolean indices of the batch
    booleanIndexBatch=zeros(size(data,1),1);
    booleanIndexBatch(randPermutations(1:batchSize))=true;
    
    %Cache the centers nearest to the batch
    [distances]=pdist2(single(c),single(M),'seuclidean');
    [minDistAll,closestCenterIndex]=min(distances);
    
    %Modify distances, if supervision is provided 
    if ~isempty(cellOfSuggestedDocs)
        
        %Update distribution of docs that contain and does not contain a given
        %boosted word. This is done for each boosted word and for each centroid
        allBooleanBatchDocIndicesInBatch = false(1,batchSize)'; 
        distAux = distances';
        finalDistances = distances';        

        for indexWord=1:length(cellOfSuggestedDocs)
          
            %Get the documents that contain the word
            docIndices = cellOfSuggestedDocs{1,indexWord};
    
            %Get boolean indices of the documents that contain word
            booleanDocIndices=zeros(size(data,1),1);
            booleanDocIndices(docIndices)=true;
            %Get boolean indices of the documents that contain word and are in the batch
            booleanBatchDocIndices = and(booleanDocIndices,booleanIndexBatch);
            %Get boolean indices of the documents that contain word and are in the batch (assuming order of the random permutation) 
            booleanBatchDocIndicesBatchMapping = booleanBatchDocIndices(randPermutations); 
            
            %Boolean index of docs in the batch and that contains the word (order and size of the batch) 
            booleanBatchDocIndicesInBatch = false(batchSize,1);
            booleanBatchDocIndicesInBatch(booleanBatchDocIndicesBatchMapping) = true;
            
            %Boolean index of docs in the batch and that does not contain the word
            invertedBooleanBatchDocIndicesInBatch = ~booleanBatchDocIndicesInBatch;
            
            % Make distances of batchDocIndices closer or larger according to the distributions values
            newDistances = distances(:,booleanBatchDocIndicesInBatch)' .* (1-(repmat(distributions{indexWord},sum(booleanBatchDocIndicesInBatch),1) ./ (Y0+factor*slope)));

            % Find closestCenterIndex
             [minDist,closestCenterIndexBatch] = min(newDistances,[],2);
                     
            % Update distributions                
            counts = arrayfun( @(x)sum(closestCenterIndexBatch==x), 1:dataClusters );
            totalCounts{indexWord} = totalCounts{indexWord} + counts;
            distributions{indexWord} = totalCounts{indexWord}/(sum(totalCounts{indexWord})+1);
                       
            %Penalize documents not containing the word for not being assigned to the clusters where the suggested words go
            oldDistances = distances(:,invertedBooleanBatchDocIndicesInBatch)' .* (1+repmat(distributions{indexWord},sum(invertedBooleanBatchDocIndicesInBatch),1) .* (mY0-factor*slope));
            
            allBooleanBatchDocIndicesInBatch = or(allBooleanBatchDocIndicesInBatch,booleanBatchDocIndicesInBatch);
            neverSuggestedDocs = and(~allBooleanBatchDocIndicesInBatch,~booleanBatchDocIndicesInBatch);
            
            distAux(booleanBatchDocIndicesInBatch,:) = newDistances;
            distAux(~booleanBatchDocIndicesInBatch,:) = oldDistances;
            finalDistances(neverSuggestedDocs,:) = bsxfun(@max,finalDistances(neverSuggestedDocs,:), distAux(neverSuggestedDocs,:));
            finalDistances(~neverSuggestedDocs,:) = bsxfun(@min,finalDistances(~neverSuggestedDocs,:), distAux(~neverSuggestedDocs,:));
        end
               
        [~,results] = min(finalDistances,[],2);
        closestCenterIndex = results';
    end
    
    %Update centroid positions
    for indexPoint=1:batchSize            
        indexCenter = closestCenterIndex(indexPoint);
        v(indexCenter) = v(indexCenter) + 1;
        nu = 1/v(indexCenter);            
        c(indexCenter,:) = (1 - nu) * c(indexCenter,:) + nu * M(indexPoint,:);
    end    
end
timeElapsed =toc;
disp(['Centroid finding with supervision: (complexity depends on amount of supervision)']);
disp(timeElapsed);


tic;
%Compute 'regular' distances for all the data to the centroids
[distances]=pdist2(single(c),single(data),'euclidean');
[minDistAll,resultsAll]=min(distances);

if ~isempty(cellOfSuggestedDocs)
    %Update distribution of docs that contain and does not contain a given
    %boosted word. This is done for each booted word and for each centroid
    
    finalDistances = distances';
    distAux = distances';
    allBooleanSelection = false(1,size(data,1))';
    
    for indexWord=1:length(cellOfSuggestedDocs)
        
        %Get the documents that contain the word
        docIndices = cellOfSuggestedDocs{1,indexWord};
        %Get boolean indices of the documents that contain word
        booleanDocIndices=false(size(data,1),1);
        booleanDocIndices(docIndices)=true;
        
        % Make distances of batchDocIndices closer or larger
        % according to the distributions values
        newDistances = distances(:,booleanDocIndices)' .*(1-(repmat(distributions{indexWord},length(docIndices),1) ./ (Y0+factor*slope)));
        
        %Penalize documents not containing the word for not being assigned to the clusters where the suggested words go
        oldDistances = distances(:,~booleanDocIndices)' .*(1+(repmat(distributions{indexWord},sum(~booleanDocIndices),1) .* (mY0-factor*slope)));
        
        allBooleanSelection = or(allBooleanSelection,booleanDocIndices);
        neverSuggestedDocs = and(~allBooleanSelection,~booleanDocIndices);
        
        distAux(booleanDocIndices,:) = newDistances;
        distAux(~booleanDocIndices,:) = oldDistances;
        finalDistances(neverSuggestedDocs,:) = bsxfun(@max,finalDistances(neverSuggestedDocs,:), distAux(neverSuggestedDocs,:));
        finalDistances(~neverSuggestedDocs,:) = bsxfun(@min,finalDistances(~neverSuggestedDocs,:), distAux(~neverSuggestedDocs,:));
        
    end

    [~,results] = min(finalDistances,[],2);
    results = results';
    
else
    results = resultsAll;
end

timeElapsed =toc;
disp(['Data point assignment (linear complexity): ']);
disp(timeElapsed);

%%Compute sums
if (sumOfDistances)%% want sum of datapoint distances for each cluster?
    sums=zeros(dataClusters,1);
    for indexCluster=1:dataClusters
        sums(indexCluster,1) = sum(distances(results==indexCluster));
    end
else
    sums=zeros(dataClusters,1);
end