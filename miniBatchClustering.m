function [results, c]= miniBatchClustering(data,dataClusters,iterations,batchSize,clusterSeeds, sumOfDistances)
%
%%The step 10 in the original mini-batch k-means algorithm by Sculley
%%http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf overwrites the
%%current centers (which shouldn't). Another variable should be used (here
%%indexCenter)

%

% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% Axel Soto, 2013
% Dalhousie University

%%Initialize centers
tic;
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

for i=1:iterations
    v = zeros(dataClusters,size(data,2));                   %Per-center counts
    randPermutations=randperm(size(data,1));
    M = data(randPermutations(1:batchSize),:);
    
    %Cache the centers nearest to the batch
    [~,closestCenterIndex]=pdist2(single(c),single(M),'euclidean','Smallest',1);
    
    for indexPoint=1:batchSize            
        indexCenter = closestCenterIndex(indexPoint);
        v(indexCenter) = v(indexCenter) + 1;
        nu = 1/v(indexCenter);
            
        c(indexCenter,:) = (1 - nu) * c(indexCenter,:) + nu * M(indexPoint,:);
    end    
end
timeElapsed =toc;
disp(['Centroid finding (constant complexity): ']);
disp(timeElapsed);

tic;
[distances,results]=pdist2(single(c),single(data),'seuclidean','Smallest',1);
timeElapsed =toc;
disp(['Data point assignment (linear complexity): ']);
disp(timeElapsed);

if (sumOfDistances)%% want sum of datapoint distances for each cluster?
    sums=zeros(dataClusters,1);
    for indexCluster=1:dataClusters
        sums(indexCluster,1) = sum(distances(results==indexCluster));
    end
end