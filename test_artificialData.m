%%Without supervision

%Create random data
mu1 = [2,2];
mu2 = [0,0];
mu3 = [-2,-2];
mu4 = [2,-2];
mu5 = [-2,2];

nInstancesPerClass = 100;

sigma = eye(2);

data=[mvnrnd(mu1,sigma*.2,nInstancesPerClass); mvnrnd(mu2,sigma*.15,nInstancesPerClass);mvnrnd(mu3,sigma*.1,nInstancesPerClass);mvnrnd(mu4,sigma*.08,nInstancesPerClass);mvnrnd(mu5,sigma*.06,nInstancesPerClass);];

nClusters=5;
minibatchSize = 10;
iter = 500;

[clusters,centroids]=miniBatchClustering(data,nClusters,iter,minibatchSize);
figure('Name','Implementation of mini-batch clustering (Sculley et al 2010)');
gscatter(data(:,1),data(:,2),clusters,[],'.ox+*sdv^<>ph');

%% With supervision

influenceFactor = 2; %Here the influence factor goes between 1 and 10 (10-100%)

%All the data points lower than 1 in both dimensions are suggested to be
%kept together
suggestedPoints{1,1}=find((data(:,1)>-0.5)&(data(:,2)>-0.5)&(data(:,1)<0.5)&(data(:,2)<0.5));

[clustersSupervision,centroidsSupervision] = miniBatchClusteringSupervision(data,nClusters,iter,minibatchSize,centroids,suggestedPoints,influenceFactor);
figure('Name','Fast user-supervised clustering (Soto et al 2014). Supervision |x|<0.5 factor = 20%');
gscatter(data(:,1),data(:,2),clustersSupervision,[],'.ox+*sdv^<>ph');

%Repeat with a higher influence factor
influenceFactor = 8; 
[clustersSupervision,centroidsSupervision] = miniBatchClusteringSupervision(data,nClusters,iter,minibatchSize,centroids,suggestedPoints,influenceFactor);
figure('Name','Fast user-supervised clustering (Soto et al 2014). Supervision |x|<0.5 factor = 80%');
gscatter(data(:,1),data(:,2),clustersSupervision,[],'.ox+*sdv^<>ph');

