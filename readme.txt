Fast user-supervised clustering readme file
-------------------------------------------

There are some fixes to the mini-batch clustering code presented by Sculley et al 2010. The per-center counts data structure is reset for each iteration as opposed to only once at the beginning (improves the stability of the algorithm). Also in Sculley's pseudoce there is a variable (c) that is overwritten when it should not.

Our Matlab implementation does not follow the pseudocode presented in the paper as it has been optimized for Matlab data structures.

This Matlab code comes with one example to run the fast user-supervised clustering

-test_artificalData.m

The example data is in two dimensions to allow visualization.

It first runs our implementation of mini-batch clustering (Sculley et al 2010) in a dataset with 50000 data points.

The supervision consists in suggesting the points that are within the sphere |x|<0.5. Two influence factors are shown. 

----------------------------------------------
Axel J. Soto
Dalhousie University
July 2014

Axel J. Soto, Ryan Kiros, Vlado Keselj, Evangelos Milios "Exploratory Visual Analysis and Interactive Pattern Extraction from Semi-Structured Data" Under review.

