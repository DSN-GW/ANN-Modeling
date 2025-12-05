PCA + KMeans using all scaled numerical features
Overview

Clustering_V1 applies a classical PCA → KMeans pipeline using all scaled numerical features.
This version focuses on linear structure and compact spherical clusters.

Features Used

All scaled numeric columns:

FSR_scaled

BIS_scaled

SRS.Raw_scaled

TDNorm_avg_PE_scaled

overall_avg_PE_scaled

TDnorm_concept_learning_scaled

overall_concept_learning_scaled

Pipeline Steps

Load & preprocess dataset

Perform PCA (n_components=2)

Fit KMeans for k = 2 to 6

Evaluate using:

Silhouette Score

Davies–Bouldin Index

ARI / NMI (vs TD/ASD)

Select best k using Silhouette

Generate visualizations:

PCA target projection

PCA cluster projection

Cluster distribution

PCA loadings

Cluster feature means & variance

Streamlit interactive PCA

Interactive PCA Cluster Explorer

Save:

processed_with_clusters.csv

metrics.json

Use Case

Baseline clustering on the full numerical dataset.
Useful when evaluating global linear separability.
