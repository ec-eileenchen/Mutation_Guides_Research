# Mutation Guides Research
## datasets
* Ranked_Guides: original dataset
* OH_Ranked_Guides: Preprocessed dataset with feature engineering and each nucleotide is one hot encoded <br>
#### cluster_analysis_datasets - all datasets output of the Clustering code file
* 2_ : 2 dimensional
* 4_ : 4 dimensional
* 7_ : 7 dimensional
* 4_agg_clustering_output: 4 dimensional aggregative hierarchical clustering (AHC)

## code_file
#### clustering
* Clustering: perform clustering using "OH_Ranked_Guides" dataset <br>
Models implemented: SpectralClustering, OPTICS, MeanShift, DBSCAN, HDBSCAN, SpectralBiclustering, and AgglomerativeClustering
* Cluster Analysis: Examination of the clustering labels with average slope, total slope, and avg T72 log2fc
* label_rank_comparison: The comparison of clustering labels and original ranks for Spectralclustering and MeanShift models

#### predictive_models
* Preprocess (One Hot Encoding): the code file that generated the OH _Ranked_Guides dataset
* clustering_label_other_algorithms: Models that predict the binary clustering labels<br>
Mpdels implemented: CatBoostClassifier, and LGBMClassifier
* clustering_label_randomforest_gridsearch: Randomforest GridSearchCV model that predict the binarhy clustering labels with logomaker implemented to generate sequence logo
* rank_ml_models: Models that predict the original ranks with XAI implemented<br>
Models implemented: RandomForestClassifier, CatBoostClassifier, KNN Classifier<br>
XAI implemented: LIME and SHAP
* 1D_cnn: 1 dimensional CNN model predicting the original ranks

## graphs
#### SHAP 
* catboost_SHAP_beeswarm : SHAP beeswarm summary results of CatboostClassifier model predicting the working guides from clustering binary labels
* randomforest_SHAP_beeswarm: SHAP beeswarm summary result of RandomForest GrindSearchCV model predicting the working guides from clustering binary labels
* randomforest_SHAP_summary_bars: SHAP bar chart summary result of RandomForest GridSearchCV model predicting the working guides from clustering binary lables.
#### sequence_logo
* 4 different types of sequence logo graphs<br>
Matrix types: weight, information, normalization, and count
