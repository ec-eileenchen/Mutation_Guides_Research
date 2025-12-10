import pandas as pd
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn
from catboost import CatBoostClassifier

# patch scikit-learn to use Intel's optimizations
patch_sklearn()

print(os.getcwd())

#guide_data = pd.read_csv('New_Ranked_Guides.csv')
guide_data = pd.read_csv('OH_Ranked_Guides.csv')

#excluding rank 3
guide_data = guide_data[guide_data['rank'] != 3]

guide_data = guide_data.reset_index(drop= True)

#3 ranks (1 = 1, 45 = 0, 6 = 6)
binary_rank = []
for i in guide_data['rank']:
    if i == 1:
        binary_rank.append(1)
    elif i == 6:
        binary_rank.append(6)
    else:
        binary_rank.append(0)

##binary ranks (1 = 1, 456 = 0)
binary_rank = []
for i in guide_data['rank']:
    if i == 1:
        binary_rank.append(1)
    else:
        binary_rank.append(0)

guide_data['Binary_Rank'] = binary_rank
guide_data.head()

guide_data.pop('name')
guide_data.pop('Guide')
guide_data.pop('avg T0 log2fc')
guide_data.pop('avg T24 log2fc')
guide_data.pop('avg T48 log2fc')
guide_data.pop('avg T72 log2fc')
guide_data.pop('rank')

rank = guide_data.pop('rank')
rank = guide_data.pop('Binary_Rank')

# split train and validation data
train_guide,val_guide, train_rank, val_rank = \
    train_test_split(guide_data, rank, stratify= rank)

#standarize data
## StandardScaler
cols = train_guide.columns
scaler = StandardScaler()
train_guide = scaler.fit_transform(train_guide)
val_guide = scaler.transform(val_guide)
train_guide = pd.DataFrame(train_guide, columns= cols)
val_guide = pd.DataFrame(val_guide, columns= cols)
## RobustScaler
cols = train_guide.columns
scaler = RobustScaler()
train_guide = scaler.fit_transform(train_guide)
val_guide = scaler.transform(val_guide)
train_guide = pd.DataFrame(train_guide, columns= cols)
val_guide = pd.DataFrame(val_guide, columns= cols)


# StratifiedKFold
kfold = StratifiedKFold()

# random oversampling pipeline
random_over_pipe = make_pipeline(
    RandomOverSampler(random_state= 50),
    RandomForestClassifier(random_state= 10, class_weight= 'balanced'))

# cross validation(over)
accuracy_score = cross_val_score(
    random_over_pipe, train_guide,
    train_rank, scoring= 'accuracy',
    cv = kfold
)
print('Accuracy: \n', accuracy_score)
print('AVG Accuracy: \n', accuracy_score.mean())

# random undersampline pipleing
random_under_pipe = make_pipeline(
    RandomUnderSampler(random_state = 50),
    RandomForestClassifier(random_state= 10, class_weight= 'balanced'))

# cross validation(under)
accuracy_score = cross_val_score(
    random_under_pipe, train_guide,
    train_rank, scoring= 'accuracy',
    cv = kfold
)
print('Accuracy: \n', accuracy_score)
print('AVG Accuracy: \n', accuracy_score.mean())

# optimize hyperparameter value
params_value = {
    'n_estimators' : [50, 100, 200],
    'random_state' : [10, 26, 30],
    'max_depth' : [5, 10, 12, 16]
}

params = {'randomforestclassifier__' \
          + key : params_value[key] \
            for key in params_value}

# oversampling
grid_over = GridSearchCV(
    random_over_pipe, 
    param_grid = params,
    cv = kfold,
    scoring = 'accuracy',
    return_train_score = True)

# oversampling
grid_over.fit(train_guide, train_rank)

# oversampling
print(grid_over.best_params_)
print(grid_over.best_score_)

# undersampling
grid_under = GridSearchCV(
    random_under_pipe, 
    param_grid = params,
    cv = kfold,
    scoring = 'accuracy',
    return_train_score = True)

# undersampling
grid_under.fit(train_guide, train_rank)

#undersampling
print(grid_under.best_params_)
print(grid_under.best_score_)


# predict
## oversampling
rank_predictions = \
    grid_over.best_estimator_.named_steps['randomforestclassifier'].predict(val_guide)

## undersampling
rank_predictions = \
    grid_under.best_estimator_.named_steps['randomforestclassifier'].predict(val_guide)

# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(val_rank, rank_predictions)
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 3, 4, 5, 6])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 4, 5, 6])
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 0, 6])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 0])
cm_display.plot()
#plt.savefig('Binary Confusion Matrix Display')
plt.show()

#score result
report = metrics.classification_report(val_rank, rank_predictions)
print(report)

accuracy = metrics.accuracy_score(val_rank, rank_predictions)
print(accuracy)
f1 = metrics.f1_score(val_rank, rank_predictions, average= 'weighted')
print(f1)
recall = metrics.recall_score(val_rank, rank_predictions, average= 'weighted')
print(recall)
precision = metrics.precision_score(val_rank, rank_predictions, average= 'weighted')
print(precision)
###################################################################################################

#lime
from lime import lime_tabular
class_names = [1, 4, 5, 6]
feature_names = list(train_guide.columns)
explainer = lime_tabular.LimeTabularExplainer(train_guide.values, feature_names= feature_names,
                                 class_names = class_names, mode= 'classification')
sample_row = val_guide.values[248]
exp = explainer.explain_instance(data_row = sample_row,
                                 predict_fn = grid_over.best_estimator_.predict_proba,
                                 num_features = len(feature_names))
exp.save_to_file('Rank_ML2_Lime_248.html')

###################################################################################################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 85)
knn.fit(train_guide, train_rank)
rank_predictions = knn.predict(val_guide)

knn.predict_proba(val_guide)
knn.score(train_guide, train_rank)
knn.score(val_guide, val_rank)


val_rank.value_counts()
rank_max = max(val_rank.value_counts())
rank_sum = sum(val_rank.value_counts())
null_accuracy = (rank_max/rank_sum)
null_accuracy

sm = SMOTE()
x_train, y_train = sm.fit_resample(train_guide, train_rank)
train_guide, train_rank = sm.fit_resample(x_train, y_train)

rus = RandomUnderSampler()
train_guide, train_rank = rus.fit_resample(train_guide, train_rank)

#############################################################################################################

# Catboost Model
model = CatBoostClassifier(
    learning_rate = 0.1,
    depth = 6,
    l2_leaf_reg  = 42,
    verbose = False,
    eval_metric = 'F1',
    n_estimators = 500,
    random_state = 42
)

model.fit(
    train_guide, 
    train_rank,
    eval_set= (val_guide, val_rank),
    verbose=True,
    plot=False
)

print('CatBoost model is fitted: ' + str(model.is_fitted()))
print('CatBoost model parameters:')
print(model.get_params())

rank_predictions = model.predict(val_guide)

import shap

explainer = shap.Explainer(model, val_guide)
shap_values = explainer(val_guide, check_additivity = False)
shap.plots.beeswarm(shap_values)
