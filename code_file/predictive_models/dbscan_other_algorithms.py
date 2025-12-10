import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import numpy as np

print(os.getcwd())
os.chdir(r'C:\Users\eilee\OneDrive\Documents\CSS 499\cluster analysis data')

guide_data = pd.read_csv('4_spectral_clustering_output.csv')
guide_data = pd.read_csv('4_agg_clustering_output.csv')

guide_data.pop('name')
guide_data.pop('avg T0 log2fc')
guide_data.pop('avg T24 log2fc')
guide_data.pop('avg T48 log2fc')
guide_data.pop('avg T72 log2fc')
guide_data.pop('rank')
guide_data.pop('slope1_score')
guide_data.pop('slope2_score')
guide_data.pop('slope3_score')
guide_data.pop('slope4_score')
guide_data.pop('slope5_score')
guide_data.pop('slope6_score')
guide_data.pop('total_slope_score')
guide_data.pop('average_slope')

rank = guide_data.pop('dbscan_label')
rank_max = max(rank.value_counts())
rank_sum = sum(rank.value_counts())
rank_max
rank.value_counts()
rank = pd.Series(np.where(rank == 1, 0, 1))
rank_max = max(rank.value_counts())
rank_sum = sum(rank.value_counts())
rank_max
rank.value_counts()

#train test split
train_guide,val_guide, train_rank, val_rank = \
    train_test_split(guide_data, rank, stratify= rank)

sl_guide = copy.deepcopy(val_guide)

train_guide.pop('Guide')
val_guide.pop('Guide')

#standardize data
cols = train_guide.columns
scaler = StandardScaler()
train_guide = scaler.fit_transform(train_guide)
val_guide = scaler.transform(val_guide)
train_guide = pd.DataFrame(train_guide, columns= cols)
val_guide = pd.DataFrame(val_guide, columns= cols)

sm = SMOTE()
x_train, y_train = sm.fit_resample(train_guide, train_rank)
train_guide, train_rank = sm.fit_resample(x_train, y_train)

rus = RandomUnderSampler()
train_guide, train_rank = rus.fit_resample(train_guide, train_rank)

# Model
model = CatBoostClassifier(
    learning_rate = 0.1,
    depth = 6,
    l2_leaf_reg  = 42,
    verbose = False,
    eval_metric = 'F1',
    n_estimators = 500,
    random_state = 42
)

model = LGBMClassifier(
    learning_rate = 0.1,
    n_estimators = 500
)

model = XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.1,
    eval_metric = 'logloss'
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

# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(val_rank, rank_predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [0, 1])
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
roc_auc = metrics.roc_auc_score(val_rank, rank_predictions)
print(roc_auc)
pr_auc = metrics.average_precision_score(val_rank, rank_predictions)
print(pr_auc)

val_rank.value_counts()
rank_max = max(val_rank.value_counts())
rank_sum = sum(val_rank.value_counts())
null_accuracy = (rank_max / rank_sum)
null_accuracy


# XAI
## SHAP
import shap

explainer = shap.Explainer(model, val_guide)
shap_values = explainer(val_guide, check_additivity = False)
shap.plots.beeswarm(shap_values) #guide that do not work
plt.savefig('beeswarm_1.png')

shap_values.shape
shap.plots.waterfall(shap_values[1, :, 1])
shap.plots.force(explainer.expected_value[1], shap_values.values[0, :, 1], matplotlib=True)

#Sequence Logo
import logomaker as lm

sl_guide = sl_guide['Guide'].str.upper()
guides_counts = lm.alignment_to_matrix(sl_guide)
guides_counts
info_matrix = lm.transform_matrix(guides_counts,
                                  from_type = 'counts',
                                  to_type = 'weight')
info_matrix = lm.transform_matrix(info_matrix, center_values = True)
info_matrix = lm.transform_matrix(guides_counts, normalize_values = True)
logo = lm.Logo(df = info_matrix,
        fade_below = 0.5,
        shade_below = 0.5)
logo.ax.set_xlabel('Position')
#plt.savefig('sequence_logo.png')
plt.show()