import pandas as pd
import os 
from sklearnex import patch_sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.metrics import Precision, Recall
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

# patch scikit-learn to use Intel's optimizations
patch_sklearn()

print(os.getcwd())

guide_data = pd.read_csv('OH_Ranked_Guides.csv')

#excluding rank 3
guide_data = guide_data[guide_data['rank'] != 3]

guide_data = guide_data.reset_index(drop= True)

guide_data.head()

#binary rank (1 = 1, 45 = 0, 6 = 6)
binary_rank = []
for i in guide_data['rank']:
    if i == 1:
        binary_rank.append(1)
    elif i == 6:
        binary_rank.append(6)
    else:
        binary_rank.append(0)

##binary rank (1 = 1, 456 = 0)
binary_rank = []
for i in guide_data['rank']:
    if i == 1:
        binary_rank.append(1)
    else:
        binary_rank.append(0)

guide_data['rank'] = binary_rank
guide_data.head()

# OH rank 
oh_encoder = OneHotEncoder(sparse_output = False)
oh_encoder_cols = pd.DataFrame(oh_encoder.fit_transform(guide_data[['rank']]))
oh_encoder_cols
## set cols name
oh_title = ['rank_1', 'rank_3', 'rank_4', 'rank_5', 'rank_6']
oh_title = ['rank_1', 'rank_4', 'rank_5', 'rank_6']
oh_title = ['rank_1', 'rank_45','rank_6']
oh_title = ['rank_1', 'rank_456']
oh_encoder_cols.columns = oh_title
## remove index
oh_encoder_cols.index = guide_data.index
## remove categorical column
guide_data = guide_data.drop('rank', axis= 1)
## add one hot encoded cols
guide_data = pd.concat([guide_data, oh_encoder_cols], axis= 1)
guide_data.columns = guide_data.columns.astype(str)

guide_data

guide_data.pop('name')
guide_data.pop('Guide')
guide_data.pop('avg T0 log2fc')
guide_data.pop('avg T24 log2fc')
guide_data.pop('avg T48 log2fc')
guide_data.pop('avg T72 log2fc')
#guide_data.pop('G/C Content')
#guide_data.pop('Runs_of_As')
#guide_data.pop('Runs_of_Ts')
guide_data.pop('Guide_in_Num')

rank = guide_data[oh_title]

guide_data = guide_data.drop(oh_title, axis= 1)


guide_data.head()
guide_data

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

# reshape train and test data
train_guide = train_guide.to_numpy().reshape(train_guide.shape[0], train_guide.shape[1], 1)
np.shape(train_guide)
train_guide

train_rank = train_rank.to_numpy()
np.shape(train_rank)

val_guide = val_guide.to_numpy().reshape(val_guide.shape[0], val_guide.shape[1], 1)
np.shape(val_guide)

val_rank = val_rank.to_numpy()
np.shape(val_rank)

# create model
model = Sequential()
model.add(layers.Conv1D(filters= 32, kernel_size= 3, strides= 1, activation= 'relu', 
                        input_shape= (train_guide.shape[1], 1)))
model.add(layers.MaxPool1D(pool_size= 2, padding= 'valid'))
model.add(layers.Conv1D(filters= 16, kernel_size= 3, strides= 1, activation= 'relu'))
model.add(layers.MaxPool1D(pool_size= 2, padding= 'valid'))
model.add(layers.Flatten())
model.add(layers.Dropout(rate= 0.3))
model.add(layers.Dense(units= 4, activation= 'softmax'))

model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy', 
              metrics= ['accuracy', Precision(name= 'precision'), Recall(name= 'recall')])
model.summary()

early_stop = EarlyStopping(monitor= 'recall', min_delta= 0.01, patience= 4, 
                           verbose= 1, mode= 'max')

model.fit(train_guide, train_rank, epochs= 50, batch_size= 128, callbacks= early_stop)

# predict 
result = model.predict(val_guide)
result = pd.DataFrame(result, columns= [1, 4, 5, 6])
result

predicted_rank = result.idxmax(axis= 1)
predicted_rank

val_rank = pd.DataFrame(val_rank, columns= [1, 4, 5, 6])
val_rank = val_rank.idxmax(axis= 1)
val_rank

#reports
accuracy = metrics.accuracy_score(val_rank, predicted_rank)
accuracy

precision = metrics.precision_score(val_rank, predicted_rank, average= 'weighted')
precision


recall = metrics.recall_score(val_rank, predicted_rank, average= 'weighted')
recall

f1 = metrics.f1_score(val_rank, predicted_rank, average= 'weighted')
f1

# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(val_rank, predicted_rank)
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 3, 4, 5, 6])
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 4, 5, 6])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [1, 0, 6])
cm_display.plot()
#plt.savefig('1D CNN Confusion Matrix Display')
plt.show()

# Null Accuracy
val_rank.value_counts()
rank_max = max(val_rank.value_counts())
rank_sum = sum(val_rank.value_counts())
rank_sum
null_accuracy = (rank_max/rank_sum)
null_accuracy

# Weighted random guessing
def random_wguess_accuracy(val_rank):
    rank_series = pd.Series(val_rank)
    rank_distribution = rank_series.value_counts(normalize= True)
    return np.sum(rank_distribution **2)

weighted_guess = random_wguess_accuracy(val_rank)
weighted_guess

###################################################################################################
#lime
from lime import lime_tabular
train_guide_df = pd.DataFrame(np.squeeze(train_guide))
class_names = [1, 4, 5, 6]
feature_names = list(guide_data.columns)
explainer = lime_tabular.LimeTabularExplainer(train_guide_df.values, feature_names= feature_names,
                                 class_names = class_names, mode= 'classification')

val_guide_df = pd.DataFrame(np.squeeze(val_guide))
sample_row = val_guide_df.values[248]
predict_fn = lambda x: model.predict(x.reshape((x.shape[0], x.shape[1], 1)))
exp = explainer.explain_instance(data_row = sample_row,
                                 predict_fn = predict_fn,
                                 num_features = len(feature_names))
exp.save_to_file('1D_cnn_Lime_248.html')
