import os
import pandas as pd
from itertools import groupby 
from sklearn.preprocessing import OneHotEncoder


print(os.getcwd())
os.chdir('/Users/eileen/Documents/CSS 499')
print(os.getcwd())

df = pd.read_csv('Ranked_Guides.csv')
df = df.iloc[:, :7]
df.head(10)

# drop rank 2 data
new_df = df.copy()
rank_2_index = []
current_index = new_df.index[0]

while current_index <= new_df.index[-1]:
    if new_df.iloc[current_index, 6] == 2:
        rank_2_index.append(current_index)
    
    current_index += 1
## 367 observations
print(len(rank_2_index))

new_df = new_df.drop(rank_2_index)

# reset index after some rows are dropped
new_df = new_df.reset_index(drop= True)

# convert guide to numbers (new col)
## G: 0, A: 1, C: 2, T: 3
guide_dict = {'G': 0, 'A': 1, 'C': 2, 'T': 3}

guide_in_num = []

for guide in new_df['Guide']:
    num = ''
    guide = guide.upper()
    for char in guide:
        num += str(guide_dict.get(char))
    guide_in_num.append(num)

new_df['Guide_in_Num'] = guide_in_num

# guide 1 - 20
title = []
for i in range(1, 21):
    title.append('Guide_' + str(i))

guides = []
for guide in new_df['Guide_in_Num']:
    guides.append(list(guide))

data = pd.DataFrame(guides)
data.columns = title
data.head(5)

new_df = pd.concat([new_df, data], axis= 1)
new_df

# G/C content 
## G: 0, C: 2

gc_content = [] 
for guide_num in new_df['Guide_in_Num']:
    gc = 0
    for num in guide_num:
        num = int(num)
        if num == 0 or num == 2:
            gc += 1
    
    gc_content.append(gc / len(guide_num))

new_df['G/C Content'] = gc_content
new_df

test = new_df[new_df['G/C Content'] > 0.60]
print(test)

# runs of nucleotides (A and T)
## A: 1, T: 3

# counting the max runs of the specified 
# nucleotide in a guide sequence.
def max_runs(row, nucleotide):
    max_length = 0
    for key, group in groupby(row):
        if key == str(nucleotide):
            group_list = list(group)
            if len(group_list) > max_length:
                max_length = len(group_list)
    
    return max_length

## runs of A's
new_df['Runs_of_As'] = \
    new_df.loc[:, 'Guide_1' : 'Guide_20' ].apply(
        max_runs, axis = 'columns', args= ('1')
    )
        
new_df[new_df['Runs_of_As'] >= 4]

## runs of T's
new_df['Runs_of_Ts'] = \
    new_df.loc[:, 'Guide_1' : 'Guide_20' ].apply(
        max_runs, axis = 'columns', args= ('3')
    )
        
new_df[new_df['Runs_of_Ts'] >= 4]

# One Hot Encoding
oh_encoder = OneHotEncoder(sparse_output = False)
object_cols = list(title)
oh_encoder_cols = pd.DataFrame(oh_encoder.fit_transform(new_df[object_cols]))
## set cols name
oh_title = []
nucleotides = ['G', 'A', 'C', 'T']
for i in range(1, 21):
    for j in nucleotides:
        oh_title.append('Pos%d_%s' %(i, j))
oh_encoder_cols.columns = oh_title
## remove index
oh_encoder_cols.index = new_df.index
## remove categorical column
new_df = new_df.drop(object_cols, axis= 1)
## add one hot encoded cols
new_df = pd.concat([new_df, oh_encoder_cols], axis= 1)
new_df.columns = new_df.columns.astype(str)

new_df.head()

# to new csv file
new_df.to_csv('OH_Ranked_Guides.csv', index= False)