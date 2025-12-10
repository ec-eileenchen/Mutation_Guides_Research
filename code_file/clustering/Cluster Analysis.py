import os
import pandas as pd
#print(os.getcwd())


# SpectralClustering 4d
s_output = pd.read_csv('4_spectral_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc', 'rank', 
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# label 1 min avg_slope -->  -0.045938
minidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = s_output.loc[minidx1, features]
cluster1min

# label 1 max avg_slope --> 0.059411
maxidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = s_output.loc[maxidx1, features]
cluster1max

# label 0 min avg_slope --> -0.182162
minidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = s_output.loc[minidx0, features]
cluster0min

# label 0 max avg_slope --> -0.006309
maxidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = s_output.loc[maxidx0, features]
cluster0max

## total_slope
# label 1 min total_slope -->  -0.045938
minidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'total_slope_score'].idxmin()
cluster1min = s_output.loc[minidx1, features]
cluster1min

# label 1 max total_slope --> 0.059411
maxidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'total_slope_score'].idxmax()
cluster1max = s_output.loc[maxidx1, features]
cluster1max

# label 0 min total_slope --> -0.182162
minidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'total_slope_score'].idxmin()
cluster0min = s_output.loc[minidx0, features]
cluster0min

# label 0 max total_slope --> -0.006309
maxidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'total_slope_score'].idxmax()
cluster0max = s_output.loc[maxidx0, features]
cluster0max

print(s_output.loc[s_output['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 6.837164503 ~ 16.56885313 
print(s_output.loc[s_output['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 11.75263585

## avg T72 log2fc
# label 1 min total_slope -->  -0.045938
minidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'avg T72 log2fc'].idxmin()
cluster1min = s_output.loc[minidx1, features]
cluster1min

# label 1 max total_slope --> 0.059411
maxidx1 = s_output.loc[s_output['dbscan_label'] == 1, 'avg T72 log2fc'].idxmax()
cluster1max = s_output.loc[maxidx1, features]
cluster1max

# label 0 min total_slope --> -0.182162
minidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'avg T72 log2fc'].idxmin()
cluster0min = s_output.loc[minidx0, features]
cluster0min

# label 0 max total_slope --> -0.006309
maxidx0 = s_output.loc[s_output['dbscan_label'] == 0, 'avg T72 log2fc'].idxmax()
cluster0max = s_output.loc[maxidx0, features]
cluster0max

s_output.groupby('dbscan_label')['rank'].value_counts()

# SpectralClustering 7d
s7_output = pd.read_csv('7_spectral_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc',
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# label 1 min avg_slope -->   -0.042854
minidx1 = s7_output.loc[s7_output['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = s7_output.loc[minidx1, features]
cluster1min

# label 1 max avg_slope --> 0.059411
maxidx1 = s7_output.loc[s7_output['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = s7_output.loc[maxidx1, features]
cluster1max

# label 0 min avg_slope --> -0.182162
minidx0 = s7_output.loc[s7_output['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = s7_output.loc[minidx0, features]
cluster0min

# label 0 max avg_slope --> -0.006309
maxidx0 = s7_output.loc[s7_output['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = s7_output.loc[maxidx0, features]
cluster0max

print(s7_output.loc[s7_output['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 6.837164503 ~ 16.56885313 
print(s7_output.loc[s7_output['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 11.75263585

s7_output.groupby('dbscan_label')['rank'].value_counts()

# SpectralClustering 2d
s2_output = pd.read_csv('2_spectral_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc',
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# label 1 min avg_slope -->   -0.041034
minidx1 = s2_output.loc[s2_output['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = s2_output.loc[minidx1, features]
cluster1min

# label 1 max avg_slope --> 0.059411
maxidx1 = s2_output.loc[s2_output['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = s2_output.loc[maxidx1, features]
cluster1max

# label 0 min avg_slope --> -0.182162
minidx0 = s2_output.loc[s2_output['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = s2_output.loc[minidx0, features]
cluster0min

# label 0 max avg_slope --> -0.005403
maxidx0 = s2_output.loc[s2_output['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = s2_output.loc[maxidx0, features]
cluster0max

print(s2_output.loc[s2_output['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 8.153812177 ~ 16.56885313 
print(s2_output.loc[s2_output['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 9.712423102

s2_output.groupby('dbscan_label')['rank'].value_counts()

s2_output.groupby('rank')['rank'].value_counts()

# Mean Shift 4d
output_4 = pd.read_csv('4_MeanShift_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc',
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# dbscan label 1 min avg_slope -->   -0.06525
minidx1 = output_4.loc[output_4['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = output_4.loc[minidx1, features]
cluster1min

# dbscan label 1 max avg_slope --> 0.059411
maxidx1 = output_4.loc[output_4['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = output_4.loc[maxidx1, features]
cluster1max

# dbscan label 0 min avg_slope --> -0.182162
minidx0 = output_4.loc[output_4['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = output_4.loc[minidx0, features]
cluster0min

# dbscan label 0 max avg_slope -->  -0.051458
maxidx0 = output_4.loc[output_4['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = output_4.loc[maxidx0, features]
cluster0max

print(output_4.loc[output_4['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 3.642206199 ~ 16.56885313

print(output_4.loc[output_4['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 8.948233691

output_4.groupby('dbscan_label')['rank'].value_counts()

# Mean Shift with 48 and 72 instead of 24
output_7 = pd.read_csv('7_MeanShift_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc',
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# dbscan label 1 min avg_slope -->  -0.06525
minidx1 = output_7.loc[output_7['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = output_7.loc[minidx1, features]
cluster1min

# dbscan label 1 max avg_slope --> 0.059411
maxidx1 = output_7.loc[output_7['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = output_7.loc[maxidx1, features]
cluster1max

# dbscan label 0 min avg_slope --> -0.182162
minidx0 = output_7.loc[output_7['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = output_7.loc[minidx0, features]
cluster0min

# dbscan label 0 max avg_slope -->  -0.044101
maxidx0 = output_7.loc[output_7['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = output_7.loc[maxidx0, features]
cluster0max

print(output_7.loc[output_7['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 3.642206199 ~ 16.56885313
print(output_7.loc[output_7['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 9.053816864

output_7.groupby('dbscan_label')['rank'].value_counts()

# Mean Shift 2D
output_2 = pd.read_csv('2_meanshift_clustering_output.csv')
features = ['name', 'Guide', 'avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc',
            'slope1_score', 'slope2_score', 'slope3_score', 'slope4_score', 'slope5_score', 'slope6_score',
            'total_slope_score', 'average_slope', 'dbscan_label']

# dbscan label 1 min avg_slope -->   -0.06221
minidx1 = output_2.loc[output_2['dbscan_label'] == 1, 'average_slope'].idxmin()
cluster1min = output_2.loc[minidx1, features]
cluster1min

# dbscan label 1 max avg_slope --> 0.059411
maxidx1 = output_2.loc[output_2['dbscan_label'] == 1, 'average_slope'].idxmax()
cluster1max = output_2.loc[maxidx1, features]
cluster1max

# dbscan label 0 min avg_slope --> -0.182162
minidx0 = output_2.loc[output_2['dbscan_label'] == 0, 'average_slope'].idxmin()
cluster0min = output_2.loc[minidx0, features]
cluster0min

# dbscan label 0 max avg_slope -->  -0.014602
maxidx0 = output_2.loc[output_2['dbscan_label'] == 0, 'average_slope'].idxmax()
cluster0max = output_2.loc[maxidx0, features]
cluster0max

print(output_2.loc[output_2['dbscan_label'] == 1,'avg T72 log2fc']) ## avg T72 log2fc 6.837164503 ~ 16.56885313
print(output_2.loc[output_2['dbscan_label'] == 0,'avg T72 log2fc']) ## avg T72 log2fc 0 ~ 8.227012593

output_2.groupby('dbscan_label')['rank'].value_counts()
