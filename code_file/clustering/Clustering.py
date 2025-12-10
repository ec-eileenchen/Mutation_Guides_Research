import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, SpectralBiclustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, OPTICS, MeanShift, estimate_bandwidth, AgglomerativeClustering
from tensorflow.python.keras.utils.vis_utils import plot_model
from hdbscan import HDBSCAN

#print(os.getcwd())

dataset = pd.read_csv('OH_Ranked_Guides.csv')
dataset.head(5)

slope1_scores = (dataset['avg T24 log2fc'] - dataset['avg T0 log2fc']) / 24
#slope1_scores
slope2_scores = (dataset['avg T48 log2fc'] - dataset['avg T24 log2fc']) / 24
slope3_scores = (dataset['avg T72 log2fc'] - dataset['avg T48 log2fc']) / 24

#slope4_scores = (dataset['avg T24 log2fc'] - dataset['avg T0 log2fc']) / 24
#slope1_scores
#slope4_scores = (dataset['avg T48 log2fc'] - dataset['avg T0 log2fc']) / 24
#slope5_scores = (dataset['avg T72 log2fc'] - dataset['avg T0 log2fc']) / 24
slope4_scores = (dataset['avg T48 log2fc'] - dataset['avg T0 log2fc']) / 48
slope5_scores = (dataset['avg T72 log2fc'] - dataset['avg T0 log2fc']) / 72
slope6_scores = (dataset['avg T72 log2fc'])

t = np.array([0, 24, 48, 72]).reshape(-1, 1) 

total_slope = np.array(dataset.loc[:, ['avg T0 log2fc', 'avg T24 log2fc', 'avg T48 log2fc', 'avg T72 log2fc']])

slope_scores = []

for row in total_slope:
    model = LinearRegression().fit(t, row)
    slope_scores.append(model.coef_[0])


dataset['slope1_score'] = slope1_scores
dataset['slope2_score'] = slope2_scores
dataset['slope3_score'] = slope3_scores
dataset['total_slope_score'] = slope_scores
dataset['slope4_score'] = slope4_scores
dataset['slope5_score'] = slope5_scores
dataset['slope6_score'] = slope6_scores
dataset['average_slope'] = (slope1_scores + slope2_scores + slope3_scores)/3



#input_4d = dataset[['slope1_score', 'slope2_score', 'slope3_score', 'avg T72 log2fc']]
input_4d = dataset[['slope1_score', 'slope2_score', 'slope3_score', 'total_slope_score']]
input_7d = dataset[['slope1_score', 'slope2_score', 'slope3_score', 'total_slope_score', 'slope4_score', 'slope5_score', 'slope6_score']]
input_2d = dataset[['slope6_score', 'average_slope']]
input_4d = input_2d
input_4d = input_7d
input_4d ## (5633, 4)

input_4d = StandardScaler().fit_transform(input_4d)


neigh = NearestNeighbors()
nbrs = neigh.fit(input_4d)
distances, indices = nbrs.kneighbors(input_4d)
k_distances = np.sort(distances[:, 4])

plt.plot(k_distances)
plt.ylabel("5-NN distance")
plt.xlabel("Points sorted by distance")
plt.title("k-distance graph")
plt.grid(True)
plt.show()

#db = DBSCAN(eps=0.3, min_samples=10).fit(input_4d)
#labels = db.labels_
#print (labels)


#db = KMeans(n_clusters= 3).fit(input_4d)
#labels = db.labels_
#print (labels)

spectral_rbf = SpectralClustering(n_clusters=2, affinity='rbf')
labels = spectral_rbf.fit_predict(input_4d)
print (labels)

sb = SpectralBiclustering(    n_clusters=2,
    method='log',          # better for many datasets
    n_components=6,        # helps stability
    random_state=42
)
sb.fit(input_4d)

print(sb.row_labels_)     # cluster labels for each row
print(sb.column_labels_)  # cluster labels for each feature/column
labels = sb.row_labels_

bandwidth = estimate_bandwidth(input_4d, quantile=0.25, n_samples=len(input_4d))
db = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(input_4d)
labels = db.labels_
#print (labels)

#db = OPTICS(min_samples=10, xi=0.0001, min_cluster_size=0.0001).fit(input_4d)
#labels = db.labels_
#print(labels)

db = HDBSCAN(
    min_cluster_size=50,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom'
).fit(input_4d)
labels = db.labels_
print(labels)

db = AgglomerativeClustering(
    distance_threshold = 80,
    n_clusters = None)
labels = db.fit_predict(input_4d)
print(labels)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

dataset['dbscan_label'] = labels
print(dataset['dbscan_label'].value_counts())

tsne = TSNE(n_components= 2)
plot_2d = tsne.fit_transform(input_4d)
#plot_2d = input_4d

plt.scatter(plot_2d[:, 0], plot_2d[:, 1])
plt.title('t-SNE Projection (4D to 2D)')
plt.xlabel('TSNE-1')
plt.ylabel('TSNE-2')
plt.grid(False)
plt.show()

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = plot_2d[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = plot_2d[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.savefig('2D.png')
plt.show()


dataset.groupby('dbscan_label')['dbscan_label'].value_counts()
dataset.groupby('dbscan_label')['rank'].value_counts()


dataset.to_csv('4_agg_clustering_output.csv')
