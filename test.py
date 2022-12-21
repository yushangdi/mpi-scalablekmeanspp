from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph
import numpy as np
import time
# SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
#                                     n_neighbors=n_neighbor,
#                                     assign_labels='discretize',
#                                     random_state=1, n_jobs=worker).fit(X)

dataset = "Mallat"
k = 330
n_components = 8
dir_addr = "/home/ubuntu/datasets/UCRArchive_2018/"

X1 = np.genfromtxt(dir_addr+dataset+"/"+dataset+"_TRAIN.tsv", delimiter="\t")
X2 = np.genfromtxt(dir_addr+dataset+"/"+dataset+"_TEST.tsv", delimiter="\t")
X = np.concatenate((X1, X2), axis=0)
true_labels = X[:,0]
X = X[:,1:]
if dataset in ["Crop", "ElectricDevices", "StarLightCurves"]:
    X, index = np.unique(X, axis=0, return_index=True)
    true_labels = true_labels[index]


start_time = time.time()
import pdb; pdb.set_trace()
connectivity = kneighbors_graph(
                X, n_neighbors=k, include_self=True
)
affinity = 0.5 * (connectivity + connectivity.T)
end_time = time.time()
print(end_time - start_time)
maps = spectral_embedding(
    affinity,
    n_components=n_components,
    eigen_solver=None,
    random_state=None,
#     eigen_tol='auto',
    drop_first=False,
)
end_time = time.time()
print(end_time - start_time)