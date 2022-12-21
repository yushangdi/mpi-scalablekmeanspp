X = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12],
          [3, 14, 15]]
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph
# SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
#                                     n_neighbors=n_neighbor,
#                                     assign_labels='discretize',
#                                     random_state=1, n_jobs=worker).fit(X)

connectivity = kneighbors_graph(
                X, n_neighbors=2, include_self=True
)
affinity = 0.5 * (connectivity + connectivity.T)
import pdb; pdb.set_trace()
maps = spectral_embedding(
    affinity,
    n_components=3,
    eigen_solver=None,
    random_state=None,
#     eigen_tol='auto',
    drop_first=False,
)