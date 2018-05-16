from qbca import QBCA
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from kemlglearn.metrics import  scatter_matrices_scores, calinski_harabasz_score, davies_bouldin_score
from kemlglearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

def test_qbca(min_k, max_k):
  data = make_blobs(
    n_samples=200,
    n_features=3,
    centers=[[0.2,0.2,0.2],[0,0,0],[-0.2,-0.2,-0.2]],
    cluster_std=[0.2,0.1,0.3]
  )[0]

  lscores_qbca = []
  lscores_kmeans = []
  lscores_gmm = []

  for k in range(min_k, max_k):
    qbca = QBCA(n_clusters=k)
    labels_qbca = qbca.fit_predict(data)

    lscores_qbca.append((
      silhouette_score(data, labels_qbca),
      calinski_harabasz_score(data, labels_qbca),
      davies_bouldin_score(data, labels_qbca)))

    kmeans = KMeans(n_clusters=k)
    labels_kmeans = kmeans.fit_predict(data)

    lscores_kmeans.append((
      silhouette_score(data, labels_kmeans),
      calinski_harabasz_score(data, labels_kmeans),
      davies_bouldin_score(data, labels_kmeans)))

    gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=0)
    gmm.fit(data)
    labels_gmm = gmm.predict(data)

    lscores_gmm.append((
      silhouette_score(data, labels_gmm),
      calinski_harabasz_score(data, labels_gmm),
      davies_bouldin_score(data, labels_gmm)))



  ax = plt.subplots(nrows=3, ncols=3)[1]

  plt.subplots_adjust(hspace=1)

  ax1, ax2, ax3 = ax[0]
  ax4, ax5, ax6 = ax[1]
  ax7, ax8, ax9 = ax[2]

  ax1.set_title('QBCA Silhouette score')
  ax2.set_title('QBCA Caliniski Harabasz score')
  ax3.set_title('QBCA Davies Boulding score')

  ax4.set_title('KMeans Silhouette score')
  ax5.set_title('KMeans Caliniski Harabasz score')
  ax6.set_title('KMeans Davies Boulding score')

  ax7.set_title('GMM Silhouette score')
  ax8.set_title('GMM Caliniski Harabasz score')
  ax9.set_title('GMM Davies Boulding score')

  ax1.plot(range(min_k,max_k), [x for x,_,_ in lscores_qbca])
  ax2.plot(range(min_k,max_k), [x for _,x,_ in lscores_qbca])
  ax3.plot(range(min_k,max_k), [x for _,_,x in lscores_qbca])

  ax4.plot(range(min_k,max_k), [x for x,_,_ in lscores_kmeans])
  ax5.plot(range(min_k,max_k), [x for _,x,_ in lscores_kmeans])
  ax6.plot(range(min_k,max_k), [x for _,_,x in lscores_kmeans])

  ax7.plot(range(min_k,max_k), [x for x,_,_ in lscores_gmm])
  ax8.plot(range(min_k,max_k), [x for _,x,_ in lscores_gmm])
  ax9.plot(range(min_k,max_k), [x for _,_,x in lscores_gmm])

  plt.show()

test_qbca(2,6)
