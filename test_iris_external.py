from qbca import QBCA
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kemlglearn.metrics import  scatter_matrices_scores, calinski_harabasz_score, davies_bouldin_score
from kemlglearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
data = iris['data']
blabels = iris['target']

def get_algorithms(k):
  qbca = QBCA(n_clusters=k, loc=0.00000001)
  kmeans = KMeans(n_clusters=k)
  gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=0)

  return [
    ('1_qbca', qbca),
    ('2_kmeans', kmeans),
    ('3_gmm', gmm)
  ]

def test_qbca_external(k):
  lscores = {
    '1_qbca': [],
    '2_kmeans': [],
    '3_gmm': [],
  }

  for name, algorithm in get_algorithms(k):
    if hasattr(algorithm, 'fit_predict'):
      labels = algorithm.fit_predict(data)
    else:
      algorithm.fit(data)
      labels = algorithm.predict(data)

    lscores[name] = [
      adjusted_mutual_info_score(blabels, labels),
      adjusted_rand_score(blabels, labels),
      normalized_mutual_info_score(blabels, labels)
    ]

  results_adj_mutual = ""
  results_rand_score = ""
  results_norm_mutual = ""

  for name in sorted(lscores):
    print(name)
    results_adj_mutual += f"{lscores[name][0]},"
    results_rand_score += f"{lscores[name][1]},"
    results_norm_mutual += f"{lscores[name][2]},"

  results_adj_mutual += "\n"
  results_rand_score += "\n"
  results_norm_mutual += "\n"


  return results_adj_mutual, results_rand_score, results_norm_mutual
  # plt.show()


def write_external_index_results(results_adj_mutual, results_rand_score, results_norm_mutual):
  filename_adj_mutual = 'iris_external_adjusted_mutual_info_index_results.csv'
  filename_rand_score = 'iris_external_rand_score_index_results.csv'
  filename_norm_mutual = 'iris_external_normalized_mutual_info_index_results.csv'

  fh = open(filename_adj_mutual, 'a')
  fh2 = open(filename_rand_score, 'a')
  fh3 = open(filename_norm_mutual, 'a')

  fh.write(results_adj_mutual)
  fh2.write(results_rand_score)
  fh3.write(results_norm_mutual)

  fh.close()
  fh2.close()
  fh3.close()

heading = "qbca,kmeans,gmm,padding\n"

results =[
  heading,
  heading,
  heading
]

res_adj_mutual, res_rand_score, res_norm_mutual = test_qbca_external(3)
results[0] += res_adj_mutual
results[1] += res_rand_score
results[2] += res_norm_mutual

write_external_index_results(results[0], results[1], results[2])

