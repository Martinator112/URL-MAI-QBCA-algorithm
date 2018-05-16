import numpy as np
import math as math

def compute_distance(A,B):
  distance = (np.array(A) - np.array(B)) ** 2
  distance = math.sqrt(distance.sum())
  return distance

def compute_cluster_diameter(cluster):
  max_distance_seen = -1
  for i in range(0, len(cluster)):
    for j in range(i + 1, len(cluster)):
      A = cluster[i]
      B = cluster[j]
      distance = compute_distance(A, B)
      if distance > max_distance_seen:
        max_distance_seen = distance
  
  return max_distance_seen

def compute_cluster_dissimilarity(cluster_one, cluster_two):
  min_distance_seen = math.inf
  for i in range(0, len(cluster_one)):
    for j in range(0, len(cluster_two)):
      A = cluster_one[i]
      B = cluster_two[j]
      distance = compute_distance(A, B)
      if distance < min_distance_seen:
        min_distance_seen = distance

  return min_distance_seen

def compute_delta_k(clusters):
  diameters_of_clusters = np.zeros(len(clusters.items()))
  for cluster_index, cluster in clusters.items():
    diameters_of_clusters[cluster_index] = compute_cluster_diameter(cluster)

  max_diameter = diameters_of_clusters.max()

  k = len(clusters)
  cluster_dissimilarities = np.repeat(np.nan, k * k).reshape(k, k)
  dissimilarities = []

  min_ratio_seen = math.inf

  for i in range(0, k):
    for j in range(i + 1, k):
      dissimilarity = compute_cluster_dissimilarity(clusters[i], clusters[j])
      cluster_dissimilarities[i,j] = dissimilarity
      dissimilarities += [dissimilarity]

      ratio = dissimilarity / max_diameter
      if ratio < min_ratio_seen:
        min_ratio_seen = ratio

  return min_ratio_seen
