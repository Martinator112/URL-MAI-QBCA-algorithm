import numpy as np
import math as math
import matplotlib.pyplot as plt
from max_heap import MaxHeap
from time import time

def compute_min_distance(histogram_bin_mins, histogram_bin_maxs, point):
  is_bin_minimum_greater = histogram_bin_mins > np.array(point)
  is_bin_maximum_lower = histogram_bin_maxs < np.array(point)

  distance = 0
  for i in range(0, len(point)):
    if is_bin_maximum_lower[i]:
      # point is lower than the minimum of the bin in given dimension
      dimension_distance = point[i] - histogram_bin_maxs[i]
    elif is_bin_minimum_greater[i]:
      # point is greater than the maximum of the bin in given dimension
      dimension_distance = point[i] - histogram_bin_mins[i]
    else:
      # point is inside bin in given dimension
      dimension_distance = 0
    dimension_distance = math.pow(dimension_distance, 2)
    distance += dimension_distance

  distance = math.sqrt(distance)
  return distance

def compute_max_distance(histogram_bin_mins, histogram_bin_maxs, point):
  bin_averages = histogram_bin_mins + histogram_bin_maxs
  bin_averages /= 2

  is_point_greater_than_average = np.array(point) > bin_averages

  distance = 0
  for i in range(0, len(point)):
    if is_point_greater_than_average[i]:
      # point is greater than the average of maximum and minimum of the bin in given dimension
      dimension_distance = point[i] - histogram_bin_mins[i]
    else:
      # point is lower than the average of maximum and minimum of the bin in given dimension
      dimension_distance = point[i] - histogram_bin_maxs[i]
    dimension_distance = math.pow(dimension_distance, 2)
    distance += dimension_distance
  distance = math.sqrt(distance)
  return distance


def compute_lowest_max_distance_for_bin(histogram_bin_mins, histogram_bin_maxs, seed_list):
  distances = np.zeros(len(seed_list))

  for i in range(0, len(seed_list)):
    seed = seed_list[i]
    distances[i] = compute_max_distance(histogram_bin_mins, histogram_bin_maxs, seed)

  return distances.min()
  
def compute_distance(A,B):
  distance = (np.array(A) - np.array(B)) ** 2
  distance = math.sqrt(distance.sum())
  return distance

def get_cluster_center_idx(point, candidate_seed_list):
  if len(candidate_seed_list) == 1:
    return candidate_seed_list[0][0]

  minimum_distance_seen = -1
  for seed_index, seed in candidate_seed_list:
    distance_from_seed = compute_distance(point, seed)
    if distance_from_seed < minimum_distance_seen or minimum_distance_seen == -1:
      cluster_center_index = seed_index
      minimum_distance_seen = distance_from_seed

  return cluster_center_index

def assign_cluster_center(histogram_bins, seed_list, num_of_dimensions, max_points):
  t0 = time()
  bin_candidate_seed_list = {}
  clusters = {}
  labels = np.empty(max_points)

  for i in range(0, len(seed_list)):
    clusters[i] = []

  for bin_index in histogram_bins['values']:

    # skip empty bins
    if histogram_bins['counts'][bin_index] == 0:
      continue

    histogram_bin = histogram_bins['values'][bin_index]
    histogram_bin_mins = histogram_bins['mins'][bin_index]
    histogram_bin_maxs = histogram_bins['maxs'][bin_index]

    points_idx = histogram_bins['ids'][bin_index]

    bin_candidate_seed_list[bin_index] = []
    lowest_max_distance_for_bin = compute_lowest_max_distance_for_bin(histogram_bin_mins, histogram_bin_maxs, seed_list)

    for seed_index, seed in enumerate(seed_list):
      min_distance_for_seed = compute_min_distance(histogram_bin_mins, histogram_bin_maxs, seed)
      if min_distance_for_seed <= lowest_max_distance_for_bin:
        bin_candidate_seed_list[bin_index] += [(seed_index, seed)]

    for point_index_in_histogram, point in enumerate(histogram_bin):
      cluster_center_index = get_cluster_center_idx(point, bin_candidate_seed_list[bin_index])
      clusters[cluster_center_index].append(point)
      labels[points_idx[point_index_in_histogram]] = cluster_center_index

  t1 = time()
  k = len(seed_list)
  cluster_centers = np.repeat(np.zeros(num_of_dimensions), len(seed_list)).reshape(k, num_of_dimensions)

  for cluster_center_index, cluster_points in clusters.items():
    recomputed_cluster_center = np.mean(cluster_points, axis=0)
    cluster_centers[cluster_center_index] = recomputed_cluster_center

  return clusters, cluster_centers, labels


def compute_alfa(prev_cluster_centers, new_cluster_centers):
  total_change = 0
  k = len(prev_cluster_centers)

  change = ((prev_cluster_centers - new_cluster_centers) ** 2).sum()
  total_change = change

  return total_change / k



def reverse_vector(vector):
  return vector[::-1]

def compute_bin_id(point, num_of_dimensions, minimum_values, lamda_val, num_of_histograms):
  bin_id = 0

  m = num_of_dimensions - 1
  eta_m = 1 if point[m] == minimum_values[m] else math.ceil((point[m] - minimum_values[m]) / lamda_val[m])
  
  for l in range(0, num_of_dimensions - 1):
    is_minimum = point[l] == minimum_values[l]
    if is_minimum:
      eta_l = 1
    else:
      eta_l = math.ceil((point[l] - minimum_values[l]) / lamda_val[l])
    bin_id += (eta_l - 1) * (num_of_histograms ** (num_of_dimensions - l - 1))  
  return bin_id + eta_m

def perform_quantization(points):
  points = np.array(points)
  bin_ids = []
  max_points = 0

  rows, cols = points.shape
  num_of_histograms = math.floor(math.log(rows, cols))

  feature_maximums = points.max(axis=0)
  feature_minimums = points.min(axis=0)
  lamda_val = (feature_maximums - feature_minimums) / num_of_histograms

  histogram_bins = {
    'values': {},
    'maxs': {},
    'mins': {},
    'counts': {},
    'ids': {}
  }

  for i in range(1, (num_of_histograms ** cols) + 1):
    histogram_bins['counts'][i] = 0
    histogram_bins['values'][i] = []
    histogram_bins['ids'][i] = []

  for point_idx, point in enumerate(points):
    bin_id = compute_bin_id(point, cols, feature_minimums, lamda_val, num_of_histograms)
    # print("Point ", point, " has bin_id: ", bin_id)
    bin_ids += [bin_id]
    histogram_bins['values'][bin_id] += [point]
    histogram_bins['counts'][bin_id] += 1
    histogram_bins['ids'][bin_id] += [point_idx]
    max_points += 1

  for bin_id, bin_values in histogram_bins['values'].items():
    if histogram_bins['counts'][bin_id] > 0 :
      histogram_bins['mins'][bin_id] = np.array(bin_values).min(axis=0)
      histogram_bins['maxs'][bin_id] = np.array(bin_values).max(axis=0)

  number_of_not_empty_bins = 0
  for histogram_count in histogram_bins['counts'].values():
    if histogram_count > 0:
      number_of_not_empty_bins += 1

  return (
    histogram_bins,
    number_of_not_empty_bins,
    max_points
  )

def initialize_cluster_centers(histogram_bins, num_of_bins, k):
  max_heap = MaxHeap()
  for bin_id in range(1, (num_of_bins) + 1):
    if histogram_bins['counts'][bin_id]:
      max_heap.heappush((bin_id, histogram_bins['values'][bin_id]), histogram_bins['counts'][bin_id])

  seed_list = []
  alternative_max_heap = MaxHeap()

  while not max_heap.is_empty():
    bin = max_heap.heappop()
    bin_id, bin_points = bin.val

    should_be_added_to_seed_list = True

    # check if neigbouring bins don't have greater cardinality
    left_neighbour_idx = bin_id - 1
    right_neighbour_idx = bin_id + 1
    if left_neighbour_idx in histogram_bins['counts'] and histogram_bins['counts'][left_neighbour_idx] > bin.priority:
      should_be_added_to_seed_list = False
    if right_neighbour_idx in histogram_bins['counts'] and histogram_bins['counts'][right_neighbour_idx] > bin.priority:
      should_be_added_to_seed_list = False

    if should_be_added_to_seed_list:
      seed_list += [bin]
    else:
      alternative_max_heap.heappush(bin.val, bin.priority)

  while k > len(seed_list):
    bin = alternative_max_heap.heappop()
    seed_list += [bin]

  seed_centers = []  

  for i in range(0, k):
    bin = seed_list[i]
    bin_points = bin.val[1]
    seed_center = np.mean(bin_points, axis=0)
    seed_centers += [seed_center]

  return seed_centers

class QBCA(object):
  def __init__(self, n_clusters, loc = 0.000001): 
    self.epsilon = loc
    self.k = n_clusters
    self.labels_ = []
    self.clusters = {}

  def fit(self, data):
    rows, cols = data.shape
    k = self.k
    epsilon = self.epsilon

    num_of_histograms = math.floor(math.log(rows, cols))
    num_of_bins = num_of_histograms ** cols

    t0 = time()
    histogram_bins, number_of_not_empty_bins, max_points = perform_quantization(data)
    t1 = time()

    max_k = number_of_not_empty_bins + 1
    min_k = 2

    if k > max_k:
      print(f"Invalid number of clusters. Max number of clusters is {max_k}. Changing k to {max_k}.")
      k = max_k
    elif k < min_k :
      print(f"Invalid number of clusters. Min number of clusters is {min_k}. Changing k to {min_k}.")

    seed_centers = initialize_cluster_centers(histogram_bins, num_of_bins, k)
    t2 = time()
    cluster_centers = assign_cluster_center(histogram_bins, seed_centers, cols, max_points)[1]
    t3 = time()

    alfa = 2 * epsilon
    iteration = 0
    prev_cluster_centers = cluster_centers
    seed_centers = cluster_centers

    while alfa > epsilon:
      iteration += 1
      clusters, cluster_centers, labels = assign_cluster_center(histogram_bins, seed_centers, cols, max_points)
      alfa = compute_alfa(prev_cluster_centers, cluster_centers)
      prev_cluster_centers = cluster_centers
      seed_centers = cluster_centers

    self.clusters = clusters
    self.labels_ = labels

    return self
  
  def fit_predict(self, data):
    self.fit(data)

    return self.labels_