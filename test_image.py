from PIL import Image
import numpy as np 
from skimage import io, color
from scipy.ndimage import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt
from qbca import QBCA
from time import time

from sklearn.cluster import KMeans

def get_image(image_path):
  """Get a numpy array of an image so that one can access values[x][y]."""
  image = Image.open(image_path, 'r')
  width, height = image.size
  pixel_values = list(image.getdata())

  print(f"Image mode is {image.mode}")
  if image.mode == 'RGB':
      channels = 3
  elif image.mode == 'L':
      channels = 1
  else:
      print("Unknown mode: %s" % image.mode)
      return None
  pixel_values = np.array(pixel_values).reshape((width, height, channels))
  return pixel_values

def get_image_lab(image_path):
  rgb = io.imread(image_path)
  lab = color.rgb2lab(rgb)

  return rgb, lab


def make_experiment(k):
  original_image_rgb, image = get_image_lab('./test_image.jpg')

  flattened_original_image_rgb = original_image_rgb.reshape(np.array(original_image_rgb.shape).prod())
  flattened_original_image_rgb_kmeans = original_image_rgb.reshape(np.array(original_image_rgb.shape).prod())
  print(original_image_rgb.shape)

  result = gaussian_filter(image, sigma=0.5)
  rows, cols, channels = result.shape
  reshaped = result.reshape(rows * cols, channels)

  t0 = time()

  qbca = QBCA(n_clusters=k, loc = 0.0001)
  qbca = qbca.fit(reshaped)

  clusters = qbca.clusters
  labels = qbca.labels_

  t1 = time()

  kmeans = KMeans(n_clusters=k, random_state=0).fit(reshaped)
  kmeans_labels = kmeans.labels_

  t2 = time()

  print(f"Algorithm took {t1-t0} s.")

  colors = [
    [150.0,150.0,150.0],
    [50.0,50.0,50.0],
    [100.0,100.0,100.0],
    [200.0,200.0,200.0],
    [100.0,0.0,0.0],
  ]

  red_color = [255.0, 0.0, 0.0]

  segmented_image = [colors[int(labels[0])]]
  last_label = labels[0]

  for idx, label in enumerate(labels[1:]):
    segmented_image += [colors[int(label)]]
    if label != last_label:
      flattened_original_image_rgb[(idx*3): ((idx*3) + 3)] = red_color
    last_label = label


  segmented_image_kmeans = [colors[int(kmeans_labels[0])]]
  last_label = kmeans_labels[0]
  for label in kmeans_labels[1:]:
    segmented_image_kmeans += [colors[int(label)]]
    if label != last_label:
      flattened_original_image_rgb_kmeans[(idx*3): ((idx*3) + 3)] = red_color
    last_label = label

  segmented_image = np.array(segmented_image).reshape((rows, cols, channels))
  segmented_image_kmeans = np.array(segmented_image_kmeans).reshape((rows, cols, channels))
  flattened_original_image_rgb = np.array(flattened_original_image_rgb).reshape((rows, cols, channels))
  flattened_original_image_rgb_kmeans = np.array(flattened_original_image_rgb_kmeans).reshape((rows, cols, channels))
  fig, ax = plt.subplots(nrows=2, ncols=2)

  ax1, ax2 = ax[0]
  ax3, ax4 = ax[1]

  ax1.imshow(segmented_image)
  ax1.set_title('Segmented image QBCA')
  ax1.get_xaxis().set_visible(False)
  ax1.get_yaxis().set_visible(False)


  ax2.imshow(segmented_image_kmeans)
  ax2.set_title('Segmented image KMeans')
  ax2.get_xaxis().set_visible(False)
  ax2.get_yaxis().set_visible(False)

  ax3.imshow(flattened_original_image_rgb)
  ax3.set_title('Original image with red dots between different clusters - QBCA')
  ax3.get_xaxis().set_visible(False)
  ax3.get_yaxis().set_visible(False)

  ax4.imshow(flattened_original_image_rgb_kmeans)
  ax4.set_title('Original image with red dots between different clusters - K-means')
  ax4.get_xaxis().set_visible(False)
  ax4.get_yaxis().set_visible(False)

  plt.show()
  # plt.savefig(f"Figure_image_clustering_k_{k}.png", dpi=300)



make_experiment(5)

