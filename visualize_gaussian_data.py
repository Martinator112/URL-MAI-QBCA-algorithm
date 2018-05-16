from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from kemlglearn.datasets import make_blobs
  
data, blabels = make_blobs(
  n_samples=200, 
  n_features=3,
  centers=[[0.2,0.2,0.2],[0,0,0],[-0.2,-0.2,-0.2]],
  cluster_std=[0.2,0.1,0.3]
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:,0], data[:,1], data[:,2], c=blabels)

plt.show()