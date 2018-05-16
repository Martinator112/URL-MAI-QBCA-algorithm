import matplotlib.pyplot as plt
import numpy as np

cols = np.arange(0, 4)

results_adj_mutual = np.genfromtxt('./results/gaussian_external_adjusted_mutual_info_index_results.csv', delimiter=',', skip_header=1, usecols=cols)
results_normalized_mutual = np.genfromtxt('./results/gaussian_external_normalized_mutual_info_index_results.csv', delimiter=',', skip_header=1, usecols=cols)
results_rand_score = np.genfromtxt('./results/gaussian_external_rand_score_index_results.csv', delimiter=',', skip_header=1, usecols=cols)

ax = plt.subplots(nrows=3, ncols=1)[1]

ax1, ax2, ax3 = ax

ax1.set_title('AMI score')
ax2.set_title('NMI score')
ax3.set_title('RI score')

plt.subplots_adjust(hspace=1)

print(results_adj_mutual)

ax1.boxplot(results_adj_mutual[:,:3], labels=['qbca', 'kmeans', 'gmm'])
ax2.boxplot(results_normalized_mutual[:,:3], labels=['qbca', 'kmeans', 'gmm'])
ax3.boxplot(results_rand_score[:,:3], labels=['qbca', 'kmeans', 'gmm'])

heading="qbca,kmeans,gmm,dbscan"
results_ami = f"{results_adj_mutual[:,0].mean()},{results_adj_mutual[:,1].mean()},{results_adj_mutual[:,2].mean()},{results_adj_mutual[:,3].mean()}"
results_nmi = f"{results_normalized_mutual[:,0].mean()},{results_normalized_mutual[:,1].mean()},{results_normalized_mutual[:,2].mean()},{results_normalized_mutual[:,3].mean()}"
results_ri = f"{results_rand_score[:,0].mean()},{results_rand_score[:,1].mean()},{results_rand_score[:,2].mean()},{results_rand_score[:,3].mean()}"

print(results_adj_mutual)
filename = './results/gaussian_external_means.csv'

result = f"{heading}\n{results_ami}\n{results_nmi}\n{results_ri}\n"
fh = open(filename, 'w')

fh.write(result)

fh.close()

plt.show()