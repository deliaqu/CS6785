import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

means = [
    np.array([-2, 0]), 
    np.array([1, 3]), 
    np.array([2, -2])
]
covariance = np.eye(2) 

samples_per_cluster = 1000
clusters = []
for mu in means:
    samples = np.random.multivariate_normal(mu, covariance, samples_per_cluster)
    clusters.append(samples)

all_samples = np.vstack(clusters)

colors = ['red', 'green', 'blue']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

plt.figure(figsize=(8, 6))
for i, (samples, color, label) in enumerate(zip(clusters, colors, labels)):
    plt.scatter(samples[:, 0], samples[:, 1], c=color, label=label, alpha=0.6)

plt.title('Scatter Plot of Samples from Gaussians')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()