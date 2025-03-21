import networkx as nx
import numpy as np
from scipy.linalg import eigh

def spectral_partitioning(graph, k):
    laplacian_matrix = nx.laplacian_matrix(graph).todense()

    # Compute the smallest k eigenvectors of the Laplacian matrix
    _, eigenvectors = eigh(laplacian_matrix, eigvals=(0, k-1))

    # Apply k-means clustering to the eigenvectors
    _, partition = kmeans(eigenvectors, k)

    return partition

def kmeans(data, k, max_iterations=100):
    # Randomly initialize cluster centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of assigned points
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Example usage
graph = nx.complete_graph(10)  # Replace with your graph
k = 2  # Replace with the desired number of partitions
result = spectral_partitioning(graph, k)
print("Final Partition:", result)
