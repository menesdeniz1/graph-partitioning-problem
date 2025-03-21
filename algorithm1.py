import networkx as nx
import itertools

def graph_partitioning_heuristic(graph, k):
    # Create an initial partition randomly
    initial_partition = {v: i % k for i, v in enumerate(graph.nodes())}

    # Perform local search to improve the partition
    final_partition = local_search(graph, initial_partition)

    return final_partition

def local_search(graph, initial_partition):
    current_partition = initial_partition.copy()
    best_partition = initial_partition.copy()
    best_cut_size = cut_size(graph, best_partition)

    # Maximum number of iterations for local search
    max_iterations = 1000
    iterations = 0

    while iterations < max_iterations:
        # Swap vertices between random partitions
        new_partition = swap_vertices(current_partition, graph)
        new_cut_size = cut_size(graph, new_partition)

        if new_cut_size < best_cut_size:
            # Update the best partition if the cut size is reduced
            best_partition = new_partition.copy()
            best_cut_size = new_cut_size

        current_partition = new_partition.copy()
        iterations += 1

    return best_partition

def swap_vertices(partition, graph):
    # Swap vertices between two random partitions
    vertices = list(graph.nodes())
    v1, v2 = random_pair(vertices)
    partition[v1], partition[v2] = partition[v2], partition[v1]
    return partition

def cut_size(graph, partition):
    # Calculate the number of edges between different partitions (cut size)
    cut_size = 0
    for edge in graph.edges():
        if partition[edge[0]] != partition[edge[1]]:
            cut_size += 1
    return cut_size

def random_pair(lst):
    # Return a random pair of elements from a list
    return tuple(random.sample(lst, 2))

# Example usage
graph = nx.complete_graph(10)  # Replace with your graph
k = 2  # Replace with the desired number of partitions
result = graph_partitioning_heuristic(graph, k)
print("Final Partition:", result)
print("Cut Size:", cut_size(graph, result))
