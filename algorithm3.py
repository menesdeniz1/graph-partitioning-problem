import networkx as nx
import random

def genetic_algorithm(graph, k, population_size=50, generations=100):
    # Generate an initial population of random partitions
    population = [random_partition(graph, k) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness of each partition in the population
        fitness_scores = [fitness(graph, partition) for partition in population]

        # Select the top-performing partitions for reproduction
        selected_indices = select_top_indices(fitness_scores, int(0.2 * population_size))
        selected_population = [population[i] for i in selected_indices]

        # Reproduce to create a new population
        new_population = reproduce(selected_population, population_size)

        # Mutate the new population to introduce genetic diversity
        new_population = [mutate(partition) for partition in new_population]

        # Replace the old population with the new one
        population = new_population

    # Return the best partition found in the final generation
    best_partition = max(population, key=lambda p: fitness(graph, p))
    return best_partition

def random_partition(graph, k):
    # Generate a random partition of the graph into k subsets
    vertices = list(graph.nodes())
    random.shuffle(vertices)
    partition = {v: i % k for i, v in enumerate(vertices)}
    return partition

def fitness(graph, partition):
    # Calculate the fitness of a partition (minimize cut size)
    cut_size = 0
    for edge in graph.edges():
        if partition[edge[0]] != partition[edge[1]]:
            cut_size += 1
    return -cut_size  # Negative cut size for maximization

def select_top_indices(scores, top_percentage):
    # Select the top-performing individuals based on fitness scores
    num_top = max(1, int(top_percentage * len(scores)))
    return sorted(range(len(scores)), key=lambda i: scores[i])[-num_top:]

def reproduce(population, target_size):
    # Reproduce the population by crossover
    new_population = []

    while len(new_population) < target_size:
        parent1, parent2 = random.sample(population, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        new_population.append(child)

    return new_population

def mutate(partition, mutation_rate=0.1):
    # Introduce random mutations to the partition
    mutated_partition = partition.copy()

    for vertex in mutated_partition:
        if random.random() < mutation_rate:
            mutated_partition[vertex] = random.choice(list(mutated_partition.values()))

    return mutated_partition

# Example usage
graph = nx.complete_graph(10)  # Replace with your graph
k = 2  # Replace with the desired number of partitions
result = genetic_algorithm(graph, k)
print("Final Partition:", result)
print("Cut Size:", -fitness(graph, result))
