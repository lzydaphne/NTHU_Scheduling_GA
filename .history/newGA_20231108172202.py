import random

# Define tournament selection
def tournament_selection(population, fitness_func, tournament_size):
    selected = []
    for _ in range(tournament_size):
        # Randomly select tournament_size individuals from the population
        participants = random.sample(population, tournament_size)
        # Evaluate the fitness of the participants
        participants_fitness = [fitness_func(individual, instance_processing_times, instance_setup_times, num_machines) for individual in participants]
        # Select the best individual among the participants
        winner_index = np.argmax(participants_fitness)
        selected.append(participants[winner_index])
    return selected

# Define cycle crossover
def cycle_crossover(parent1, parent2):
    child = [-1] * len(parent1)  # Start with a child with no assigned jobs
    cycle_start = parent1[0]  # Start cycle with the first job of parent1
    index = 0
    while child[index] == -1:
        child[index] = parent1[index]  # Assign job from parent1 to the child
        index = parent2.index(child[index])  # Find the position of this job in parent2
    # Continue until we return to the cycle start position
    while parent1[index] != cycle_start:
        child[index] = parent1[index]  # Assign job from parent1 to the child
        index = parent2.index(child[index])  # Find the position of this job in parent2
    # Fill in remaining jobs from parent2
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

# Define swap mutation
def swap_mutation(chromosome):
    # Randomly select two indices to swap
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    # Swap the jobs at these indices
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# Define survivor selection (fitness-based selection)
def survivor_selection(population, fitness_scores, num_survivors):
    # Sort the population based on the fitness scores in descending order
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    # Select the top num_survivors to survive to the next generation
    survivors = sorted_population[:num_survivors]
    return survivors

# Now we can define a function to run one generation of the genetic algorithm
def run_generation(population, fitness_func, tournament_size, num_survivors):
    # Selection
    parents = tournament_selection(population, fitness_func, tournament_size)
    
    # Crossover
    children = []
    while len(children) < len(population):
        parent1, parent2 = random.sample(parents, 2)
        child = cycle_crossover(parent1, parent2)
        children.append(child)
    
    # Mutation
    for child in children:
        if random.random() < 0.1:  # Assuming a mutation probability of 10%
            child = swap_mutation(child)
    
    # Calculate fitness for the new population
    fitness_scores = [fitness_func(individual, instance_processing_times, instance_setup_times, num_machines) for individual in children]
    
    # Survivor Selection
    new_population = survivor_selection(children + population, fitness_scores + [fitness(ind, instance_processing_times, instance_setup_times, num_machines) for ind in population], num_survivors)
    
    return new_population

# Run one generation as a test
tournament_size = 3  # Size of tournament for selection
num_survivors = len(initial_population)  # The number of survivors to keep for the next generation

# Running a single generation
new_population = run_generation(initial_population, fitness, tournament_size, num_survivors)

# Calculate fitness for the new population to check the performance
new_population_fitness = [fitness(ind, instance_processing_times, instance_setup_times, num_machines) for ind in new_population]

new_population, new_population_fitness
