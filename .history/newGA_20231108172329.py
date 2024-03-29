import random
import numpy as np

# Assume we have the number of machines and instances from the data
num_machines = 4  # As mentioned in the problem statement
num_jobs = 20     # As mentioned in the problem statement

# Group the data by instance to get processing times for each instance
grouped_data = data.groupby('instance')['Processing Time'].apply(list)

# Define a function to initialize a population of chromosomes
def init_population(population_size, num_jobs):
    population = []
    for _ in range(population_size):
        # Generate a random sequence of job numbers for each chromosome
        chromosome = np.random.permutation(num_jobs).tolist()
        population.append(chromosome)
    return population

# Define a function to calculate the makespan
def calculate_makespan(chromosome, processing_times, setup_times, num_machines):
    # Initialize machine times
    machine_times = [0] * num_machines
    last_family = [-1] * num_machines  # To track the last family processed by each machine
    
    for job in chromosome:
        # Find the machine with the minimum load
        machine_index = machine_times.index(min(machine_times))
        # Check if setup time is needed
        setup_time = 0 if last_family[machine_index] == setup_times[job][1] else setup_times[job][0]
        # Update machine time with the processing time and setup time
        machine_times[machine_index] += processing_times[job] + setup_time
        # Update the last family processed on this machine
        last_family[machine_index] = setup_times[job][1]

    # The makespan is the maximum time among all machines
    makespan = max(machine_times)
    return makespan

# Define a function to calculate the fitness (utilization) of a chromosome
def fitness(chromosome, processing_times, setup_times, num_machines):
    total_processing_time = sum(processing_times)
    makespan = calculate_makespan(chromosome, processing_times, setup_times, num_machines)
    utilization = total_processing_time / (num_machines * makespan)
    return utilization

# For the setup times, we'll assume that the setup time is the same for all jobs within a family
# and that this information is given in a separate structure (e.g., a dictionary where the key is the family ID)
# For the sake of this example, let's mock this data
# (In a real scenario, this should be extracted from your Excel data)
mock_setup_times = {1: (984, 1), 2: (984, 2), 3: (984, 3), 4: (984, 4)}

# Testing the initialization and fitness function with a mock chromosome for a single instance
# Here we'll use 'in1' instance as an example
instance_processing_times = grouped_data['in1']
instance_setup_times = [mock_setup_times[family] for family in data[data['instance'] == 'in1']['family']]

# Generate an initial population
population_size = 10  # For illustration, we'll use a small population size
initial_population = init_population(population_size, num_jobs)

# Calculate fitness for the first chromosome in the population
example_chromosome = initial_population[0]
chromosome_fitness = fitness(example_chromosome, instance_processing_times, instance_setup_times, num_machines)

# initial_population, chromosome_fitness


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

# new_population, new_population_fitness
