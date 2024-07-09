import numpy as np
import matplotlib.pyplot as plt

# Definir el problema: implementar la función objetivo
def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 1

# Inicializar la población
def initialize_population(pop_size, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, 1))

# Evaluar aptitud
def evaluate_fitness(population):
    return objective_function(population)

# Selección
def select_parents(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
        parents[i, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -999999  # Para no volver a seleccionarlo
    return parents

# Crossover
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

# Mutación
def mutate(offspring, mutation_rate, bounds):
    for idx in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            random_value = np.random.uniform(bounds[0], bounds[1])
            offspring[idx] = random_value
    return offspring

# Configuración del algoritmo genético
pop_size = 50
num_generations = 100
num_parents_mating = 10
bounds = [0, 1]
mutation_rate = 0.1

# Inicializar la población
population = initialize_population(pop_size, bounds)
best_outputs = []
print("población Inicial")
text_population = ""
for pop in population:
    text_population += f"{pop}, "
print(text_population)
# Iteración a través de generaciones
for generation in range(num_generations):
    fitness = evaluate_fitness(population)
    best_outputs.append(np.max(fitness))
    
    parents = select_parents(population, fitness.copy(), num_parents_mating)
    offspring_crossover = crossover(parents, (pop_size - parents.shape[0], 1))
    offspring_mutation = mutate(offspring_crossover, mutation_rate, bounds)
    
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = offspring_mutation

# Mejor solución encontrada
best_solution = population[np.argmax(evaluate_fitness(population))]
best_fitness = np.max(evaluate_fitness(population))

# Visualización de la evolución de la mejor aptitud
plt.plot(best_outputs)
plt.xlabel('Generaciones')
plt.ylabel('Mejor aptitud')
plt.title('Evolución de la mejor aptitud')
plt.show()

print(f"Mejor solución: x = {best_solution[0]}, f(x) = {best_fitness}")
