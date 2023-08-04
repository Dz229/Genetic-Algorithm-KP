import numpy as np
import random

class GeneticAlgorithm:
  def __init__(self, n, v, c, C, m = 1000, T = 100, crossover_propability = 0.9, mutation_propability = 0.001, seed = 0):
    self.n = n #n - problem size
    self.v = v #v - value
    self.c = c #c - cost
    self.C = C #C - capacity
    self.m = m #m - population size
    self.T = T #T - iterations
    self.crossover_propability = crossover_propability
    self.mutation_propability = mutation_propability
    self.seed = seed

    self.population = []

  #Population generation
  def generate_initial_population(self, population_size):
    for i in range (population_size):
      individual = [random.choice([0,1]) for _ in self.v]
      self.population.append(individual)
    
  #Fitness function
  def fitness(self, individual):
    total_value = 0;
    total_cost = 0
    for i in range(self.n):
      total_value += individual[i] * self.v[i]
      total_cost += individual[i] * self.c[i]
    #Drastyczna kara 
    if(total_cost > self.C):
      return 0
    #Miękka kara
    # if(total_cost > self.C):
    #   total_value -= 10000
    return total_value

  #Fitness + weight
  def fitness_and_weight(self, individual):
    total_value = 0;
    total_cost = 0
    for i in range(self.n):
      total_value += individual[i] * self.v[i]
      total_cost += individual[i] * self.c[i]
    return total_value, total_cost

  #Roulette selection
  # def selection(self):
  #   max = sum([self.fitness(c) for c in self.population])
  #   selection_probs = [self.fitness(c)/max for c in self.population]
  #   parents = self.population[np.random.choice(self.m, 2, p=selection_probs)]
  #   return parents
    
  #Tournament selection between 4 individuals 
  def selection(self):
    parents = []
    random.shuffle(self.population)
    #First fight
    if (self.fitness(self.population[0]) > self.fitness(self.population[1])):
        parents.append(self.population[0])
    else:
        parents.append(self.population[1])
    #Second fight
    if (self.fitness(self.population[2]) > self.fitness(self.population[3])):
        parents.append(self.population[2])
    else:
        parents.append(self.population[3])
    #Returning winners
    return parents

  #One-point Crossover
  # def crossover(self, parents):
  #   first_point = random.randint(0, self.n)
  #   first_child = np.concatenate((np.array(parents[0][:first_point]), np.array(parents[1][first_point:])))
  #   second_child = np.concatenate((np.array(parents[1][:first_point]), np.array(parents[0][first_point:])))
  #   return [first_child, second_child]
    
  #Two-point Crossover
  def crossover(self, parents):
    first_point = random.randint(0, self.n)
    second_point = random.randint(0, self.n)
    #first_child = np.concatenate(parents[0][:first_point], parents[1][first_point:second_point], parents[0][second_point:])
    first_child = np.concatenate((np.array(parents[0][:first_point]), np.array(parents[1][first_point:second_point]), np.array(parents[0][second_point:])))
    #second_child = np.concatenate(parents[1][:first_point], parents[0][first_point:second_point], parents[1][second_point:])
    second_child = np.concatenate((np.array(parents[1][:first_point]), np.array(parents[0][first_point:second_point]), np.array(parents[1][second_point:])))
    return [first_child, second_child]

  #Mutation
  def mutation(self, individual):
    for i in range(self.n):
      if (random.random() < self.mutation_propability):
        if individual[i] == 0:
          individual[i] = 1
        else:
          individual[i] = 0
    return individual
    
  #Next generation
  def create_next_generation(self):
    next_generation = []
    while len(next_generation) < self.m:
      children = []
      parents = self.selection()
      if random.random() < self.crossover_propability:
        children = self.crossover(parents)
      else:
        children = parents
      if random.random() < self.mutation_propability:
        children[0] = self.mutation(children[0])
        children[1] = self.mutation(children[1])
      next_generation.extend(children)
    return next_generation[:self.m]

  def run(self):
    random.seed(self.seed)
    self.generate_initial_population(self.m)
    avg_fitness = []
    avg_weight = []
    for _ in range(self.T):
      fit = 0
      wei = 0
      for individual in self.population:
        one, two = self.fitness_and_weight(individual)
        fit += one
        wei += two
      fit = fit/self.m
      wei = wei/self.m
      avg_fitness.append(fit)
      avg_weight.append(wei)
      self.population = self.create_next_generation()

    best_result = self.population[0]
    for i in range(self.m):
      if(self.fitness(best_result) < self.fitness(self.population[i])):
        best_result = self.population[i]
    a, b = self.fitness_and_weight(best_result)
    return best_result, a, b, avg_fitness, avg_weight
    
  