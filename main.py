import numpy as np
import time
from gen import GeneticAlgorithm
import matplotlib.pyplot as plt

def random_dkp(n, scale = 2000, seed=0):
  np.random.seed(seed)
  items = np.ceil(scale * np.random.rand(n, 2)).astype("int32")
  C = int(np.ceil(0.5 * 0.5 * n * scale))
  v = items[:, 0]
  c = items[:, 1]
  return v, c, C  

if __name__ == '__main__':
  n = 100
  v, c, C = random_dkp(n = n, scale = 2000)

  ga = GeneticAlgorithm(n, v, c, C , m=1000, T=100, crossover_propability=0.9, mutation_propability=0.01, seed=0)
  solution, best_value, best_weight, avg_fitness, avg_weight = ga.run()

  print(f"BEST PACK VALUE: {best_value}")
  print(f"BEST PACK WEIGHT: {best_weight}")
  print(f"SOLUTION: {solution}")

  x = np.arange(0, len(avg_fitness), 1)
  fig, axs = plt.subplots(2)
  fig.suptitle('Top - fitness, Bottom - weight')
  axs[0].plot(x, avg_fitness)
  axs[1].plot(x, avg_weight)
  plt.show()