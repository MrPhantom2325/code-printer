"""Central answer store for the codeprinter package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

Answer = Union[str, int, float, dict[str, Any], list[Any], Callable[[], Any]]

ANSWERS: dict[int, Answer] = {
    1: ('''Question 1:
Implementation of Fuzzy Logic System

Answer 1:
import numpy as np

R = {
    "Low Temp": np.array([0.8, 0.5, 0.3]),
    "Medium Temp": np.array([0.6, 0.7, 0.4]),
    "High Temp": np.array([0.3, 0.6, 0.9])
}

S = {
    "Dry": np.array([0.7, 0.4, 0.3]),
    "Normal": np.array([0.5, 0.6, 0.4]),
    "Humid": np.array([0.2, 0.5, 0.8])
}

temperature_input = "Low Temp"
humidity_input = "Dry"

mu_R = R[temperature_input]
mu_S = S[humidity_input]

composed_result = np.minimum(mu_R, mu_S)

cooling_action = ["Low Cooling", "Medium Cooling", "High Cooling"]
action_index = np.argmax(composed_result)

print("Composed Membership:", composed_result)
print("Selected Action:", cooling_action[action_index])

'''),
    2: ('''Question 2:
Implementation of Defuzzification

Answer 2:

fuzzy_set = {1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0, 5: 1.0, 6: 0.7, 7: 0.3}

# 1. Lambda Cut
def lambda_cut(fs, lam):
    return [x for x, mu in fs.items() if mu >= lam]

# 2. Mean of Maximum (MOM)
def mom(fs):
    max_mu = max(fs.values())
    return sum(x for x, mu in fs.items() if mu == max_mu) / \
           len([1 for mu in fs.values() if mu == max_mu])

# 3. Center of Gravity (COG)
def cog(fs):
    num = sum(x * mu for x, mu in fs.items())
    den = sum(fs.values())
    return num / den if den != 0 else 0

print("Lambda Cut:", lambda_cut(fuzzy_set, 0.8))
print("MOM:", mom(fuzzy_set))
print("COG:", cog(fuzzy_set))
'''),
    3:('''
Question 3:
Implementation of Ant Colony Optimization

Answer 3:
import numpy as np
import random

dist = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

n = len(dist)

ants = 5
iterations = 50
alpha = 1
beta = 2
evap = 0.5

pheromone = np.ones((n, n))

def choose_next(curr, visited):
    probs = []
    for j in range(n):
        if j not in visited:
            tau = pheromone[curr][j] ** alpha
            eta = (1 / dist[curr][j]) ** beta
            probs.append(tau * eta)
        else:
            probs.append(0)

    probs = np.array(probs)
    probs = probs / probs.sum()
    return np.random.choice(range(n), p=probs)

def path_cost(path):
    cost = 0
    for i in range(len(path)-1):
        cost += dist[path[i]][path[i+1]]
    cost += dist[path[-1]][path[0]]
    return cost

best_path = None
best_cost = float('inf')

for _ in range(iterations):
    all_paths = []

    for _ in range(ants):
        start = random.randint(0, n-1)
        path = [start]

        while len(path) < n:
            next_city = choose_next(path[-1], path)
            path.append(next_city)

        cost = path_cost(path)
        all_paths.append((path, cost))

        if cost < best_cost:
            best_cost = cost
            best_path = path

    pheromone *= (1 - evap)

    for path, cost in all_paths:
        for i in range(len(path)-1):
            pheromone[path[i]][path[i+1]] += 1 / cost

print("Best Path:", best_path)
print("Best Cost:", best_cost)
    '''),
    4: ('''
Question 4:
Implementation of Particle Swarm Optimization
Answer 4:
import numpy as np

def f(x):
    return x**2   

particles = 5
iterations = 50
w = 0.5      
c1 = 1      
c2 = 1      
        
pos = np.random.uniform(-10, 10, particles)
vel = np.zeros(particles)

pbest = pos.copy()

gbest = pbest[np.argmin(f(pbest))]
        
for _ in range(iterations):
    for i in range(particles):
        r1, r2 = np.random.rand(), np.random.rand()
        vel[i] = (w * vel[i] +
                  c1 * r1 * (pbest[i] - pos[i]) +
                  c2 * r2 * (gbest - pos[i]))

        pos[i] += vel[i]
        if f(pos[i]) < f(pbest[i]):
            pbest[i] = pos[i]
        
    gbest = pbest[np.argmin(f(pbest))]


print("Best position:", gbest)
print("Best value:", f(gbest))
    '''),
    5:('''
Question 5:
Implementation of Genetic Algorithms
   
Answer 5:
import random
def fitness(x):
    return x**2  

population_size = 6
generations = 20
mutation_rate = 0.1

population = [random.randint(-10, 10) for _ in range(population_size)]

for _ in range(generations):
    population = sorted(population, key=fitness, reverse=True)
    
    parents = population[:2]
    
    new_population = parents.copy()
    
    while len(new_population) < population_size:
        child = (parents[0] + parents[1]) // 2
        
        if random.random() < mutation_rate:
            child += random.randint(-2, 2)
        
        new_population.append(child)
    
    population = new_population

best = max(population, key=fitness)

print("Best solution:", best)
print("Best fitness:", fitness(best))
    '''),
    6:('''
Question 6:
Implementation of Grey Wolf Optimizer

Answer 6:
import numpy as np

def f(x):
    return x**2   

wolves = 5
iterations = 50

pos = np.random.uniform(-10, 10, wolves)

for t in range(iterations):

    sorted_pos = sorted(pos, key=f)
    alpha, beta, delta = sorted_pos[:3]

    a = 2 - t * (2 / iterations) 

    new_pos = []
    for x in pos:
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha - x)
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta - x)
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta - x)
        X3 = delta - A3 * D_delta

        new_x = (X1 + X2 + X3) / 3
        new_pos.append(new_x)

    pos = np.array(new_pos)

best = min(pos, key=f)

print("Best position:", best)
print("Best value:", f(best))

    '''),
    7:('''
    
    Question 7:
Implementation of Intelligent Droplet

Answer 7:
import numpy as np
import random

dist = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

n = len(dist)

drops = 5
iterations = 50

soil = np.ones((n, n)) * 0.1

def choose_next(curr, visited):
    probs = []
    for j in range(n):
        if j not in visited:
            probs.append(1 / (soil[curr][j] + 1e-6))  # less soil = better
        else:
            probs.append(0)

    probs = np.array(probs)
    probs = probs / probs.sum()
    return np.random.choice(range(n), p=probs)

def path_cost(path):
    return sum(dist[path[i]][path[i+1]] for i in range(len(path)-1)) + dist[path[-1]][path[0]]

best_path = None
best_cost = float('inf')

for _ in range(iterations):
    all_paths = []

    for _ in range(drops):
        start = random.randint(0, n-1)
        path = [start]

        while len(path) < n:
            next_city = choose_next(path[-1], path)
            path.append(next_city)

        cost = path_cost(path)
        all_paths.append((path, cost))

        if cost < best_cost:
            best_cost = cost
            best_path = path

    for path, cost in all_paths:
        for i in range(len(path)-1):
            soil[path[i]][path[i+1]] = max(soil[path[i]][path[i+1]], 1e-6)

print("Best Path:", best_path)
print("Best Cost:", best_cost)
       
    '''),
    8:('''
Question 8:
Implementation of Firefly Algorithm

Answer 8:
import numpy as np

def f(x):
    return x**2   

fireflies = 5
iterations = 50
alpha = 0.2   
beta0 = 1     
gamma = 1     

pos = np.random.uniform(-10, 10, fireflies)

for _ in range(iterations):
    for i in range(fireflies):
        for j in range(fireflies):
            if f(pos[j]) < f(pos[i]):
                r = abs(pos[i] - pos[j])

                beta = beta0 * np.exp(-gamma * r**2)
       
                pos[i] += beta * (pos[j] - pos[i]) + alpha * np.random.randn()

best = min(pos, key=f)

print("Best position:", best)
print("Best value:", f(best))

    '''),
    9:('''
Implementation of Artificial Bee Colony

Answer 9:
import numpy as np
def f(x):
    return x**2   # Minimum at x = 0

food_sources = 5
iterations = 50
limit = 10   # abandonment limit

pos = np.random.uniform(-10, 10, food_sources)
fitness = np.array([f(x) for x in pos])
trial = np.zeros(food_sources)

for _ in range(iterations):


    for i in range(food_sources):
        k = np.random.randint(food_sources)
        phi = np.random.uniform(-1, 1)

        new = pos[i] + phi * (pos[i] - pos[k])

        if f(new) < fitness[i]:
            pos[i] = new
            fitness[i] = f(new)
            trial[i] = 0
        else:
            trial[i] += 1

    prob = fitness / fitness.sum()
    for i in range(food_sources):
        if np.random.rand() < prob[i]:
            k = np.random.randint(food_sources)
            phi = np.random.uniform(-1, 1)

            new = pos[i] + phi * (pos[i] - pos[k])

            if f(new) < fitness[i]:
                pos[i] = new
                fitness[i] = f(new)
                trial[i] = 0
            else:
                trial[i] += 1

    for i in range(food_sources):
        if trial[i] > limit:
            pos[i] = np.random.uniform(-10, 10)
            fitness[i] = f(pos[i])
            trial[i] = 0

best = pos[np.argmin(fitness)]

print("Best position:", best)
print("Best value:", f(best))

    ''')
}

