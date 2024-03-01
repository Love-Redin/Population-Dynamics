import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import random

# Constants
HERBIVORE, LOW_PREDATOR, TOP_PREDATOR, EMPTY = 0, 1, 2, 3

# Initialize a global variable to track starvation counters
starvation_counters = None

# Parameters
grid_size = 20
herbivore_prob = 0.4491522715542885
low_predator_prob = 0.01434260589107006
top_predator_prob = 0.5365051225546414

reproduction_rate = {
    HERBIVORE: 0.2846552315592564,
    LOW_PREDATOR: 0.2087841998956354,
    TOP_PREDATOR: 0.9890669724345257
}
starvation_limit = {
    LOW_PREDATOR: 5,
    TOP_PREDATOR: 19
}

# Function to initialize the grid with specified probabilities
def initialize_grid(grid_size, herbivore_prob, low_predator_prob, top_predator_prob):
    empty_prob = 1 - (herbivore_prob + low_predator_prob + top_predator_prob)
    if empty_prob < 0:
        raise ValueError("Sum of probabilities exceeds 1")

    probabilities = [herbivore_prob, low_predator_prob, top_predator_prob, empty_prob]
    return np.random.choice([HERBIVORE, LOW_PREDATOR, TOP_PREDATOR, EMPTY], (grid_size, grid_size), p=probabilities)

# Function to get neighboring cells
def get_neighbors(grid, x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % grid.shape[0], (y + dy) % grid.shape[1]
            neighbors.append((nx, ny))
    return neighbors

# Modified simulate_step function to include dynamic reproduction rates and starvation limits
def simulate_step(grid, starvation_counters, reproduction_rate, starvation_limit):
    new_grid = grid.copy()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            cell = grid[x, y]
            neighbors = get_neighbors(grid, x, y)
            if cell == HERBIVORE:
                # Reproduction
                if random.random() < reproduction_rate[HERBIVORE]:
                    empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
                    if empty_neighbors:
                        nx, ny = random.choice(empty_neighbors)
                        new_grid[nx, ny] = HERBIVORE

            elif cell in [LOW_PREDATOR, TOP_PREDATOR]:
                # Feeding
                prey = HERBIVORE if cell == LOW_PREDATOR else LOW_PREDATOR
                prey_neighbors = [n for n in neighbors if grid[n] == prey]
                if prey_neighbors:
                    nx, ny = random.choice(prey_neighbors)
                    new_grid[nx, ny] = cell
                    starvation_counters[x, y] = 0
                    # Reproduction
                    if random.random() < reproduction_rate[cell]:
                        empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
                        if empty_neighbors:
                            ex, ey = random.choice(empty_neighbors)
                            new_grid[ex, ey] = cell
                else:
                    starvation_counters[x, y] += 1
                    if starvation_counters[x, y] > starvation_limit[cell]:
                        new_grid[x, y] = EMPTY
                        
    return new_grid, starvation_counters

# Function to simulate and return lifespan
def simulate_lifespan(grid_size, n_iterations, herbivore_prob, low_predator_prob, top_predator_prob,
                      reproduction_rate, starvation_limit):
    grid = initialize_grid(grid_size, herbivore_prob, low_predator_prob, top_predator_prob)
    starvation_counters = np.zeros((grid_size, grid_size), dtype=int)

    for iteration in range(n_iterations):
        grid, starvation_counters = simulate_step(grid, starvation_counters, reproduction_rate, starvation_limit)
        if not all(np.sum(grid == actor) > 0 for actor in [HERBIVORE, LOW_PREDATOR, TOP_PREDATOR]):
            return iteration  # Return the iteration number when one species dies out
    
    return n_iterations  # All species survived throughout the simulation

max_grid_size = 50
iterations = 100
max_iter = 100


results = {}

for grid_size in range(1, max_grid_size + 1, 10):
    grid_size_results = []
    for iteration in range(iterations):
        lifespan = simulate_lifespan(grid_size, max_iter, herbivore_prob, low_predator_prob, 
                                     top_predator_prob, reproduction_rate, starvation_limit)
        grid_size_results.append(lifespan)
        
    mean_lifespan = np.mean(grid_size_results)
    std_deviation = np.std(grid_size_results)
    share_stable = grid_size_results.count(max_iter) / iterations
    
    results[grid_size] = (share_stable * 100, mean_lifespan, std_deviation)
    print(f'Grid size: {grid_size}, stable share: {share_stable}, mean: {mean_lifespan}')

grid_sizes = list(results.keys())
share_stable_values = [result[0] for result in results.values()]
mean_lifespan_values = [result[1] for result in results.values()]
std_deviation_values = [result[2] for result in results.values()]

# Plotting the results
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Grid Size')
ax1.set_ylabel(f'Stability (Share of Lifespan = {max_iter})', color=color)
ax1.plot(grid_sizes, share_stable_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Mean Lifespan', color=color)
ax2.errorbar(grid_sizes, mean_lifespan_values, yerr=std_deviation_values, color=color, fmt='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()