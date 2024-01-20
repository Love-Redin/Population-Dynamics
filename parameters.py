import numpy as np
import random
import pandas as pd

# Constants
HERBIVORE, LOW_PREDATOR, TOP_PREDATOR, EMPTY = 0, 1, 2, 3

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

# Function to run the simulation with random parameters
def run_simulation_with_random_params(n_runs, grid_size, n_iterations):
    results = []
    for i in range(n_runs):
        # Randomly generate two numbers and sort them
        cuts = sorted([random.random(), random.random()])

        # Use these to create three segments
        herbivore_prob = cuts[0]  # The first segment length
        low_predator_prob = cuts[1] - cuts[0]  # The second segment length
        top_predator_prob = 1 - cuts[1]  # The third segment length

        # Adjust reproduction rates and starvation limits
        reproduction_rate = {
            HERBIVORE: random.uniform(0, 1),
            LOW_PREDATOR: random.uniform(0, 1),
            TOP_PREDATOR: random.uniform(0, 1)
        }

        starvation_limit = {
            LOW_PREDATOR: random.randint(1, 50),
            TOP_PREDATOR: random.randint(1, 50)
        }
        
        reproduction_rate_herb = reproduction_rate[HERBIVORE]
        reproduction_rate_low = reproduction_rate[LOW_PREDATOR]
        reproduction_rate_top = reproduction_rate[TOP_PREDATOR]
        
        starvation_limit_low = starvation_limit[LOW_PREDATOR]
        starvation_limit_top = starvation_limit[TOP_PREDATOR]
        
        # Run simulation
        lifespan = simulate_lifespan(grid_size, n_iterations, herbivore_prob, low_predator_prob, 
                                     top_predator_prob, reproduction_rate, starvation_limit)
        results.append([lifespan, herbivore_prob, low_predator_prob, top_predator_prob, reproduction_rate_herb, reproduction_rate_low, 
                        reproduction_rate_top, starvation_limit_low, starvation_limit_top])
        
        print(f"Run {i}, lifespan: {lifespan}, Parameters: {herbivore_prob}, {low_predator_prob}, {top_predator_prob}, {reproduction_rate_herb}, {reproduction_rate_low}, {reproduction_rate_top}, {starvation_limit_low}, {starvation_limit_top}")

    return results

# Running the simulation with random parameters
n_runs = 5000  # Number of simulations to run
grid_size = 50
n_iterations = 500
simulation_results = run_simulation_with_random_params(n_runs, grid_size, n_iterations)

simulation_columns = ['lifespan', 'herbivore_initial_probability', 'low_predator_initial_probability', 'top_predator_initial_probability', 
           'herbivore_reproduction_rate', 'low_predator_reproduction_rate', 'top_predator_reproduction_rate', 'low_predator_starvation_rate', 
           'top_predator_starvation_rate']

simulation_df = pd.DataFrame(simulation_results, columns = simulation_columns)
simulation_df.to_csv(f"parameters_{n_runs}_{n_iterations}.txt", sep=",")
print(simulation_df)