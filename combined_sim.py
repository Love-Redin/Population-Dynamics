import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import random

# Constants
HERBIVORE, LOW_PREDATOR, TOP_PREDATOR, EMPTY = 0, 1, 2, 3

# Initialize the grid with specified probabilities
def initialize_grid(grid_size, herbivore_prob, low_predator_prob, top_predator_prob):
    empty_prob = 1 - (herbivore_prob + low_predator_prob + top_predator_prob)
    probabilities = [herbivore_prob, low_predator_prob, top_predator_prob, empty_prob]
    init_grid = np.random.choice([HERBIVORE, LOW_PREDATOR, TOP_PREDATOR, EMPTY], (grid_size, grid_size), p=probabilities)
    return init_grid

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

def simulate_step(grid, reproduction_rate, starvation_limit):
    global starvation_counters  # Declare the variable as global

    if starvation_counters is None:
        starvation_counters = np.zeros_like(grid)
    
    new_grid = grid.copy()
    
    # Shuffle the order of cell processing to reduce bias
    cells_to_process = list(np.ndindex(grid.shape))
    random.shuffle(cells_to_process)
    
    for x, y in cells_to_process:
        cell = grid[x, y]
        neighbors = get_neighbors(grid, x, y)
        
        if cell == HERBIVORE:
            # Herbivore reproduction logic
            if random.random() < reproduction_rate[HERBIVORE]:
                empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
                if empty_neighbors:
                    nx, ny = random.choice(empty_neighbors)
                    new_grid[nx, ny] = HERBIVORE

        elif cell == LOW_PREDATOR:
            # Low-level predator eating and reproduction logic
            prey = HERBIVORE
            prey_neighbors = [n for n in neighbors if grid[n] == prey]
            if prey_neighbors:
                # Predator eats randomly from available prey
                nx, ny = random.choice(prey_neighbors)
                new_grid[nx, ny] = LOW_PREDATOR  # Set the new cell where predator ate
                # Remove the predator from its current cell
                new_grid[x, y] = EMPTY
                starvation_counters[nx, ny] = 0
                # Reproduction
                if random.random() < reproduction_rate[LOW_PREDATOR]:
                    empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
                    if empty_neighbors:
                        ex, ey = random.choice(empty_neighbors)
                        new_grid[ex, ey] = LOW_PREDATOR
            else:
                starvation_counters[x, y] += 1
                if starvation_counters[x, y] > starvation_limit[LOW_PREDATOR]:
                    new_grid[x, y] = EMPTY

        elif cell == TOP_PREDATOR:
            # Top-level predator eating and reproduction logic
            prey = LOW_PREDATOR
            prey_neighbors = [n for n in neighbors if grid[n] == prey]
            if prey_neighbors:
                # Predator eats randomly from available prey
                nx, ny = random.choice(prey_neighbors)
                new_grid[nx, ny] = TOP_PREDATOR  # Set the new cell where predator ate
                # Remove the predator from its current cell
                new_grid[x, y] = EMPTY
                starvation_counters[nx, ny] = 0
                # Reproduction
                if random.random() < reproduction_rate[TOP_PREDATOR]:
                    empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
                    if empty_neighbors:
                        ex, ey = random.choice(empty_neighbors)
                        new_grid[ex, ey] = TOP_PREDATOR
            else:
                starvation_counters[x, y] += 1
                if starvation_counters[x, y] > starvation_limit[TOP_PREDATOR]:
                    new_grid[x, y] = EMPTY
                    
    return new_grid

# Initialize a global variable to track starvation counters
starvation_counters = None

# Parameters
grid_size = 50
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

# Initialize a global variable to track whether the animation is paused
animation_paused = False

# Adjusted on_click function
def on_click(event):
    global animation_paused, ani
    if not animation_paused:
        ani.event_source.stop()
        animation_paused = True
    else:
        ani.event_source.start()
        animation_paused = False

# Set up the initial grid and figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Create two subplots
grid = initialize_grid(grid_size, herbivore_prob, low_predator_prob, top_predator_prob)
cmap = ListedColormap(['green', 'yellow', 'red', 'white'])  # Colors

mat = ax1.matshow(grid, cmap=cmap)

# Define legend patches for each category
legend_patches = [
    plt.Rectangle((0, 0), 1, 1, color='green', label='Herbivore'),
    plt.Rectangle((0, 0), 1, 1, color='yellow', label='Low-level Predator'),
    plt.Rectangle((0, 0), 1, 1, color='red', label='Top-level Predator'),
    plt.Rectangle((0, 0), 1, 1, color='white', label='Empty cell')
]

# Create the legend below the left animation plot
legend = ax1.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=4, facecolor='lightgrey')
legend.set_title("Species in System")
        
# Connect the click event to the figure, not just the subplot
fig.canvas.mpl_connect('button_press_event', on_click)

# Population dynamics plot
populations = {HERBIVORE: [], LOW_PREDATOR: [], TOP_PREDATOR: []}#, EMPTY: []}
lines = {
    HERBIVORE: ax2.plot([], [], label="Herbivores", color="green")[0],
    LOW_PREDATOR: ax2.plot([], [], label="Low-level Predators", color="yellow")[0],
    TOP_PREDATOR: ax2.plot([], [], label="Top-level Predators", color="red")[0]#,
}
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Population")
ax2.set_title("Population Dynamics")
ax2.legend()

# Animation update function
def update(frame):
    global grid, populations
    grid = simulate_step(grid, reproduction_rate, starvation_limit)
    mat.set_data(grid)
    
    # Update population counts
    for actor in [HERBIVORE, LOW_PREDATOR, TOP_PREDATOR]:#, EMPTY]:
        populations[actor].append(np.sum(grid == actor))
        lines[actor].set_data(range(len(populations[actor])), populations[actor])

    ax2.relim()  # Recalculate the limits based on the updated data
    ax2.autoscale_view()  # Rescale the view
    
    mat.autoscale()
    
    return [mat, *lines.values()]

# Run the animation
ani = animation.FuncAnimation(fig, update, interval=100, save_count=50)

# This line is critical to start the animation
ani.event_source.start()


plt.show()
