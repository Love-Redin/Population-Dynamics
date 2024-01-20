from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

colormap = "plasma_r"

parameters_file_name = "parameters_5000_500.txt"

simulation_df = pd.read_csv(parameters_file_name, sep=",", index_col=0)
max_lifespan = simulation_df.lifespan.max()
print(f"Max lifespan: {max_lifespan}")
print(simulation_df)
simulation_columns = simulation_df.columns

# Normalize only the parameters, not the lifespan
parameters = simulation_df.drop('lifespan', axis=1)
scaler = StandardScaler()
parameters_normalized = scaler.fit_transform(parameters)
print(parameters_normalized)

# Applying PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(parameters_normalized)

# Creating a DataFrame for PCA results
pca_df = pd.DataFrame(data=principalComponents, columns=['PCA1', 'PCA2'])
pca_df['Lifespan'] = simulation_df['lifespan']

print(pca_df)

# Define the normalization for the color bar based on lifespan values
norm = mpl.colors.Normalize(vmin=pca_df['Lifespan'].min(), vmax=pca_df['Lifespan'].max())

# Sort the DataFrame based on the 'Lifespan' column
# Points with the highest lifespan will be plotted last and, therefore, will appear on top.
pca_df_sorted = pca_df.sort_values('Lifespan')

plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(data=pca_df_sorted, x='PCA1', y='PCA2', hue='Lifespan', palette=colormap, legend=False, hue_norm=norm)
plt.title("PCA of Simulation Parameters with Lifespan")

# Create a color bar with the correct scale
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Lifespan')

plt.show()

# Filter the DataFrame to include only rows where Lifespan equals max_lifespan
filtered_df = pca_df[pca_df['Lifespan'] == max_lifespan]

# Create a color palette for the points with Lifespan = 500
highlight_palette = sns.color_palette(colormap, as_cmap=True)
highlight_color = highlight_palette(0.5)  # Adjust the index as needed

plt.figure(figsize=(10, 8))

# Scatterplot for all points with reduced alpha
scatter = sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', color='gray', alpha=0.2, legend=False)

# Scatterplot for points with Lifespan = max_lifespan with full alpha
sns.scatterplot(data=filtered_df, x='PCA1', y='PCA2', color=highlight_color, legend=False)

plt.title(f"PCA of Simulation Parameters with Lifespan {max_lifespan} Highlighted", fontsize=20)  # Increase the title font size

# Increase the font size for x and y axis labels
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)

# Increase the font size for tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# PCA loading plot

pca.fit(parameters_normalized)

# Extracting PCA loadings (components)
loadings = pca.components_.T  # Transpose to align with original variables


num_vars = loadings.shape[0]
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(num_vars):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.1)
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, simulation_columns[1:][i], color='r', fontsize=12)  # Increase the text font size

# Setting the plot limits
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Adding labels and title
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.title('PCA Loading Plot', fontsize=20)  # Increase the title font size
plt.grid()

# Increase the font size for tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()


# Step 1: Filter the DataFrame for points with 'Lifespan' of 500
lifespan_value = 500  # Replace with the lifespan you're interested in
filtered_df = pca_df[pca_df['Lifespan'] == lifespan_value]

# Step 2: Calculate the weighted middle point (centroid) in the PCA space
centroid_pca = filtered_df[['PCA1', 'PCA2']].mean().values.reshape(1, -1)

# Step 3: Inverse transform the centroid from PCA space to the normalized parameter space
centroid_normalized = pca.inverse_transform(centroid_pca)

# Step 4: Inverse transform from normalized parameter space to original parameter space
centroid_original = scaler.inverse_transform(centroid_normalized)

# Converting the centroid array back into a DataFrame with appropriate column names
centroid_df = pd.DataFrame(centroid_original, columns=parameters.columns)

# Printing the centroid in the original parameter space
print(centroid_df)

for i in range(len(centroid_df.columns)):
    print(f"{centroid_df.columns[i]}: {centroid_df.iloc[0][i]}")


# Calculate the correlation matrix
correlation_matrix = simulation_df.corr()

# Extract only the correlations with respect to 'lifespan'
lifespan_correlations = correlation_matrix['lifespan'].drop('lifespan')  # Drop the self-correlation of lifespan

# Sort the values to see the most positively and negatively correlated parameters first
lifespan_correlations_sorted = lifespan_correlations.sort_values(ascending=False)

# Print the sorted correlations with 'lifespan'
print(lifespan_correlations_sorted)

plt.figure(figsize=(10, 8))  # Increase the height if necessary
sns.heatmap(lifespan_correlations_sorted.to_frame(), annot=True, cmap='coolwarm', cbar=True, annot_kws={"size": 20})  # Adjust the fontsize here

# Adjust the title position
plt.title('Correlation with Lifespan', y=1.05, fontsize=20)  # Adjust the fontsize here and use 'fontsize' parameter for the title

plt.tight_layout()  # This will adjust the layout of the plot
plt.show()


# Random forest regression
rf = RandomForestRegressor(n_estimators=100)
rf.fit(parameters, simulation_df['lifespan'])

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(parameters.shape[1]):
    print(f"{f + 1}. feature {parameters.columns[indices[f]]} ({importances[indices[f]]})")

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

# Plot the feature importances of the forest
plt.title("Feature importances", fontsize=20)  # Increase the title font size
plt.bar(range(len(importances)), importances[indices], align="center")

# Rotate the feature names on the x-axis so they don't overlap
plt.xticks(range(len(importances)), parameters.columns[indices], rotation=45, ha='right', fontsize=20)  # Increase the x-axis label font size

# Increase the y-axis label font size
plt.yticks(fontsize=14)

# Automatically adjust subplot params so the subplot(s) fits into the figure area.
plt.tight_layout()

plt.show()