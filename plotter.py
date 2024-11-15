import matplotlib.pyplot as plt
import numpy as np

# Define the datasets and m values (excluding Earthquakes)
datasets = ["Ecg200", "Ecg5000", "FordA", "Worms"]
m_values = [1, 3, 5, 10, 15]
positions_colors = {"Start": "red", "Center": "blue", "End": "green"}

# Initialize data structure to store parsed results
data = {
    "False": {dataset: {"Start": [], "Center": [], "End": []} for dataset in datasets},
    "True": {dataset: {"Start": [], "Center": [], "End": []} for dataset in datasets}
}

# Read and parse the file
with open("out.txt", "r") as file:
    for line in file:
        parts = line.strip().split(", ")
        position = parts[0].strip()
        stack = parts[1].strip() == "T"
        accuracies = list(map(int, parts[2:]))  # Convert accuracy values to integers

        # Store data for each dataset (excluding Earthquakes)
        for i, dataset in enumerate(datasets):
            start_idx = (i + 1) * 5  # Shift start index to skip Earthquakes data
            end_idx = start_idx + 5
            data[str(stack)][dataset][position].extend(accuracies[start_idx:end_idx])

# Plot the data
fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharex=True, constrained_layout=True)  # 4 rows for remaining datasets
fig.suptitle("Accuracy Plots Across Datasets and Stack Configurations", fontsize=16)

# Set up x-axis labels as categorical
x_pos = range(len(m_values))  # Evenly spaced positions for categorical labels

for i, dataset in enumerate(datasets):
    for j, stack in enumerate(["False", "True"]):
        ax = axes[i, j]

        # Gather all values for dynamic y-axis scaling
        y_values = []

        # Calculate average accuracy for each m across all positions
        avg_accuracy_per_m = []

        # Plot each position with its specific color and opacity
        for position, color in positions_colors.items():
            y_data = data[stack][dataset][position]
            ax.scatter(x_pos, y_data, color=color, alpha=0.7, label=position)
            y_values.extend(y_data)

        # Calculate the average accuracy for each m value across the three positions
        for m_idx in range(len(m_values)):
            avg_accuracy = np.mean([data[stack][dataset][pos][m_idx] for pos in positions_colors.keys()])
            avg_accuracy_per_m.append(avg_accuracy)

        # Find the m value with the highest average accuracy
        max_avg_accuracy = max(avg_accuracy_per_m)
        max_m = m_values[avg_accuracy_per_m.index(max_avg_accuracy)]

        # Dynamic y-axis based on min/max accuracy values
        y_min, y_max = min(y_values), max(y_values)
        ax.set_ylim(y_min - 50, y_max + 50)

        # Plot formatting with max average accuracy info
        ax.set_title(f"{dataset} (Stack={stack})\nAvg Accuracy: {np.mean(y_values):.2f}, Max Avg Accuracy at m={max_m}: {max_avg_accuracy:.2f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(m_values)  # Set m values as categorical labels: 1, 3, 5, 10, 15
        ax.set_xlabel("m")
        ax.set_ylabel("Accuracy")

        # Only add legend once per stack column
        if i == 0:
            ax.legend()

# Save the plot to an image file
plt.savefig("positioningResults.png")  # Save as PNG (change the extension to .jpg, .pdf, etc., if desired)

# Display the plot (optional, remove if you only need to save without displaying)
plt.show()
