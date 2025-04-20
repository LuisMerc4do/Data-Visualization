import csv
import numpy as np
import random
import matplotlib.pyplot as plt  # Only for plotting

# Global Variables To modify the algorithm
MAX_ITER = 100  # Max number of iterations in order to prevent infinite loop, it could finish befor reaching 100 iterations.

# ------------------------------
# Read the CSV file
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line (header)
        for row in reader:
            if row:  # If empty, skip
                # If not empty convert to float and append [tAssets, aIncome]
                data.append([float(row[0]), float(row[1])])
    # Returns all rows as a numpy array for better math handling
    return np.array(data)

# ------------------------------
# Calculate distance between two points
# According to the assesment directions we need to use the euclidean formula  -> √((x2-x1)² + (y2-y1)²)

def euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# ------------------------------
# Calculating the closest centroid and assign the point to it
def assign_clusters(data, centroids):
    # Empty list for each cluster
    clusters = [[] for _ in range(len(centroids))] 
    # Empty arr To track which cluster each point belongs to 
    labels = []  

    for point in data:
        # Measure how far this point is from each centroid
        # we call the euclidean formula and calculate the euclidean distance from point in data to all the centroids
        distances = [euclidean(point, c) for c in centroids]
        # Choose the closest centroid
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)
        labels.append(cluster_idx)

    return clusters, np.array(labels)

# ------------------------------
# Update Centroids
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            # Average of all points in the cluster
            new_centroids.append(np.mean(cluster, axis=0))
        else:
            # If a cluster is empty, keep it at (0, 0)
            new_centroids.append(np.zeros(2))
    return np.array(new_centroids)

# ------------------------------
# K-mean algorithm 
def kmeans(data, k):
    # Randomly select K initial centroids from the data
    centroids = data[random.sample(range(len(data)), k)]

    for _ in range(MAX_ITER):
        # Assign all data points to the nearest centroid
        clusters, labels = assign_clusters(data, centroids)

        # Update the centroids to be the mean of each cluster
        new_centroids = update_centroids(clusters)

        # If the centroids didn't change, its done so break the loop
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters, labels

# ------------------------------
# Calculate inertia (sum of squared distances of samples to their closest centroid)
def calculate_inertia(data, centroids, labels):
    inertia = 0
    for i, point in enumerate(data):
        inertia += euclidean(point, centroids[labels[i]]) ** 2
    return inertia

# ------------------------------
# Calculate minimum inter-cluster distance
def min_inter_cluster_distance(centroids):
    min_dist = float('inf')
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = euclidean(centroids[i], centroids[j])
            if dist < min_dist and dist > 0:  # Avoid same centroid
                min_dist = dist
    return min_dist

# ------------------------------
# Calculate maximum intra-cluster distance (diameter of a cluster)
def max_intra_cluster_distance(clusters):
    max_dist = 0
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        
        # Convert to numpy array for vectorized operations
        cluster = np.array(cluster)
        
        # Calculate pairwise distances within the cluster
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                dist = euclidean(cluster[i], cluster[j])
                max_dist = max(max_dist, dist)
    
    return max_dist

# ------------------------------
# Calculate Dunn Index
def calculate_dunn_index(centroids, clusters):
    min_inter = min_inter_cluster_distance(centroids)
    max_intra = max_intra_cluster_distance(clusters)
    
    # Handle division by zero
    if max_intra == 0:
        return float('inf')
    
    return min_inter / max_intra

# ------------------------------
# Save centroids information (assets and income)
def save_centroids_info(centroids, k):
    # Sort centroids by assets (ascending)
    sorted_centroids = sorted(enumerate(centroids), key=lambda x: x[1][0])
    
    # Also print to console
    print(f"\nCentroids for K = {k}:")
    print("Cluster\tAssets\t\tIncome")
    print("-" * 30)
    
    for i, centroid in sorted_centroids:
        print(f"{i+1}\t{centroid[0]:.2f}\t\t{centroid[1]:.2f}")

# ------------------------------
# Visualization of the results
def plot_clusters(data, centroids, labels, k):
    # Colors for visualization, can add more if needed
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors[i % len(colors)], label=f'Cluster {i+1}')

    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

    # Add axis labels and title
    plt.xlabel('Household Total Assets ($)')
    plt.ylabel('Annual Household Income ($)')
    plt.title(f'K-Means Clustering (K={k})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'clusters_k{k}.png')  # Save figure for the report
    plt.close()

# ------------------------------
# Main Function
def main():
    # Step 1: Load the data
    data = read_csv('HouseholdWealth.csv')
    
    # Step 2: Run K-Means for K from 2 to 10
    results = []
    all_centroids = {}
    
    for k in range(2, 11):
        print(f"Processing K = {k}...")
        
        # Run K-Means
        centroids, clusters, labels = kmeans(data, k)
        
        # Save centroids information
        save_centroids_info(centroids, k)
        all_centroids[k] = centroids
        
        # Calculate Dunn Index
        dunn_index = calculate_dunn_index(centroids, clusters)
        
        # Calculate Inertia
        inertia = calculate_inertia(data, centroids, labels)
        
        # Store results
        results.append((k, dunn_index, inertia))
        
        # Create visualization
        plot_clusters(data, centroids, labels, k)
    
    # Step 3: Print and save results
    print("\nResults Table:")
    print("K\tDunn Index\tInertia")
    print("-" * 30)
    
    for k, dunn, inertia in results:
        print(f"{k}\t{dunn:.6f}\t{inertia:.2f}")
    
    # Find optimal K values
    optimal_k_dunn = max(results, key=lambda x: x[1])[0]
    optimal_k_inertia = min(results, key=lambda x: x[2])[0]
    
    print("\nOptimal K based on Dunn Index:", optimal_k_dunn)
    print("Optimal K based on Inertia:", optimal_k_inertia)
    

# Run the program
if __name__ == '__main__':
    main()