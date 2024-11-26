import os
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, KMeans
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.cluster import sk_learn
from zen_mapper import mapper
from zen_mapper.adapters import to_networkx

def load_single_point_cloud(filename, index=0):
    """Load a single point cloud from the HDF5 file"""
    with h5py.File(filename, 'r') as f:
        points = f[f'points_3d/points_{index}'][:]
        label = f[f'points_3d/points_{index}'].attrs['label']
    return points, label

def compute_mapper_graph(point_cloud):
    # Define projection (using z-coordinate)
    projection = point_cloud[:, 2]

    # Define cover
    cover_scheme = Width_Balanced_Cover(
        n_elements=4,
        percent_overlap=0.4
    )

    # Define clusterer
    sk = KMeans(n_clusters=3, random_state=42)
    clusterer = sk_learn(sk)

    # Compute mapper graph
    result = mapper(
        data=point_cloud,
        projection=projection,
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=1
    )

    return result

def compute_centroids(point_cloud, result):
    """Compute centroids for each vertex in the mapper graph"""
    centroids = {}

    # Convert to networkx graph
    graph = to_networkx(result.nerve)

    # Create mapping from vertex ID to node index
    vertex_to_node = {}
    for i, cover_nodes in enumerate(result.cover):
        for node_id in cover_nodes:
            vertex_to_node[node_id] = i

    # For each vertex in the graph
    for vertex_id in graph.nodes():
        # Get points for this vertex from the nodes list
        points = result.nodes[vertex_id]  # This contains the indices into point_cloud
        points_in_vertex = point_cloud[points]
        centroid = np.mean(points_in_vertex, axis=0)
        centroids[vertex_id] = centroid

    return centroids

def visualize_results(point_cloud, label, result, centroids):
    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 6))

    # Convert to networkx graph
    graph = to_networkx(result.nerve)

    # 1. Plot point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=point_cloud[:, 2],
        cmap='gray',
        alpha=0.1,
        s=1
    )
    ax1.set_title(f'MNIST Digit {label}: Point Cloud')

    # 2. Plot abstract mapper graph
    ax2 = fig.add_subplot(132)
    nx.draw(graph, ax=ax2, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8)
    ax2.set_title('Abstract Mapper Graph')

    # 3. Plot embedded mapper complex
    ax3 = fig.add_subplot(133, projection='3d')

    # Plot point cloud in background
    ax3.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c='gray',
        alpha=0.1,
        s=1
    )

    # Plot vertices (centroids)
    centroid_points = np.array(list(centroids.values()))
    ax3.scatter(
        centroid_points[:, 0],
        centroid_points[:, 1],
        centroid_points[:, 2],
        c='red',
        s=100,
        alpha=1.0,
        marker='*',
        label='Vertices'
    )

    # Plot edges
    for edge in graph.edges():
        v1, v2 = edge
        p1 = centroids[v1]
        p2 = centroids[v2]
        ax3.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            'b-',
            alpha=0.5,
            linewidth=1
        )

    ax3.set_title('Embedded Mapper Complex')

    # Set common properties for 3D plots
    for ax in [ax1, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    plt.tight_layout()
    plt.show()

def main():
    # File paths
    data_filename = 'data/raw/preprocessed_data/mnist_3d_cloud.h5'

    # Load a single point cloud
    point_cloud, label = load_single_point_cloud(data_filename, index=0)

    # Print basic statistics
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Coordinate ranges:")
    print(f"X range: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
    print(f"Y range: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
    print(f"Z range: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")

    # Compute mapper graph
    result = compute_mapper_graph(point_cloud)

    # Compute centroids for visualization
    centroids = compute_centroids(point_cloud, result)

    # Visualize
    visualize_results(point_cloud, label, result, centroids)

if __name__ == "__main__":
    main()
