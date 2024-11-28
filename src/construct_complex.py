import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from zen_mapper import mapper
from zen_mapper.adapters import to_networkx
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover

from src.simplicial_complex import SimplicialComplex


def load_single_point_cloud(filename, index=0):
    with h5py.File(filename, "r") as f:
        points = f[f"points_3d/points_{index}"][:]
        label = f[f"points_3d/points_{index}"].attrs["label"]
    return points, label


def compute_mapper_graph(point_cloud, dimension=2):
    # Define projection (using PCA-1)
    pca = PCA(n_components=1)
    projection = pca.fit_transform(point_cloud)
    # projection = point_cloud[:, 2]

    # Define cover
    cover_scheme = Width_Balanced_Cover(n_elements=20, percent_overlap=0.5)

    # Define clusterer
    sk = AffinityPropagation(
        damping=0.5, max_iter=300, convergence_iter=30, preference=None
    )

    # sk = DBSCAN(eps=0.05, min_samples=3)
    clusterer = sk_learn(sk)

    # Compute mapper graph
    result = mapper(
        data=point_cloud,
        projection=projection,
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=dimension,
    )

    return result


def compute_centroids(point_cloud, result):
    """Compute centroids for each vertex in the mapper graph"""
    centroids = {}

    for vertex_id in range(len(result.nodes)):
        points = result.nodes[vertex_id]  # Contains indices into point_cloud
        points_in_vertex = point_cloud[points]
        centroid = np.mean(points_in_vertex, axis=0)
        centroids[vertex_id] = centroid

    return centroids


def create_simplicial_complex(result, centroids, point_cloud):
    """Create a SimplicialComplex from mapper result"""
    try:
        complex = SimplicialComplex()

        # Add vertices with their centroids
        for vertex_id in range(len(result.nodes)):
            complex.add_vertex(vertex_id, centroids[vertex_id])
            complex.set_vertex_function(vertex_id, centroids[vertex_id][2])

        # Add simplices from nerve
        for dim in range(result.nerve.dim + 1):
            for simplex in result.nerve[dim]:
                try:
                    complex.add_simplex(simplex, dim + 1)
                except ValueError as e:
                    print(f"Warning: Could not add simplex {simplex}: {e}")

        complex.extend_function(method="mean")
        return complex

    except Exception as e:
        print(f"Error creating simplicial complex: {e}")
        raise


def print_topological_info(complex):
    """Prints some information about the complex"""
    print("\nTopological Information:")
    print(f"Complex dimension: {complex.dimension}")

    # Print number of simplices in each dimension
    for dim in range(complex.dimension + 1):
        print(f"{dim}-simplices: {len(complex._simplices[dim])}")

    print("\nEuler characteristics:")
    print(f"Full: {complex.euler_characteristic()}")

    # Calculate Euler characteristic at different thresholds
    thresholds = [-0.5, 0.0, 0.25, 0.5]
    for t in thresholds:
        chi = complex.euler_characteristic(threshold=t)
        print(f"At threshold {t:.2f}: {chi}")


def visualize_results(point_cloud, label, result, centroids, complex):
    """
    Visualize the results of the MNIST digit point cloud, nerve complex,
        and embedded complex.
    Parameters:
    :point_cloud (numpy.ndarray): The point cloud data of the MNIST digit,
        with shape (n_points, 3).
    :label (int): The label of the MNIST digit.
    :result (object): The result object containing the nerve complex and nodes.
    :centroids (numpy.ndarray): The centroids of the clusters in the point cloud.
    :complex (object): The complex object containing vertex coordinates,
        vertex functions, and simplices.
    Returns:
    :matplotlib.figure.Figure: The figure object containing the visualizations.
    """

    fig = plt.figure(figsize=(20, 6))

    # 1. Plot point cloud
    ax1 = fig.add_subplot(131, projection="3d")
    scatter = ax1.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=point_cloud[:, 2],  # Color by height
        cmap="viridis",
        alpha=1,
        s=1,
    )
    ax1.set_title(f"MNIST Digit {label}: Point Cloud")

    # 2. Plot nerve complex with function values
    ax2 = fig.add_subplot(132)
    graph = to_networkx(result.nerve)

    # Color nodes by function value
    node_colors = [complex.vertex_functions.get(v, 0) for v in graph.nodes()]

    # Size nodes by number of points they represent
    node_sizes = [len(result.nodes[v]) * 10 for v in graph.nodes()]

    nx.draw(
        graph,
        ax=ax2,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        cmap=plt.cm.viridis,
        edge_color="gray",
        width=0.5,
    )
    ax2.set_title("Nerve Complex\n(size: cluster size, color: function value)")

    # 3. Plot embedded complex
    ax3 = fig.add_subplot(133, projection="3d")

    # Plot original points in background
    ax3.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c="lightgray",
        alpha=0.5,
        s=1,
    )

    # Plot vertices (centroids)
    vertex_coords = np.array(
        [complex.vertex_coords[v] for v in range(len(result.nodes))]
    )
    vertex_colors = [complex.vertex_functions[v] for v in range(len(result.nodes))]
    scatter = ax3.scatter(
        vertex_coords[:, 0],
        vertex_coords[:, 1],
        vertex_coords[:, 2],
        c=vertex_colors,
        s=100,
        alpha=1.0,
        marker="o",
        cmap=plt.cm.viridis,
    )

    # Plot 1-simplices (edges)
    for simplex in complex._simplices[1]:
        v1, v2 = simplex
        p1 = complex.vertex_coords[v1]
        p2 = complex.vertex_coords[v2]
        color = plt.cm.viridis(
            (complex.simplex_functions.get(simplex, 0) - min(vertex_colors))
            / (max(vertex_colors) - min(vertex_colors))
        )
        ax3.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            c=color,
            alpha=0.5,
            linewidth=1,
        )

    ax3.set_title("Embedded Complex\n(color: function value)")
    plt.colorbar(scatter, ax=ax3, label="Function Value")

    # Set common properties for 3D plots
    for ax in [ax1, ax3]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)  # Set viewing angle

        # Set limits based on point cloud bounds
        margin = 0.1
        for i, dim in enumerate(["x", "y", "z"]):
            min_val = point_cloud[:, i].min()
            max_val = point_cloud[:, i].max()
            range_val = max_val - min_val
            getattr(ax, f"set_{dim}lim")(
                [min_val - margin * range_val, max_val + margin * range_val]
            )

    plt.tight_layout()
    return fig


def main():
    """Main function to process and visualize MNIST point cloud data"""
    # Set backend to avoid Wayland issues
    import os

    os.environ["QT_QPA_PLATFORM"] = "xcb"

    # File paths
    data_filename = "data/raw/preprocessed_data/mnist_3d_cloud.h5"

    # Load point cloud
    point_cloud, label = load_single_point_cloud(data_filename, index=1)

    # Print data statistics
    print("\nData Statistics:")
    print(f"Point cloud shape: {point_cloud.shape}")
    print("Coordinate ranges:")
    for i, dim in enumerate(["X", "Y", "Z"]):
        print(
            f"{dim} range: [{point_cloud[:,i].min():.3f},{point_cloud[:,i].max():.3f}]"
        )

    # Compute mapper graph
    print("\nComputing Mapper graph...")
    result = compute_mapper_graph(point_cloud, dimension=1)

    # Compute centroids
    print("Computing centroids...")
    centroids = compute_centroids(point_cloud, result)

    # Create simplicial complex
    print("Creating simplicial complex...")
    complex = create_simplicial_complex(result, centroids, point_cloud)

    # Print complex information
    print("\nComplex Information:")
    print(f"Dimension: {complex.dimension}")
    for dim in range(complex.dimension + 1):
        print(f"{dim}-simplices: {len(complex._simplices[dim])}")

    print("\nEuler Characteristics:")
    print(f"Full: {complex.euler_characteristic()}")
    for threshold in [-0.5, 0.0, 0.25, 0.5]:
        print(
            f"At threshold {threshold:.2f}: {complex.euler_characteristic(
                threshold=threshold
            )}"
        )

    # Visualize
    print("\nGenerating visualization...")
    plt.show()


if __name__ == "__main__":
    main()
