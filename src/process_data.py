import h5py
import numpy as np
from tqdm import tqdm

from src.construct_complex import (
    compute_centroids,
    compute_mapper_graph,
    create_simplicial_complex,
)


def create_processed_data(input_filename, output_filename):
    """
    Create processed data file containing Euler characteristics for different
    directions and thresholds.
    """

    # Generate uniform directions on unit sphere using spherical coordinates
    n_directions = 64
    n_thresholds = 64

    # Generate angles
    phi = np.linspace(0, 2 * np.pi, n_directions // 2)
    theta = np.linspace(0, np.pi, n_directions // 2)
    phi, theta = np.meshgrid(phi, theta)

    # Convert to Cartesian coordinates
    directions = (
        np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        .reshape(3, -1)
        .T[:n_directions]
    )

    # Normalize directions
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # Generate threshold values
    thresholds = np.linspace(-1, 1, n_thresholds)

    print("Opening input file...")
    with h5py.File(input_filename, "r") as f:
        n_samples = len(f["points_3d"].keys())
        print(f"Found {n_samples} samples")

        print("Creating output file...")
        with h5py.File(output_filename, "w") as out_f:
            # Create datasets
            features = out_f.create_dataset(
                "features",
                shape=(n_samples, n_directions, n_thresholds),
                dtype=np.int32,
            )
            labels = out_f.create_dataset("labels", shape=(n_samples,), dtype=np.int32)

            # Process each point cloud with progress bar
            for idx in tqdm(range(n_samples), desc="Processing point clouds"):
                # Load point cloud and label
                points = f[f"points_3d/points_{idx}"][:]
                label = f[f"points_3d/points_{idx}"].attrs["label"]

                # Compute mapper graph and complex
                result = compute_mapper_graph(points, dimension=2)
                centroids = compute_centroids(points, result)
                complex = create_simplicial_complex(result, centroids, points)

                # Compute Euler characteristics for each direction and threshold
                for i, direction in enumerate(directions):
                    # Compute dot products for vertex function
                    for vertex_id in complex.vertex_coords:
                        dot_product = np.dot(
                            complex.vertex_coords[vertex_id], direction
                        )
                        complex.set_vertex_function(vertex_id, dot_product)

                    # Extend function to higher simplices
                    complex.extend_function(method="mean")

                    # Compute Euler characteristic for each threshold
                    for j, threshold in enumerate(thresholds):
                        chi = complex.euler_characteristic(threshold=threshold)
                        features[idx, i, j] = chi

                # Store label
                labels[idx] = label

            # Store metadata
            out_f.attrs["n_directions"] = n_directions
            out_f.attrs["n_thresholds"] = n_thresholds
            out_f.attrs["directions"] = directions
            out_f.attrs["thresholds"] = thresholds


def main():
    input_filename = "data/raw/preprocessed_data/mnist_3d_cloud.h5"
    output_filename = "data/processed/processed_data.h5"

    print("Creating processed data file...")
    create_processed_data(input_filename, output_filename)
    print("Done!")


if __name__ == "__main__":
    main()
