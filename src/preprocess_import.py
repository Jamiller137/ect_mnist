import os

import h5py
import numpy as np
from tqdm import tqdm  # For progress bar

from src.mnist_loader import MNISTLoader
from src.transform_3d import ImageTo3D


def process_and_save_data():
    """
    Process MNIST data, convert images to 3D point clouds, and save to an HDF5 file.

    Returns:
        str: The filename of the saved HDF5 file containing the processed data.
    """
    # Create save directory
    save_dir = "data/raw/preprocessed_data"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize loader and transformer
    loader = MNISTLoader()
    transformer = ImageTo3D()

    # Get training data only
    print("\nLoading training data...")
    X_train, y_train = loader.get_data(train=True)

    # Create HDF5 file for training data
    h5_filename = os.path.join(save_dir, "mnist_3d_cloud.h5")
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset("labels", data=y_train)

        # Create groups for original images and 3D points
        orig_group = f.create_group("original_images")
        points_group = f.create_group("points_3d")

        # Process each image with progress bar
        print("\nConverting images to 3D points...")
        for idx in tqdm(range(len(X_train))):
            # Get image and convert to 3D
            image = loader.get_image(idx)
            points_3d = transformer.convert_image(image)

            # Save to HDF5
            orig_group.create_dataset(f"img_{idx}", data=image)
            points_group.create_dataset(f"points_{idx}", data=points_3d)

            # Store the label in the points dataset attributes
            points_group[f"points_{idx}"].attrs["label"] = y_train[idx]

    print(f"\nSaved training data to {h5_filename}")
    return h5_filename


def load_data(filename):
    """
    Load data from the HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.

    Returns:
        tuple: A tuple containing:
            - all_points (np.ndarray): The 3D point clouds.
            - labels (np.ndarray): The corresponding labels.
    """
    with h5py.File(filename, "r") as f:
        # Load labels
        labels = f["labels"][:]

        # Get number of samples
        n_samples = len(labels)

        # Load first sample to get dimensions
        first_points = f["points_3d"]["points_0"][:]
        points_dim = first_points.shape[1]  # Should be 3 for x, y, z

        # Initialize arrays
        all_points = np.zeros((n_samples, first_points.shape[0], points_dim))

        # Load all point clouds
        for i in range(n_samples):
            all_points[i] = f[f"points_3d/points_{i}"][:]

    return all_points, labels


def main():
    """_summary_
    Process MNIST data, convert images to 3D point clouds, save to an HDF5 file, and
    then load and verify the saved data.
    """
    print("Starting data processing and saving...")
    h5_filename = process_and_save_data()

    # Load and verify the saved data
    print("\nVerifying saved data...")
    points, labels = load_data(h5_filename)

    print("\nData structure:")
    print(f"Number of training examples: {len(labels)}")
    print(f"Point cloud shape: {points.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    # Example of accessing data
    print("\nExample of first point cloud:")
    print(f"Label: {labels[0]}")
    print(f"Points shape: {points[0].shape}")
    print(f"First few points:\n{points[0][:5]}")


if __name__ == "__main__":
    main()
