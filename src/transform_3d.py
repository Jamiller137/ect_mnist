import numpy as np


class ImageTo3D:
    def __init__(self):
        self.center_value = 0.5
        self.grayscale_weight = 10  # adds a weight for the grayscale.

    def convert_image(self, image):
        """
        Converts a 2D image to 3D points.

        This function takes a 2D image and converts it into a set of 3D points where
        each point consists of the x and y coordinates and the grayscale value of
        the image at that point.

        :param image: A 2D numpy array representing the grayscale image.
        :type image: numpy.ndarray
        :return: A numpy array of shape (height*width, 3) where each row represents
            a 3D point (x, y, grayscale).
        :rtype: numpy.ndarray
        """

        height, width = image.shape

        # Create a grid of points

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Create 3D points (x, y, grayscale)
        points = np.column_stack(tup=(x.ravel(), y.ravel(), image.ravel()))

        return self.normalize_to_unit_ball(points)

    def normalize_to_unit_ball(self, points):
        """
        Centers the coordinates of the points and scales them to fit
        within a unit ball. The z-coordinates are scaled by a grayscale weight
        before normalization.

        :param points: The points to be normalized. Expected to be a numpy array with
            shape (n, 3), where n is the number of points and
            each point has x, y, and z coordinates.
        :type points: numpy.ndarray

        :return: The normalized points.
        :rtype: numpy.ndarray
        """

        # Center the coordinates
        points[:, 0] -= 13.5  # (28-1)/2
        points[:, 1] -= 13.5

        # Scale to unit ball
        points[:, 2] *= self.grayscale_weight
        max_radius = np.sqrt(
            2 * 14**2 + (self.center_value * self.grayscale_weight) ** 2
        )
        points /= max_radius

        return points
