import numpy as np


class ImageTo3D:
    def __init__(self):
        self.center_value = 0.5
        self.grayscale_weight = 10 # adds a weight for the grayscale.

    def convert_image(self, image):
        """Converts a 2D image to 3D points.
        
        :param self: Description
        :type self:  
        :param image: Description
        :type image:  """

        height,width = image.shape

        # Create a grid of points

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Create 3D points (x, y, grayscale)
        points = np.column_stack(tup=(
            x.ravel(),
            y.ravel(),
            image.ravel()
        ))

        return self.normalize_to_unit_ball(points)
    
    def normalize_to_unit_ball(self, points):
        """Normalizes the points to the unit ball with center pixel at the origin.
        
        :param self: Description
        :type self:  
        :param points: Description
        :type points:  """

        # Center the coordinates
        points[:, 0] -= 13.5 # (28-1)/2
        points[:, 1] -= 13.5
         
        # Scale to unit ball
        points[:, 2] *= self.grayscale_weight
        max_radius = np.sqrt(2 * 14**2 + (self.center_value * self.grayscale_weight)**2)
        points /= max_radius
        
        return points
    