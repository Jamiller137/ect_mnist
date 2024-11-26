from sklearn.datasets import fetch_openml

class MNISTLoader:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        # Normalize pixel values to [0, 1]
        self.X = self.X / 255.0

    def get_data(self, train=True):
        if train:
            return self.X[:60000], self.y[:60000]
        return self.X[60000:], self.y[60000:]
    
    def get_image(self, index):
        # Reshape the image to 28x28
        return self.X[index].reshape(28, 28)