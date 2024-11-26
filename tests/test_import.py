import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.mnist_loader import MNISTLoader
from src.transform_3d import ImageTo3D

def visualize_digit(image, points_3d, digit_label=None):
    """
    Create an interactive visualization using plotly
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "image"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f'Original Image (Digit: {digit_label})' if digit_label is not None else 'Original Image',
            '3D Representation'
        )
    )

    # Add 2D image
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(image),
            colorscale='gray',
            showscale=False
        ),
        row=1, col=1
    )

    # Create unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Add transparent unit sphere
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.2,
            showscale=False,
            colorscale=[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']]
        ),
        row=1, col=2
    )

    # Add 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_3d[:, 2],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title='Intensity')
            )
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="MNIST Digit Visualization",
        showlegend=False,
        width=1200,
        height=600,
    )

    # Update 3D scatter layout
    fig.update_scenes(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        aspectmode='cube'
    )

    return fig

def main():
    # Load MNIST data
    loader = MNISTLoader()
    X_train, y_train = loader.get_data(train=True)

    # Create transformer
    transformer = ImageTo3D()

    # Get first image
    image = loader.get_image(5)
    points_3d = transformer.convert_image(image)
    digit_label = y_train[5]

    # Create visualization
    fig = visualize_digit(image, points_3d, digit_label)

    # Show the plot in browser
    fig.show()

    # Also save as HTML file for later viewing
    fig.write_html("./tests/mnist_3d_visualization.html")
    print("\nVisualization has been saved to 'mnist_3d_visualization.html'")
    print("You can open this file in your web browser at any time.")

if __name__ == "__main__":
    main()
