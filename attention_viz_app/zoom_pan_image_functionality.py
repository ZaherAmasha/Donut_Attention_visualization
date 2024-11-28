import numpy as np
import io

# import base64
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class AttentionVisualizer:
    def __init__(self):
        self.original_plot = None  # Store the original matplotlib figure
        self.last_zoom_params = {"zoom_level": 1.0, "x_offset": 0.5, "y_offset": 0.5}

    def create_mini_map_overlay(self, image, zoom_level, x_offset, y_offset):
        """
        Create a mini-map showing the current zoomed view of the image
        """
        if image is None:
            return None

        # Convert matplotlib figure to image array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image_array = np.array(Image.open(buf))
        plt.close()

        orig_height, orig_width = image_array.shape[:2]
        mini_map = image_array.copy()  # copy to draw on the minimap

        mini_map_pil = Image.fromarray(mini_map)
        draw = ImageDraw.Draw(mini_map_pil)

        # Calculate the size of the zoomed view rectangle
        view_width = int(orig_width / zoom_level)
        view_height = int(orig_height / zoom_level)

        # Calculate the top-left corner of the view rectangle
        view_x = int((orig_width - view_width) * x_offset)
        view_y = int((orig_height - view_height) * y_offset)

        # Draw the view rectangle
        draw.rectangle(
            [view_x, view_y, view_x + view_width, view_y + view_height],
            outline="red",
            width=3,
        )

        return np.array(mini_map_pil)

    def zoom_pan_plot(self, zoom_level, x_offset, y_offset):
        """
        Zoom and pan the matplotlib plot
        """
        if self.original_plot is None:
            return None, None

        # Store current zoom parameters
        self.last_zoom_params = {
            "zoom_level": zoom_level,
            "x_offset": x_offset,
            "y_offset": y_offset,
        }

        # Create a new figure with the same size and content as the original
        fig = plt.figure(
            figsize=self.original_plot.get_size_inches(),
            dpi=self.original_plot.get_dpi(),
        )

        # Copy the original plot's axes
        ax = fig.add_subplot(111)

        # Get the image from the original plot
        image = self.original_plot.axes[0].get_images()[0]
        data = image.get_array()

        # Calculate zooming
        height, width = data.shape[:2]
        new_width = int(width * zoom_level)
        new_height = int(height * zoom_level)

        # Calculate crop coordinates
        x_start = int((new_width - width) * x_offset)
        y_start = int((new_height - height) * y_offset)

        # Resize and crop
        from scipy.ndimage import zoom as image_zoom

        zoomed_data = image_zoom(data, (zoom_level, zoom_level, 1), order=1)

        # Crop back to original size
        cropped_data = zoomed_data[
            y_start : y_start + height, x_start : x_start + width
        ]

        # Display the cropped image
        ax.imshow(cropped_data)
        ax.set_title(self.original_plot.axes[0].get_title())
        ax.axis("off")

        # Removes any padding around the image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Create mini map
        mini_map = self.create_mini_map_overlay(
            cropped_data, zoom_level, x_offset, y_offset
        )

        return fig, mini_map

    def store_original_plot(self, plot):
        """
        Store the original plot for zooming and panning
        """
        self.original_plot = plot
        return plot


# Create a global instance of the visualizer
attention_visualizer = AttentionVisualizer()


def zoom_pan_image(zoom_level, x_offset, y_offset):
    # def zoom_pan_image(plot, zoom_level, x_offset, y_offset):
    """
    Wrapper function for Gradio interface
    """
    # Validate and normalize slider values
    zoom_level = max(0.25, min(zoom_level, 15))
    x_offset = max(0, min(x_offset, 1))
    y_offset = max(0, min(y_offset, 1))

    # Zoom and pan the plot
    return attention_visualizer.zoom_pan_plot(zoom_level, x_offset, y_offset)
