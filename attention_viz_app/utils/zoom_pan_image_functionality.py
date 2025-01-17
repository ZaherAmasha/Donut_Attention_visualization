import numpy as np

# import base64
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class ZoomPanImageTracker:
    """
    Tracks the original plot (image + attention map) for the zoom and pan functionality.
    Has methods to do that
    """

    def __init__(self):
        self.original_plot = None  # Store the original matplotlib figure
        self.last_zoom_params = {"zoom_level": 1.0, "x_offset": 0.5, "y_offset": 0.5}

    def create_mini_map_overlay(self, original_image, zoom_level, x_offset, y_offset):
        """
        Create a mini-map showing the current zoomed view of the image
        """
        if original_image is None:
            return None

        mini_map = original_image.copy()
        mini_map_pil = Image.fromarray((mini_map * 255).astype(np.uint8))
        draw = ImageDraw.Draw(mini_map_pil)

        # Calculate view rectangle dimensions
        orig_height, orig_width = original_image.shape[:2]
        view_width = int(orig_width / zoom_level)
        view_height = int(orig_height / zoom_level)

        # Calculate view rectangle position
        # Adjust offset calculation to center the view
        max_x_offset = orig_width - view_width
        max_y_offset = orig_height - view_height
        view_x = int(max_x_offset * x_offset)
        view_y = int(max_y_offset * y_offset)

        # Ensure view rectangle stays within bounds
        view_x = max(0, min(view_x, orig_width - view_width))
        view_y = max(0, min(view_y, orig_height - view_height))

        # Draw view rectangle
        draw.rectangle(
            [view_x, view_y, view_x + view_width, view_y + view_height],
            outline="red",
            width=2,
        )

        return np.array(mini_map_pil) / 255.0

    def zoom_pan_plot(self, zoom_level, x_offset, y_offset):
        """
        Zoom and pan the matplotlib plot
        """
        if self.original_plot is None:
            return None, None

        # Store current parameters
        self.last_zoom_params = {
            "zoom_level": zoom_level,
            "x_offset": x_offset,
            "y_offset": y_offset,
        }

        # Render the original plot at high DPI
        original_dpi = self.original_plot.get_dpi()
        high_res_dpi = original_dpi * max(1, zoom_level)  # Scale DPI with zoom

        # Create a high-res version of the original plot
        self.original_plot.set_dpi(high_res_dpi)

        # Get image data from original plot
        self.original_plot.canvas.draw()
        w, h = self.original_plot.canvas.get_width_height()
        buf = np.frombuffer(self.original_plot.canvas.buffer_rgba(), dtype=np.uint8)
        original_image = buf.reshape(h, w, 4)[:, :, :3].astype(np.float32) / 255.0

        # Reset the original plot's DPI
        self.original_plot.set_dpi(original_dpi)

        # Calculate dimensions
        height, width = original_image.shape[:2]

        # Calculate the visible region
        view_width = int(width / zoom_level)
        view_height = int(height / zoom_level)

        # Calculate offsets for the region to display
        max_x_offset = width - view_width
        max_y_offset = height - view_height
        x_start = int(max_x_offset * x_offset)
        y_start = int(max_y_offset * y_offset)

        # Ensure we stay within bounds
        x_start = max(0, min(x_start, width - view_width))
        y_start = max(0, min(y_start, height - view_height))

        # Extract the region to zoom
        region_to_zoom = original_image[
            y_start : y_start + view_height, x_start : x_start + view_width
        ]

        # Create zoomed view
        zoomed_fig = plt.figure(
            figsize=self.original_plot.get_size_inches(),
            dpi=self.original_plot.get_dpi(),
        )
        ax = zoomed_fig.add_subplot(111)
        ax.imshow(region_to_zoom, interpolation="nearest")
        ax.set_title(f"Zoom: {zoom_level:.1f}x")
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Create mini-map
        mini_map = self.create_mini_map_overlay(
            original_image, zoom_level, x_offset, y_offset
        )

        return zoomed_fig, mini_map

    def store_original_plot(self, plot: plt.figure):
        """
        Store the original plot for zooming and panning
        """
        self.original_plot = plot
        return plot


# Create a global instance of the visualizer
zoom_pan_image_tracker = ZoomPanImageTracker()


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
    return zoom_pan_image_tracker.zoom_pan_plot(zoom_level, x_offset, y_offset)
