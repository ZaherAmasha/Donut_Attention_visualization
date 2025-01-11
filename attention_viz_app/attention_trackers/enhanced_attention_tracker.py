import os
import torch
from datetime import datetime
from scipy.ndimage import zoom
import textwrap
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
import math
import matplotlib.gridspec as gridspec
import sys

# Get the parent directory of the current file and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))
)

from attention_viz_app.attention_trackers.cross_attention_tracker import (
    CrossAttentionTracker,
)
from attention_trackers.swin_attention_tracker import SwinAttentionTracker


class EnhancedAttentionVisualizer:
    def __init__(self, model, processor, max_length=100, output_dir="attention_plots"):
        self.model = model
        self.processor = processor
        self.max_length = max_length
        self.attention_tracker = CrossAttentionTracker(model)
        self.swin_tracker = SwinAttentionTracker(model)
        self.model.eval()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.cached_results = {
            "image": None,
            "outputs": None,
            "attention_weights": None,
            "tokens": [],
            "generated_text": None,
            "feature_size": None,
            "swin_attentions": None,
        }

    def generate_and_cache(self, image):
        """Generate text and cache results for later visualization"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.model.device)

            # Clear previous Swin attention patterns
            self.swin_tracker.swin_attentions = []

            # removed this because I'm seeing that there is a possible duplication here
            # First, run encoder only to get Swin attention patterns
            # encoder_outputs = self.model.encoder(pixel_values,/ output_attentions=True)

            generation_kwargs = {
                "pixel_values": pixel_values,
                "max_length": self.max_length,
                "num_beams": 1,
                "do_sample": False,
            }

            print("Generating text and collecting attention weights...")
            outputs, all_attention_weights = self.attention_tracker.collect_attentions(
                **generation_kwargs
            )

            generated_text = self.processor.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
            tokens = self.processor.tokenizer.convert_ids_to_tokens(
                outputs.sequences[0]
            )

            # Calculate feature size from the attention weights shape
            feature_size = int(np.sqrt(all_attention_weights[0][0].shape[-1]))
            print(f"Feature size calculated: {feature_size}")

            self.cached_results = {
                "image": image,
                "outputs": outputs,
                "attention_weights": all_attention_weights,
                "tokens": tokens,
                "generated_text": generated_text,
                "feature_size": feature_size,
                "swin_attentions": self.swin_tracker.swin_attentions,
            }

            return self.cached_results

    # def visualize_swin_attention(self) -> Tuple[plt.Figure, str]:
    #     """Visualize attention patterns within Swin Transformer blocks"""
    #     if not self.cached_results["swin_attentions"]:
    #         raise ValueError(
    #             "No Swin attention weights available. Run generate_and_cache first."
    #         )

    #     # Get original image size
    #     image_height, image_width = self.cached_results["image"].shape[:2]

    #     # Generate visualization using the new method
    #     fig, save_path = self.swin_tracker.visualize_attention_maps(
    #         image_size=(image_height, image_width), output_dir=self.output_dir
    #     )

    #     return fig, save_path

    def aggregate_attention_heads(
        self, attention_weights: torch.Tensor, method: str = "mean"
    ) -> torch.Tensor:
        """Aggregate attention weights across heads using different methods"""
        if method == "mean":
            return torch.mean(attention_weights, dim=1)
        elif method == "max":
            return torch.max(attention_weights, dim=1)[0]
        elif method == "weighted":
            # Compute head importance scores based on attention entropy
            entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(-1)
            weights = torch.softmax(-entropy, dim=1)  # Lower entropy -> higher weight
            return (attention_weights * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def analyze_head_specialization(self) -> Dict[str, List[float]]:
        """Analyze the specialization of different attention heads"""
        if self.cached_results["attention_weights"] is None:
            raise ValueError(
                "No attention weights available. Run generate_and_cache first."
            )

        head_metrics = {"entropy": [], "sparsity": [], "spatial_focus": []}

        for layer_idx in range(len(self.cached_results["attention_weights"][0])):
            layer_attention = self.cached_results["attention_weights"][0][layer_idx]

            for head_idx in range(layer_attention.shape[1]):
                head_attention = layer_attention[0, head_idx].numpy()

                # Calculate entropy
                entropy = (
                    -(head_attention * np.log(head_attention + 1e-9)).sum(-1).mean()
                )
                head_metrics["entropy"].append(entropy)

                # Calculate sparsity (% of attention weights above threshold)
                sparsity = (head_attention > 0.1).mean()
                head_metrics["sparsity"].append(sparsity)

                # Calculate spatial focus (variance of attention distribution)
                spatial_focus = np.var(head_attention)
                head_metrics["spatial_focus"].append(spatial_focus)

        return head_metrics

    def visualize_token_attention(self, image=None, layer_idx=0, head_idx=0):
        """Visualize cross-attention for each generated token"""
        if image is not None and (
            self.cached_results["image"] is None
            or not np.array_equal(image, self.cached_results["image"])
        ):
            self.generate_and_cache(image)
        elif self.cached_results["image"] is None:
            raise ValueError("No image provided and no cached results available")

        attention_weights = self._process_attention_weights(
            self.cached_results["attention_weights"], layer_idx, head_idx
        )
        print(
            f"Aggregated attention shape from the normal visualization: {attention_weights.shape}"
        )
        print("Creating visualization...")
        fig = self._create_visualization(
            self.cached_results["image"],
            attention_weights,
            self.cached_results["tokens"],
            self.cached_results["feature_size"],
            self.cached_results["generated_text"],
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"attention_plot_{timestamp}.png")
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, save_path

    def visualize_aggregate_attention(
        self, aggregation_method: str = "weighted"
    ) -> Tuple[plt.Figure, str]:
        """Visualize aggregated attention across all heads"""
        print(
            "type of the cached attention weights: ",
            type(self.cached_results["attention_weights"]),
        )
        print(
            f'Length of the cached attention weights list: {len(self.cached_results["attention_weights"])}'
        )
        print("Cached attention weights: ", self.cached_results["attention_weights"])

        # attention_weights = torch.tensor(self.cached_results["attention_weights"])
        attention_weights_list = self.cached_results["attention_weights"]
        # Check the data type of the first element
        first_element = attention_weights_list[0]
        print(f"Type of the first element: {type(first_element)}")
        if isinstance(first_element, torch.Tensor):
            # If the elements are already PyTorch tensors, stack them directly
            attention_weights = torch.stack(attention_weights_list, dim=0)
        else:
            # If the elements are not PyTorch tensors, convert them
            attention_weights = torch.stack(
                [torch.tensor(step[0]) for step in attention_weights_list], dim=0
            )
            # attention_weights = torch.stack([torch.tensor(step) for step in attention_weights_list], dim=0)

        # attention_weights = torch.stack([torch.tensor(step) for step in attention_weights_list], dim=0)
        aggregated_attention = self.aggregate_attention_heads(
            attention_weights, aggregation_method
        )
        print(
            f"Aggregated attention shape from the aggregated visualization: {aggregated_attention.shape}"
        )

        # Create visualization using aggregated attention
        fig = self._create_visualization(
            self.cached_results["image"],
            aggregated_attention.numpy(),
            self.cached_results["tokens"],
            self.cached_results["feature_size"],
            self.cached_results["generated_text"],
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.output_dir, f"aggregate_attention_{aggregation_method}_{timestamp}.png"
        )
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, save_path

    def _process_attention_weights(self, all_attention_weights, layer_idx, head_idx):
        """Process collected attention weights for a specific layer and head"""
        attention_weights = []

        for step_attentions in all_attention_weights:
            layer_attention = step_attentions[layer_idx]
            head_attention = layer_attention[0, head_idx].numpy()
            attention_weights.append(head_attention)

        return np.stack(attention_weights)

    def _create_visualization(
        self, image, attention_weights, tokens, feature_size, generated_text
    ):
        """Create visualization grid of attention maps"""
        num_tokens = len(tokens)
        num_cols = 5
        num_rows = math.ceil(num_tokens / num_cols)

        fig = plt.figure(figsize=(4 * num_cols, 3 * num_rows))
        plt.suptitle(
            f"Generated Text:\n{self._wrap_text(generated_text)}", fontsize=12, y=0.98
        )

        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.3, hspace=0.5)

        image_height, image_width = image.shape[:2]

        for idx in range(num_tokens):
            if idx == 0:  # Skip first token (usually BOS token)
                continue

            # Get attention weights for current token
            token_attention = attention_weights[idx - 1]

            # Reshape attention weights to square feature map
            token_attention = token_attention.reshape(feature_size, feature_size)

            # Upsample attention map to match image dimensions
            upsampled_attention = self._upsample_attention_map(
                token_attention, (image_height, image_width)
            )

            ax = plt.subplot(gs[idx // num_cols, idx % num_cols])

            # Plot image and attention overlay
            ax.imshow(image)
            attention_mask = ax.imshow(
                upsampled_attention, cmap="magma", alpha=0.5, interpolation="bilinear"
            )
            print(
                f"[INFO]: Added the colorbar from inside the enhanced_attention_tracker"
            )
            plt.colorbar(attention_mask, ax=ax, fraction=0.046, pad=0.04)

            # Add token information
            token_display = tokens[idx].replace("Ġ", " ")
            ax.set_title(f'[{idx}] "{token_display}"', fontsize=8, pad=5)

            # Add attention strength
            avg_attention = np.mean(token_attention)
            ax.text(
                0.02,
                0.98,
                f"Att: {avg_attention:.3f}",
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.7),
            )

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _upsample_attention_map(self, attention_map, target_shape):
        """Upsample attention map to match image dimensions"""
        zoom_h = target_shape[0] / attention_map.shape[0]
        zoom_w = target_shape[1] / attention_map.shape[1]

        upsampled = zoom(attention_map, (zoom_h, zoom_w), order=3)
        upsampled = (upsampled - upsampled.min()) / (
            upsampled.max() - upsampled.min() + 1e-8
        )

        return upsampled

    def _wrap_text(self, text, width=100):
        """Wrap text to specified width"""
        return "\n".join(textwrap.wrap(text, width=width))
