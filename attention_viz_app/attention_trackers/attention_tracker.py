import os
import torch
from scipy.ndimage import zoom
import numpy as np
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


class AttentionVisualizer:
    def __init__(self, model, processor, max_length=100, output_dir="attention_plots"):
        self.model = model
        self.processor = processor
        self.max_length = max_length
        self.cross_attention_tracker = CrossAttentionTracker(model)
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

            generation_kwargs = {
                "pixel_values": pixel_values,
                "max_length": self.max_length,
                "num_beams": 1,
                "do_sample": False,
            }

            print("Generating text and collecting attention weights...")
            outputs, all_attention_weights = (
                self.cross_attention_tracker.collect_attentions(**generation_kwargs)
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

    def _process_attention_weights(self, all_attention_weights, layer_idx, head_idx):
        """Process collected attention weights for a specific layer and head"""
        attention_weights = []

        for step_attentions in all_attention_weights:
            layer_attention = step_attentions[layer_idx]
            head_attention = layer_attention[0, head_idx].numpy()
            attention_weights.append(head_attention)

        return np.stack(attention_weights)

    def _upsample_attention_map(self, attention_map, target_shape):
        """Upsample attention map to match image dimensions"""
        zoom_h = target_shape[0] / attention_map.shape[0]
        zoom_w = target_shape[1] / attention_map.shape[1]

        upsampled = zoom(attention_map, (zoom_h, zoom_w), order=3)
        upsampled = (upsampled - upsampled.min()) / (
            upsampled.max() - upsampled.min() + 1e-8
        )

        return upsampled
