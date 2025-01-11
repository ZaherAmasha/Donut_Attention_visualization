import matplotlib.pyplot as plt
import math
import torch
from typing import Tuple
import matplotlib.gridspec as gridspec
import os
from datetime import datetime


class SwinAttentionTracker:
    def __init__(self, model):
        self.model = model
        self.swin_attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            # if hasattr(module, "attn") and hasattr(module.attn, "softmax_output"):
            #     # Some implementations store the attention weights directly
            #     attention_weights = module.attn.softmax_output.detach().cpu()
            #     self.swin_attentions.append(attention_weights)
            #     print("entered the attn block")
            # elif hasattr(module, "attentions"):
            #     # Alternative implementation where weights are stored in 'attentions'
            #     attention_weights = module.attentions[-1].detach().cpu()
            #     print("entered the attentions block")
            #     self.swin_attentions.append(attention_weights)
            # elif hasattr(module, "attn") and hasattr(module.attn, "_attention_weights"):
            #     # Another common implementation pattern
            #     attention_weights = module.attn._attention_weights.detach().cpu()
            #     self.swin_attentions.append(attention_weights)
            #     print("entered the attn and _attention_weights block")
            # else:
            if output[0] is not None:
                # it should be output[1] to extract the attentions, but we have to set output_attentions=True in the model() method
                self.swin_attentions.append(output[1].detach().cpu())
                print(
                    f"Swin attention extracted size: {output[1].detach().cpu().shape}"
                )
                # print(f'Swin attention extracted: {output[0].detach().cpu()}')
            print(
                "didn't enter the attentions attributes blocks so it should've entered the donutswinlayer"
            )
            return output

        # Register hooks for all window attention modules in the Swin Transformer
        def find_attention_modules(model):
            attention_modules = []
            for name, module in model.named_modules():
                # Common class names for Swin attention modules
                if any(
                    attention_type in module.__class__.__name__.lower()
                    for attention_type in ["donutswinlayer"]
                ):
                    attention_modules.append(module)
            return attention_modules

        # Register hooks for all identified attention modules
        attention_modules = find_attention_modules(self.model.encoder)
        for module in attention_modules:
            self.hooks.append(module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.swin_attentions = []

    def get_attention_maps(self):
        """
        Process and return the collected attention maps in a standardized format

        Returns:
            List[torch.Tensor]: List of attention weight tensors, one per layer
        """
        processed_attentions = []
        counter = 0
        for attention in self.swin_attentions:
            print(f"shape of swin attention weights: {attention.shape}")
            counter += 1
            # Ensure attention weights are in the expected shape [batch, num_heads, seq_len, seq_len]
            if len(attention.shape) == 4:
                processed_attentions.append(attention)
            elif len(attention.shape) == 3:
                # Add batch dimension if missing
                processed_attentions.append(attention.unsqueeze(0))
            else:
                # Reshape if necessary (e.g., for window attention)
                batch_size = 1  # Assuming single image processing
                num_windows = attention.shape[0]
                num_heads = attention.shape[1] if len(attention.shape) > 2 else 1
                window_size = int(math.sqrt(attention.shape[-1]))

                reshaped_attention = attention.view(
                    batch_size,
                    num_windows,
                    num_heads,
                    window_size * window_size,
                    window_size * window_size,
                )
                processed_attentions.append(reshaped_attention)
        print(f"Printed the shape of attention weights for: {counter} times")
        return processed_attentions

    def visualize_attention_maps(
        self, image_size: Tuple[int, int], output_dir: str = "attention_plots"
    ) -> Tuple[plt.Figure, str]:
        """
        Visualize Swin Transformer attention patterns

        Args:
            image_size: Tuple of (height, width) of the original image
            output_dir: Directory to save the visualization

        Returns:
            Tuple of (matplotlib figure, save path)
        """
        attention_maps = self.get_attention_maps()
        if not attention_maps:
            raise ValueError("No attention maps collected. Run forward pass first.")

        num_layers = 20  # len(attention_maps)
        # num_heads = attention_maps[0].shape[2]  # Assuming all layers have same number of heads
        # num_heads = attention_maps[0].shape[1]  # Assuming all layers have same number of heads
        num_heads = 32  # 15 #14
        # Create grid for visualization
        fig = plt.figure(figsize=(20, 4 * num_layers))
        gs = gridspec.GridSpec(num_layers, num_heads + 1)  # +1 for averaged attention
        gs.update(wspace=0.3, hspace=0.4)

        for layer_idx, layer_attention in enumerate(attention_maps):
            print(f"INFO - Layer index: {layer_idx}")
            # The layers have different number of heads so we need to define this
            num_heads = layer_attention.shape[1]

            # Process window attention into image-sized attention map
            layer_attention = layer_attention.squeeze(0)  # Remove batch dimension

            # Calculate window layout
            num_windows = layer_attention.shape[0]
            window_size = int(math.sqrt(layer_attention.shape[-1]))
            windows_per_row = int(math.sqrt(num_windows))

            # Create subplot for averaged attention (across all heads)
            ax_avg = plt.subplot(gs[layer_idx, 0])
            avg_attention = layer_attention.mean(dim=1)  # Average across heads
            self._plot_attention_map(
                ax_avg,
                avg_attention,
                window_size,
                windows_per_row,
                image_size,
                title=f"Layer {layer_idx}\nAverage",
            )

            # Plot individual head attentions
            print(
                f"Number of heads from the visualize swin attention maps: {num_heads}"
            )
            for head_idx in range(num_heads):
                ax = plt.subplot(gs[layer_idx, head_idx - 1])
                # ax = plt.subplot(gs[layer_idx, head_idx + 1])
                head_attention = layer_attention[:, head_idx]
                self._plot_attention_map(
                    ax,
                    head_attention,
                    window_size,
                    windows_per_row,
                    image_size,
                    title=f"Layer {layer_idx}\nHead {head_idx}",
                )

        plt.tight_layout()

        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"swin_attention_{timestamp}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, save_path

    def _plot_attention_map(
        self, ax, attention_weights, window_size, windows_per_row, image_size, title=""
    ):
        """Helper method to plot a single attention map"""
        # Reshape windows into 2D grid
        full_attention = self._reconstruct_attention_map(
            attention_weights, window_size, windows_per_row, image_size
        )
        print(f"[INFO]: Added the colorbar from inside the SwinAttentionTracker")
        # Plot attention map
        im = ax.imshow(full_attention, cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    def _reconstruct_attention_map(
        self, attention_weights, window_size, windows_per_row, image_size
    ):
        """
        Reconstruct full attention map from window attention patterns
        """
        num_windows = attention_weights.shape[0]
        h, w = image_size

        # Calculate padded image size (multiple of window_size)
        h_padded = math.ceil(h / window_size) * window_size
        w_padded = math.ceil(w / window_size) * window_size

        # Initialize full attention map
        full_attention = torch.zeros(h_padded, w_padded)

        # Place each window's attention in the correct position
        for window_idx in range(num_windows):
            # Calculate window position
            row = (window_idx // windows_per_row) * window_size
            col = (window_idx % windows_per_row) * window_size

            # Get window attention (average across sequence length)
            # print(
            #     f"This is the window attention before the mean: {attention_weights[window_idx]}"
            # )
            # print(
            #     f"This is the size of the window attention before the mean: {attention_weights[window_idx].shape}"
            # )
            window_attention = attention_weights[window_idx]  # .mean(-1)
            # print(
            #     f"This is the size of the window attention after the mean: {window_attention.shape}"
            # )
            # print(f"This is the window attention after the mean: {window_attention}")
            window_attention = window_attention.view(window_size, window_size)

            # Place window in full attention map
            full_attention[row : row + window_size, col : col + window_size] = (
                window_attention
            )

        # Crop to original image size
        full_attention = full_attention[:h, :w]
        return full_attention.numpy()
