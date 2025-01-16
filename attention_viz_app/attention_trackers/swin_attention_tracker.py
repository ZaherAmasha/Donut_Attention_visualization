class SwinAttentionTracker:
    def __init__(self, model):
        self.model = model
        self.swin_attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if output[0] is not None:
                # it should be output[1] to extract the attentions, but we have to set output_attentions=True in the model() method
                self.swin_attentions.append(output[1].detach().cpu())
                # example Swin attention extracted size: torch.Size([1444, 4, 100, 100]), 1444 is the number of windows in the image
                # (38 by 38) this is true if a single image is passed, the 1444 would become 2888 if two images were passed, 4
                # is the number of heads, 100 is the number of patches per window and multiplying the key and query matrices we get
                # a 100 by 100 matrix. The 100 by 100 matrix represents how each patch is paying attention to every other patch in
                # the specific window.
                print(
                    f"Swin attention extracted size: {output[1].detach().cpu().shape}"
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

    def _reconstruct_attention_map(self, attention_weights):
        """
        Reconstruct full attention map from window attention patterns
        input shape of attention map for a specific head of a specific layer:
            [num_windows_for_entire_image, num_patches_per_window, num_patches_per_window]
        output shape to be displayed in a figure:
            [num_patches_per_row, num_patches_per_col]
        """

        # Get the max attention on the last column of the 100 by 100 matrix (100 being the number of patches per window)
        # This column represents the attention between different image patches inside a particular window. I'm taking the
        # max along the same column that is normalized by a softmax to sum to one. Taking the max here may not be the most
        # optimal/accurate way of doing this, I could be wrong. The shape becomes [num_windows, heads (in this case 1 since
        # we are specifying it earlier), num_patches_per_window (which is 100)]
        attention_weights, _ = attention_weights.max(axis=-1)
        print(
            f"This is the full attention weights matrix shape after getting the max along the last row: {attention_weights.shape}"
        )
        # This is the attention for a single head because we are specifying which head to choose earlier
        num_windows, num_patches = attention_weights.shape

        # Reshape the attention_weights to unravel the num_patches per window so that (,100) becomes (, 10, 10)
        attention_weights = attention_weights.reshape(
            num_windows, int(num_patches**0.5), int(num_patches**0.5)
        )
        print(
            f"This is the full attention weights matrix shape after reshaping the num_patches dimension: {attention_weights.shape}"
        )
        num_windows_per_row = int(num_windows**0.5)
        h_patches_per_window = attention_weights.shape[-1]
        h_display = num_windows_per_row * h_patches_per_window
        h_windows_per_image = h_display // h_patches_per_window
        print(
            f"This is the number of windows per row (same as per column for square images): {num_windows_per_row}"
        )
        print(
            f"This is the height of the attention map image to be displayed (same as width for square images): {h_display}"
        )

        attention_weights = attention_weights.contiguous().view(
            h_windows_per_image,
            h_windows_per_image,
            h_patches_per_window,
            h_patches_per_window,
        )
        attention_weights = attention_weights.permute(0, 2, 1, 3)
        h = attention_weights.shape[0] * attention_weights.shape[1]
        attention_weights = attention_weights.contiguous().view(h, h)
        print(
            f"This is the final shape of the attention weights: {attention_weights.shape}"
        )
        return attention_weights.numpy()
