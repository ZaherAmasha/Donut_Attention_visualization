import gradio as gr
import matplotlib.pyplot as plt
import math
import html
import re

from attention_trackers.enhanced_attention_tracker import EnhancedAttentionVisualizer
from attention_viz_app.zoom_pan_image_functionality import (
    attention_visualizer,
    zoom_pan_image,
)

# We have 4 decoder layers


# to show which swin stage block we are at
def signify_which_swin_stage_is_selected(layer_idx):
    html_output = """
    <strong>Current Swin Block:</strong>
    <div style="display: flex; gap: 10px; font-size: 1.5em;">
        <div style="width: 200px; height: 40px;
                    background-color: {color1}; text-align: center; line-height: 40px;">Swin Stage 1</div>
        <div style="width: 200px; height: 40px;
                    background-color: {color2}; text-align: center; line-height: 40px;">Swin Stage 2</div>
        <div style="width: 200px; height: 40px;
                    background-color: {color3}; text-align: center; line-height: 40px;">Swin Stage 3</div>
        <div style="width: 200px; height: 40px;
                    background-color: {color4}; text-align: center; line-height: 40px;">Swin Stage 4</div>
    </div>
    """.format(
        color1="green" if layer_idx >= 1 and layer_idx <= 2 else "gray",
        color2="green" if layer_idx > 2 and layer_idx <= 4 else "gray",
        color3="green" if layer_idx > 4 and layer_idx <= 18 else "gray",
        color4="green" if layer_idx > 18 and layer_idx <= 20 else "gray",
    )
    return html_output


def create_gradio_interface(model, processor):
    visualizer = EnhancedAttentionVisualizer(model, processor)

    def initial_process(image, max_length):
        """Initial processing of image - generates and caches results"""
        visualizer.max_length = max_length
        cached_results = visualizer.generate_and_cache(image)
        return cached_results["generated_text"], gr.Slider(
            maximum=len(cached_results["tokens"]) - 1
        )  # -1 to exclude BOS token

    def create_highlighted_text(token_idx):
        """Create HTML with highlighted token"""
        if (
            visualizer.cached_results is None
            or "tokens" not in visualizer.cached_results
        ):
            return ""

        tokens = visualizer.cached_results["tokens"][1:]  # Skip BOS token
        words = []
        # print(f'These are the tokens to be highlighted: {len(tokens)}')
        for i, token in enumerate(tokens):
            # Clean up token
            # Remove underscore prefix only if followed by a number
            if re.match(r"^▁\d", token):
                token = token[1:]

            cleaned_token = html.escape(token)
            # Highlight the selected token
            if i == token_idx:
                word = f'<span style="background-color: #ffd700; padding: 0.2em 0.4em; border-radius: 0.2em; font-weight: bold;">{cleaned_token}</span>'
            else:
                word = cleaned_token
            words.append(word)

        return "".join(words)

    def visualize_cross_attention(token_idx, layer_idx, head_idx):
        """Generate cross-attention visualization for specific token, layer and head"""
        if visualizer.cached_results["image"] is None:
            return None

        token_attention = visualizer._process_attention_weights(
            visualizer.cached_results["attention_weights"], layer_idx, head_idx
        )

        # Get attention for specific token
        token_attention = token_attention[token_idx]

        # Create visualization for single token
        fig = plt.figure(figsize=(8, 6))

        # Get image and feature dimensions
        image = visualizer.cached_results["image"]
        image_height, image_width = image.shape[:2]
        feature_size = visualizer.cached_results["feature_size"]

        # Reshape and upsample attention
        token_attention = token_attention.reshape(feature_size, feature_size)
        upsampled_attention = visualizer._upsample_attention_map(
            token_attention, (image_height, image_width)
        )

        # Plot
        plt.imshow(image)
        attention_mask = plt.imshow(upsampled_attention, cmap="magma", alpha=0.5)
        plt.colorbar(attention_mask)

        token = visualizer.cached_results["tokens"][
            token_idx + 1
        ]  # +1 because we skipped BOS
        plt.title(f'Attention for token: "{token}"')
        plt.axis("off")

        # Storing the image of the figure for the zoom and pan functionality later on
        attention_visualizer.store_original_plot(fig)

        # Create highlighted text
        highlighted_text = create_highlighted_text(token_idx)

        return fig, highlighted_text

    def visualize_swin_attention(layer_idx, head_idx):
        """Generate Swin attention visualization for specific layer and head"""
        if visualizer.cached_results["image"] is None:
            return None

        image_height, image_width = visualizer.cached_results["image"].shape[:2]

        # Create figure for single layer/head combination
        fig = plt.figure(figsize=(8, 6))
        print(f"layer index: {layer_idx}")
        if layer_idx == "":
            layer_idx = 1
        layer_idx = int(layer_idx)
        attention_weights = visualizer.swin_tracker.swin_attentions[layer_idx]

        # Extract specific head's attention
        head_attention = attention_weights[0, head_idx].numpy()

        # Process attention weights (using existing _reconstruct_attention_map method)
        window_size = int(math.sqrt(head_attention.shape[-1]))
        windows_per_row = int(math.sqrt(attention_weights.shape[0]))

        full_attention = visualizer.swin_tracker._reconstruct_attention_map(
            attention_weights[0, head_idx],
            window_size,
            windows_per_row,
            (image_height, image_width),
        )

        plt.imshow(full_attention, cmap="viridis")
        plt.colorbar()
        plt.title(f"Swin Attention - Layer {layer_idx}, Head {head_idx}")
        plt.axis("off")

        return fig, ""

    # Create Gradio interface
    with gr.Blocks() as demo:
        with gr.Row():
            # Input column
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Input Image")

            # Visualization controls column
            with gr.Column():
                attention_type = gr.Radio(
                    choices=["Cross Attention", "Swin Attention"],
                    label="Attention Type",
                    value="Cross Attention",
                )

                # Common controls
                layer_slider = gr.Slider(
                    minimum=1, maximum=4, step=1, value=0, label="Layer Index"
                )
                head_slider = gr.Slider(
                    minimum=1, maximum=16, step=1, value=0, label="Head Index"
                )
                length_slider = gr.Slider(
                    minimum=10,
                    maximum=2000,
                    step=10,
                    value=100,
                    label="Max Sequence Length",
                )
                process_btn = gr.Button("Process Image")
                current_swin_block_display = gr.HTML(
                    label="Current Swin Block"
                )  # label is not showing for some reason
                generated_text = gr.Textbox(label="Generated Text")
                number_of_generated_tokens = gr.Textbox(
                    label="Number of Generated Tokens"
                )

                # Cross-attention specific controls
                with gr.Group() as cross_attention_controls:
                    token_slider = gr.Slider(
                        minimum=1,
                        maximum=100,  # Will be updated after processing
                        step=1,
                        value=1,
                        label="Token Index",
                    )
                    with gr.Row():
                        previous_token_button = gr.Button(value="Previous token")
                        next_token_button = gr.Button(value="Next token")

        # Text display with highlighted token
        with gr.Row():
            highlighted_text = gr.HTML(
                label="Generated Text (Selected token is highlighted)"
            )

        # Visualization output
        with gr.Row():
            attention_plot = gr.Plot(label="Attention Visualization")
        with gr.Group() as grp:
            zoom_in_out_slider = gr.Slider(
                minimum=0.25, maximum=4, value=1, label="Zoom"
            )
            horizontol_move_along_image_slider = gr.Slider(
                minimum=0, maximum=1, value=0.5, label="Horizontal Pan"
            )
            vertical_move_along_image_slider = gr.Slider(
                minimum=0, maximum=1, value=0.5, label="Vertical Pan (right goes down)"
            )

            reset_zoom_button = gr.Button(value="Reset Zoom")

            mini_map_output = gr.Image(label="Mini Map")

        visualizer.cached_results["tokens"] = []

        def get_token_count():
            """Retrieve and return the number of tokens."""
            return gr.Textbox(value=len(visualizer.cached_results["tokens"]))

        # Process image and update token slider range
        process_btn.click(
            fn=initial_process,
            inputs=[image_input, length_slider],
            outputs=[generated_text, token_slider],
        ).then(
            fn=get_token_count, inputs=[], outputs=[number_of_generated_tokens]
        ).then(
            fn=lambda: gr.Slider(value=4), inputs=[], outputs=[token_slider]
        )

        # Update visualization based on controls
        def update_visualization(attention_choice, token_idx, layer_idx, head_idx):
            if attention_choice == "Cross Attention":

                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        gr.Slider(minimum=1, maximum=4),
                        gr.Slider(minimum=1, maximum=16),
                        "",
                    )

                fig, html = visualize_cross_attention(
                    token_idx, layer_idx - 1, head_idx - 1
                )  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                return (
                    fig,
                    html,
                    gr.Slider(minimum=1, maximum=4),
                    gr.Slider(minimum=1, maximum=16),
                    "",
                )
            else:
                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        gr.Slider(minimum=1, maximum=20),
                        gr.Slider(minimum=1, maximum=32),
                        signify_which_swin_stage_is_selected(1),
                    )
                return (
                    *visualize_swin_attention(
                        layer_idx - 1, head_idx - 1
                    ),  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                    gr.Slider(
                        minimum=1, maximum=20
                    ),  # layers for the swin transformer here are [2,2,14,2] with a total of 20
                    gr.Slider(minimum=1, maximum=32),  # head count is [4, 8, 16, 32]
                    signify_which_swin_stage_is_selected(layer_idx),
                )

        attention_type.change(
            fn=update_visualization,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                layer_slider,
                head_slider,
                current_swin_block_display,
            ],
        ).then(
            fn=lambda: (gr.Slider(value=1), gr.Slider(value=1)),
            inputs=[],
            outputs=[layer_slider, head_slider],
        )

        token_slider.change(
            fn=update_visualization,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                layer_slider,
                head_slider,
                current_swin_block_display,
            ],
        )
        layer_slider.change(
            fn=update_visualization,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                layer_slider,
                head_slider,
                current_swin_block_display,
            ],
        ).then(
            fn=lambda attention_type_input, layer_idx: (
                signify_which_swin_stage_is_selected(
                    layer_idx
                )  # setting the selection to be on the first stage as this would be the same as the default selection of the layer slider
                if attention_type_input == "Swin Attention"
                else None
            ),
            inputs=[attention_type, layer_slider],
            outputs=[current_swin_block_display],
        )

        head_slider.change(
            fn=update_visualization,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                layer_slider,
                head_slider,
                current_swin_block_display,
            ],
        )
        previous_token_button.click(
            fn=lambda x: gr.Slider(value=x - 1),
            inputs=[token_slider],
            outputs=[token_slider],
        )
        next_token_button.click(
            fn=lambda x: gr.Slider(value=x + 1),
            inputs=[token_slider],
            outputs=[token_slider],
        )

        # Zoom and Pan Function Trigger
        gr.on(
            triggers=[
                zoom_in_out_slider.change,
                vertical_move_along_image_slider.change,
                horizontol_move_along_image_slider.change,
            ],
            fn=zoom_pan_image,
            inputs=[
                zoom_in_out_slider,
                horizontol_move_along_image_slider,
                vertical_move_along_image_slider,
            ],
            outputs=[attention_plot, mini_map_output],
        )

        # Reset Zoom Functionality
        reset_zoom_button.click(
            fn=lambda: [1, 0.5, 0.5],
            outputs=[
                zoom_in_out_slider,
                horizontol_move_along_image_slider,
                vertical_move_along_image_slider,
            ],
        )
    return demo
