import gradio as gr
import matplotlib.pyplot as plt
import math

from utils.zoom_pan_image_functionality import (
    zoom_pan_image_tracker,
    zoom_pan_image,
)
from attention_trackers.enhanced_attention_tracker import EnhancedAttentionVisualizer

from utils.gradio_utils import (
    adjust_head_slider_according_to_the_current_layer,
    signify_which_swin_stage_is_selected,
)
from utils.visualization_utils import visualize_cross_attention

# We have 4 decoder layers


def create_gradio_interface(model, processor):

    visualizer = EnhancedAttentionVisualizer(model, processor)

    def initial_process(image, max_length):
        """Initial processing of image - generates and caches results"""
        visualizer.max_length = max_length
        cached_results = visualizer.generate_and_cache(image)
        return cached_results["generated_text"], gr.Slider(
            maximum=len(cached_results["tokens"]) - 1
        )  # -1 to exclude BOS token

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
        # TODO: continue debugging from here, see how the attentions are extracted for the different layers and heads
        # and check if the layer and head indices are off by 1 or not, because I say earlier that sometimes they are set to -1 for some reason
        # print(
        #     f"This is the original swin attention weights matrix for layer {layer_idx} and head {head_idx}: {visualizer.swin_tracker.swin_attentions.shape}"
        # )

        # Extract specific head's attention
        head_attention = attention_weights[0, head_idx].numpy()

        # Process attention weights (using existing _reconstruct_attention_map method)
        window_size = int(math.sqrt(head_attention.shape[-1]))
        windows_per_row = int(math.sqrt(attention_weights.shape[0]))

        print(
            f"This is the swin attention weights matrix for layer {layer_idx} and head {head_idx}: {attention_weights[0, head_idx].shape}"
        )

        full_attention = visualizer.swin_tracker._reconstruct_attention_map(
            attention_weights[0, head_idx],
            window_size,
            windows_per_row,
            (image_height, image_width),
        )

        print(
            f"This is the full swin attention matrix for layer {layer_idx} and head {head_idx}: {full_attention.shape}"
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
        def update_visualization_on_attention_type_change(
            attention_choice, token_idx, layer_idx, head_idx
        ):
            if attention_choice == "Cross Attention":

                # clipping the index values from swin attentions tab's sliders so not to encounter an index error
                if layer_idx > 4:
                    layer_idx = 4
                if head_idx > 16:
                    head_idx = 16

                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        gr.Slider(minimum=1, maximum=4),
                        gr.Slider(minimum=1, maximum=16),
                        "",
                        # for the zoom and pan
                        gr.Slider(visible=True),
                        gr.Slider(visible=True),
                        gr.Slider(visible=True),
                        gr.Button(visible=True),
                        gr.Image(visible=True),
                        gr.Slider(visible=True),
                        gr.Button(visible=True),
                        gr.Button(visible=True),
                    )

                fig, html = visualize_cross_attention(
                    visualizer,
                    zoom_pan_image_tracker,
                    token_idx,
                    layer_idx - 1,
                    head_idx - 1,
                )  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                return (
                    fig,
                    html,
                    gr.Slider(minimum=1, maximum=4),
                    gr.Slider(minimum=1, maximum=16),
                    "",
                    # for the zoom and pan
                    gr.Slider(visible=True),
                    gr.Slider(visible=True),
                    gr.Slider(visible=True),
                    gr.Button(visible=True),
                    gr.Image(visible=True),
                    gr.Slider(visible=True),
                    gr.Button(visible=True),
                    gr.Button(visible=True),
                )
            else:
                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        gr.Slider(minimum=1, maximum=20),
                        gr.Slider(minimum=1, maximum=4),
                        signify_which_swin_stage_is_selected(layer_idx),
                        # for the zoom and pan
                        gr.Slider(visible=False),
                        gr.Slider(visible=False),
                        gr.Slider(visible=False),
                        gr.Button(visible=False),
                        gr.Image(visible=False),
                        gr.Slider(visible=False),
                        gr.Button(visible=False),
                        gr.Button(visible=False),
                    )
                return (
                    *visualize_swin_attention(
                        layer_idx - 1, head_idx - 1
                    ),  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                    gr.Slider(
                        minimum=1, maximum=20
                    ),  # layers for the swin transformer here are [2,2,14,2] with a total of 20
                    gr.Slider(minimum=1, maximum=4),  # head count is [4, 8, 16, 32]
                    signify_which_swin_stage_is_selected(layer_idx),
                    # for the zoom and pan
                    gr.Slider(visible=False),
                    gr.Slider(visible=False),
                    gr.Slider(visible=False),
                    gr.Button(visible=False),
                    gr.Image(visible=False),
                    gr.Slider(visible=False),
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                )

        # Update visualization based on controls
        def update_visualization_on_control_sliders_change(
            attention_choice, token_idx, layer_idx, head_idx
        ):
            if attention_choice == "Cross Attention":

                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        "",
                    )

                fig, html = visualize_cross_attention(
                    visualizer,
                    zoom_pan_image_tracker,
                    token_idx,
                    layer_idx - 1,
                    head_idx - 1,
                )  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                return (
                    fig,
                    html,
                    "",
                )
            else:
                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        signify_which_swin_stage_is_selected(layer_idx),
                    )
                return (
                    *visualize_swin_attention(
                        layer_idx - 1, head_idx - 1
                    ),  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                    signify_which_swin_stage_is_selected(layer_idx),
                )

        # Update visualization based on controls
        def update_visualization_on_layer_and_head_sliders_change(
            attention_choice, token_idx, layer_idx, head_idx
        ):
            if attention_choice == "Cross Attention":

                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        "",
                    )

                fig, html = visualize_cross_attention(
                    visualizer,
                    zoom_pan_image_tracker,
                    token_idx,
                    layer_idx - 1,
                    head_idx - 1,
                )  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                return (
                    fig,
                    html,
                    "",
                )
            else:
                # To be able to move freely error-free through the UI before uploading an input image
                if visualizer.cached_results["image"] is None:
                    return (
                        None,
                        "",
                        signify_which_swin_stage_is_selected(layer_idx),
                    )
                return (
                    *visualize_swin_attention(
                        layer_idx - 1, head_idx - 1
                    ),  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
                    # layers for the swin transformer here are [2,2,14,2] with a total of 20
                    # head count is [4, 8, 16, 32]
                    signify_which_swin_stage_is_selected(layer_idx),
                )

        attention_type.change(
            fn=update_visualization_on_attention_type_change,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                layer_slider,
                head_slider,
                current_swin_block_display,
                # for the zoom functionality
                zoom_in_out_slider,
                horizontol_move_along_image_slider,
                vertical_move_along_image_slider,
                reset_zoom_button,
                mini_map_output,
                # for the token slider, not needed for the Swin Attention viz
                token_slider,
                previous_token_button,
                next_token_button,
            ],
        ).then(
            fn=lambda: (gr.Slider(value=1), gr.Slider(value=1)),
            inputs=[],
            outputs=[layer_slider, head_slider],
        )

        token_slider.change(
            fn=update_visualization_on_control_sliders_change,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
                current_swin_block_display,
            ],
        )
        layer_slider.change(
            fn=update_visualization_on_layer_and_head_sliders_change,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
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
        ).then(  # to adjust the max of the head slider to match the specific layer that is selected by the layer slider
            fn=adjust_head_slider_according_to_the_current_layer,
            inputs=[layer_slider, head_slider, attention_type],
            outputs=[head_slider],
        )

        head_slider.change(
            fn=update_visualization_on_layer_and_head_sliders_change,
            inputs=[attention_type, token_slider, layer_slider, head_slider],
            outputs=[
                attention_plot,
                highlighted_text,
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
