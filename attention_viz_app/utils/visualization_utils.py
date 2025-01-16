import matplotlib.pyplot as plt
import re
import html
import gradio as gr

from attention_trackers.attention_tracker import AttentionVisualizer

# from main import visualizer
from utils.zoom_pan_image_functionality import zoom_pan_image_tracker
from utils.gradio_utils import signify_which_swin_stage_is_selected


def create_highlighted_text(visualizer: AttentionVisualizer, token_idx):
    """Create HTML with highlighted token"""
    if visualizer.cached_results is None or "tokens" not in visualizer.cached_results:
        return ""

    tokens = visualizer.cached_results["tokens"][1:]  # Skip BOS token
    words = []
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


def visualize_cross_attention(
    visualizer: AttentionVisualizer,
    zoom_pan_image_tracker,
    token_idx,
    layer_idx,
    head_idx,
):
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
    zoom_pan_image_tracker.store_original_plot(fig)

    # Create highlighted text
    highlighted_text = create_highlighted_text(visualizer, token_idx)

    return fig, highlighted_text


def visualize_swin_attention(visualizer: AttentionVisualizer, layer_idx, head_idx):
    """Generate Swin attention visualization for specific layer and head"""
    if visualizer.cached_results["image"] is None:
        return None

    # Create figure for single layer/head combination
    fig = plt.figure(figsize=(8, 6))
    if layer_idx == "":
        layer_idx = 1
    layer_idx = int(layer_idx)
    attention_weights = visualizer.swin_tracker.swin_attentions[layer_idx]
    print(
        f"[INFO]: This is the original swin attention weights matrix for layer {layer_idx} and head {head_idx}: {len(visualizer.swin_tracker.swin_attentions)}"
    )

    # Process attention weights (using existing _reconstruct_attention_map method)
    print(
        f"This is the full attention matrix for layer {layer_idx+1} and head {head_idx+1}: {attention_weights.shape}"
    )

    full_attention = visualizer.swin_tracker._reconstruct_attention_map(
        attention_weights[:, head_idx],
    )

    print(
        f"This is the full swin attention matrix for layer {layer_idx+1} and head {head_idx+1}: {full_attention.shape}"
    )

    plt.imshow(full_attention, cmap="viridis")
    plt.colorbar()
    plt.title(f"Swin Attention - Layer {layer_idx+1}, Head {head_idx+1}")
    plt.axis("off")

    return fig, ""


def update_visualization_on_attention_type_change(
    visualizer: AttentionVisualizer,
    attention_choice,
    token_idx,
    layer_idx,
    head_idx,
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
                visualizer, layer_idx - 1, head_idx - 1
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
def update_visualization_on_layer_and_head_sliders_change(
    visualizer: AttentionVisualizer,
    attention_choice,
    token_idx,
    layer_idx,
    head_idx,
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
                visualizer, layer_idx - 1, head_idx - 1
            ),  # the -1 is because the numpy arrays of the attention matrices are 0 indexed
            # layers for the swin transformer here are [2,2,14,2] with a total of 20
            # head count is [4, 8, 16, 32]
            signify_which_swin_stage_is_selected(layer_idx),
        )
