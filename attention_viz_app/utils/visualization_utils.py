import matplotlib.pyplot as plt
import re
import html


def create_highlighted_text(visualizer, token_idx):
    """Create HTML with highlighted token"""
    if visualizer.cached_results is None or "tokens" not in visualizer.cached_results:
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


def visualize_cross_attention(
    visualizer, zoom_pan_image_tracker, token_idx, layer_idx, head_idx
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
