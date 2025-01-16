# Donut Attention Visualization

A Python application that visualizes attention mechanisms in the Donut OCR model, specifically focusing on cross-attention between text tokens and image tokens, as well as internal Swin attention patterns. The visualization includes interactive heatmaps overlaid on input images, providing insights into how the model processes and understands document images.

## Video Demo
[Video Demo Coming Soon]

## Overview

This project provides tools to inspect and visualize how the Donut model attends to different parts of an image during text generation. It focuses on two main types of attention:

1. **Cross Attention**: Visualizes how text tokens attend to image patches during generation
2. **Swin Attention**: Shows the internal attention patterns in the Swin Transformer backbone

## Features

### Cross Attention Visualization

- Interactive selection of MBart decoder layers (3 layers) and attention heads (16 heads per layer)
- Token-by-token visualization of attention patterns
- Dynamic HTML overlay showing current token position in generated JSON output
- Interactive navigation through generated tokens using Previous/Next buttons
- Zoom and pan functionality for detailed inspection:
  - Adjustable zoom level
  - Vertical and horizontal pan controls
  - Reset zoom capability
  - Minimap showing current viewport position
- Attention map caching for efficient visualization of different settings

Key observation: Head 8 in the third layer of the MBart decoder shows the most focused attention patterns when generating text tokens.

### Swin Attention Visualization

- Layer and head selection across all four Swin stages
- Dynamic UI adaptation based on selected layer:
  - Stage 1: 2 blocks, 2 heads
  - Stage 2: 2 blocks, 8 heads
  - Stage 3: 14 blocks, 16 heads
  - Stage 4: 2 blocks, 32 heads
- Visual stage tracker showing current Swin stage (1-4)

## Project Structure

```
.
├── attention_trackers/
│   ├── attention_tracker.py
│   ├── enhanced_attention_tracker.py
│   └── swin_attention_tracker.py
├── utils/
│   ├── gradio_utils.py
│   ├── visualization_utils.py
│   └── zoom_pan_image_functionality.py
├── requirements.txt
├── .env
├── gradio_app.py
├── load_donut_model.py
└── main.py
```

## Implementation Details

The application uses register hooks to extract attention maps from the transformer model during image processing. These maps are cached for each image to enable efficient visualization with different settings without reprocessing.

The visualization interface is built using Gradio components, including:
- Radio buttons for attention type selection
- Layer and head selection controls
- Token navigation slider and buttons
- Zoom and pan controls with minimap
- HTML overlays for token tracking and stage visualization

## Roadmap

1. **Short-term Goals**
   - Add utility function for loading model from HuggingFace Hub
   - Implement AttenLRP for more accurate cross-attention relevancy scoring

2. **Future Enhancements**
   - Improve visualization performance
   - Add batch processing capabilities
   - Enhance user interface and controls

## Note to Users

This project is currently in development. To use this application, you'll need:
1. Access to a fine-tuned Donut model checkpoint
2. Example mindmap images for testing
3. Python environment with required dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the Apache-2.0 License. See LICENSE file for more information.

