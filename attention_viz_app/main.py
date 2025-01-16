from gradio_app import create_gradio_interface
from load_donut_model import donut_test
from attention_trackers.attention_tracker import AttentionVisualizer

model = donut_test.model
processor = donut_test.processor
visualizer = AttentionVisualizer(model, processor)

demo = create_gradio_interface(visualizer)
demo.launch(debug=True)
