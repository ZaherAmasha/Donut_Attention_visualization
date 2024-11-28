from gradio_app import create_gradio_interface
from load_donut_model import donut_test


model = donut_test.model
processor = donut_test.processor
demo = create_gradio_interface(model, processor)
demo.launch(debug=True)

# TODO: need to fix the swin attention visualization (still not working). And I need to configure the z44oom functionality properly
# for now It zooms the original image without the attention overlayed on top of it, probably something that has to do with
# how I'm saving the figure. It's saving the first image only.
