import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
import math
from typing import List, Dict, Tuple, Any
import os
from datetime import datetime
import textwrap
import gradio as gr
import sys


from gradio_app import create_gradio_interface
from load_donut_model import donut_test


model = donut_test.model
processor = donut_test.processor
demo = create_gradio_interface(model, processor)
demo.launch(debug=True)
# TODO: need to fix the swin attention visualization (still not working). And I need to configure the z44oom functionality properly
# for now It zooms the original image without the attention overlayed on top of it, probably something that has to do with
# how I'm saving the figure. It's saving the first image only.
