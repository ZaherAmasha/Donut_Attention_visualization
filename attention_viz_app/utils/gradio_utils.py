import gradio as gr


def get_swin_stage_max_head_idx(layer_idx) -> dict:
    if 1 <= layer_idx <= 2:
        return {"maximum_head_idx": 4}
    elif 2 < layer_idx <= 4:
        return {"maximum_head_idx": 8}
    elif 4 < layer_idx <= 18:
        return {"maximum_head_idx": 16}
    elif 18 < layer_idx <= 20:
        return {"maximum_head_idx": 32}
    else:
        raise ValueError("Layer slider should be between 1 and 20 for Swin Layers")


def adjust_head_slider_according_to_the_current_layer(
    layer_idx, head_idx, attention_type
):
    print(
        f"Head idx from inside the adjust head slider according to current layer function: {head_idx}"
    )
    if attention_type == "Swin Attention":
        return gr.Slider(
            minimum=1,
            maximum=get_swin_stage_max_head_idx(layer_idx)["maximum_head_idx"],
            value=1,
        )

    else:
        return gr.Slider(minimum=1, maximum=16, value=head_idx)


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
