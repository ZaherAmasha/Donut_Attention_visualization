class CrossAttentionTracker:
    def __init__(self, model):
        self.model = model
        self.collected_attentions = []
        self._original_forward = None

    def _attention_hook(self, module, input, output):
        if hasattr(output, "cross_attentions") and output.cross_attentions is not None:
            attention_weights = [
                attn.detach().cpu() for attn in output.cross_attentions
            ]
            self.collected_attentions.append(attention_weights)
        return output

    def collect_attentions(self, **generation_kwargs):
        self.collected_attentions = []
        original_forward = self.model.forward

        def modified_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            self._attention_hook(None, None, outputs)
            return outputs

        try:
            self.model.forward = modified_forward
            generation_kwargs.update(
                {
                    "output_attentions": True,
                    "output_hidden_states": True,
                    "return_dict_in_generate": True,
                }
            )

            outputs = self.model.generate(**generation_kwargs)

            all_attention_weights = []
            for step_attentions in self.collected_attentions:
                step_weights = []
                for layer_attentions in step_attentions:
                    step_weights.append(layer_attentions)
                all_attention_weights.append(step_weights)

            return outputs, all_attention_weights

        finally:
            self.model.forward = original_forward
