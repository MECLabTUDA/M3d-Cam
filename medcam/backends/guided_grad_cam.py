import numpy as np
from medcam.backends.grad_cam import GradCAM
from medcam.backends.guided_backpropagation import GuidedBackPropagation
from medcam import medcam_utils
import torch


class GuidedGradCam():
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        self.model_GCAM = GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)
        self.model_GBP = GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph)

    def generate_attention_map(self, batch, label):
        """Handles the generation of the attention map from start to finish."""
        output, self.attention_map_GCAM, output_batch_size, output_channels, output_shape = self.model_GCAM.generate_attention_map(batch.clone(), label)
        #_, self.attention_map_GBP, _, _, _ = self.model_GBP.generate_attention_map(batch.clone(), label)
        self.attention_map_GBP = self._generate_gbp(batch, label)[""]
        #self.attention_map_GBP = self.attention_map_GBP[""]
        attention_map = self.generate()
        return output, attention_map, output_batch_size, output_channels, output_shape

    def get_registered_hooks(self):
        """Returns every hook that was able to register to a layer."""
        return self.model_GCAM.get_registered_hooks()

    def generate(self):  # TODO: Redo ggcam, find a solution for normalize_per_channel
        """Generates an attention map."""
        for layer_name in self.attention_map_GCAM.keys():
            if self.attention_map_GBP.shape == self.attention_map_GCAM[layer_name].shape:
                self.attention_map_GCAM[layer_name] = np.multiply(self.attention_map_GCAM[layer_name], self.attention_map_GBP)
            else:
                attention_map_GCAM_tmp = medcam_utils.interpolate(self.attention_map_GCAM[layer_name], self.attention_map_GBP.shape[2:])
                self.attention_map_GCAM[layer_name] = np.multiply(attention_map_GCAM_tmp, self.attention_map_GBP)
            self.attention_map_GCAM[layer_name] = self._normalize_per_channel(self.attention_map_GCAM[layer_name])
        return self.attention_map_GCAM

    def _generate_gbp(self, batch, label):
        output = self.model_GBP.forward(batch)
        self.model_GBP.backward(label=label)

        attention_map = self.model_GBP.data.grad.clone()
        self.model_GBP.data.grad.zero_()
        B, _, *data_shape = attention_map.shape
        attention_map = attention_map.view(B, 1, -1, *data_shape)
        attention_map = torch.mean(attention_map, dim=2)  # TODO: mean or sum?
        attention_map = attention_map.repeat(1, self.model_GBP.output_channels, *[1 for _ in range(self.model_GBP.input_dim)])
        attention_map = attention_map.cpu().numpy()
        attention_maps = {}
        attention_maps[""] = attention_map
        return attention_maps

    def _normalize_per_channel(self, attention_map):
        if np.min(attention_map) == np.max(attention_map):
            return np.zeros(attention_map.shape)
        # Normalization per channel
        B, C, *data_shape = attention_map.shape
        attention_map = np.reshape(attention_map, (B, C, -1))
        attention_map_min = np.min(attention_map, axis=2, keepdims=True)[0]
        attention_map_max = np.max(attention_map, axis=2, keepdims=True)[0]
        attention_map -= attention_map_min
        attention_map /= (attention_map_max - attention_map_min)
        attention_map = np.reshape(attention_map, (B, C, *data_shape))
        return attention_map
