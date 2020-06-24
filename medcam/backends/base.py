import torch
import numpy as np
from torch.nn import functional as F
from medcam import medcam_utils


class _BaseWrapper():

    def __init__(self, model, postprocessor=None, retain_graph=False):
        """A base wrapper of common functions for the backends."""
        self.device = next(model.parameters()).device
        self.retain_graph = retain_graph
        self.model = model
        self.forward_handlers = []
        self.backward_handlers = []
        self.postprocessor = postprocessor
        self.registered_hooks = {}

    def generate_attention_map(self, batch, label):
        """Handles the generation of the attention map from start to finish."""
        output = self.forward(batch)
        self.backward(label=label)
        attention_map = self.generate()
        return output, attention_map, self.output_batch_size, self.output_channels, self.output_shape

    def forward(self, batch):
        """Calls the forward() of the model."""
        self.model.zero_grad()
        self.logits = self.model.model_forward(batch)
        self._extract_metadata(batch, self.logits)
        self._set_postprocessor_and_label(self.logits)
        self.remove_hook(forward=True, backward=False)
        return self.logits

    def backward(self, label=None):
        """Applies postprocessing and class discrimination on the model output and then backwards it."""
        if label is None:
            label = self.model.medcam_dict['label']


        self.mask = self._isolate_class(self.logits, label)
        self.logits.backward(gradient=self.mask, retain_graph=self.retain_graph)
        self.remove_hook(forward=True, backward=True)

    def _isolate_class(self, output, label):
        """Isolates a desired class on the channel dim by creating a mask that is applied on the gradients during backward."""
        if label is None:
            return torch.ones(output.shape).to(self.device)
        if label == "best":
            if self.output_batch_size > 1:
                raise RuntimeError("Best label mode works only with a batch size of one. You need to choose a specific label or None with a batch size bigger than one.")
            B, C, *data_shape = output.shape
            if len(data_shape) > 0:
                _output = output.view(B, C, -1)
                _output = torch.sum(_output, dim=2)
                label = torch.argmax(_output, dim=1).item()
            else:
                label = torch.argmax(output, dim=1).item()
        if callable(label):
            mask = label(output) * 1.0
            print(mask.dtype)
        else:
            mask = torch.zeros(output.shape).to(self.device)
            mask[:, label] = 1
        return mask

    def _extract_metadata(self, input, output):  # TODO: Does not work for classification output (shape: (1, 1000)), merge with the one in medcam_inject
        """Extracts metadata like batch size, number of channels and the data shape from the output batch."""
        self.input_dim = len(input.shape[2:])
        self.output_batch_size = output.shape[0]
        if self.model.medcam_dict['channels'] == 'default':
            self.output_channels = output.shape[1]
        else:
            self.output_channels = self.model.medcam_dict['channels']
        if self.model.medcam_dict['data_shape'] == 'default':
            if len(output.shape) == 2:  # Classification -> Cannot convert attention map to classifiaction
                self.output_shape = None
            else:  # Output is an 2D/3D image
                self.output_shape = output.shape[2:]
        else:
            self.output_shape = self.model.medcam_dict['data_shape']

    def _normalize_per_channel(self, attention_map):
        if torch.min(attention_map) == torch.max(attention_map):
            return torch.zeros(attention_map.shape)
        # Normalization per channel
        B, C, *data_shape = attention_map.shape
        attention_map = attention_map.view(B, C, -1)
        attention_map_min = torch.min(attention_map, dim=2, keepdim=True)[0]
        attention_map_max = torch.max(attention_map, dim=2, keepdim=True)[0]
        attention_map -= attention_map_min
        attention_map /= (attention_map_max - attention_map_min)
        attention_map = attention_map.view(B, C, *data_shape)
        return attention_map

    def generate(self):
        """Generates an attention map."""
        raise NotImplementedError

    def remove_hook(self, forward, backward):
        """
        Remove all the forward/backward hook functions
        """
        if forward:
            for handle in self.forward_handlers:
                handle.remove()
            self.forward_handlers = []
        if backward:
            for handle in self.backward_handlers:
                handle.remove()
            self.backward_handlers = []

    def layers(self, reverse=False):
        """Returns the layers of the model. Optionally reverses the order of the layers."""
        return medcam_utils.get_layers(self.model, reverse)

    def _set_postprocessor_and_label(self, output):
        if self.model.medcam_dict['label'] is None:
            if output.shape[0] == self.output_batch_size and len(output.shape) == 2:  # classification
                self.model.medcam_dict['label'] = "best"
