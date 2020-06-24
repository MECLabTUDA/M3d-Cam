from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from medcam.backends.base import _BaseWrapper
from medcam import medcam_utils

# Changes the used method to hook into backward
ENABLE_MODULE_HOOK = False

class GradCAM(_BaseWrapper):

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        """
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
        https://arxiv.org/pdf/1610.02391.pdf
        Look at Figure 2 on page 4
        """
        super(GradCAM, self).__init__(model, postprocessor=postprocessor, retain_graph=retain_graph)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self._target_layers = target_layers
        if target_layers == 'full' or target_layers == 'auto':
            target_layers = medcam_utils.get_layers(self.model)
        elif isinstance(target_layers, str):
            target_layers = [target_layers]
        self.target_layers = target_layers

    def _register_hooks(self):
        """Registers the forward and backward hooks to the layers."""
        def forward_hook(key):
            def forward_hook_(module, input, output):
                self.registered_hooks[key][0] = True
                # Save featuremaps
                if not isinstance(output, torch.Tensor):
                    print("Cannot hook layer {} because its gradients are not in tensor format".format(key))

                if not ENABLE_MODULE_HOOK:
                    def _backward_hook(grad_out):
                        self.registered_hooks[key][1] = True
                        # Save the gradients correspond to the featuremaps
                        self.grad_pool[key] = grad_out.detach()

                    # Register backward hook directly to the output
                    # Handle must be removed afterwards otherwise tensor is not freed
                    if not self.registered_hooks[key][1]:
                        _backward_handle = output.register_hook(_backward_hook)
                        self.backward_handlers.append(_backward_handle)
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        # This backward hook method looks prettier but is currently bugged in pytorch (04/25/2020)
        # Handle does not need to be removed, tensors are freed automatically
        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                self.registered_hooks[key][1] = True
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()  # TODO: Still correct with batch size > 1?

            return backward_hook_

        self.remove_hook(forward=True, backward=True)
        for name, module in self.model.named_modules():
            if self.target_layers is None or name in self.target_layers:
                self.registered_hooks[name] = [False, False]
                self.forward_handlers.append(module.register_forward_hook(forward_hook(name)))
                if ENABLE_MODULE_HOOK:
                    self.backward_handlers.append(module.register_backward_hook(backward_hook(name)))

    def get_registered_hooks(self):
        """Returns every hook that was able to register to a layer."""
        registered_hooks = []
        for layer in self.registered_hooks.keys():
            if self.registered_hooks[layer][0] and self.registered_hooks[layer][1]:
                registered_hooks.append(layer)
        self.remove_hook(forward=True, backward=True)
        if self._target_layers == 'full' or self._target_layers == 'auto':
            self.target_layers = registered_hooks
        return registered_hooks

    def forward(self, data):
        """Calls the forward() of the base."""
        self._register_hooks()
        return super(GradCAM, self).forward(data)

    def generate(self):
        """Generates an attention map."""
        self.remove_hook(forward=True, backward=True)
        attention_maps = {}
        if self._target_layers == "auto":
            layer, fmaps, grads = self._auto_layer_selection()
            self._check_hooks(layer)
            attention_map = self._generate_helper(fmaps, grads, layer).cpu().numpy()
            attention_maps = {layer: attention_map}
        else:
            for layer in self.target_layers:
                self._check_hooks(layer)
                if self.registered_hooks[layer][0] and self.registered_hooks[layer][1]:
                    fmaps = self._find(self.fmap_pool, layer)
                    grads = self._find(self.grad_pool, layer)
                    attention_map = self._generate_helper(fmaps, grads, layer)
                    attention_maps[layer] = attention_map.cpu().numpy()
        if not attention_maps:
            raise ValueError("None of the hooks registered to the target layers")
        return attention_maps

    def _auto_layer_selection(self):
        """Selects the last layer from which attention maps can be generated."""
        # It's ugly but it works ;)
        module_names = self.layers(reverse=True)
        found_valid_layer = False

        counter = 0
        for layer in module_names:
            try:
                fmaps = self._find(self.fmap_pool, layer)
                grads = self._find(self.grad_pool, layer)
                nonzeros = np.count_nonzero(grads.detach().cpu().numpy())  # TODO: Add except here with description, replace nonzero with sum == 0?
                self._compute_grad_weights(grads)
                if nonzeros == 0 or not isinstance(fmaps, torch.Tensor) or not isinstance(grads, torch.Tensor):
                    counter += 1
                    continue
                print("Selected module layer: {}".format(layer))
                found_valid_layer = True
                break
            except ValueError:
                counter += 1
            except RuntimeError:
                counter += 1
            except IndexError:
                counter += 1

        if not found_valid_layer:
            raise ValueError("Could not find a valid layer. "
                             "Check if base.logits or the mask result of base._mask_output() contains only zeros. "
                             "Check if requires_grad flag is true for the batch input and that no torch.no_grad statements effects medcam. "
                             "Check if the model has any convolution layers.")

        return layer, fmaps, grads

    def _find(self, pool, target_layer):
        """Returns the feature maps or gradients for a specific layer."""
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        """Computes the weights based on the gradients by average pooling."""
        if self.input_dim == 2:
            return F.adaptive_avg_pool2d(grads, 1)
        else:
            return F.adaptive_avg_pool3d(grads, 1)

    def _generate_helper(self, fmaps, grads, layer):
        weights = self._compute_grad_weights(grads)
        attention_map = torch.mul(fmaps, weights)
        B, _, *data_shape = attention_map.shape
        try:
            attention_map = attention_map.view(B, self.output_channels, -1, *data_shape)
        except RuntimeError:
            raise RuntimeError("Number of set channels ({}) is not a multiple of the feature map channels ({}) in layer: {}".format(self.output_channels, fmaps.shape[1], layer))
        attention_map = torch.sum(attention_map, dim=2)
        attention_map = F.relu(attention_map)
        attention_map = self._normalize_per_channel(attention_map)
        return attention_map

    def _check_hooks(self, layer):
        """Checks if all hooks registered."""
        if not self.registered_hooks[layer][0] and not self.registered_hooks[layer][1]:
            raise ValueError("Neither forward hook nor backward hook did register to layer: " + str(layer))
        elif not self.registered_hooks[layer][0]:
            raise ValueError("Forward hook did not register to layer: " + str(layer))
        elif not self.registered_hooks[layer][1]:
            raise ValueError("Backward hook did not register to layer: " + str(layer) + ", Check if the hook was registered to a layer that is skipped during backward and thus no gradients are computed")
