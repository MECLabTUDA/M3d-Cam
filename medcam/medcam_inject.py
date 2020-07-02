import torch
from pathlib import Path
import types
import pickle
from medcam import medcam_utils
from medcam.backends.guided_backpropagation import GuidedBackPropagation
from medcam.backends.grad_cam import GradCAM
from medcam.backends.guided_grad_cam import GuidedGradCam
from medcam.backends.grad_cam_pp import GradCamPP
from collections import defaultdict
from medcam.evaluation.evaluator import Evaluator
import copy
import numpy as np

def inject(model, output_dir=None, backend='medcam', layer='auto', label=None, data_shape='default',
           save_maps=False, save_pickle=False, save_scores=False, evaluate=False, metric='wioa', threshold='otsu', retain_graph=False,
           return_score=False, replace=False, cudnn=True, enabled=True):
    """
    Injects a model with medcam functionality to extract attention maps from it. The model can be used as usual.
    Whenever model(input) or model.forward(input) is called medcam will extract the corresponding attention maps.
    Args:
        model: A CNN-based model that inherits from torch.nn.Module
        output_dir: The directory to save any results to
        backend: One of the implemented visualization backends.

                'gbp': Guided-Backpropagation

                'medcam': Grad-Cam

                'ggcam': Guided-Grad-Cam

                'gcampp': Grad-Cam++

        layer: One or multiple layer names of the model from which attention maps will be extracted.

                'auto': Selects the last layer from which attention maps can be extracted.

                'full': Selects every layer from which attention maps can be extracted.

                (layer name): A layer name of the model as string.

                [(layer name 1), (layer name 2), ...]: A list of layer names of the model as string.

            Note: Guided-Backpropagation ignores this parameter.

        label: A class label of interest. Alternatively this can be a class discriminator that creates a mask with only the non masked logits being backwarded through the model.

                Example for class label: label=1
                Example for discriminator: label=lambda x: 0.5 < x

        data_shape: The shape of the resulting attention maps. The given shape should exclude batch and channel dimension.

                'default': The shape of the current input data, excluding batch and channel dimension.

        save_maps: If the attention maps should be saved sorted by layer in the output_dir.

        save_pickle: If the attention maps should be saved as a pickle file in the output_dir.

        save_scores: If the evaluation scores should be saved as an excel file in the output_dir.

        evaluate: If the attention maps should be evaluated. This requires a corresponding mask when calling model.forward().

        metric: An evaluation metric for comparing the attention map with the mask.

                'wioa': Weighted intersection over attention. Most suited for classification.

                'ioa': Intersection over attention.

                'iou': Intersection over union. Not suited for classification.

                (A function): An evaluation function.

        threshold: A threshold used during evaluation for ignoring low attention. Most models have low amounts of attention everywhere in an attention map due to the nature of CNN-based models. The threshold can be used to ignore these low amounts if wanted.

                'otsu': Uses the otsu algorithm to determine a threshold.

                (float): A value between 0 and 1 that is used as threshold.

        retain_graph: If the computation graph should be retained or not.

        return_score: If the evaluation evaluation of the current input should be returned in addition to the model output.

        replace: If the model output should be replaced with the extracted attention map.

        cudnn: If cudnn should be disabled. Some models (e.g. LSTMs) crash when using medcam with enabled cudnn.

        enabled: If medcam should be enabled.

    Returns: A shallow copy of the model injected with medcam functionality.

    """

    if _already_injected(model):
        return

    if not cudnn:
        torch.backends.cudnn.enabled = False

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_clone = copy.copy(model)
    model_clone.eval()
    # Save the original forward of the model
    # This forward will be called by the backend, so if someone writes a new backend they only need to call model.model_forward and not model.medcam_dict['model_forward']
    setattr(model_clone, 'model_forward', model_clone.forward)

    # Save every other attribute in a dict which is added to the model attributes
    # It is ugly but it avoids name conflicts
    medcam_dict = {}

    medcam_dict['output_dir'] = output_dir
    medcam_dict['layer'] = layer
    medcam_dict['counter'] = 0
    medcam_dict['save_scores'] = save_scores
    medcam_dict['save_maps'] = save_maps
    medcam_dict['save_pickle'] = save_pickle
    medcam_dict['evaluate'] = evaluate
    medcam_dict['metric'] = metric
    medcam_dict['return_score'] = return_score
    medcam_dict['_replace_output'] = replace
    medcam_dict['threshold'] = threshold
    medcam_dict['label'] = label
    medcam_dict['channels'] = 1  # TODO: Remove in a later version
    medcam_dict['data_shape'] = data_shape
    medcam_dict['pickle_maps'] = []
    if evaluate:
        medcam_dict['Evaluator'] = Evaluator(output_dir + "/", metric=metric, threshold=threshold, layer_ordering=medcam_utils.get_layers(model_clone))
    medcam_dict['current_attention_map'] = None
    medcam_dict['current_layer'] = None
    medcam_dict['device'] = next(model_clone.parameters()).device
    medcam_dict['tested'] = False
    medcam_dict['enabled'] = enabled
    setattr(model_clone, 'medcam_dict', medcam_dict)

    if output_dir is None and (save_scores or save_maps or save_pickle or evaluate):
        raise ValueError("output_dir needs to be set if save_scores, save_maps, save_pickle or evaluate is set to true")

    # Append methods methods to the model
    model_clone.get_layers = types.MethodType(get_layers, model_clone)
    model_clone.get_attention_map = types.MethodType(get_attention_map, model_clone)
    model_clone.save_attention_map = types.MethodType(save_attention_map, model_clone)
    model_clone.replace_output = types.MethodType(replace_output, model_clone)
    model_clone.dump = types.MethodType(dump, model_clone)
    model_clone.forward = types.MethodType(forward, model_clone)
    model_clone.enable_medcam = types.MethodType(enable_medcam, model_clone)
    model_clone.disable_medcam = types.MethodType(disable_medcam, model_clone)
    model_clone.test_run = types.MethodType(test_run, model_clone)

    model_clone._assign_backend = types.MethodType(_assign_backend, model_clone)
    model_clone._process_attention_maps = types.MethodType(_process_attention_maps, model_clone)
    model_clone._save_attention_map = types.MethodType(_save_attention_map, model_clone)
    model_clone._replace_output = types.MethodType(_replace_output, model_clone)

    model_backend, heatmap = _assign_backend(backend, model_clone, layer, None, retain_graph)  # TODO: Remove postprocessor in a later version
    medcam_dict['model_backend'] = model_backend
    medcam_dict['heatmap'] = heatmap

    return model_clone

def get_layers(self, reverse=False):
    """Returns the layers of the model. Optionally reverses the order of the layers."""
    return self.medcam_dict['model_backend'].layers(reverse)

def get_attention_map(self):
    """Returns the current attention map."""
    return self.medcam_dict['current_attention_map']

def save_attention_map(self, attention_map):
    """Saves an attention map."""
    medcam_utils.save_attention_map(filename=self.medcam_dict['output_dir'] + "/" + self.medcam_dict['current_layer'] + "/attention_map_" +
                                             str(self.medcam_dict['counter']), attention_map=attention_map, heatmap=self.medcam_dict['heatmap'])
    self.medcam_dict['counter'] += 1

def replace_output(self, replace):
    """If the output should be replaced with the corresponiding attention map."""
    self.medcam_dict['_replace_output'] = replace

def dump(self):
    """Saves all of the collected data to the output directory."""
    if self.medcam_dict['save_pickle']:
        with open(self.medcam_dict['output_dir'] + '/attention_maps.pkl', 'wb') as handle:  # TODO: Save every 1GB
            pickle.dump(self.medcam_dict['pickle_maps'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if self.medcam_dict['save_scores']:
        self.medcam_dict['Evaluator'].dump()

def forward(self, batch, label=None, mask=None, raw_input=None):
    """
    Generates attention maps for a given batch input.
    Args:
        batch: An input batch of shape (BxCxHxW) or (BxCxDxHxW).
        label: A class label (int) or a class discriminator function to different attention maps for every class.
        mask: A ground truth mask corresponding to the input batch. Only needed when evaluate is set to true.

    Returns: Either the normal output of the model or an attention map.

    """
    if self.medcam_dict['enabled']:
        self.test_run(batch, internal=True)
        if self.medcam_dict['layer'] == 'full' and not self.medcam_dict['tested']:
            raise ValueError("Layer mode 'full' requires a test run either during injection or by calling test_run() afterwards")
        with torch.enable_grad():
            output, attention_map, batch_size, channels, data_shape = self.medcam_dict['model_backend'].generate_attention_map(batch, label)
            if attention_map:
                if len(attention_map.keys()) == 1:
                    self.medcam_dict['current_attention_map'] = attention_map[list(attention_map.keys())[0]]
                    self.medcam_dict['current_layer'] = list(attention_map.keys())[0]
                scores = self._process_attention_maps(attention_map, mask, batch_size, channels, raw_input)
                output = self._replace_output(output, attention_map, data_shape)
            else:  # If no attention maps could be extracted
                self.medcam_dict['current_attention_map'] = None
                self.medcam_dict['current_layer'] = None
                scores = None
                if self.medcam_dict['_replace_output']:
                    raise ValueError("Unable to extract any attention maps")
            self.medcam_dict['counter'] += 1
            if self.medcam_dict['return_score']:
                return output, scores
            else:
                return output
    else:
        return self.model_forward(batch)

def test_run(self, batch, internal=False):
    """Performs a test run. This allows medcam to determine for which layers it can generate attention maps."""
    registered_hooks = []
    if batch is not None and not self.medcam_dict['tested']:
        with torch.enable_grad():
            _ = self.medcam_dict['model_backend'].generate_attention_map(batch, None)
            registered_hooks = self.medcam_dict['model_backend'].get_registered_hooks()
        self.medcam_dict['tested'] = True
        if not internal:
            print("Successfully registered to the following layers: ", registered_hooks)
            if self.medcam_dict['output_dir'] is not None:
                np.savetxt(self.medcam_dict['output_dir'] + '/registered_layers.txt', np.asarray(registered_hooks).astype(str), fmt="%s")
    return registered_hooks

def disable_medcam(self):
    """Disables medcam."""
    self.medcam_dict['enabled'] = False

def enable_medcam(self):
    """Enables medcam."""
    self.medcam_dict['enabled'] = True

def _already_injected(model):
    """Checks if the model is already injected with medcam."""
    try:  # try/except is faster than hasattr, if inject method is called repeatedly
        model.medcam_dict  # Check if attribute exists
        return True
    except AttributeError:
        return False

def _assign_backend(backend, model, target_layers, postprocessor, retain_graph):
    """Assigns a chosen backend."""
    if backend == "gbp":
        return GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcam":
        return GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), True
    elif backend == "ggcam":
        return GuidedGradCam(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcampp":
        return GradCamPP(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), True
    else:
        raise ValueError("Backend does not exist")

def _process_attention_maps(self, attention_map, mask, batch_size, channels, raw_input):
    """Handles all the stuff after the attention map has been generated. Like creating dictionaries, saving the attention map and doing the evaluation."""
    batch_scores = defaultdict(list) if self.medcam_dict['evaluate'] else None
    raw_input_single = None
    for layer_name in attention_map.keys():
        layer_output_dir = None
        if self.medcam_dict['output_dir'] is not None and self.medcam_dict['save_maps']:
            if layer_name == "":
                layer_output_dir = self.medcam_dict['output_dir']
            else:
                layer_output_dir = self.medcam_dict['output_dir'] + "/" + layer_name
            Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
        for j in range(batch_size):
            for k in range(channels):
                attention_map_single = attention_map[layer_name][j][k]
                if raw_input is not None:
                    raw_input_single = raw_input[j]
                self._save_attention_map(attention_map_single, layer_output_dir, j, k, raw_input_single)
                if self.medcam_dict['evaluate']:
                    if mask is None:
                        raise ValueError("Mask cannot be none in evaluation mode")
                    score = self.medcam_dict['Evaluator'].comp_score(attention_map_single, mask[j][k].squeeze(), layer=layer_name, class_label=k)
                    batch_scores[layer_name].append(score)
    return batch_scores

def _save_attention_map(self, attention_map, layer_output_dir, j, k, raw_input):
    """Internal method for saving saving an attention map."""
    if self.medcam_dict['save_pickle']:
        self.medcam_dict['pickle_maps'].append(attention_map)
    if self.medcam_dict['save_maps']:
        medcam_utils.save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.medcam_dict['counter']) + "_" + str(j) + "_" + str(k), attention_map=attention_map, heatmap=self.medcam_dict['heatmap'], raw_input=raw_input)

def _replace_output(self, output, attention_map, data_shape):
    """Replaces the model output with the current attention map."""
    if self.medcam_dict['_replace_output']:
        if len(attention_map.keys()) == 1:
            output = torch.tensor(self.medcam_dict['current_attention_map']).to(str(self.medcam_dict['device']))
            if data_shape is not None:  # If data_shape is None then the task is classification -> return unchanged attention map
                output = medcam_utils.interpolate(output, data_shape)
        else:
            raise ValueError("Not possible to replace output when layer is 'full', only with 'auto' or a manually set layer")
    return output