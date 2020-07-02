import cv2
import numpy as np
import matplotlib.cm as cm
import nibabel as nib
import torch
from torch.nn import functional as F
from functools import reduce
import operator

MIN_SHAPE = (500, 500)

def save_attention_map(filename, attention_map, heatmap, raw_input):
    """
    Saves an attention maps.
    Args:
        filename: The save path, including the name, excluding the file extension.
        attention_map: The attention map in HxW or DxHxW format.
        heatmap: If the attention map should be saved as a heatmap. True for gcam and gcampp. False for gbp and ggcam.
    """
    dim = len(attention_map.shape)
    attention_map = normalize(attention_map.astype(np.float))
    attention_map = generate_attention_map(attention_map, heatmap, dim, raw_input)
    _save_file(filename, attention_map, dim)

def generate_attention_map(attention_map, heatmap, dim, raw_input):
    if dim == 2:
        if heatmap:
            return generate_gcam2d(attention_map, raw_input)
        else:
            return generate_guided_bp2d(attention_map)
    elif dim == 3:
        if heatmap:
            return generate_gcam3d(attention_map)
        else:
            return generate_guided_bp3d(attention_map)
    else:
        raise RuntimeError("Unsupported dimension. Only 2D and 3D data is supported.")

def generate_gcam2d(attention_map, raw_input):
    assert(len(attention_map.shape) == 2)  # No batch dim
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor

    if raw_input is not None:
        attention_map = overlay(raw_input, attention_map)
    else:
        attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
        attention_map = cm.jet_r(attention_map)[..., :3] * 255.0
    return np.uint8(attention_map)

def generate_guided_bp2d(attention_map):
    assert(len(attention_map.shape) == 2)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor

    attention_map *= 255.0
    attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
    return np.uint8(attention_map)

def generate_gcam3d(attention_map, data=None):
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor
    assert(isinstance(data, np.ndarray) or data is None)  # Not PIL
    assert(data is None or len(data.shape) == 3)

    attention_map *= 255.0
    return np.uint8(attention_map)

def generate_guided_bp3d(attention_map):
    assert(len(attention_map.shape) == 3)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor

    attention_map *= 255.0
    return np.uint8(attention_map)

def _load_data(data_path):
    if isinstance(data_path, str):
        return cv2.imread(data_path)
    else:
        return data_path

def _resize_attention_map(attention_map, min_shape):
    attention_map_shape = attention_map.shape[:2]
    if min(min_shape) < min(attention_map_shape):
        attention_map = cv2.resize(attention_map, tuple(np.flip(attention_map_shape)))
    else:
        resize_factor = int(min(min_shape) / min(attention_map_shape))
        data_shape = (attention_map_shape[0] * resize_factor, attention_map_shape[1] * resize_factor)
        attention_map = cv2.resize(attention_map, tuple(np.flip(data_shape)))
    return attention_map

def normalize(x):
    """Normalizes data both numpy or tensor data to range [0,1]."""
    if isinstance(x, torch.Tensor):
        if torch.min(x) == torch.max(x):
            return torch.zeros(x.shape)
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    else:
        if np.min(x) == np.max(x):
            return np.zeros(x.shape)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

def _save_file(filename, attention_map, dim):
    if dim == 2:
        cv2.imwrite(filename + ".png", attention_map)
    else:
        attention_map = attention_map.transpose(1, 2, 0)
        attention_map = nib.Nifti1Image(attention_map, affine=np.eye(4))
        nib.save(attention_map, filename + ".nii.gz")

def get_layers(model, reverse=False):
    """Returns the layers of the model. Optionally reverses the order of the layers."""
    layer_names = []
    for name, _ in model.named_modules():
        layer_names.append(name)

    if layer_names[0] == "":
        layer_names = layer_names[1:]

    index = 0
    sub_index = 0
    while True:
        if index == len(layer_names) - 1:
            break
        if sub_index < len(layer_names) - 1 and layer_names[index] == layer_names[sub_index + 1][:len(layer_names[index])]:
            sub_index += 1
        elif sub_index > index:
            layer_names.insert(sub_index, layer_names.pop(index))
            sub_index = index
        else:
            index += 1
            sub_index = index

    if reverse:
        layer_names.reverse()

    return layer_names

def interpolate(data, shape, squeeze=False):
    """Interpolates data to the size of a given shape. Optionally squeezes away the batch and channel dim if the data was given in HxW or DxHxW format."""
    if isinstance(data, np.ndarray):
        # Lazy solution, numpy and scipy have multiple interpolate methods with only linear or nearest, so I don't know which one to use... + they don't work with batches
        # Should be redone with numpy or scipy though
        data_type = data.dtype
        data = torch.FloatTensor(data)
        data = _interpolate_tensor(data, shape, squeeze)
        data = data.numpy().astype(data_type)
    elif isinstance(data, torch.Tensor):
        data = _interpolate_tensor(data, shape, squeeze)
    else:
        raise ValueError("Unsupported data type for interpolation")
    return data

def _interpolate_tensor(data, shape, squeeze):
    """Interpolates data to the size of a given shape. Optionally squeezes away the batch and channel dim if the data was given in HxW or DxHxW format."""
    if (len(shape) == 2 and len(data.shape) == 2) or ((len(shape) == 3 and len(data.shape) == 3)):  # Add batch and channel dim
        data = data.unsqueeze(0).unsqueeze(0)
        _squeeze = 2
    elif (len(shape) == 2 and len(data.shape) == 3) or ((len(shape) == 3 and len(data.shape) == 4)):  # Add batch dim
        data = data.unsqueeze(0)
        _squeeze = 1
    if len(shape) == 2:
        data = F.interpolate(data, shape, mode="bilinear", align_corners=False)
    else:
        data = F.interpolate(data, shape, mode="trilinear", align_corners=False)
    if squeeze:  # Remove unnecessary dims
        for i in range(_squeeze):
            data = data.squeeze(0)
    return data

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def overlay(raw_input, attention_map):
    if isinstance(raw_input, torch.Tensor):
        raw_input = raw_input.detach().cpu().numpy()
        if raw_input.shape[0] == 1 or raw_input.shape[0] == 3:
            raw_input = raw_input.transpose(1, 2, 0)
    if np.max(raw_input) > 1:
        raw_input = raw_input.astype(np.float)
        raw_input /= 255
    attention_map = cv2.resize(attention_map, tuple(np.flip(raw_input.shape[:2])))
    attention_map = cm.jet_r(attention_map)[..., :3]
    attention_map = (attention_map.astype(np.float) + raw_input.astype(np.float)) / 2
    attention_map *= 255
    return attention_map
