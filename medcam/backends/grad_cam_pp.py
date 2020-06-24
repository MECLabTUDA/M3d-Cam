import torch
from torch.nn import functional as F
from medcam.backends.grad_cam import GradCAM
from medcam import medcam_utils
from medcam.medcam_utils import prod


class GradCamPP(GradCAM):

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        """
        "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
        https://arxiv.org/abs/1710.11063
        """
        super(GradCamPP, self).__init__(model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)

    def _generate_helper(self, fmaps, grads, layer):
        B, C, *data_shape = grads.size()

        alpha_num = grads.pow(2)
        tmp = fmaps.mul(grads.pow(3))
        tmp = tmp.view(B, C, prod(data_shape))
        tmp = tmp.sum(-1, keepdim=True)
        if self.input_dim == 2:
            tmp = tmp.view(B, C, 1, 1)
        else:
            tmp = tmp.view(B, C, 1, 1, 1)
        alpha_denom = grads.pow(2).mul(2) + tmp
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)

        if self.mask is not None:
            mask = self.mask.squeeze()
        if self.mask is None:  # Classification
            prob_weights = torch.tensor(1.0)
        elif len(mask.shape) == 1:  # Classification best/index
            prob_weights = self.logits.squeeze()[torch.argmax(mask)]
        else:  # Segmentation
            masked_logits = self.logits * self.mask
            prob_weights = medcam_utils.interpolate(masked_logits, grads.shape[2:])  # TODO: Still removes channels...

        positive_gradients = F.relu(torch.mul(prob_weights.exp(), grads))
        weights = (alpha * positive_gradients).view(B, C, -1).sum(-1)
        if self.input_dim == 2:
            weights = weights.view(B, C, 1, 1)
        else:
            weights = weights.view(B, C, 1, 1, 1)

        attention_map = (weights * fmaps)
        try:
            attention_map = attention_map.view(B, self.output_channels, -1, *data_shape)
        except RuntimeError:
            raise RuntimeError("Number of set channels ({}) is not a multiple of the feature map channels ({}) in layer: {}".format(self.output_channels, fmaps.shape[1], layer))
        attention_map = torch.sum(attention_map, dim=2)
        attention_map = F.relu(attention_map).detach()
        attention_map = self._normalize_per_channel(attention_map)

        return attention_map