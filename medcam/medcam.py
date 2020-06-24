from medcam import medcam_inject
from medcam import medcam_utils
from medcam.evaluation import evaluation_utils, evaluator
from functools import wraps


@wraps(medcam_inject.inject)
def inject(*args, **kwargs):
    return medcam_inject.inject(*args, **kwargs)


@wraps(medcam_utils.get_layers)
def get_layers(model, reverse=False):
    return medcam_utils.get_layers(model, reverse)


@wraps(evaluation_utils.comp_score)
def compute_score(attention_map, mask, metric="wioa", threshold='otsu'):
    return evaluation_utils.comp_score(attention_map, mask, metric, threshold)


@wraps(evaluator.Evaluator)
def Evaluator(save_path, metric="wioa", threshold='otsu', layer_ordering=None):
    return evaluator.Evaluator(save_path, metric, threshold, layer_ordering)


@wraps(medcam_utils.save_attention_map)
def save(attention_map, filename, heatmap):
    medcam_utils.save_attention_map(filename, attention_map, heatmap)