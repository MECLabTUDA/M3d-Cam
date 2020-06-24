from medcam.evaluation import evaluation_utils
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Evaluator():

    def __init__(self, save_path, metric="wioa", threshold='otsu', layer_ordering=None):
        """
        Evaluator class that evaluates attention maps with the corresponding ground truth masks.
        Args:
            save_path: The save path for the results.

            metric: The metric to use.

                    'wioa': Weighted intersection over attention. How much of weighted total attention is inside the ground truth mask.

                    'ioa': Intersection over attention. How much of total attention is inside the ground truth mask.

                    'iou': Intersection over union.

                    (A function): A metric function that takes the arguments attention map and ground truth mask.

            threshold: A threshold used during evaluation for ignoring low attention. Most models have low amounts of attention everywhere in an attention map due to the nature of CNN-based models. The threshold can be used to ignore these low amounts if wanted.

                    'otsu': Uses the otsu algorithm to determine a threshold.

                    (float): A value between 0 and 1 that is used as threshold.

            layer_ordering: A list of layer names that is used for ordering the mean scores.
        """
        self.scores = pd.DataFrame(columns=['evaluation', 'layer', 'class_label', 'name'])
        self.save_path = save_path
        self.metric = metric
        self.threshold = threshold
        self.layer_ordering = layer_ordering

    def comp_score(self, attention_map, mask, layer=None, class_label=None, name=None):
        """
        Evaluates an attention maps with the corresponding ground truth mask.
        Args:
            attention_map: The attention map.
            mask: The ground truth mask.
            layer: The layer name.
            class_label: The class label.
            name: A name for better identification (e.g. the filename of the ground truth mask)

        Returns: The evaluation score

        """
        score = evaluation_utils.comp_score(attention_map, mask, self.metric, self.threshold)
        self._add(score, layer, class_label, name)
        return score

    def _add(self, score, layer=None, class_label=None, name=None):
        """Adds a score to the score dataframe."""
        self.scores = self.scores.append({'evaluation': score, 'layer': layer, 'class_label': class_label, 'name': name}, ignore_index=True)

    def dump(self, mean_only=False, layer=None, class_label=None):
        """
        Saves the scores and mean scores as an excel table.
        Args:
            mean_only: Save only the mean scores.
            layer: Filter by a single layer.
            class_label: Filter by a single class label.

        """
        scores = self.scores
        if layer is not None:
            scores = scores[scores['layer']==layer]
        if class_label is not None:
            scores = scores[scores['class_label'] == class_label]
        mean_scores = self._comp_means(scores)
        with pd.ExcelWriter(self.save_path + 'scores.xlsx') as writer:
            if not mean_only:
                scores.to_excel(writer, sheet_name='Scores', na_rep='NaN')
            mean_scores.to_excel(writer, sheet_name='Mean Scores', na_rep='NaN')

    def _comp_means(self, scores):
        """Computes the mean scores."""
        mean_scores = pd.DataFrame(columns=['mean_score', 'layer', 'class_label'])
        mean_score_list = []
        unique_layers = pd.unique(scores['layer'])
        if self.layer_ordering is not None and unique_layers[0] != "":
            unique_layers = sorted(set(self.layer_ordering).intersection(unique_layers), key=lambda x: self.layer_ordering.index(x))
        for unique_layer in unique_layers:
            _scores = scores[scores['layer'] == unique_layer]
            unique_class_labels = pd.unique(_scores['class_label'])
            for unique_class_label in unique_class_labels:
                __scores = _scores[_scores['class_label'] == unique_class_label]
                mean_score = __scores['evaluation'].to_numpy().astype(np.float)
                mean_score = mean_score[~np.isnan(mean_score)]
                mean_score = np.mean(mean_score)
                mean_score_list.append(mean_score)
                mean_scores = mean_scores.append({'mean_score': mean_score, 'layer': unique_layer, 'class_label': unique_class_label}, ignore_index=True)
        self._plot_mean_scores(unique_layers, mean_score_list)
        return mean_scores

    def _plot_mean_scores(self, layers, mean_scores):
        """Plots the mean scores as a nice graph."""
        layers, mean_scores = self._reduce(layers, mean_scores)
        fig, ax = plt.subplots(1)
        plt.plot(range(len(layers)), mean_scores)
        x_ticks = np.arange(len(layers))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(layers)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.savefig(self.save_path + 'mean_scores.png', dpi=300)
        fig.clf()

    def _reduce(self, layers, mean_scores):
        """Reduces the number of total layers by removing sub layers. Except for the very first and last layer."""
        saved = [[layers[0], mean_scores[0]], [layers[len(layers)-1], mean_scores[len(layers)-1]]]
        i = 0
        while i < len(layers):
            j = 0
            while i < len(layers) and j < len(layers):
                if layers[i] in layers[j] and i != j:
                    del layers[j]
                    del mean_scores[j]
                    if j < i:
                        i -= 1
                else:
                    j += 1
            i += 1
        layers.insert(0, saved[0][0])
        mean_scores.insert(0, saved[0][1])
        layers.append(saved[1][0])
        mean_scores.append(saved[1][1])
        return layers, mean_scores