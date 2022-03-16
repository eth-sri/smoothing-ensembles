'''
- this is the core file which supports ensembles for Randomized Smoothing
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothEnsemble(object):

    ABSTAIN = -1

    def __init__(self, base_classifiers, num_classes, sigma, aggregation_scheme):
        self.base_classifiers = base_classifiers
        self.num_classifiers = len(base_classifiers)
        self.num_classes = num_classes
        self.sigma = sigma
        self.aggregation_scheme = aggregation_scheme

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        for base_classifier in self.base_classifiers:
            base_classifier.eval()
        counts_selections = self._sample_noises(x, n0, batch_size)
        cAHats = [counts_selection.argmax().item() for counts_selection in counts_selections]
        counts_estimations = self._sample_noises(x, n, batch_size)
        nAs = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats, counts_estimations)]
        pABars = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs]
        certified_radii = [(SmoothEnsemble.ABSTAIN, 0.0) if pABar < 0.5
                          else (cAHat, self.sigma * norm.ppf(pABar))
                          for pABar, cAHat in zip(pABars, cAHats)]
        return certified_radii

    def _sample_noises(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros((self.num_classifiers*2, self.num_classes), dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                
                batch = x.repeat((this_batch_size, 1, 1, 1))
                batch = batch.to('cuda')
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                inputs = batch+noise
                outputs = [base_classifier(inputs) for base_classifier in self.base_classifiers]
                predictions = self._get_predictions(outputs, this_batch_size)
                for i, prediction in enumerate(predictions):
                    counts[i] += self._count_arr(prediction.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    # Returns the list of predictions of single models (0 mod 2) and ensemble models (1 mod 2)
    def _get_predictions(self, outputs, batch_size):
        ensemble_outputs = torch.zeros(batch_size, self.num_classes)
        ensemble_outputs = ensemble_outputs.to('cuda')
        cohen_weights = [0.0321, 0.1202, 0.0386, 0.0592, 0.1218, 0.0269, 0.0999, 0.0124, 0.1224, 0.1075]
        consistency_weights = [0.0229,  0.0568,  0.2844,  0.0997, -0.1398,  0.1760,  0.1653, -0.0079, 0.1170,  0.1249]
        predictions = []
        for i, output in enumerate(outputs):
            for j in range(batch_size):
                if self.aggregation_scheme == 0: # soft-voting without softmax - default
                    ensemble_outputs[j] += output[j]
                elif self.aggregation_scheme == 1: # hard voting
                    ensemble_outputs[j][torch.argmax(output[j])] += 1
                elif self.aggregation_scheme == 2: # soft-voting with/after softmax
                    ensemble_outputs[j] += torch.nn.functional.softmax(output[j], 0)
                elif self.aggregation_scheme == 3: # learned weights for pretrained Gaussian resnet110 (sigma=0.25)
                    ensemble_outputs[j] += output[j] * cohen_weights[i]
                elif self.aggregation_scheme == 4: # learned weights for pretrained consistency resnet110 (sigma=0.25)
                    ensemble_outputs[j] += output[j] * consistency_weights[i]
            if self.aggregation_scheme == 1: # if draw after hard voting, use randomness
                ensemble_outputs = ensemble_outputs + torch.rand(batch_size, self.num_classes, device='cuda') * 0.1
            predictions.append(output.argmax(1))
            predictions.append(ensemble_outputs.argmax(1))
            
        return predictions
