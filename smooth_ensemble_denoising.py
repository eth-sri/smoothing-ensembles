'''
- this is the core file which supports ensembles of denoised models for Randomized Smoothing
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothEnsembleDenoising(object):

    ABSTAIN = -1

    def __init__(self, base_classifiers, core_model, num_classes, sigma): # base classifiers are the denoisers
        self.base_classifiers = base_classifiers
        self.num_classifiers = len(base_classifiers)
        self.num_classes = num_classes
        self.sigma = sigma
        self.core_model = core_model

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        for base_classifier in self.base_classifiers:
            base_classifier.eval()
        self.core_model.eval()
        counts_selections = self._sample_noises(x, n0, batch_size)
        cAHats = [counts_selection.argmax().item() for counts_selection in counts_selections]
        counts_estimations = self._sample_noises(x, n, batch_size)
        nAs = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats, counts_estimations)]
        pABars = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs]
        certified_radii = [(SmoothEnsembleDenoising.ABSTAIN, 0.0) if pABar < 0.5
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
                new_inputs = [base_classifier(inputs) for base_classifier in self.base_classifiers]
                outputs = [self.core_model(new_input) for new_input in new_inputs]
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
    
    def _get_predictions(self, outputs, batch_size):
        ensemble_outputs = torch.zeros(batch_size, self.num_classes)
        ensemble_outputs = ensemble_outputs.to('cuda')
        predictions = []
        for i, output in enumerate(outputs):
            for j in range(batch_size):
                ensemble_outputs[j] += output[j]
            predictions.append(output.argmax(1))
            predictions.append(ensemble_outputs.argmax(1))
        return predictions

