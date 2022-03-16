'''
- this is the core file which supports Adaptive Sampling for Randomied Smoothing for individual models, and for ensembles
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothEnsembleAdaptive(object):

    ABSTAIN = -1

    def __init__(self, base_classifiers, num_classes, sigma):
        self.base_classifiers = base_classifiers
        self.num_classifiers = len(base_classifiers)
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0, n_original, n_list, alpha, beta, batch_size, radius_to_certify) -> (int, float):
        n_num = len(n_list)
        estimated_pa = 0.0
        for i, n in enumerate(n_list):
            cAHat, nA = self._certify_help(x, n0, n, batch_size)
            pABar = self._lower_confidence_bound(nA, n, alpha/n_num)
            radius = self.sigma * norm.ppf(pABar)
            estimated_pa = float(nA / n)
            if pABar > 0.5 and radius >= radius_to_certify:
                return 1, cAHat, i, estimated_pa # certifiably predicts cAHat after round i
            if i < n_num-1:
                should_stop = self._should_stop(nA, n, beta, n_num, radius_to_certify)
                if should_stop:
                    return -1, -1, i, estimated_pa # abstained after round i, not worth continuing    
        return -1, -1, i, estimated_pa # abstained after last round
    
    def _should_stop(self, nA, n, beta, n_num, radius_to_certify):
        upper_confidence_bound = proportion_confint(nA, n, alpha=2 * beta/(n_num-1), method='beta')[1]
        radius_worth_pursuing = self.sigma * norm.ppf(upper_confidence_bound)
        if radius_worth_pursuing < radius_to_certify:
            return True
        return False
        
    def _certify_help(self, x, n0, n, batch_size):
        for base_classifier in self.base_classifiers:
            base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        return cAHat, nA

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                inputs = batch+noise
                outputs = [base_classifier(inputs) for base_classifier in self.base_classifiers]
                predictions = self._get_predictions(outputs, this_batch_size)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
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
        for i, output in enumerate(outputs):
            for j in range(batch_size):
                ensemble_outputs[j] += output[j]
        return ensemble_outputs.argmax(1)
    
    