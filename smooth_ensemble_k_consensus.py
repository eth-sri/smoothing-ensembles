'''
- this is the core file which supports ensembles with optional K-consensus aggregation for Randomized Smoothing
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothEnsembleKConsensus(object):

    ABSTAIN = -1

    def __init__(self, base_classifiers, num_classes, sigma, k):
        self.base_classifiers = base_classifiers
        self.num_classifiers = len(base_classifiers)
        self.num_classes = num_classes
        self.sigma = sigma
        self.k = k
        self.number_of_early_stoppings = 0
        self.number_of_additional_batches_needed = 0

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        self.number_of_early_stoppings = 0
        self.number_of_additional_batches_needed = 0
        for base_classifier in self.base_classifiers:
            base_classifier.eval()
        counts_selection = self._sample_noises(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        counts_estimation = self._sample_noises(x, n, batch_size)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        estimated_pr = nA / n
        if pABar < 0.5:
            return SmoothEnsembleKConsensus.ABSTAIN, 0.0, self.number_of_early_stoppings / (n + n0), self.number_of_additional_batches_needed, estimated_pr
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, self.number_of_early_stoppings / (n + n0), self.number_of_additional_batches_needed, estimated_pr

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SmoothEnsemble.ABSTAIN
        else:
            return top2[0]

    def _sample_noises(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=int)
        with torch.no_grad():
            remained_inputs = []
            remained_outputs = []
            for _ in range(self.k):
                remained_outputs.append([])
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                
                batch = x.repeat((this_batch_size, 1, 1, 1))
                batch = batch.to('cuda')
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                inputs = batch+noise
                outputs = [base_classifier(inputs) for base_classifier in self.base_classifiers[:self.k]]
                should_stop = self._should_stop(outputs, this_batch_size)
                for i, stopping_result in enumerate(should_stop):
                    if stopping_result >= 0:
                        counts[int(stopping_result)] += 1
                        self.number_of_early_stoppings += 1
                    else:
                        remained_inputs.append(inputs[i])
                        for j in range(self.k):
                            remained_outputs[j].append(outputs[j][i])
                if len(remained_inputs) >= batch_size:
                    self.number_of_additional_batches_needed += 1
                    predictions, remained_inputs, remained_outputs = self._get_additional_predictions(remained_inputs, remained_outputs, batch_size)
                    for prediction in predictions:
                        counts[prediction.cpu().numpy()] += 1
            if len(remained_inputs) > 0:
                self.number_of_additional_batches_needed += 1
                predictions, _, _ = self._get_additional_predictions(remained_inputs, remained_outputs, len(remained_inputs))
                for prediction in predictions:
                    counts[prediction.cpu().numpy()] += 1
        return counts
    
    def _get_additional_predictions(self, remained_inputs, remained_outputs, batch_size):
        new_inputs = torch.stack(remained_inputs[:batch_size])
        remained_inputs = remained_inputs[batch_size:]
        all_outputs = []
        for i in range(self.k):
            to_add_0 = torch.stack(remained_outputs[i][:batch_size])
            all_outputs.append(to_add_0)
            remained_outputs[i] = remained_outputs[i][batch_size:]
        for i in range(self.k, len(self.base_classifiers)):
            to_add = self.base_classifiers[i](new_inputs)
            all_outputs.append(to_add)
        predictions = self._get_predictions(all_outputs, batch_size)
        return predictions, remained_inputs, remained_outputs
        

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    # returns a list of integers
    # if >= 0, then that's the prediction of all first k models, so stop
    # otherwise (i.e. -1), we need to run the noise through all models
    def _should_stop(self, outputs, batch_size):
        should_stop = np.ones(batch_size) * -1
        predictions = []
        for output in outputs:
            predictions.append(output.argmax(1))
        for i in range(batch_size):
            should_stop[i] = predictions[0][i]
            for j in range(1, len(predictions)):
                if predictions[j][i] != should_stop[i]:
                    should_stop[i] = -1
        return should_stop
    
    def _get_predictions(self, outputs, batch_size):
        ensemble_outputs = torch.zeros(batch_size, self.num_classes)
        ensemble_outputs = ensemble_outputs.to('cuda')
        for i, output in enumerate(outputs):
            for j in range(batch_size):
                ensemble_outputs[j] += output[j]
        return ensemble_outputs.argmax(1)
