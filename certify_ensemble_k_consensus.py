'''
- this is the file which does certification for the SmoothEnsembleKConsensus class (smooth_ensemble_k_consensus.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_ensemble_k_consensus import SmoothEnsembleKConsensus
from time import time
import torch
import datetime
from architectures import get_architecture

import random
import numpy as np

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("base_classifiers", type=str, help="path to saved pytorch models of base classifiers", nargs='+')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
parser.add_argument("--voting_size", type=int, default=1, help="number of classifiers who vote in the first round (K in K-consensus aggregation)")
args = parser.parse_args()

seed = args.seed
print('seed: ', seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # load the base classifiers
    base_classifiers = []
    for PATH in args.base_classifiers:
        checkpoint = torch.load(PATH)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifiers.append(base_classifier)

    # create the smooothed ensemble classifier g
    smoothed_classifier = SmoothEnsembleKConsensus(base_classifiers, get_num_classes(args.dataset), args.sigma, args.voting_size)

    # prepare output file
    output_dir = os.path.dirname(args.outfile)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime\tearly_stopping_ratio\tadditional_batches\testimated_pr", file=f, flush=True)
        
    # to save results in numpy list
    results_list = []

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        # certify the prediction of g around x
        before_time = time()
        x = x.cuda()
        certification_results = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        time_elapsed = datetime.timedelta(seconds=(after_time - before_time))
        
        prediction = certification_results[0]
        correct = int(prediction == label)
        radius = certification_results[1]
        early_stopping_ratio = certification_results[2]
        additional_batches_needed = certification_results[3]
        estimated_pr = certification_results[4]
            
        results_list.append([i, label, prediction, radius, correct, time_elapsed, early_stopping_ratio, additional_batches_needed, estimated_pr])
        print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed, early_stopping_ratio, additional_batches_needed, estimated_pr), file=f, flush=True)
        if len(results_list) % 100 == 0:
            np.save(args.outfile, np.array(results_list))
        
        np.save(args.outfile, np.array(results_list))

    f.close()
