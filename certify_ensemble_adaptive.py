'''
- this is the file which does certification for the SmoothEnsembleAdaptive class (smooth_ensemble_adaptive.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_ensemble_adaptive import SmoothEnsembleAdaptive
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np
import random

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("base_classifiers", type=str, help="path to saved pytorch models of base classifiers", nargs='+')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")

parser.add_argument("--N1", type=int, default=1000)
parser.add_argument("--N2", type=int, default=10000)
parser.add_argument("--N3", type=int, default=0)
parser.add_argument("--N4", type=int, default=0)
parser.add_argument("--N5", type=int, default=0)
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--beta", type=float, default=0.0001, help="failure probability 2")
parser.add_argument("--radius_to_certify", type=float, default=0.25)
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
    
    # load the classifiers
    base_classifiers = []
    for PATH in args.base_classifiers:
        checkpoint = torch.load(PATH)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifiers.append(base_classifier)

    # create the smooothed classifier g
    smoothed_classifier_adaptive = SmoothEnsembleAdaptive(base_classifiers, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    output_dir = os.path.dirname(args.outfile)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius proved\tcorrect\tlevels_needed\testimated_pr\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    count_levels = [0, 0, 0, 0, 0]
    count_results = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    num_certified = 0
    results_np = []
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
        n_list = [args.N1]
        if args.N2 > 0:
            n_list.append(args.N2)
        if args.N3 > 0:
            n_list.append(args.N3)
        if args.N4 > 0:
            n_list.append(args.N4)
        if args.N5 > 0:
            n_list.append(args.N5)
        certification_results = smoothed_classifier_adaptive.certify(x, args.N0, args.N, n_list, args.alpha, args.beta, args.batch, args.radius_to_certify)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        prediction = str(certification_results[1]) # label we predict/consider for certification
        correct = str(int(certification_results[1] == label)) # whether prediction is correct
        levels_needed = str(certification_results[2]) # levels we needed for the whole process
        certified = str(certification_results[0]) # whether we certified at given radius
        estimated_pr = str(certification_results[3])
        print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}".format(
            i, label, prediction, certified, correct, levels_needed, estimated_pr, time_elapsed), file=f, flush=True)
        
        # collecting some data for logging
        results_np.append([i, label, prediction, certified, correct, levels_needed, estimated_pr, time_elapsed])
        is_certified_and_correct = 0
        if certification_results[1] == label and certification_results[0]: # if certified and correct
            num_certified += 1
            is_certified_and_correct = 1
        count_levels[certification_results[2]] += 1
        count_results[certification_results[2]][is_certified_and_correct] += 1

    # various logging outputs
    print("correctly certified: ", num_certified)
    print("levels needed (1, 2 or 3): ", count_levels)
    print("levels needed and outcome (e.g. stopped at level 1 and didn't certify/certify: ", count_results)
    np.save(args.outfile, np.array(results_np))
    f.close()
