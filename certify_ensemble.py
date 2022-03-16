'''
- this is the file which does certification for the SmoothEnsemble class (smooth_ensemble.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_ensemble import SmoothEnsemble
from time import time
import torch
import datetime
from architectures import get_architecture
from architectures import get_architecture_center_layer

import random
import numpy as np

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("base_classifiers", type=str, help="path to saved pytorch models of base classifiers", nargs='+')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--skip_offset", type=int, default=0, help="which mod to consider while skipping")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
parser.add_argument("--aggregation_scheme", type=int, default=0, help="0 is default softvoting; 1 is hard voting; 2 is softvoting after softmax, 3 and 4 are weightings according to prelearned weights for specific models")
parser.add_argument("--center_layer", type=int, default=0, help="set to 1 if model requires centering the layer")
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
        if args.center_layer > 0:
            base_classifier = get_architecture_center_layer(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifiers.append(base_classifier)

    # create the smooothed ensemble classifier g
    smoothed_classifier = SmoothEnsemble(base_classifiers, get_num_classes(args.dataset), args.sigma, args.aggregation_scheme)

    # prepare output file
    output_files = []
    output_dir = os.path.dirname(args.outfile)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i in range(len(base_classifiers)*2):
        f = open(args.outfile+"_"+str(i), 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
        output_files.append(f)
        
    results_list = []
    for i in range(len(base_classifiers)*2):
        results_list.append([])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != args.skip_offset:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the predictions of g around x
        x = x.cuda()
        certification_results = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()

        time_elapsed = datetime.timedelta(seconds=(after_time - before_time))
        for j, f in enumerate(output_files):
            prediction = certification_results[j][0]
            correct = int(prediction == label)
            radius = certification_results[j][1]
            
            # approximates running times, assuming that all individual models approximately require the same running time
            current_time_elapsed = time_elapsed
            if j % 2 == 0:
                current_time_elapsed /= len(base_classifiers)
            else:
                current_time_elapsed = ((j+1)/2) / len(base_classifiers) * time_elapsed
            current_time_elapsed = str(current_time_elapsed)
            
            results_list[j].append([i, label, prediction, radius, correct, current_time_elapsed])
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, current_time_elapsed), file=f, flush=True)
            
            if len(results_list[j]) % 100 == 0:
                np.save(args.outfile+"_"+str(j), np.array(results_list[j]))
        
        for j in range(len(base_classifiers)*2):
            np.save(args.outfile+"_"+str(j), np.array(results_list[j]))

    f.close()
