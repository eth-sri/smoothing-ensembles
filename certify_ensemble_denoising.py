'''
- this is the file which does certification for the SmoothEnsembleDenoising class (smooth_ensemble_denoising.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_ensemble_denoising import SmoothEnsembleDenoising
from time import time
import torch
import datetime
from architectures import get_architecture
from archs.dncnn import DnCNN

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("core_classifier", type=str, help="path to saved core pytorch model (which was trained on unperturbed data)")
parser.add_argument("base_classifiers", type=str, help="path to saved pytorch models of base classifiers", nargs='+')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifiers
    base_classifiers = []
    for PATH in args.base_classifiers:
        checkpoint = torch.load(PATH)
        base_classifier = DnCNN(image_channels=3, depth=17, n_channels=128).cuda()
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifiers.append(base_classifier)
        
    # loading the core model
    checkpoint = torch.load(args.core_classifier)
    core_classifier = get_architecture(checkpoint["arch"], args.dataset)
    core_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = SmoothEnsembleDenoising(base_classifiers, core_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    output_files = []
    output_dir = os.path.dirname(args.outfile)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i in range(len(base_classifiers)*2):
        f = open(args.outfile+"_"+str(i), 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
        output_files.append(f)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        certification_results = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        for j, f in enumerate(output_files):
            prediction = certification_results[j][0]
            correct = int(prediction == label)
            radius = certification_results[j][1]
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()

