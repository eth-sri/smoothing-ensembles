'''
- contains various functions to transform log files into valuable data for ensembles
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/analyze.py written by Jeremy Cohen
'''

import os
import numpy as np

# transforms log file into a numpy array
def get_data(file_path, data_type):
    nums = {'ensemble': 5, 'ensemble_k_consensus': 7}
    num = nums[data_type]
    f = open(file_path, 'r')
    lines = f.readlines()
    data = np.array([line.split('\t')[:num] for line in lines[1:]]).astype(float)
    return data

# computes certified accuracy at a given radius
def get_certified_accuracy(data, radius):
    certified_accuracy = 0
    for i in range(len(data)):
        if data[i][4] and data[i][3] >= radius:
            certified_accuracy += 1
    certified_accuracy = 100 * certified_accuracy / len(data)
    return certified_accuracy

# computes acr (average certified radius)
def get_acr(data):
    acr = 0.0
    for i in range(len(data)):
        if data[i][4]:
            acr += data[i][3]
    acr /= len(data)
    return acr

# computes kcr (percentages of perturbed samples where we stop after k models)
def get_kcr(data):
    return 100 * np.mean(data[:, 5])

# computes time needed for all samples [h]
def get_time(file_path, data_type):
    nums = {'ensemble': 5, 'ensemble_k_consensus': 7}
    num = nums[data_type]
    
    f = open(file_path, 'r')
    lines = f.readlines()
    time_list = np.array([line.split('\t')[:num+1] for line in lines[1:]]).astype(str)[:, num]
    
    seconds = 0.0
    for t in time_list:
        ts = [float(s) for s in t.split(':')]
        seconds += (((ts[0]*60)+ts[1])*60+ts[2])
    return seconds / 3600

# creates a latex table for ensemble logs, possibly with k-consensus (various parameter settings possible)
def get_latex_table_ensemble(output_path, file_paths, sigmas, radii, use_k_consensus=False, ks = [], time_baseline=1):
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("sigma")
    if use_k_consensus:
        f.write(" & k")
    f.write(" & ACR")
    for radius in radii:
        f.write(" & " + str(radius))
    if use_k_consensus:
        f.write(" KCR [%] & TimeRF")
    f.write(" & Time")
    f.write("\\\\\n\midrule\n")
    
    # add data to file
    for j, file_path in enumerate(file_paths):
        if not use_k_consensus:
            data = get_data(file_path, 'ensemble')
        else:
            data = get_data(file_path, 'ensemble_k_consensus')
        f.write("{}".format(sigmas[j]))
        if use_k_consensus:
            f.write(" & {}".format(ks[j]))
        acr = get_acr(data)
        f.write(" & {:.3f}".format(acr))
        for radius in radii:
            certified_accuracy = get_certified_accuracy(data, radius)
            f.write(" & {:.1f}".format(certified_accuracy))
        if not use_k_consensus:
            time_needed = get_time(file_path, 'ensemble')
        else:
            time_needed = get_time(file_path, 'ensemble_k_consensus')
            time_reduction_factor = time_baseline / time_needed
            kcr = get_kcr(data)
            f.write(" & {:.1f}".format(kcr))
            f.write(" & {:.2f}".format(time_reduction_factor))
        f.write(" & {:.2f}".format(time_needed))
        f.write("\\\\\n")
    f.close()
