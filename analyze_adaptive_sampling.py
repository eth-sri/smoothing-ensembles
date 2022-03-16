'''
- contains various functions to transform log files into valuable data for adaptive sampling
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/analyze.py written by Jeremy Cohen
'''

import os
import numpy as np

# transforms log file into a numpy array
def get_data(file_path, data_type):
    nums = {'adaptive': 8, 'adaptive_and_k_consensus': 9}
    num = nums[data_type]
    f = open(file_path, 'r')
    lines = f.readlines()
    data = np.array([line.split('\t')[:num] for line in lines[1:]]).astype(float)
    return data

# computes certified accuracy for given radius
def get_certified_accuracy(data):
    return np.mean([1 if row[3] and row[4] else 0 for row in data])

# computes the sample reduction factor compared to sampling with n0 and n
def get_sample_reduction_factor(data, n0=100, n=100000):
    samples_needed = np.sum([row[7] for row in data])
    return (n+n0) * len(data) / samples_needed

# computes the ratios of the phases we stopped in
def get_step_percentages(data, num_steps = 3):
    steps = np.zeros(num_steps)
    for i in range(len(data)):
        steps[int(data[i][5])] += 1
    return 100 * steps / len(data)

# computes the ratios of the phases we stopped in (differentiates between certification and abstain)
def get_step_percentages_detailed(data, num_steps = 3):
    steps = np.zeros((num_steps, 2))
    for i in range(len(data)):
        if data[i][2] != -1:
            steps[int(data[i][5])][0] += 1
        else:
            steps[int(data[i][5])][1] += 1
    return 100 * steps / len(data)

# returns the ratio for which only the first k samples were evaluated
def get_kcr(data):
    samples_needed = np.sum(data[:, 7])
    early_stoppings = np.sum(data[:, 8])
    return 100 * early_stoppings / samples_needed

# computes time needed for all samples [h]
def get_time(file_path, data_type):
    nums = {'adaptive': 8, 'adaptive_and_k_consensus': 9}
    num = nums[data_type]
    
    f = open(file_path, 'r')
    lines = f.readlines()
    time_list = np.array([line.split('\t')[:num+1] for line in lines[1:]]).astype(str)[:, num]
    
    seconds = 0.0
    for t in time_list:
        ts = [float(s) for s in t.split(':')]
        seconds += (((ts[0]*60)+ts[1])*60+ts[2])
    return seconds / 3600

# generates latex table from adaptive sampling log files (various parameter settings are possible)
def get_latex_table_adaptive(output_path, file_paths, radii, sigmas, schedules, num_steps=3, detailed=False, time_baseline=1, n0=100, n=100000):
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("Radius & sigma & Schedule & Certified Accuracy [\%] ")
    if not detailed:
        for i in range(1, num_steps+1):
            f.write("& ACR" + str(i) + " ")
    else:
        for i in range(1, num_steps+1):
            f.write("& ACRC" + str(i) + " ")
            f.write("& ACRA" + str(i) + " ")
    f.write("& SampleRF & TimeRF & Time")
    f.write("\\\\\n\midrule\n")
    
    # add data to file
    for j, file_path in enumerate(file_paths):
        data = get_data(file_path, "adaptive")
        f.write("{:.2f}".format(radii[j]))
        f.write(" & {:.2f}".format(sigmas[j]))
        f.write(" & " + schedules[j])
        certified_accuracy = get_certified_accuracy(data)
        f.write(" & {:.1f}".format(100*certified_accuracy))
        if not detailed:
            step_ratios = get_step_percentages(data, num_steps)
            for i in range(num_steps):
                f.write(" & {:.1f}".format(step_ratios[i]))
        else:
            step_ratios_detailed = get_step_percentages_detailed(data, num_steps)
            for i in range(num_steps):
                f.write(" & {:.1f}".format(step_ratios_detailed[i][0]))
                f.write(" & {:.1f}".format(step_ratios_detailed[i][1]))
        sample_reduction_factor = get_sample_reduction_factor(data, n0, n)
        f.write(" & {:.2f}".format(sample_reduction_factor))
    
        time_needed = get_time(file_path, "adaptive")
        time_reduction_factor = time_baseline / time_needed
        f.write(" & {:.2f}".format(time_reduction_factor))
        f.write(" & {:.2f}".format(time_needed))
        f.write("\\\\\n")
    f.close()
    
 # generates latex table from adaptive sampling and k consensus log files (various parameter settings are possible)
def get_latex_table_adaptive_and_k_consensus(output_path, file_paths, radii, sigmas, schedules, num_steps=3, detailed=False, time_baseline=1, n0=100, n=100000):
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("Radius & sigma & Schedule & Certified Accuracy [\%] ")
    if not detailed:
        for i in range(1, num_steps+1):
            f.write("& ACR" + str(i) + " ")
    else:
        for i in range(1, num_steps+1):
            f.write("& ACRC" + str(i) + " ")
            f.write("& ACRA" + str(i) + " ")
    f.write("& KCR [\%] & SampleRF & TimeRF & Time")
    f.write("\\\\\n\midrule\n")
    
    # add data to file
    for j, file_path in enumerate(file_paths):
        data = get_data(file_path, "adaptive_and_k_consensus")
        f.write("{:.2f}".format(radii[j]))
        f.write(" & {:.2f}".format(sigmas[j]))
        f.write(" & " + schedules[j])
        certified_accuracy = get_certified_accuracy(data)
        f.write(" & {:.1f}".format(100*certified_accuracy))
        if not detailed:
            step_ratios = get_step_percentages(data, num_steps)
            for i in range(num_steps):
                f.write(" & {:.1f}".format(step_ratios[i]))
        else:
            step_ratios_detailed = get_step_percentages_detailed(data, num_steps)
            for i in range(num_steps):
                f.write(" & {:.1f}".format(step_ratios_detailed[i][0]))
                f.write(" & {:.1f}".format(step_ratios_detailed[i][1]))
        kcr = get_kcr(data)
        f.write(" & {:.1f}".format(kcr))
        sample_reduction_factor = get_sample_reduction_factor(data, n0, n)
        f.write(" & {:.2f}".format(sample_reduction_factor))
        time_needed = get_time(file_path, "adaptive_and_k_consensus")
        time_reduction_factor = time_baseline / time_needed
        f.write(" & {:.2f}".format(time_reduction_factor))
        f.write(" & {:.2f}".format(time_needed))
        f.write("\\\\\n")
    f.close()

