import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
import datetime
import csv
from itertools import product

LOGS_DIR = 'logs4_maze'

ENV = 'MazeF3-v0'

PER_MEASURED_STAT = 'reward'
PER_PRIORITY_FUNCTION = 'proportional'

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BETAS = [None, 0.25, 0.5, 0.75]

BPER_DIST = 'euclidean'

BPER_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BPER_KS = [0.25, 0.5, 1.0, 2.0, 5.0]
NR_CLUSTERS = [2, 3, 5, 7]

# MODE = 'explore'

# FILENAMES_EXPLORE = [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(ENV)))[0]),
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(ENV)))[0]),
# ] + [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_{}_bper_k_{}_nr_clusters_{}_dist__{}_*_acs2bper_explore_metrics.json'.format# (ENV, alpha, ks, nr_clusters, BPER_DIST)))[0]) for alpha, ks, nr_clusters in product(BPER_ALPHAS, BPER_KS, NR_CLUSTERS)
# ]
# 
# print(FILENAMES_EXPLORE)
# print(len(FILENAMES_EXPLORE))

# FILENAMES_EXPLOIT = [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(ENV)))[0]),
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(ENV)))[0]),
# ] + [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format# (ENV, alpha, ks, nr_clusters, BPER_DIST)))[0]) for alpha, ks, nr_clusters in product(BPER_ALPHAS, BPER_KS, NR_CLUSTERS)
# ]
# 
# print(FILENAMES_EXPLOIT)
# print(len(FILENAMES_EXPLOIT))

# assert len(FILENAMES_EXPLORE) == len(FILENAMES_EXPLOIT)

FILENAMES_EXPLORE = [
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(ENV)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(ENV)))[0]),
] + [
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(ENV, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION, alpha, beta)))[0]) for alpha, beta in product(ALPHAS, BETAS)
]

print(FILENAMES_EXPLORE)
print(len(FILENAMES_EXPLORE))

# FILENAMES_EXPLOIT = [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(ENV)))[0]),
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(ENV)))[0]),
# ] + [
#     os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_exploit_metrics.json'.format(ENV, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION, alpha, beta)))[0]) for alpha, beta in product(ALPHAS, BETAS)
# ]

# print(FILENAMES_EXPLOIT)
# print(len(FILENAMES_EXPLOIT))

# TITLES = [
#     'ACS2'
#     'ACS2ER',
# ] + [
#     'ACS2PER (alpha={}, beta={})'.format(alpha, beta) for alpha, beta in product(ALPHAS, BETAS)
# ]

COL_AGENTS = [
    'ACS2',
    'ACS2ER',
] + [
    'ACS2BPER'
] * len(BPER_ALPHAS) * len(BPER_KS) * len(NR_CLUSTERS)

COL_PARAMS = [
    '',
    '',
] + [
    'alpha={} | k={} | l. klastrów={}'.format(alpha, bper_k, nr_clusters) for alpha, bper_k, nr_clusters in product(BPER_ALPHAS, BPER_KS, NR_CLUSTERS)
]

def compute_classifiers(data):
    assert len(data) == 30
    avg_max_classifiers = mean([max([trial['population'] for trial in episode]) for episode in data])
    end_nr_classifers = mean([episode[-1]['population'] for episode in data])
    end_nr_reliables = mean([episode[-1]['reliable'] for episode in data])
    difference = end_nr_classifers - end_nr_reliables
    return avg_max_classifiers, end_nr_classifers, end_nr_reliables, difference


def compute_avg_metric(data, metric_key='reward'):
    numerator = 0.0
    denominator = 0.0
    for episode in data:
        for trial in episode:
            if metric_key in trial:
                numerator += trial[metric_key]
                denominator += 1.0
    if denominator == 0.0:
        return None
    return numerator / denominator


def compute_avg_episode_perf_time(data):
    numerator = 0.0
    denominator = 0.0
    for episode in data:
        denominator += 1.0
        for trial in episode:
            if 'perf_time' in trial:
                numerator += trial['perf_time']
    if denominator == 0.0:
        return None
    return numerator / denominator

def compute_avg_num_of_trials_for_knowledge_lvl(data, knowledge_lvl, lower_bound_est=False):
    numerator = 0.0
    denominator = 0.0
    for episode in data:
        smallest_trial = next((trial['trial'] for trial in episode if 'knowledge' in trial and trial['knowledge'] / 100.0 >= knowledge_lvl), None)
        # print(smallest_trial)
        if smallest_trial is not None:
            numerator += smallest_trial
            denominator += 1.0
        elif lower_bound_est:
            numerator += 300
            denominator += 1.0
    if denominator == 0.0:
        return None, None
    return numerator / denominator, denominator

OUTPUT_DIR = 'reports'

if __name__ == "__main__":
    print("REPORT FOR {}".format(ENV))
    print("=========================================================")
    # with open(os.path.join(OUTPUT_DIR, '{}_{}_{}_per_episode_perf_time_report.csv'.format(ENV, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION)), 'w') as csvfile:
    # with open(os.path.join(OUTPUT_DIR, '{}_{}_explore_bper_classifiers_report.csv'.format(ENV, BPER_DIST)), 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     # writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.', 'L. klas. wiarygodnych', 'Różnica'])
    #     writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.', 'L. klas. wiarygodnych', 'Różnica'])
    #     # for file_explore, file_exploit, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, COL_AGENTS, COL_PARAMS):
    #     for file_explore, col_agent, col_param in zip(FILENAMES_EXPLORE, COL_AGENTS, COL_PARAMS):
    #         # with open(os.path.join(LOGS_DIR, file_explore), 'r') as f_re, open(os.path.join(LOGS_DIR, file_exploit), 'r') as f_it:
    #         with open(os.path.join(LOGS_DIR, file_explore), 'r') as f_re:
    #             data_re = json.load(f_re)
    #             # data_it = json.load(f_it)
    #             stats = compute_classifiers(data_re)
    #             # stats_re = compute_avg_metric(data_re, 'steps_in_trial')
    #             # stats_it = compute_avg_metric(data_it, 'steps_in_trial')
    #             writer.writerow([col_agent, col_param] + list(stats) )
    #             # writer.writerow([col_agent, col_param, stats_re, stats_it] )
    #             # print(col_agent, col_param, stats_re, stats_it)
    #             # print(col_agent, col_param, stats)

    
    
    print("REPORT FOR {}".format(ENV))
    print("=========================================================")
    for file, title, param in list(zip(FILENAMES_EXPLORE, COL_AGENTS, COL_PARAMS)):
        print(title+param)
        print("FILE: {}".format(file))
        with open(os.path.join(LOGS_DIR, file), 'r') as f:
            data = json.load(f)
            print("AVG REWARD: {}".format(compute_avg_metric(data)))
            print("AVG STEPS IN TRIAL: {}".format(compute_avg_metric(data, 'steps_in_trial')))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 1.0)
            print("AVG TRIALS FOR 100% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.95)
            print("AVG TRIALS FOR 95% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.9)
            print("AVG TRIALS FOR 90% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.8)
            print("AVG TRIALS FOR 80% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.75)
            print("AVG TRIALS FOR 75% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            print()
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 1.0, lower_bound_est=True)
            print("AVG TRIALS FOR 100% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.95, lower_bound_est=True)
            print("AVG TRIALS FOR 95% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.9, lower_bound_est=True)
            print("AVG TRIALS FOR 90% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.8, lower_bound_est=True)
            print("AVG TRIALS FOR 80% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
            avg_knowledge, denominator = compute_avg_num_of_trials_for_knowledge_lvl(data, 0.75, lower_bound_est=True)
            print("AVG TRIALS FOR 75% KNOWLEDGE: {} ({} episodes)".format(avg_knowledge, denominator))
