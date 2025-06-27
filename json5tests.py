from scipy import stats
import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
import datetime
import csv
from itertools import product

LOGS_DIR = 'logs5_maze_750'

ENVS = ['Maze6-v0', 'MazeF3-v0', 'MazeT2-v0']

# ENVS = ['Maze6-v0']
# ENVS = ['corridor-20-v0']
# ENVS = ['boolean-multiplexer-6bit-v0', 'boolean-multiplexer-11bit-v0']

RSI = 250

PER_MEASURED_STATS = ['imm_reward', 'reward']
PER_PRIORITY_FUNCTIONS = ['proportional', 'ranked']

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BETAS = [None, 0.25, 0.5, 0.75]

BPER_DISTS = ['cityblock', 'euclidean']
BPER_KS = [0.25, 0.5, 1.0, 2.0, 5.0]
NR_CLUSTERS = [2, 3, 5, 7]

OUTPUT_DIR = 'tests'

metrics = [
    lambda data: [episode[-1]['knowledge'] for episode in data],                             # max_knowledges
    lambda data: [max([trial['population'] for trial in episode]) for episode in data],      # max population
    lambda data: [episode[-1]['population'] for episode in data],                            # end population
    lambda data: [episode[-1]['reliable'] for episode in data],                              # end reliable
    lambda data: [mean([trial['steps_in_trial'] for trial in episode]) for episode in data], # mean steps in trial
    lambda data: [mean([trial['reward'] for trial in episode]) for episode in data],         # mean reward
    lambda data: [mean([trial['perf_time'] for trial in episode]) for episode in data],      # mean perf_time                
]

metric_names = [
    'max_knowledge',
    'max_population',
    'end_population',
    'end_reliable',
    'mean_steps_in_trial',
    'mean_reward',
    'mean_perf_time',
]

p_val = 0.05

# x = [1, 1, 1]
# y = [2, 2, 2]
# 
# z = stats.kruskal(x, y)
# 
# print(z)
# print(z.statistic)
# print(z.pvalue)
# help(z)



if __name__ == "__main__":
    for env in ENVS:
        FILENAME_ER = os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0])

        FILENAMES_UT = [
            os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_{}_beta_{}_rsi_{}_*_acs2per_explore_metrics.json'.format(env, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION, alpha, beta, RSI)))[0]) for alpha, beta, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION in product(ALPHAS, BETAS, PER_MEASURED_STATS, PER_PRIORITY_FUNCTIONS)
        ] + [
            os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(env, alpha, bper_k, nr_clusters, BPER_DIST)))[0]) for alpha, bper_k, nr_clusters, BPER_DIST in product(ALPHAS, BPER_KS, NR_CLUSTERS, BPER_DISTS)
        ]

        print(len(FILENAMES_UT))
        print("p_val_corrected:", p_val / len(FILENAMES_UT))

        with open(os.path.join(LOGS_DIR, FILENAME_ER)) as file_er:
            data_er = json.load(file_er)

        datas_ut = []
        for filename_ut in FILENAMES_UT:
            with open(os.path.join(LOGS_DIR, filename_ut)) as file_ut:
                datas_ut.append(json.load(file_ut))
        
        COL_AGENTS = ['ACS2PER'] * len(ALPHAS) * len(BETAS) * len(PER_MEASURED_STATS) * len(PER_PRIORITY_FUNCTIONS) + ['ACS2BPER'] * len(ALPHAS) * len(BPER_KS) * len(NR_CLUSTERS) * len(BPER_DISTS)

        COL_PARAMS = [
            f'\'{PER_MEASURED_STAT}\' | \'{PER_PRIORITY_FUNCTION}\' | alpha_per={alpha} | beta_per={beta}' for alpha, beta, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION in product(ALPHAS, BETAS, PER_MEASURED_STATS, PER_PRIORITY_FUNCTIONS)
        ] + [
            f'\'{BPER_DIST}\' | alpha_bper={alpha} | bper_k={bper_k} | nr_clusters={nr_clusters}' for alpha, bper_k, nr_clusters, BPER_DIST in product(ALPHAS, BPER_KS, NR_CLUSTERS, BPER_DISTS)
        ]

        for func, func_name in zip(metrics, metric_names):
            print(f"Writing tests file for {env}, {func_name}")
            with open(os.path.join(OUTPUT_DIR, f'{env}_{func_name}_tests.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Statystyka T', 'Wartość p', 'Istotność'])

                for col_agent, col_param, data_ut in zip(COL_AGENTS, COL_PARAMS, datas_ut):
                    try:
                        test_result = stats.kruskal(func(data_er), func(data_ut))
                        writer.writerow([col_agent, col_param, round(test_result.statistic, 2), round(test_result.pvalue, 3) if test_result.pvalue >= 0.001 else "<0.001", '*' if test_result.pvalue < p_val / len(FILENAMES_UT) else ''])
                    except ValueError:
                        writer.writerow([col_agent, col_param, '-', '-', ''])
                    