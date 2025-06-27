import matplotlib.pyplot as plt
import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
from itertools import product

LOGS_DIR = 'logs4_maze'

ENV = 'MazeF3-v0'
DIST_TYPES = ['cityblock', 'euclidean']
# MEASURED_STAT = 'imm_reward'
# PRIORITY_FUNCTION = 'proportional'
# BETA = 0.75
# different bper_ks
# ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BPER_KS = [0.25, 0.5, 1.0, 2.0, 5.0]
NRS_OF_CLUSTERS = [2, 3, 5, 7]

# RECALC_SET_INTERVAL = 250

for DIST_TYPE, BPER_K, NR_OF_CLUSTERS in product(DIST_TYPES, BPER_KS, NRS_OF_CLUSTERS):

    print("DIST_TYPE: {}, BPER_K: {}, NR_OF_CLUSTERS: {}".format(DIST_TYPE, BPER_K, NR_OF_CLUSTERS))

    # SUBSTRING = 'beta_{}_rsi_{}'.format(BETA if BETA else None, RECALC_SET_INTERVAL)
    # SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
    SUBSTRING_BPER_K = 'bper_k_{}'.format(BPER_K)
    SUBSTRING_NR_OF_CLUSTERS = 'nr_clusters_{}'.format(NR_OF_CLUSTERS)

    FILENAMES_EXPLORE = [
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(ENV)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(ENV)))[0]),

        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.25_{}_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.5_{}_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.75_{}_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_1.0_{}_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_1.25_{}_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),

        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_0.25_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_0.5_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_1.0_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_2.0_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_5.0_{}_dist__{}_*_acs2bper_explore_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
    ]

    FILENAMES_EXPLOIT = [
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(ENV)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(ENV)))[0]),

        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_nr_clusters_2_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_BPER_K, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_nr_clusters_3_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_BPER_K, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_nr_clusters_5_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_BPER_K, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_nr_clusters_7_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_BPER_K, DIST_TYPE)))[0]),
    
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.25_{}_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.5_{}_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_0.75_{}_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_1.0_{}_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_1.25_{}_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),

        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_0.25_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_0.5_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_1.0_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_2.0_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
        # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_bper_k_5.0_{}_dist_{}_*_acs2bper_exploit_metrics.json'.format(ENV, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, DIST_TYPE)))[0]),
    ]



    TITLE = '{}, {}'.format(ENV, DIST_TYPE)
    # ADD_TITLE = ' (beta={}, recalc_set_interval={})'.format(BETA, RECALC_SET_INTERVAL)
    ADD_TITLE = ' (bper_k={}, nr_clusters={})'.format(BPER_K, NR_OF_CLUSTERS)
    AVG_WINDOW_EXPLORE = 25
    AVG_WINDOW_EXPLOIT = 1
    # for maze & corridor
    METRIC_KEYS = ['steps_in_trial', 'reward', 'perf_time', 'population', 'reliable', 'knowledge']
    METRIC_NAMES = ['LICZBA KROKÓW', 'NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW', 'LICZBA WIARYGODNYCH KLASYFIKATORÓW', 'WIEDZA [%]']

    # for mux
    # METRIC_KEYS = ['reward', 'perf_time', 'population']
    # METRIC_NAMES = ['NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW']

    if __name__ == "__main__":
        assert len(METRIC_KEYS) == len(METRIC_NAMES), "METRIC_KEYS and METRIC_NAMES must have the same length"
        assert len(FILENAMES_EXPLORE) == len(FILENAMES_EXPLOIT), "FILENAMES_EXPLORE and FILENAMES_EXPLOIT must have the same length"
        assert len(FILENAMES_EXPLORE) > 0, "FILENAMES_EXPLORE and FILENAMES_EXPLOIT must have more than 0 elements"
        print(FILENAMES_EXPLORE)
        print(len(FILENAMES_EXPLORE))
        print(FILENAMES_EXPLOIT)
        print(len(FILENAMES_EXPLOIT))
        fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
        fig.suptitle(TITLE+ADD_TITLE)
        colors = ["black", "red", "orange", "yellow", "lime", "green", "blue", "purple"]
        legend = ['ACS2', 'ACS2ER', 'ACS2BPER, alpha=0.25', 'ACS2BPER, alpha=0.5', 'ACS2BPER, alpha=0.75', 'ACS2BPER, alpha=1.0', 'ACS2BPER, alpha=1.25']
        # legend = ['ACS2', 'ACS2ER', 'ACS2PER, beta OFF', 'ACS2PER, beta=0.25', 'ACS2PER, beta=0.5', 'ACS2PER, beta=0.75']

        handles = []

        for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
            path = os.path.join(LOGS_DIR, file)
            with open(path, 'r') as fp:
                data = json.load(fp)
            xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLORE)
            for metric_id, (metric_key, metric_name) in enumerate(zip(METRIC_KEYS, METRIC_NAMES)):
                metric_mean = np.array([
                    mean(
                        [
                            mean(
                                [
                                    run[j][metric_key]
                                    for run in data
                                ]
                            )
                            for j in range(i, i + AVG_WINDOW_EXPLORE)
                        ]
                    )
                    for i in range(len(data[0]) + 1 - AVG_WINDOW_EXPLORE)
                ])
                metric_std = np.array([
                    mean(
                        [
                            pstdev(
                                [
                                    run[j][metric_key]
                                    for run in data
                                ]
                            )
                            for j in range(i, i + AVG_WINDOW_EXPLORE)
                        ]
                    )
                    for i in range(len(data[0]) + 1 - AVG_WINDOW_EXPLORE)
                ])
                line, = axs[0][metric_id].plot(xs, metric_mean, color=color)
                if metric_id == 0:
                    handles.append(line)
                axs[0][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                axs[0][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')
                # axs[metric_id].fill_between(xs,
                #                             metric_mean - metric_std,
                #                             metric_mean + metric_std,
                #                             facecolor='none',
                #                             edgecolor=color,
                #                             # alpha=0.01,
                #                             linestyle='dotted',
                #                             linewidth=1,
                #                             # label=None,
                #                             )
                axs[0][metric_id].set_title(metric_key)
                axs[0][metric_id].set_xlabel('EPIZOD')
                axs[0][metric_id].set_ylabel(metric_name)

        for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
            path = os.path.join(LOGS_DIR, file)
            with open(path, 'r') as fp:
                data = json.load(fp)
            xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
            for metric_id, (metric_key, metric_name) in enumerate(zip(METRIC_KEYS, METRIC_NAMES)):
                # metric_values = np.array([
                #     result[metric_key] for result in data
                # ])
                metric_mean = np.array([
                    mean(
                        [
                            mean(
                                [
                                    run[j][metric_key]
                                    for run in data
                                ]
                            )
                            for j in range(i, i + AVG_WINDOW_EXPLOIT)
                        ]
                    )
                    for i in range(len(data[0]) + 1 - AVG_WINDOW_EXPLOIT)
                ])
                metric_std = np.array([
                    mean(
                        [
                            pstdev(
                                [
                                    run[j][metric_key]
                                    for run in data
                                ]
                            )
                            for j in range(i, i + AVG_WINDOW_EXPLOIT)
                        ]
                    )
                    for i in range(len(data[0]) + 1 - AVG_WINDOW_EXPLOIT)
                ])
                # line, = axs[metric_id].plot(xs, metric_mean, color=color)

                # axs[1][metric_id].plot(xs, metric_values, color=color)


                line, = axs[1][metric_id].plot(xs, metric_mean, color=color)

                # if metric_id == 0:
                #     handles.append(line)
                axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')
                # axs[metric_id].fill_between(xs,
                #                             metric_mean - metric_std,
                #                             metric_mean + metric_std,
                #                             facecolor='none',
                #                             edgecolor=color,
                #                             # alpha=0.01,
                #                             linestyle='dotted',
                #                             linewidth=1,
                #                             # label=None,
                #                             )
                axs[1][metric_id].set_title(metric_key)
                axs[1][metric_id].set_xlabel('EPIZOD')
                axs[1][metric_id].set_ylabel(metric_name)
        axs[0][0].legend(handles, legend)
        axs[1][1].legend(handles, legend)
        # plt.show()
        # plt.savefig('../plots_pyalcs/output_mazeF3/per/different_alphas/Figure_1_{}_{}_{}_beta_{}_recalc_set_interval_{}_avg_window_{}.png'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, BETA, RECALC_SET_INTERVAL, AVG_WINDOW_EXPLORE), dpi=300, bbox_inches='tight')
        plt.savefig('../plots_pyalcs/output_mazeF3/bper/different_bper_alpha/Figure_1_{}_{}_bper_k_{}_nr_clusters_{}_avg_window_{}.png'.format(ENV, DIST_TYPE, BPER_K, NR_OF_CLUSTERS, AVG_WINDOW_EXPLORE), dpi=300, bbox_inches='tight')
