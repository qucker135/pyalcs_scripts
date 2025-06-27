import matplotlib.pyplot as plt
import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
import shutil
from itertools import product

LOGS_DIR = "logs5_mux_750"
OUTPUT_DIR = "plots"

RSI = 50

# ENVS = ['Maze6-v0', 'MazeF3-v0', 'MazeT2-v0']

# ENVS = ['Maze6-v0']

# ENVS = ['corridor-20-v0']

ENVS = ['boolean-multiplexer-6bit-v0', 'boolean-multiplexer-11bit-v0']

AVG_WINDOW_EXPLORE = 10
AVG_WINDOW_EXPLOIT = 1

# For Maze & corridor
# METRIC_KEYS = ['steps_in_trial', 'reward', 'perf_time', 'population', 'reliable', 'knowledge']
# METRIC_NAMES = ['LICZBA KROKÓW', 'NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW', 'LICZBA WIARYGODNYCH KLASYFIKATORÓW', 'WIEDZA [%]']

# For mux
METRIC_KEYS = ['reward', 'perf_time', 'population']
METRIC_NAMES = ['NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW']


MEASURED_STATS = ['imm_reward', 'reward']
PRIORITY_FUNCTIONS = ['proportional', 'ranked']

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BETAS = [None, 0.25, 0.5, 0.75]

DIST_TYPES = ['cityblock', 'euclidean']
BPER_KS = [0.25, 0.5, 1.0, 2.0, 5.0]
NRS_OF_CLUSTERS = [2, 3, 5, 7]

# SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
# SUBSTRING_BETA = 'beta_{}'.format(BETA)
SUBSTRING_RSI = 'rsi_{}'.format(RSI)

# TITLE = '{}, {}, {}'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION)
# ADD_TITLE_BETA = ' (beta={}, recalc_set_interval={})'.format(BETA, RECALC_SET_INTERVAL)
# ADD_TITLE_ALPHA = ' (alpha={}, recalc_set_interval={})'.format(ALPHA, RECALC_SET_INTERVAL)

if __name__ == "__main__":
    for env in ENVS:
        # os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}"), exist_ok=True)
        # os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "per"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "per", "different_alphas"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "per", "different_betas"), exist_ok=True)
        # os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "bper"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_bper_alpha"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_bper_k"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_nr_clusters"), exist_ok=True)

        # PER - different alphas
        colors = ["black", "red", "orange", "yellow", "lime", "green", "blue", "purple"]
        legend = ['ACS2', 'ACS2ER']
        for ALPHA in ALPHAS:
            legend.append('ACS2PER, alpha={}'.format(ALPHA))
        for BETA, MEASURED_STAT, PRIORITY_FUNCTION in product(BETAS, MEASURED_STATS, PRIORITY_FUNCTIONS):
            print(f"Processing {env}, {MEASURED_STAT}, {PRIORITY_FUNCTION}, BETA={BETA}")
            SUBSTRING_BETA = 'beta_{}'.format(BETA if BETA is not None else 'None')

            TITLE = '{}, {}, {}'.format(env, MEASURED_STAT, PRIORITY_FUNCTION)
            ADD_TITLE_BETA = ' (beta_per={}, rsi_per={})'.format(BETA if BETA is not None else 'OFF', RSI)
            
            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ]
            for ALPHA in ALPHAS:
                SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
                # print(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2per_explore_metrics.json'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_BETA))))
                FILENAMES_EXPLORE.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}_{}*_acs2per_explore_metrics.json'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_BETA, SUBSTRING_RSI)))[0]))
                FILENAMES_EXPLOIT.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}_{}*_acs2per_exploit_metrics.json'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_BETA, SUBSTRING_RSI)))[0]))
            
            assert len(FILENAMES_EXPLORE) == len(ALPHAS) + 2, "Number of FILENAMES_EXPLORE does not match number of ALPHAS"
            assert len(FILENAMES_EXPLOIT) == len(ALPHAS) + 2, "Number of FILENAMES_EXPLOIT does not match number of ALPHAS"

            handles = []

            fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
            fig.suptitle(TITLE+ADD_TITLE_BETA)

            for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
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
                    axs[0][metric_id].set_title(metric_key)
                    axs[0][metric_id].set_xlabel('EPIZOD')
                    axs[0][metric_id].set_ylabel(metric_name)

            # print("Exploit files:", FILENAMES_EXPLOIT)

            for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)

                # print("Processing file:", file)
                # print("Data length:", len(data))
                # print("Data[0]:", data[0])
                xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
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

                    # print("Metric mean:", metric_mean)
                    # print("Metric std:", metric_std)
                    # print("X values:", list(xs))

                    line, = axs[1][metric_id].plot(xs, metric_mean, color=color)
                    # axs[1][metric_id].plot(xs, metric_mean, color=color)
                    axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                    axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')

                    axs[1][metric_id].set_title(metric_key)
                    axs[1][metric_id].set_xlabel('EPIZOD')
                    axs[1][metric_id].set_ylabel(metric_name)
            fig.legend(handles, legend, loc='outside upper center', bbox_to_anchor=(0.5, .95), ncol=len(legend), fontsize='small')
            # axs[1][1].legend(handles, legend)

            # plt.savefig('../plots_pyalcs/output_mazeF3/per/different_betas/Figure_1_{}_{}_{}_alpha_{}_recalc_set_interval_{}_avg_window_{}.png'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, ALPHA, RECALC_SET_INTERVAL, AVG_WINDOW_EXPLORE), dpi=300, bbox_inches='tight')

            plt.savefig(os.path.join(OUTPUT_DIR, f"output_{env}", "per", "different_alphas", 'Figure_1_{}_{}_{}_{}_{}_avg_window_{}.png'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_BETA, SUBSTRING_RSI, AVG_WINDOW_EXPLORE)), dpi=300, bbox_inches='tight')

            plt.close(fig)  # Close the figure to free memory

        # PER - different betas
        legend = ['ACS2', 'ACS2ER']
        for BETA in BETAS:
            legend.append('ACS2PER, beta={}'.format(BETA if BETA is not None else 'OFF'))
        for ALPHA, MEASURED_STAT, PRIORITY_FUNCTION in product(ALPHAS, MEASURED_STATS, PRIORITY_FUNCTIONS):
            print(f"Processing {env}, {MEASURED_STAT}, {PRIORITY_FUNCTION}, ALPHA={ALPHA}")
            SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)

            TITLE = '{}, {}, {}'.format(env, MEASURED_STAT, PRIORITY_FUNCTION)
            ADD_TITLE_ALPHA = ' (alpha_per={}, rsi_per={})'.format(ALPHA, RSI)

            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ]
            for BETA in BETAS:
                SUBSTRING_BETA = 'beta_{}'.format(BETA if BETA is not None else 'None')
                FILENAMES_EXPLORE.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}_{}*_acs2per_explore_metrics.json'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_BETA, SUBSTRING_RSI)))[0]))
                FILENAMES_EXPLOIT.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}_{}*_acs2per_exploit_metrics.json'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_BETA, SUBSTRING_RSI)))[0]))

            assert len(FILENAMES_EXPLORE) == len(BETAS) + 2, "Number of FILENAMES_EXPLORE does not match number of BETAS"
            assert len(FILENAMES_EXPLOIT) == len(BETAS) + 2, "Number of FILENAMES_EXPLOIT does not match number of BETAS"

            handles = []

            fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
            fig.suptitle(TITLE+ADD_TITLE_ALPHA)

            for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
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
                    axs[0][metric_id].set_title(metric_key)
                    axs[0][metric_id].set_xlabel('EPIZOD')
                    axs[0][metric_id].set_ylabel(metric_name)

            for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)

                xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
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

                    line, = axs[1][metric_id].plot(xs, metric_mean, color=color)
                    axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                    axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')

                    axs[1][metric_id].set_title(metric_key)
                    axs[1][metric_id].set_xlabel('EPIZOD')
                    axs[1][metric_id].set_ylabel(metric_name)

            fig.legend(handles, legend, loc='outside upper center', bbox_to_anchor=(0.5, .95), ncol=len(legend), fontsize='small')

            plt.savefig(os.path.join(OUTPUT_DIR, f"output_{env}", "per", "different_betas", 'Figure_2_{}_{}_{}_{}_{}_avg_window_{}.png'.format(env, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI, AVG_WINDOW_EXPLORE)), dpi=300, bbox_inches='tight')

            plt.close(fig)  # Close the figure to free memory


        
        # BPER - different alphas
        legend = ['ACS2', 'ACS2ER']
        for ALPHA in ALPHAS:
            legend.append('ACS2BPER, alpha={}'.format(ALPHA))
        for DIST_TYPE, BPER_K, NR_OF_CLUSTERS in product(DIST_TYPES, BPER_KS, NRS_OF_CLUSTERS):
            print(f"Processing {env}, DIST_TYPE={DIST_TYPE}, BPER_K={BPER_K}, NR_OF_CLUSTERS={NR_OF_CLUSTERS}")
            SUBSTRING_DIST_TYPE = 'dist_{}'.format(DIST_TYPE)
            SUBSTRING_BPER_K = 'bper_k_{}'.format(BPER_K)
            SUBSTRING_NR_OF_CLUSTERS = 'nr_clusters_{}'.format(NR_OF_CLUSTERS)

            TITLE = '{}, {} (bper_k={}, nr_clusters={})'.format(env, DIST_TYPE, BPER_K, NR_OF_CLUSTERS)
            # ADD_TITLE_ALPHA = ' (alpha_bper={}, rsi_bper={})'.format(ALPHA, RSI)

            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ]
            for ALPHA in ALPHAS:
                SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
                FILENAMES_EXPLORE.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_explore_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))
                FILENAMES_EXPLOIT.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_exploit_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))
                

            assert len(FILENAMES_EXPLORE) == len(ALPHAS) + 2, "Number of FILENAMES_EXPLORE does not match number of ALPHAS"
            assert len(FILENAMES_EXPLOIT) == len(ALPHAS) + 2, "Number of FILENAMES_EXPLOIT does not match number of ALPHAS"

            handles = []
            fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
            fig.suptitle(TITLE)

            for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
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
                    axs[0][metric_id].set_title(metric_key)
                    axs[0][metric_id].set_xlabel('EPIZOD')
                    axs[0][metric_id].set_ylabel(metric_name)

            for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
                xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
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

                    line, = axs[1][metric_id].plot(xs, metric_mean, color=color)
                    axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                    axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')

                    axs[1][metric_id].set_title(metric_key)
                    axs[1][metric_id].set_xlabel('EPIZOD')
                    axs[1][metric_id].set_ylabel(metric_name)

            fig.legend(handles, legend, loc='outside upper center', bbox_to_anchor=(0.5, .95), ncol=len(legend), fontsize='small')

            plt.savefig(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_bper_alpha", 'Figure_3_{}_{}_{}_{}_avg_window_{}.png'.format(env, SUBSTRING_DIST_TYPE, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, AVG_WINDOW_EXPLORE)), dpi=300, bbox_inches='tight')

            plt.close(fig)  # Close the figure to free memory




        # BPER - different bper_k
        legend = ['ACS2', 'ACS2ER']
        for BPER_K in BPER_KS:
            legend.append('ACS2BPER, bper_k={}'.format(BPER_K))
        for ALPHA, DIST_TYPE, NR_OF_CLUSTERS in product(ALPHAS, DIST_TYPES, NRS_OF_CLUSTERS):
            print(f"Processing {env}, ALPHA={ALPHA}, DIST_TYPE={DIST_TYPE}, NR_OF_CLUSTERS={NR_OF_CLUSTERS}")
            SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
            SUBSTRING_DIST_TYPE = 'dist_{}'.format(DIST_TYPE)
            SUBSTRING_NR_OF_CLUSTERS = 'nr_clusters_{}'.format(NR_OF_CLUSTERS)
            TITLE = '{}, {} (alpha= {}, nr_clusters={})'.format(env, DIST_TYPE, ALPHA, NR_OF_CLUSTERS)

            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ]

            for BPER_K in BPER_KS:
                SUBSTRING_BPER_K = 'bper_k_{}'.format(BPER_K)
                FILENAMES_EXPLORE.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_explore_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))
                FILENAMES_EXPLOIT.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_exploit_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))

            assert len(FILENAMES_EXPLORE) == len(BPER_KS) + 2, "Number of FILENAMES_EXPLORE does not match number of BPER_KS"
            assert len(FILENAMES_EXPLOIT) == len(BPER_KS) + 2, "Number of FILENAMES_EXPLOIT does not match number of BPER_KS"

            handles = []

            fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
            fig.suptitle(TITLE)

            for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
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
                    axs[0][metric_id].set_title(metric_key)
                    axs[0][metric_id].set_xlabel('EPIZOD')
                    axs[0][metric_id].set_ylabel(metric_name)
            
            for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
                xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
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

                    line, = axs[1][metric_id].plot(xs, metric_mean, color=color)
                    axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                    axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')

                    axs[1][metric_id].set_title(metric_key)
                    axs[1][metric_id].set_xlabel('EPIZOD')
                    axs[1][metric_id].set_ylabel(metric_name)

            fig.legend(handles, legend, loc='outside upper center', bbox_to_anchor=(0.5, .95), ncol=len(legend), fontsize='small')

            plt.savefig(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_bper_k", 'Figure_4_{}_{}_{}_{}_avg_window_{}.png'.format(env, SUBSTRING_DIST_TYPE, SUBSTRING_ALPHA, SUBSTRING_NR_OF_CLUSTERS, AVG_WINDOW_EXPLORE)), dpi=300, bbox_inches='tight')

            plt.close(fig)  # Close the figure to free memory




        # BPER - different nr_clusters
        legend = ['ACS2', 'ACS2ER']
        for NR_OF_CLUSTERS in NRS_OF_CLUSTERS:
            legend.append('ACS2BPER, nr_clusters={}'.format(NR_OF_CLUSTERS))
        for ALPHA, DIST_TYPE, BPER_K in product(ALPHAS, DIST_TYPES, BPER_KS):
            print(f"Processing {env}, ALPHA={ALPHA}, DIST_TYPE={DIST_TYPE}, BPER_K={BPER_K}")
            SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
            SUBSTRING_DIST_TYPE = 'dist_{}'.format(DIST_TYPE)
            SUBSTRING_BPER_K = 'bper_k_{}'.format(BPER_K)
            TITLE = '{}, {} (alpha= {}, bper_k={})'.format(env, DIST_TYPE, ALPHA, BPER_K)

            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ]

            for NR_OF_CLUSTERS in NRS_OF_CLUSTERS:
                SUBSTRING_NR_OF_CLUSTERS = 'nr_clusters_{}'.format(NR_OF_CLUSTERS)
                FILENAMES_EXPLORE.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_explore_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))
                FILENAMES_EXPLOIT.append(os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_{}*_acs2bper_exploit_metrics.json'.format(env, SUBSTRING_ALPHA, SUBSTRING_BPER_K, SUBSTRING_NR_OF_CLUSTERS, SUBSTRING_DIST_TYPE)))[0]))

            assert len(FILENAMES_EXPLORE) == len(NRS_OF_CLUSTERS) + 2, "Number of FILENAMES_EXPLORE does not match number of NRS_OF_CLUSTERS"
            assert len(FILENAMES_EXPLOIT) == len(NRS_OF_CLUSTERS) + 2, "Number of FILENAMES_EXPLOIT does not match number of NRS_OF_CLUSTERS"

            handles = []

            fig, axs = plt.subplots(2, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 2*5))
            fig.suptitle(TITLE)

            for file, caption, color in zip(FILENAMES_EXPLORE, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
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
                    axs[0][metric_id].set_title(metric_key)
                    axs[0][metric_id].set_xlabel('EPIZOD')
                    axs[0][metric_id].set_ylabel(metric_name)
                
            for file, caption, color in zip(FILENAMES_EXPLOIT, legend, colors):
                with open(os.path.join(LOGS_DIR, file), 'r') as f:
                    data = json.load(f)
                xs = range(1, len(data[0]) + 2 - AVG_WINDOW_EXPLOIT)
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

                    line, = axs[1][metric_id].plot(xs, metric_mean, color=color)
                    axs[1][metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
                    axs[1][metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')

                    axs[1][metric_id].set_title(metric_key)
                    axs[1][metric_id].set_xlabel('EPIZOD')
                    axs[1][metric_id].set_ylabel(metric_name)

            fig.legend(handles, legend, loc='outside upper center', bbox_to_anchor=(0.5, .95), ncol=len(legend), fontsize='small')

            plt.savefig(os.path.join(OUTPUT_DIR, f"output_{env}", "bper", "different_nr_clusters", 'Figure_5_{}_{}_{}_{}_avg_window_{}.png'.format(env, SUBSTRING_DIST_TYPE, SUBSTRING_ALPHA, SUBSTRING_BPER_K, AVG_WINDOW_EXPLORE)), dpi=300, bbox_inches='tight')

            plt.close(fig)

    print("Alles gute, Volks!")
