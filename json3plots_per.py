import matplotlib.pyplot as plt
import os
import json
from statistics import mean, pstdev
import numpy as np
import glob

LOGS_DIR = 'logs4_maze'

ENV = 'MazeF3-v0'
MEASURED_STAT = 'imm_reward'
PRIORITY_FUNCTION = 'proportional'
# BETA = 0.75
ALPHA = 1.25
RECALC_SET_INTERVAL = 250

# SUBSTRING = 'beta_{}_rsi_{}'.format(BETA if BETA else None, RECALC_SET_INTERVAL)
SUBSTRING_ALPHA = 'alpha_{}'.format(ALPHA)
SUBSTRING_RSI = 'rsi_{}'.format(RECALC_SET_INTERVAL)

FILENAMES_EXPLORE = [
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(ENV)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(ENV)))[0]),
    
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.25_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.5_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.75_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_1.0_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_1.25_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),

    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_None_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.25_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.5_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.75_{}*_acs2per_explore_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
]

FILENAMES_EXPLOIT = [
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(ENV)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(ENV)))[0]),
    
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.25_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.5_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_0.75_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_1.0_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),
    # os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_1.25_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING)))[0]),

    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_None_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.25_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.5_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
    os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_{}_beta_0.75_{}*_acs2per_exploit_metrics.json'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, SUBSTRING_ALPHA, SUBSTRING_RSI)))[0]),
]



TITLE = '{}, {}, {}'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION)
# ADD_TITLE = ' (beta={}, recalc_set_interval={})'.format(BETA, RECALC_SET_INTERVAL)
ADD_TITLE = ' (alpha={}, recalc_set_interval={})'.format(ALPHA, RECALC_SET_INTERVAL)
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
    legend = ['ACS2', 'ACS2ER', 'ACS2PER, alpha=0.25', 'ACS2PER, alpha=0.5', 'ACS2PER, alpha=0.75', 'ACS2PER, alpha=1.0', 'ACS2PER, alpha=1.25']
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
    plt.savefig('../plots_pyalcs/output_mazeF3/per/different_betas/Figure_1_{}_{}_{}_alpha_{}_recalc_set_interval_{}_avg_window_{}.png'.format(ENV, MEASURED_STAT, PRIORITY_FUNCTION, ALPHA, RECALC_SET_INTERVAL, AVG_WINDOW_EXPLORE), dpi=300, bbox_inches='tight')
