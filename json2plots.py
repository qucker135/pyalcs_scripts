import matplotlib.pyplot as plt
import os
import json
from statistics import mean, pstdev
import numpy as np


LOGS_DIR = 'logs1_maze7'
# FILENAMES = ['logs_maze_er_07_03_2025__01_50_02.json', # ER
#              'logs_maze_per_imm_reward_proportional_07_03_2025__04_18_07.json', # PER 0.2
#              'logs_maze_per_imm_reward_proportional_07_03_2025__07_27_50.json', # PER 0.4
#              'logs_maze_per_imm_reward_proportional_07_03_2025__10_30_02.json', # PER 0.6
#              'logs_maze_per_imm_reward_proportional_07_03_2025__13_56_08.json', # PER 0.8
#              'logs_maze_per_imm_reward_proportional_07_03_2025__17_13_24.json', # PER 1.0
#              'logs_maze_per_imm_reward_proportional_07_03_2025__20_33_40.json', # PER 1.2
#              'logs_maze_per_imm_reward_proportional_07_03_2025__23_46_38.json'  # PER 1.4
#              ]

# FILENAMES = [
#                 'logs_maze_er_07_03_2025__01_50_02.json', # ER
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__04_36_16.json', # PER 0.2
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__08_01_11.json', # PER 0.4
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__11_28_25.json', # PER 0.6
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__14_52_05.json', # PER 0.8
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__18_38_30.json', # PER 1.0
#                 'logs_maze_per_imm_reward_ranked_07_03_2025__22_40_50.json', # PER 1.2
#                 'logs_maze_per_imm_reward_ranked_08_03_2025__03_27_50.json'  # PER 1.4
#             ]

# FILENAMES = [
#                 'logs_maze_er_07_03_2025__01_50_02.json', # ER
#                 'logs_maze_per_reward_proportional_07_03_2025__06_31_52.json', # PER 0.2
#                 'logs_maze_per_reward_proportional_07_03_2025__11_42_48.json', # PER 0.4
#                 'logs_maze_per_reward_proportional_07_03_2025__17_01_34.json', # PER 0.6
#                 'logs_maze_per_reward_proportional_07_03_2025__22_33_55.json', # PER 0.8
#                 'logs_maze_per_reward_proportional_08_03_2025__03_47_22.json', # PER 1.0
#                 'logs_maze_per_reward_proportional_08_03_2025__08_54_17.json', # PER 1.2
#                 'logs_maze_per_reward_proportional_08_03_2025__14_01_15.json'  # PER 1.4
# ]

# FILENAMES = [
#                 'logs_maze_er_07_03_2025__01_50_02.json', # ER
#                 'logs_maze_per_reward_ranked_07_03_2025__06_32_06.json', # PER 0.2
#                 'logs_maze_per_reward_ranked_07_03_2025__11_48_03.json', # PER 0.4
#                 'logs_maze_per_reward_ranked_07_03_2025__17_27_36.json', # PER 0.6
#                 'logs_maze_per_reward_ranked_07_03_2025__22_50_58.json', # PER 0.8
#                 'logs_maze_per_reward_ranked_08_03_2025__04_39_49.json', # PER 1.0
#                 'logs_maze_per_reward_ranked_08_03_2025__10_29_38.json', # PER 1.2
#                 'logs_maze_per_reward_ranked_10_03_2025__03_34_16.json'  # PER 1.4 # weird file
# ]

# FILENAMES = [
#     'logs_mpx_er_boolean-multiplexer-6bit-v0_10_03_2025__00_19_10.json', # ER
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__01_00_30_0.2.json',   # PER 0.2
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__01_48_48_0.4.json',   # PER 0.4
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__02_38_29_0.6.json',   # PER 0.6
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__03_28_35_0.8.json',   # PER 0.8
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__04_16_16_1.0.json',   # PER 1.0
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__05_04_29_1.2.json',   # PER 1.2
#     'logs_mpx_per_imm_reward_proportional_10_03_2025__05_52_26_1.4.json'    # PER 1.4
# ]

# FILENAMES = [
#     'logs_mpx_er_boolean-multiplexer-6bit-v0_10_03_2025__00_19_10.json', # ER
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__01_05_58_0.2.json',  # PER 0.2
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__02_06_23_0.4.json',  # PER 0.4
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__03_10_00_0.6.json',  # PER 0.6
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__04_14_05_0.8.json',  # PER 0.8
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__05_15_16_1.0.json',  # PER 1.0
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__06_13_27_1.2.json',  # PER 1.2
#     'logs_mpx_per_imm_reward_ranked_10_03_2025__07_09_23_1.4.json'   # PER 1.4
# ]

# FILENAMES = [
#     'logs_mpx_er_boolean-multiplexer-6bit-v0_10_03_2025__00_19_10.json', # ER
#     'logs_mpx_per_reward_proportional_10_03_2025__01_16_40_0.2.json',  # PER 0.2
#     'logs_mpx_per_reward_proportional_10_03_2025__02_20_31_0.4.json',  # PER 0.4
#     'logs_mpx_per_reward_proportional_10_03_2025__03_24_41_0.6.json',  # PER 0.6
#     'logs_mpx_per_reward_proportional_10_03_2025__04_28_21_0.8.json',  # PER 0.8
#     'logs_mpx_per_reward_proportional_10_03_2025__05_33_48_1.0.json',  # PER 1.0
#     'logs_mpx_per_reward_proportional_10_03_2025__06_34_40_1.2.json',  # PER 1.2
#     'logs_mpx_per_reward_proportional_10_03_2025__07_33_30_1.4.json'   # PER 1.4
# ]

# FILENAMES = [
#     'logs_mpx_er_boolean-multiplexer-6bit-v0_10_03_2025__00_19_10.json', # ER
#     'logs_mpx_per_reward_ranked_10_03_2025__01_21_21_0.2.json', # PER 0.2
#     'logs_mpx_per_reward_ranked_10_03_2025__02_37_29_0.4.json', # PER 0.4
#     'logs_mpx_per_reward_ranked_10_03_2025__03_55_32_0.6.json', # PER 0.6
#     'logs_mpx_per_reward_ranked_10_03_2025__05_15_46_0.8.json', # PER 0.8
#     'logs_mpx_per_reward_ranked_10_03_2025__06_32_12_1.0.json', # PER 1.0
#     'logs_mpx_per_reward_ranked_10_03_2025__07_44_10_1.2.json', # PER 1.2
#     'logs_mpx_per_reward_ranked_10_03_2025__08_50_43_1.4.json'  # PER 1.4
# ]

# FILENAMES = [
#     'logs_corridor_er_corridor-20-v0_10_03_2025__00_30_39.json', # ER
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__01_06_31_0.2.json', # PER 0.2
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__01_42_57_0.4.json', # PER 0.4
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__02_20_40_0.6.json', # PER 0.6
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__02_57_52_0.8.json', # PER 0.8
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__03_35_36_1.0.json', # PER 1.0
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__04_12_42_1.2.json', # PER 1.2
#     'logs_corridor_per_corridor-20-v0_imm_reward_proportional_10_03_2025__04_50_47_1.4.json'  # PER 1.4
# ]

# FILENAMES = [
#     'logs_corridor_er_corridor-20-v0_10_03_2025__00_30_39.json', # ER
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__01_08_27_0.2.json', # PER 0.2
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__01_47_02_0.4.json', # PER 0.4
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__02_25_24_0.6.json', # PER 0.6
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__03_04_05_0.8.json', # PER 0.8
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__03_44_17_1.0.json', # PER 1.0
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__04_25_31_1.2.json', # PER 1.2
#     'logs_corridor_per_corridor-20-v0_imm_reward_ranked_10_03_2025__05_09_02_1.4.json'  # PER 1.4
# ]

# FILENAMES = [
#     'logs_corridor_er_corridor-20-v0_10_03_2025__00_30_39.json', # ER
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__01_25_23_0.2.json', # PER 0.2
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__02_19_56_0.4.json', # PER 0.4
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__03_16_39_0.6.json', # PER 0.6
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__04_15_15_0.8.json', # PER 0.8
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__05_15_32_1.0.json', # PER 1.0
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__06_15_17_1.2.json', # PER 1.2
#     'logs_corridor_per_corridor-20-v0_reward_proportional_10_03_2025__07_13_51_1.4.json', # PER 1.4
# ]

# FILENAMES = [
#     'logs_corridor_er_corridor-20-v0_10_03_2025__00_30_39.json', # ER
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__01_23_25_0.2.json', # PER 0.2
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__02_19_22_0.4.json', # PER 0.4
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__03_14_47_0.6.json', # PER 0.6
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__04_10_26_0.8.json', # PER 0.8
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__05_06_05_1.0.json', # PER 1.0
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__06_05_43_1.2.json', # PER 1.2
#     'logs_corridor_per_corridor-20-v0_reward_ranked_10_03_2025__07_07_31_1.4.json'  # PER 1.4
# ]

FILENAMES = [
    'logs_maze_er_Maze7-v0_10_03_2025__17_12_09.json', # ER
    'logs_maze_per_Maze7-v0_imm_reward_proportional_10_03_2025__20_23_50_0.2.json', # PER 0.2
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__00_14_15_0.4.json', # PER 0.4
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__04_11_53_0.6.json', # PER 0.6
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__08_17_02_0.8.json', # PER 0.8
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__12_12_28_1.0.json', # PER 1.0
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__16_07_40_1.2.json', # PER 1.2
    'logs_maze_per_Maze7-v0_imm_reward_proportional_11_03_2025__19_59_43_1.4.json', # PER 1.4
]

TITLE = 'Maze7, imm_reward, proportional'
AVG_WINDOW = 25
# for maze & corridor
METRIC_KEYS = ['steps_in_trial', 'reward', 'perf_time', 'population', 'knowledge']
METRIC_NAMES = ['LICZBA KROKÓW', 'NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW', 'WIEDZA [%]']

# for mux
# METRIC_KEYS = ['reward', 'perf_time', 'population']
# METRIC_NAMES = ['NAGRODA', 'CZAS [s]', 'LICZBA KLASYFIKATORÓW']

if __name__ == "__main__":
    fig, axs = plt.subplots(1, len(METRIC_KEYS), figsize=(5*len(METRIC_KEYS), 5))
    fig.suptitle(TITLE)
    colors = ["black", "red", "orange", "yellow", "lime", "green", "blue", "purple"]
    legend = ['ER', 'PER, alpha=0.2', 'PER, alpha=0.4', 'PER, alpha=0.6', 'PER, alpha=0.8', 'PER, alpha=1.0', 'PER, alpha=1.2', 'PER, alpha=1.4']
    
    handles = []

    for file, caption, color in zip(FILENAMES, legend, colors):
        path = os.path.join(LOGS_DIR, file)
        with open(path, 'r') as fp:
            data = json.load(fp)
        xs = range(1, len(data[0]) + 2 - AVG_WINDOW)
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
                        for j in range(i, i + AVG_WINDOW)
                    ]
                )
                for i in range(len(data[0]) + 1 - AVG_WINDOW)
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
                        for j in range(i, i + AVG_WINDOW)
                    ]
                )
                for i in range(len(data[0]) + 1 - AVG_WINDOW)
            ])
            line, = axs[metric_id].plot(xs, metric_mean, color=color)
            if metric_id == 0:
                handles.append(line)
            axs[metric_id].plot(xs, metric_mean + metric_std, color=color, linestyle='dotted')
            axs[metric_id].plot(xs, metric_mean - metric_std, color=color, linestyle='dotted')
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
            axs[metric_id].set_title(metric_key)
            axs[metric_id].set_xlabel('EPIZOD')
            axs[metric_id].set_ylabel(metric_name)
    axs[0].legend(handles, legend)
    plt.show()