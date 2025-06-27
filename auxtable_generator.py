from scipy import stats
import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
import datetime
import csv
from itertools import product

ENVS = [('Maze6-v0', r'\textit{Maze 6}', 'logs5_maze_750', 250),
        ('MazeF3-v0', r'\textit{Maze F3}', 'logs5_maze_750', 250),
        ('MazeT2-v0', r'\textit{Maze T2}', 'logs5_maze_750', 250),
        ('boolean-multiplexer-6bit-v0', r'\textit{Mux 6}', 'logs5_mux_750', 50),
        ('boolean-multiplexer-11bit-v0', r'\textit{Mux 11}', 'logs5_mux_750', 50),
        ('corridor-20-v0', r'\textit{Corridor 20}', 'logs5_corridor_750', 200)]

PER_MEASURED_STATS = ['imm_reward', 'reward']
PER_PRIORITY_FUNCTIONS = ['proportional', 'ranked']

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
BETAS = [None, 0.25, 0.5, 0.75]

BPER_DISTS = ['cityblock', 'euclidean']
BPER_KS = [0.25, 0.5, 1.0, 2.0, 5.0]
NR_CLUSTERS = [2, 3, 5, 7]

OUTPUT_DIR = 'aux_tables'

metrics = [
    lambda data: [episode[-1]['population'] for episode in data],                            # end population
    lambda data: [max([trial['population'] for trial in episode]) for episode in data],      # max population
    lambda data: [mean([trial['reward'] for trial in episode]) for episode in data],         # mean reward
    lambda data: [episode[-1]['reliable'] for episode in data],                              # end reliable
    lambda data: [episode[-1]['knowledge'] for episode in data],                             # max_knowledges
    lambda data: [mean([trial['steps_in_trial'] for trial in episode]) for episode in data], # mean steps in trial
    
    # lambda data: [mean([trial['perf_time'] for trial in episode]) for episode in data],      # mean perf_time                
]

metric_names = [
    'end_population',
    'max_population',
    'mean_reward',
    'end_reliable',
    'max_knowledge',
    'mean_steps_in_trial',
    
    # 'mean_perf_time',
]

metric_nr_cols = {
    'end_population': 6,
    'max_population': 6,
    'mean_reward': 6,
    'end_reliable': 4,
    'max_knowledge': 4,
    'mean_steps_in_trial': 4,
    
    # 'mean_perf_time': 3,
}

# def mean_std(data):
#     """Calculate the mean and standard deviation of a list of numbers."""
#     if not data:
#         return None, None
#     return mean(data), pstdev(data)

TOP_VALUES = {
    'end_population': [407.73333333333335, 89.8, 86.7, 265.8333333333333, 2007.9, 38],
    'max_population': [493.8333333333333, 101.2, 97.56666666666666, 267.6666666666667, 2008.3333333333333, 38],
    'mean_reward': [961.1111111111111, 994.1333333333333, 997.9111111111112, 700.3555555555555, 637.2888888888889, 993.6444444444444],
    'end_reliable': [443.3333333333333, 101, 100.83333333333333, 38],
    'max_knowledge': [99.62962962962963, 100.0, 100.0, 100.0],
    'mean_steps_in_trial': [14.032533333333333, 7.494355555555556, 5.841333333333333, 47.82906666666667],
}



COL_AGENTS = ['ACS2PER'] * len(PER_MEASURED_STATS) * len(PER_PRIORITY_FUNCTIONS) * len(ALPHAS) * len(BETAS) + \
            ['ACS2BPER'] * len(BPER_DISTS) * len(ALPHAS) * len(BPER_KS) * len(NR_CLUSTERS)

p_val = 0.05

effective_p_val = p_val / len(COL_AGENTS)  # Bonferroni correction for multiple comparisons

bs = '\''

COL_PARAMS = [fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}{bs}${per_measured_stat}${bs} \\ {bs}${per_priority_function}${bs} \\ $alpha\_per={alpha}$ \\ $beta\_per={beta}$\end{{tabular}}' for per_measured_stat, per_priority_function, alpha, beta in product(PER_MEASURED_STATS, PER_PRIORITY_FUNCTIONS, ALPHAS, BETAS)] + \
            [fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}{bs}${bper_dist}${bs} \\ $alpha\_per={alpha}$ \\ $k\_bper={bper_k}$ \\ $nr\_clusters={nr_clusters}$\end{{tabular}}' for bper_dist, alpha, bper_k, nr_clusters in product(BPER_DISTS, ALPHAS, BPER_KS, NR_CLUSTERS)]

def generate_auxtables():
    global metrics, metric_names, COL_AGENTS, COL_PARAMS, bs, metric_nr_cols
    for metric, metric_name in zip(metrics, metric_names):
        matrix_means = []
        metric_name_formatted = metric_name.replace("_", r"\_")
        caption = fr'Wartości metryki \textit{{{metric_name_formatted}}}'
        label = fr'tab:aux_{metric_name}'
        nr_cols = metric_nr_cols[metric_name]
    
        if nr_cols == 6:
            firsthead = r'\hline \multicolumn{2}{|c|}{\textbf{Wariant}} & \multicolumn{1}{c|}{\textbf{Maze 6}} & \multicolumn{1}{c|}{\textbf{Maze F3}} & \multicolumn{1}{c|}{\textbf{Maze T2}} & \multicolumn{1}{c|}{\textbf{Mux 6}} & \multicolumn{1}{c|}{\textbf{Mux 11}} & \multicolumn{1}{c|}{\textbf{Corridor 20}} \\ \hline'
        elif nr_cols == 4:
            firsthead = r'\hline \multicolumn{2}{|c|}{\textit{Wariant}} & \multicolumn{1}{c|}{\textbf{Maze 6}} & \multicolumn{1}{c|}{\textbf{Maze F3}} & \multicolumn{1}{c|}{\textbf{Maze T2}} & \multicolumn{1}{c|}{\textbf{Corridor 20}} \\ \hline'
        else:
            raise ValueError(f"Unsupported number of columns: {nr_cols}")

        TABLE_HEADER = fr'''
            {{\tiny
            \begin{{center}}
        \begin{{longtable}}{{{'|c' * (nr_cols+2) + '|'}}} 
        \caption{{{caption}}} \label{{{label}}} \\

        {firsthead}
        \endfirsthead

        \multicolumn{{{nr_cols+2}}}{{c}}%
        {{{{ \tablename\ \thetable{{}} -- {caption}}}}} \\
        {firsthead}
        \endhead

        \hline \multicolumn{{{nr_cols+2}}}{{|r|}}{{{{Ciąg dalszy na następnej stronie}}}} \\ \hline
        \endfoot

        \hline \hline
        \endlastfoot

        '''

        TABLE_FOOTER = fr'''
        \end{{longtable}}
        \end{{center}}
        }}
        '''

        TABLE_CONTENT = ''
        # ER CASE
        if nr_cols == 6:
            FILENAMES_ER = [
                'Maze6-v0_13_05_2025__17_50_51_acs2er_explore_metrics.json',
                'MazeF3-v0_13_05_2025__16_22_50_acs2er_explore_metrics.json',
                'MazeT2-v0_13_05_2025__16_20_15_acs2er_explore_metrics.json',
                'boolean-multiplexer-6bit-v0_20_05_2025__00_11_25_acs2er_explore_metrics.json',
                'boolean-multiplexer-11bit-v0_20_05_2025__01_19_44_acs2er_explore_metrics.json',
                'corridor-20-v0_22_05_2025__16_20_39_acs2er_explore_metrics.json'
                ]
            datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES_ER[:3]] + \
                          [json.load(open(os.path.join('logs5_mux_750', filename), 'r')) for filename in FILENAMES_ER[3:5]] + \
                            [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES_ER[5:6]]
        elif nr_cols == 4:
            FILENAMES_ER = [
                'Maze6-v0_13_05_2025__17_50_51_acs2er_explore_metrics.json',
                'MazeF3-v0_13_05_2025__16_22_50_acs2er_explore_metrics.json',
                'MazeT2-v0_13_05_2025__16_20_15_acs2er_explore_metrics.json',
                # 'boolean-multiplexer-6bit-v0_20_05_2025__00_11_25_acs2er_explore_metrics.json',
                # 'boolean-multiplexer-11bit-v0_20_05_2025__01_19_44_acs2er_explore_metrics.json',
                'corridor-20-v0_22_05_2025__16_20_39_acs2er_explore_metrics.json'
                  ]
            datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES_ER[:3]] + \
                          [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES_ER[3:4]]
        else:
            raise ValueError(f"Unsupported number of columns: {nr_cols}")
        
        METRICS_VALUES_ER = [metric(data) for data in datas]
        # print(metrics_values)
        for metric_value_list in METRICS_VALUES_ER:
            assert len(metric_value_list) == 30, "Metrics length mismatch across datasets"
        means = [mean(metric(data)) for data in datas]
        stds = [pstdev(metric(data)) for data in datas]

        means_for_top = TOP_VALUES[metric_name]

        mask = [np.isclose(mean, top_value, rtol=0.0, atol=1e-5) for mean, top_value in zip(means, means_for_top)]



        # TABLE_CONTENT += fr'''
        # ACS2ER &  & {means[0]:.2f}\pm{stds[0]:.2f} & {stds[0]:.2f} & * \\ \hline
        # '''

        TABLE_CONTENT += ('ACS2ER & & ' + ' & '.join([fr'$\mathbf{{{mean:.2f}\pm{std:.2f}}}$' if bit_m else fr'${mean:.2f}\pm{std:.2f}$' for mean, std, bit_m in zip(means, stds, mask)]) + r' \\ \hline ')

        # PER loop
        for per_measured_stat, per_priority_function, alpha, beta in product(PER_MEASURED_STATS, PER_PRIORITY_FUNCTIONS, ALPHAS, BETAS):
            if nr_cols == 6:
                FILENAMES = [
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'Maze6-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeF3-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeT2-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-6bit-v0_{}_{}_alpha_{}_beta_{}_rsi_50_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-11bit-v0_{}_{}_alpha_{}_beta_{}_rsi_50_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_corridor_750', 'corridor-20-v0_{}_{}_alpha_{}_beta_{}_rsi_200_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                      ]
                assert len(FILENAMES) == 6, "Expected 6 filenames for 6 columns"
                datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES[:3]] + \
                          [json.load(open(os.path.join('logs5_mux_750', filename), 'r')) for filename in FILENAMES[3:5]] + \
                            [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES[5:6]]
            elif nr_cols == 4:
                FILENAMES = [
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'Maze6-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeF3-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeT2-v0_{}_{}_alpha_{}_beta_{}_rsi_250_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    # os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-6bit-v0_{}_{}_alpha_{}_beta_{}_rsi_50_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    # os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-11bit-v0_{}_{}_alpha_{}_beta_{}_rsi_50_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_corridor_750', 'corridor-20-v0_{}_{}_alpha_{}_beta_{}_rsi_200_*_acs2per_explore_metrics.json'.format(per_measured_stat, per_priority_function, alpha, beta)))[0]),
                      ]
                assert len(FILENAMES) == 4, "Expected 4 filenames for 4 columns"
                datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES[:3]] + \
                          [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES[3:4]]
            else:
                raise ValueError(f"Unsupported number of columns: {nr_cols}")
            
            metrics_values = [metric(data) for data in datas]
            for metric_value_list in metrics_values:
                assert len(metric_value_list) == 30, "Metrics length mismatch across datasets"
            means = [mean(metric(data)) for data in datas]
            stds = [pstdev(metric(data)) for data in datas]
            matrix_means.append(means)

            tests = ['(*)' if len(set(mv1 + mv2)) > 1 and stats.kruskal(mv1, mv2).pvalue < effective_p_val else '' for mv1, mv2 in zip(METRICS_VALUES_ER, metrics_values)]
            assert len(tests) == nr_cols, "Tests length mismatch with number of columns"

            means_for_top = TOP_VALUES[metric_name]
            mask = [np.isclose(mean, top_value, rtol=0.0, atol=1e-5) for mean, top_value in zip(means, means_for_top)]

            per_measured_stat_formatted = per_measured_stat.replace('_', r'\_')
            TABLE_CONTENT += fr'''ACS2PER & \begin{{tabular}}[c]{{@{{}}c@{{}}}}{bs}{per_measured_stat_formatted}{bs} \\ {bs}{per_priority_function}{bs} \\ alpha\_per={alpha} \\ beta\_per={beta} \end{{tabular}} & ''' + ' & '.join([fr'$\mathbf{{{mean:.2f}\pm{std:.2f}{tv}}}$' if bit_m else fr'${mean:.2f}\pm{std:.2f}{tv}$' for mean, std, tv, bit_m in zip(means, stds, tests, mask)]) + fr''' \\ \hline''' + '\n'

        # BPER loop
        for dist_type, alpha, bper_k, nr_cluster in product(BPER_DISTS, ALPHAS, BPER_KS, NR_CLUSTERS):
            if nr_cols == 6:
                FILENAMES = [
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'Maze6-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeF3-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeT2-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-6bit-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_mux_750', 'boolean-multiplexer-11bit-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_corridor_750', 'corridor-20-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                      ]
                assert len(FILENAMES) == 6, "Expected 6 filenames for 6 columns"
                datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES[:3]] + \
                          [json.load(open(os.path.join('logs5_mux_750', filename), 'r')) for filename in FILENAMES[3:5]] + \
                            [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES[5:6]]
            elif nr_cols == 4:
                FILENAMES = [
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'Maze6-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeF3-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'MazeT2-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    # os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'boolean-multiplexer-6bit-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    # os.path.basename(glob.glob(os.path.join('logs5_maze_750', 'boolean-multiplexer-11bit-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                    os.path.basename(glob.glob(os.path.join('logs5_corridor_750', 'corridor-20-v0_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(alpha, bper_k, nr_cluster, dist_type)))[0]),
                      ]
                assert len(FILENAMES) == 4, "Expected 4 filenames for 4 columns"
                datas = [json.load(open(os.path.join('logs5_maze_750', filename), 'r')) for filename in FILENAMES[:3]] + \
                          [json.load(open(os.path.join('logs5_corridor_750', filename), 'r')) for filename in FILENAMES[3:4]]
            else:
                raise ValueError(f"Unsupported number of columns: {nr_cols}")
            
            metrics_values = [metric(data) for data in datas]
            for metric_value_list in metrics_values:
                assert len(metric_value_list) == 30, "Metrics length mismatch across datasets"
            means = [mean(metric(data)) for data in datas]
            stds = [pstdev(metric(data)) for data in datas]
            matrix_means.append(means)

            tests = ['(*)' if len(set(mv1 + mv2)) > 1 and stats.kruskal(mv1, mv2).pvalue < effective_p_val else '' for mv1, mv2 in zip(METRICS_VALUES_ER, metrics_values)]
            assert len(tests) == nr_cols, "Tests length mismatch with number of columns"

            means_for_top = TOP_VALUES[metric_name]
            mask = [np.isclose(mean, top_value, rtol=0.0, atol=1e-5) for mean, top_value in zip(means, means_for_top)]

            TABLE_CONTENT += fr'''ACS2BPER & \begin{{tabular}}[c]{{@{{}}c@{{}}}}{bs}{dist_type}{bs} \\ alpha\_bper={alpha} \\ k\_bper={bper_k} \\ nr\_clusters={nr_cluster} \end{{tabular}} & ''' + ' & '.join([fr'$\mathbf{{{mean:.2f}\pm{std:.2f}{tv}}}$' if bit_m else fr'${mean:.2f}\pm{std:.2f}{tv}$' for mean, std, tv, bit_m in zip(means, stds, tests, mask)]) + fr''' \\ \hline''' + '\n'
            
            
            # find minimum value in each column of matrix_means
            
            # find all indexes of minimum values in each column
            # def find_indexes(lst, value):
            #     return [i for i, x in enumerate(lst) if x == value]
            # min_indexes = [find_indexes(col, min_value) for col, min_value in zip(zip(*matrix_means), min_values)]
            # print(f'Minimum indexes: {min_indexes}')
            pass

        
        if metric_name in ['end_population', 'max_population', 'mean_steps_in_trial']:
            min_values = [min(col) for col in zip(*matrix_means)]
            print(f'Metric: {metric_name}')
            print(min_values)
        elif metric_name in ['mean_reward', 'end_reliable', 'max_knowledge']:
            max_values = [max(col) for col in zip(*matrix_means)]
            print(f'Metric: {metric_name}')
            print(max_values)
        else:
            raise ValueError(f"Unsupported metric name: {metric_name}")
        
        # print(f'Minimum values: {min_values}')

        result = '\n'.join([TABLE_HEADER, TABLE_CONTENT, TABLE_FOOTER])

        with open(os.path.join('aux_tables', f"{metric_name}.tex"), 'w') as f:
            f.write(result)
    
    

if __name__ == "__main__":
    generate_auxtables()
    pass