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

OUTPUT_DIR = 'tables'

def compute_classifiers(data, compute_reliables=True):
    assert len(data) == 30
    avg_max_classifiers = mean([max([trial['population'] for trial in episode]) for episode in data])
    end_nr_classifers = mean([episode[-1]['population'] for episode in data])
    if compute_reliables:
        end_nr_reliables = mean([episode[-1]['reliable'] for episode in data])
        difference = end_nr_classifers - end_nr_reliables
        return avg_max_classifiers, end_nr_classifers, end_nr_reliables, difference
    else:
        return avg_max_classifiers, end_nr_classifers, None, None


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

def compute_highest_knowledge(data):
    return mean(
        [episode[-1]['knowledge']/100.0 for episode in data]
    )

PER_COL_AGENTS = [
    'ACS2',
    'ACS2ER',
] + [
    'ACS2PER'
] * len(ALPHAS) * len(BETAS)

PER_COL_PARAMS = [
    '',
    '',
] + [
    'alpha_per={} | beta_per={}'.format(alpha, beta) for alpha, beta in product(ALPHAS, BETAS)
]

BPER_COL_AGENTS = [
    'ACS2',
    'ACS2ER',
] + [
    'ACS2BPER'
] * len(ALPHAS) * len(BPER_KS) * len(NR_CLUSTERS)

BPER_COL_PARAMS = [
    '',
    '',
] + [
    'alpha_bper={} | k={} | l. klastrów={}'.format(alpha, bper_k, nr_clusters) for alpha, bper_k, nr_clusters in product(ALPHAS, BPER_KS, NR_CLUSTERS)
]

if __name__ == "__main__":
    for env in ENVS:
        os.makedirs(os.path.join(OUTPUT_DIR, f"output_{env}"), exist_ok=True)
        # PER
        for PER_MEASURED_STAT, PER_PRIORITY_FUNCTION in product(PER_MEASURED_STATS, PER_PRIORITY_FUNCTIONS):
            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ] + [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_{}_beta_{}_rsi_{}_*_acs2per_explore_metrics.json'.format(env, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION, alpha, beta, RSI)))[0]) for alpha, beta in product(ALPHAS, BETAS)
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ] + [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_{}_{}_alpha_{}_beta_{}_rsi_{}_*_acs2per_exploit_metrics.json'.format(env, PER_MEASURED_STAT, PER_PRIORITY_FUNCTION, alpha, beta, RSI)))[0]) for alpha, beta in product(ALPHAS, BETAS)
            ]

            assert len(FILENAMES_EXPLORE) == len(PER_COL_AGENTS), "Number of filenames for explore does not match number of agents"
            assert len(FILENAMES_EXPLOIT) == len(PER_COL_AGENTS), "Number of filenames for exploit does not match number of agents"

            # classifiers data
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{PER_MEASURED_STAT}_{PER_PRIORITY_FUNCTION}_classifiers_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.', 'L. klas. wiarygodnych', 'Różnica'])
                
                # For MUX:
                # writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.'])

                for filename_re, col_agent, col_param in zip(FILENAMES_EXPLORE, PER_COL_AGENTS, PER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f:
                        data = json.load(f)
                        avg_max_classifiers, end_nr_classifers, end_nr_reliables, difference = compute_classifiers(data)
                        writer.writerow([col_agent, col_param, round(avg_max_classifiers, 2), round(end_nr_classifers, 2), round(end_nr_reliables, 2), round(difference, 2)])
                        
                        # For mux:

                        # avg_max_classifiers, end_nr_classifers, _, _ = compute_classifiers(data, compute_reliables=False)
                        # writer.writerow([col_agent, col_param, round(avg_max_classifiers, 2), round(end_nr_classifers, 2)])


            # perf_time
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{PER_MEASURED_STAT}_{PER_PRIORITY_FUNCTION}_perf_time_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średni czas epizodu (explore) [s]', 'Średni czas epizodu (exploit) [s]'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, PER_COL_AGENTS, PER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_perf_time_re = compute_avg_episode_perf_time(data_re)
                        avg_perf_time_it = compute_avg_episode_perf_time(data_it)
                        writer.writerow([col_agent, col_param, round(avg_perf_time_re, 2), round(avg_perf_time_it, 2)])

            # nagroda
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{PER_MEASURED_STAT}_{PER_PRIORITY_FUNCTION}_reward_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średnia nagroda (explore)', 'Średnia nagroda (exploit)'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, PER_COL_AGENTS, PER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_reward_re = compute_avg_metric(data_re, metric_key='reward')
                        avg_reward_it = compute_avg_metric(data_it, metric_key='reward')
                        writer.writerow([col_agent, col_param, round(avg_reward_re, 2), round(avg_reward_it, 2)])

            # kroki
            #'''
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{PER_MEASURED_STAT}_{PER_PRIORITY_FUNCTION}_steps_int_trial_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średnia liczba kroków (explore)', 'Średnia liczba kroków (exploit)'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, PER_COL_AGENTS, PER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_steps_re = compute_avg_metric(data_re, metric_key='steps_in_trial')
                        avg_steps_it = compute_avg_metric(data_re, metric_key='steps_in_trial')
                        writer.writerow([col_agent, col_param, round(avg_steps_re, 2), round(avg_steps_it, 2)])
            #'''
            # knowledge
            #'''
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{PER_MEASURED_STAT}_{PER_PRIORITY_FUNCTION}_knowledge_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', '80% w trial', '95% w trial', '100% w trial', 'Najlepszy wynik [%]'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, PER_COL_AGENTS, PER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re:
                        data_re = json.load(f_re)
                        assert len(data_re) == 30, "Expected 30 episodes in the data for {}".format(filename_re)
                        # data_it = json.load(f_it)
                        avg_knowledge_080_re, eps_080 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 0.8)
                        avg_knowledge_095_re, eps_095 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 0.95)
                        avg_knowledge_100_re, eps_100 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 1.0)
                        highest_know = compute_highest_knowledge(data_re)
                        writer.writerow([col_agent, col_param,
                                         "-" if eps_080 is None else (round(avg_knowledge_080_re, 2) if eps_080 == len(data_re) else f"<{round(avg_knowledge_080_re, 2)} ({int(eps_080)})"),
                                         "-" if eps_095 is None else (round(avg_knowledge_095_re, 2) if eps_095 == len(data_re) else f"<{round(avg_knowledge_095_re, 2)} ({int(eps_095)})"),
                                         "-" if eps_100 is None else (round(avg_knowledge_100_re, 2) if eps_100 == len(data_re) else f"<{round(avg_knowledge_100_re, 2)} ({int(eps_100)})"),
                                         round(highest_know * 100.0, 2)
                                         ])
            #'''

        # BPER
        for BPER_DIST in BPER_DISTS:
            FILENAMES_EXPLORE = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_explore_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_explore_metrics.json'.format(env)))[0]),
            ] + [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_explore_metrics.json'.format(env, alpha, bper_k, nr_clusters, BPER_DIST)))[0]) for alpha, bper_k, nr_clusters in product(ALPHAS, BPER_KS, NR_CLUSTERS)
            ]
            FILENAMES_EXPLOIT = [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2_exploit_metrics.json'.format(env)))[0]),
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}*_acs2er_exploit_metrics.json'.format(env)))[0]),
            ] + [
                os.path.basename(glob.glob(os.path.join(LOGS_DIR, '{}_alpha_{}_bper_k_{}_nr_clusters_{}_dist_{}*_acs2bper_exploit_metrics.json'.format(env, alpha, bper_k, nr_clusters, BPER_DIST)))[0]) for alpha, bper_k, nr_clusters in product(ALPHAS, BPER_KS, NR_CLUSTERS)
            ]
            
            assert len(FILENAMES_EXPLORE) == len(BPER_COL_AGENTS), "Number of filenames for explore does not match number of agents"
            assert len(FILENAMES_EXPLOIT) == len(BPER_COL_AGENTS), "Number of filenames for exploit does not match number of agents"

            # classifiers data
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{BPER_DIST}_classifiers_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
                writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.', 'L. klas. wiarygodnych', 'Różnica'])
                
                # For MUX:
                # writer.writerow(['Agent', 'Parametry', 'Śr. maks. l. klas.', 'L. klas.'])

                for filename_re, col_agent, col_param in zip(FILENAMES_EXPLORE, BPER_COL_AGENTS, BPER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f:
                        data = json.load(f)
                        avg_max_classifiers, end_nr_classifers, end_nr_reliables, difference = compute_classifiers(data)
                        writer.writerow([col_agent, col_param, round(avg_max_classifiers, 2), round(end_nr_classifers, 2), round(end_nr_reliables, 2), round(difference, 2)])
                        
                        # For mux:
                        # avg_max_classifiers, end_nr_classifers, _, _ = compute_classifiers(data, compute_reliables=False)
                        # writer.writerow([col_agent, col_param, round(avg_max_classifiers, 2), round(end_nr_classifers, 2)])

            # perf_time
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{BPER_DIST}_perf_time_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średni czas epizodu (explore) [s]', 'Średni czas epizodu (exploit) [s]'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, BPER_COL_AGENTS, BPER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_perf_time_re = compute_avg_episode_perf_time(data_re)
                        avg_perf_time_it = compute_avg_episode_perf_time(data_it)
                        writer.writerow([col_agent, col_param, round(avg_perf_time_re, 2), round(avg_perf_time_it, 2)])

            # nagroda
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{BPER_DIST}_reward_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średnia nagroda (explore)', 'Średnia nagroda (exploit)'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, BPER_COL_AGENTS, BPER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_reward_re = compute_avg_metric(data_re, metric_key='reward')
                        avg_reward_it = compute_avg_metric(data_it, metric_key='reward')
                        writer.writerow([col_agent, col_param, round(avg_reward_re, 2), round(avg_reward_it, 2)])

            # kroki
            # '''
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{BPER_DIST}_steps_int_trial_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', 'Średnia liczba kroków (explore)', 'Średnia liczba kroków (exploit)'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, BPER_COL_AGENTS, BPER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re, open(os.path.join(LOGS_DIR, filename_it), 'r') as f_it:
                        data_re = json.load(f_re)
                        data_it = json.load(f_it)
                        avg_steps_re = compute_avg_metric(data_re, metric_key='steps_in_trial')
                        avg_steps_it = compute_avg_metric(data_re, metric_key='steps_in_trial')
                        writer.writerow([col_agent, col_param, round(avg_steps_re, 2), round(avg_steps_it, 2)])
            # '''

            # knowledge
            # '''
            with open(os.path.join(OUTPUT_DIR, f"output_{env}", f"{env}_{BPER_DIST}_knowledge_report.csv"), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Agent', 'Parametry', '80% w trial', '95% w trial', '100% w trial', 'Najlepszy wynik [%]'])
                
                for filename_re, filename_it, col_agent, col_param in zip(FILENAMES_EXPLORE, FILENAMES_EXPLOIT, BPER_COL_AGENTS, BPER_COL_PARAMS):
                    with open(os.path.join(LOGS_DIR, filename_re), 'r') as f_re:
                        data_re = json.load(f_re)
                        assert len(data_re) == 30, "Expected 30 episodes in the data for {}".format(filename_re)
                        # data_it = json.load(f_it)
                        avg_knowledge_080_re, eps_080 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 0.8)
                        avg_knowledge_095_re, eps_095 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 0.95)
                        avg_knowledge_100_re, eps_100 = compute_avg_num_of_trials_for_knowledge_lvl(data_re, 1.0)
                        highest_know = compute_highest_knowledge(data_re)
                        writer.writerow([col_agent, col_param,
                                         "-" if eps_080 is None else (round(avg_knowledge_080_re, 2) if eps_080 == len(data_re) else f"<{round(avg_knowledge_080_re, 2)} ({int(eps_080)})"),
                                         "-" if eps_095 is None else (round(avg_knowledge_095_re, 2) if eps_095 == len(data_re) else f"<{round(avg_knowledge_095_re, 2)} ({int(eps_095)})"),
                                         "-" if eps_100 is None else (round(avg_knowledge_100_re, 2) if eps_100 == len(data_re) else f"<{round(avg_knowledge_100_re, 2)} ({int(eps_100)})"),
                                         round(highest_know * 100.0, 2)
                                         ])
            # '''

        print(f"Reports for {env} generated in {OUTPUT_DIR}/output_{env}/")
