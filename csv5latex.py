import os
import json
from statistics import mean, pstdev
import numpy as np
import glob
import datetime
import csv
from itertools import product

# \textbf{{Agent}} & \textbf{{Parametry}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. maks.\\ l. klas.\end{{tabular}}}} & \textbf{{L. klas.}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}L. klas.\\ wiarygodnych\end{{tabular}}}} & \textbf{{Różnica}} \\ \hline
# ACS2 &  & 483.36 & 430.3 & 258.96 & 171.33 \\ \hline
# ACS2ER &  & 534.53 & 450.2 & 419.96 & 30.23 \\ \hline
# ACS2BPER & \begin{{tabular}}[c]{{@{{}}c@{{}}}}alpha\_bper=0.25\\ k=0.25\\ l. klastrów=2\end{{tabular}} & 522.4 & 438.86 & 411.36 & 27.5 \\ \hline
# ACS2BPER & \begin{{tabular}}[c]{{@{{}}c@{{}}}}alpha\_bper=0.25\\ k=0.25\\ l. klastrów=3\end{{tabular}} & 539.8 & 445.73 & 416.73 & 29.0 \\ \hline
# ACS2 &  & 483.36 & 430.3 & 258.96 & 171.33 \\ \hline
# ACS2ER &  & 534.53 & 450.2 & 419.96 & 30.23 \\ \hline
# ACS2BPER & \begin{{tabular}}[c]{{@{{}}c@{{}}}}alpha\_bper=0.25\\ k=0.25\\ l. klastrów=2\end{{tabular}} & 522.4 & 438.86 & 411.36 & 27.5 \\ \hline

TABLE_FOOTER = fr'''
\end{{longtable}}
\end{{center}}
'''

ENVS_DICT = {
    'Maze6-v0': 'Maze 6',
    'MazeF3-v0': 'Maze F3',
    'MazeT2-v0': 'Maze T2',
    'corridor-20-v0': 'Corridor 20',
    'boolean-multiplexer-6bit-v0': 'Boolean Multiplexer 6',
    'boolean-multiplexer-11bit-v0': 'Boolean Multiplexer 11',
            }

METRICS_DICT = {
    'classifiers_report': 'Liczba klasyfikatorów',
    'knowledge_report': 'Wiedza',
    'perf_time_report': 'Średni czas',
    'reward_report': 'Średnia nagroda',
    'steps_int_trial_report': 'Średnia liczba kroków',
    'end_population_tests': 'Testy populacji końcowej',
    'max_population_tests': 'Testy maksymalnej populacji',
    'max_knowledge_tests': 'Testy maksymalnej wiedzy',
    'end_reliable_tests': 'Testy końcowej liczby klasyfikatorów wiarygodnych',
    'mean_perf_time_tests': 'Testy średniego czasu wykonania jednego z trzydziestu epizodów',
    'mean_reward_tests': 'Testy średniej nagrody',
    'mean_steps_in_trial_tests': 'Testy średniej liczby kroków',
}

SPECIAL_HEADERS = {
    'Śr. maks. l. klas.': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. maks.\\ l. klas.\end{{tabular}}',
    'L. klas. wiarygodnych': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}L. klas.\\ wiarygodnych\end{{tabular}}',
    'Średni czas epizodu (explore) [s]': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. czas\\ (explore) [s]\end{{tabular}}',
    'Średni czas epizodu (exploit) [s]': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. czas\\ (exploit) [s]\end{{tabular}}',
    'Średnia nagroda (explore)': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. nagroda\\ (explore)\end{{tabular}}',
    'Średnia nagroda (exploit)': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. nagroda\\ (exploit)\end{{tabular}}',
    'Średnia liczba kroków (explore)': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. liczba\\ kroków (explore)\end{{tabular}}',
    'Średnia liczba kroków (exploit)': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Śr. liczba\\ kroków (exploit)\end{{tabular}}',
    '80% w trial': '80\% w trial',
    '95% w trial': '95\% w trial',
    '100% w trial': '100\% w trial',
    'Najlepszy wynik [%]': fr'\begin{{tabular}}[c]{{@{{}}c@{{}}}}Najlepszy\\ wynik [\%]\end{{tabular}}'
}

def generate_latex_table(csv_dir, csv_filename, output_dir):
    # TABLE_CONTENT = ''

    with open(os.path.join(csv_dir, csv_filename), 'r') as csvfile:
        lines = csvfile.readlines()
        headers = lines[0].strip().split(';')
        print(f"Headers: {headers}")
        nr_cols = len(headers)
        column_style = '|' + 'c|' * nr_cols
        env_key, metric_dict = None, None
        for k, v in ENVS_DICT.items():
            if k in csv_filename:
                env_key = k
                break
        assert env_key is not None, f"Environment key not found in filename: {csv_filename}"
        for k, v in METRICS_DICT.items():
            if k in csv_filename:
                metric_dict = v
                break
        assert metric_dict is not None, f"Metric dictionary key not found in filename: {csv_filename}"
        # supplement = '({})' if 'reports' in csv_filename else ''
        # supplement = '(ACS2PER, imm_reward, ranked)'
        if_acs2bper = 'euclidean' in csv_filename or 'cityblock' in csv_filename
        if_cityblock = 'cityblock' in csv_filename
        if_imm_reward = 'imm_reward' in csv_filename
        if_ranked = 'ranked' in csv_filename
        supplement = ('(' + (('ACS2BPER, ' + ('$cityblock$' if if_cityblock else '$euclidean$')) if if_acs2bper else ('ACS2PER, ' + ('$imm\_reward$, ' if if_imm_reward else '$reward$, ') + ('$ranked$' if if_ranked else '$proportional$'))) + ')') if 'report' in csv_filename else ''

        caption = fr"\textit{{{ENVS_DICT[env_key]}}} - {metric_dict} {supplement}"
        print(f"Caption: {caption}")
        label = f"tab:{csv_filename.replace('.csv', '').replace('_', '-')}"
        firsthead = r'\hline ' + fr'\multicolumn{{1}}{{|c|}}{{\textbf{{{headers[0]}}}}} & ' + ' & '.join([fr'\multicolumn{{1}}{{c|}}{{\textbf{{{header if header not in SPECIAL_HEADERS.keys() else SPECIAL_HEADERS[header]}}}}}' for header in headers[1:]]) + r' \\ \hline'

        # \begin{tabular}[c]{@{}c@{}}alpha\_bper=0.25\\ k=0.25\\ l. klastrów=2\end{tabular}

        # field.replace('_', r'\_')

        bs = r' \\ '
        cs = r'\_'

        TABLE_CONTENT = '\n'.join([' & '.join([(
            fr"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{bs.join([subfield.strip().replace('_', cs) for subfield in field.split('|')])}\end{{tabular}}"
        ) if field_nr == 1 and field.strip() != '' else field.strip() for field_nr, field in enumerate(line.strip().split(";"))]) + r' \\ \hline' for line in lines[1:]])


    TABLE_HEADER = fr'''
    \begin{{center}}
    \begin{{longtable}}{{{column_style}}} 
    \caption{{{caption}}} \label{{{label}}} \\
    
    {firsthead}
    \endfirsthead
    
    \multicolumn{{{nr_cols}}}{{c}}%
    {{{{\bfseries \tablename\ \thetable{{}} -- {caption}}}}} \\
    {firsthead}
    \endhead
    
    \hline \multicolumn{{{nr_cols}}}{{|r|}}{{{{Ciąg dalszy na następnej stronie}}}} \\ \hline
    \endfoot
    
    \hline \hline
    \endlastfoot
    
    '''
    TABLE_FOOTER = fr'''
    \end{{longtable}}
    \end{{center}}
    '''

    result = '\n'.join([TABLE_HEADER, TABLE_CONTENT, TABLE_FOOTER])

    with open(os.path.join(output_dir, f"{csv_filename}.tex"), 'w') as f:
        f.write(result)

    # print(result)

    # return result




def generate_latex_figure(main_dir, env, agent, different_variant, filename, output_dir):
    DICT_ENVS = {
        'output_Maze6-v0': 'Maze 6',
        'output_MazeF3-v0': 'Maze F3',
        'output_MazeT2-v0': 'Maze T2',
        'output_corridor-20-v0': 'Corridor 20',
        'output_boolean-multiplexer-6bit-v0': 'Boolean Multiplexer 6',
        'output_boolean-multiplexer-11bit-v0': 'Boolean Multiplexer 11',
    }
    
    DICT_DIFFERENT = {
        'different_bper_alpha': fr'alpha_{{bper}}',
        'different_bper_k': fr'k_{{bper}}',
        'different_nr_clusters': fr'nr\_clusters',
        'different_alphas': fr'alpha_{{per}}',
        'different_betas': fr'beta_{{per}}',
    }

    if_imm_reward = (agent == "per" and 'imm_reward' in filename)
    if_ranked = (agent == "per" and 'ranked' in filename)
    
    filename_splitted = filename.split('_')

    constants = {
        'different_bper_alpha': ('$cityblock$' if 'cityblock' in filename else '$euclidean$') + fr', $k_{{bper}}=' + filename_splitted[7] + fr'$, $nr\_clusters=' + filename_splitted[10] + fr'$',
        'different_bper_k': ('$cityblock$' if 'cityblock' in filename else '$euclidean$') + fr', $alpha_{{bper}}=' + filename_splitted[6] + fr'$, $nr\_clusters=' + filename_splitted[9] + fr'$',
        'different_nr_clusters': ('$cityblock$' if 'cityblock' in filename else '$euclidean$') + fr', $alpha_{{bper}}=' + filename_splitted[6] + fr'$, $k_{{bper}}=' + filename_splitted[9] + fr'$',
        'different_alphas': (fr'$imm\_reward$' if if_imm_reward else '$reward$') + (fr', $ranked$' if if_ranked else ', $proportional$') + fr', $beta_{{per}}=' + (filename_splitted[6 + int(if_imm_reward)] if filename_splitted[6 + int(if_imm_reward)] != 'None' else 'OFF') + fr'$',
        'different_betas': (fr'$imm\_reward$' if if_imm_reward else '$reward$') + (fr', $ranked$' if if_ranked else ', $proportional$') + fr', $alpha_{{per}}=' + filename_splitted[6 + int(if_imm_reward)] + fr'$',
    }[different_variant]

    caption = f'{DICT_ENVS[env]} - Wykresy metryk dla ACS2{agent.upper()}, {constants}, zmienne ${DICT_DIFFERENT[different_variant]}$'
    
    FIGURE = fr'''
\begin{{figure}}[]
  \centering
    \includegraphics[width=\textwidth]{{{os.path.join('images', main_dir, env, agent, different_variant, filename)}}}
    \caption{{{caption}}}
    \label{{fig:{filename.replace('.png', '').replace('_', '-')}}}
\end{{figure}}
    '''
    # return FIGURE

    print(FIGURE)

    with open(os.path.join(output_dir, f"{filename}.tex"), 'w') as f:
        f.write(FIGURE)


if __name__ == "__main__":
    # print(TABLE_TEMPLATE)
    # generate_latex_table('tables/output_Maze6-v0', 'Maze6-v0_cityblock_classifiers_report.csv', 'tex_sources/reports')
    # generate_latex_table('tests', 'corridor-20-v0_end_population_tests.csv', 'tex_sources/tests')
    
    for dirname in os.listdir('tables'):
        for filename in os.listdir(os.path.join('tables', dirname)):
            generate_latex_table(os.path.join('tables', dirname), filename, 'tex_sources/reports')
            print(os.path.join('tables', dirname, filename))
# 
#     for filename in os.listdir('tests'):
#         generate_latex_table('tests', filename, 'tex_sources/tests')
#         print(os.path.join('tests', filename))

#     generate_latex_figure(
#         'plots',
#         'output_Maze6-v0',
#         'bper',
#         'different_bper_alpha',
#         'Figure_3_Maze6-v0_dist_cityblock_bper_k_0.5_nr_clusters_2_avg_window_25.png',
#         'tex_sources/figures'
#     )


#def generate_latex_figure(main_dir, env, agent, different_variant, filename, output_dir):
#     MAIN_DIR = 'plots'
#     for env in os.listdir(MAIN_DIR):
#         for agent in os.listdir(os.path.join(MAIN_DIR, env)):
#             for different_variant in os.listdir(os.path.join(MAIN_DIR, env, agent)):
#                 for filename in os.listdir(os.path.join(MAIN_DIR, env, agent, different_variant)):
#                     if filename.endswith('.png'):
#                         generate_latex_figure(
#                             MAIN_DIR,
#                             env,
#                             agent,
#                             different_variant,
#                             filename,
#                             'tex_sources/figures'
#                         )
# 
    pass