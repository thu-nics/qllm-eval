import os

import numpy as np
import matplotlib.pyplot as plt

from importlib import import_module


save_path = './results/radar_plot/'

normalize = False

mode = 'w_only'
modes = ['w_only', 'w_a', 'kv_cache']

selected_bitwidth = 'w3'

models_to_plot = {
    'opt':      ['OPT_125M', 'OPT_1B3', 'OPT_2B7', 'OPT_6B7', 'OPT_13B', 'OPT_30B', 'OPT_66B'],
    'falcon':   ['Falcon_7B', 'Falcon_40B', 'Falcon_180B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_13B', 'LlaMA2_70B'],
    'bloom':    ['Bloom_560M', 'Bloom_1B1', 'Bloom_1B7', 'Bloom_3B', 'Bloom_7B1', 'Bloom_175B'],
    'bloomz':   ['Bloomz_560M', 'Bloomz_1B1', 'Bloomz_1B7', 'Bloomz_3B', 'Bloomz_7B1', 'Bloomz_175B'],
    'chatglm3': ['ChatGLM3_6B'],
}
'''
models_to_plot = {
    'opt':      ['OPT_2B7', 'OPT_66B'],
    'falcon':   ['Falcon_7B', 'Falcon_40B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_70B'],
    'bloom':    ['Bloom_3B', 'Bloom_175B'],
    'bloomz':   ['Bloomz_3B', 'Bloomz_175B'],
    'chatglm3': ['ChatGLM3_6B'],
}
'''

# datasets=['chid', 'winogrande', 'race', 'lambada', 'rte', 'piqa', 'siqa']
datasets=['chid', 'winogrande', 'race', 'lambada', 'piqa', 'siqa']

skip_combinations = ['chid-OPT',
                     'winograde-OPT_125M', 'winograde-OPT_1B3', 'winograde-Bloom_560M', 'winograde-Bloom_1B1',
                     'winograde-Bloom_1B7', 'winograde-Bloom_3B', 'winograde-Bloomz_560M', 'winograde-Bloomz_1B1',
                     'winograde-Bloomz_1B7', 'winograde-Bloomz_3B',
                     'race-Falcon_7B', 'race-OPT', 'race-Bloom', 'race-Bloomz_560M', 'race-Bloomz_1B1', 'race-ChatGLM3_6B']
if mode == 'w_only' and selected_bitwidth == 'w3':
    skip_combinations.extend(['chid-Bloomz_1B7', 'chid-Bloomz_3B', 'siqa-OPT', 'lambada-OPT', 'siqa-OPT', 'winogrande-OPT'])

legend_loc = 'lower left'

def normalize_results(raw_results, fp_idx=0, minimal=None, range=1, w_fp=True):
    # raw_results = result_pad(raw_results)
    has_nonzero_fp_result = raw_results[fp_idx] is not None and raw_results[fp_idx] != 0
    if has_nonzero_fp_result and w_fp:
        # do not consider the minimal value of the dataset
        fp_result = raw_results[fp_idx]
        if minimal is None:
            norm_results = [i / fp_result if i is not None else None for i in raw_results]
            # print('Result Normalization Succeeded.')
        else:
            norm_results = []
            for raw_result in raw_results:
                if raw_result is None:
                    norm_results.append(None)
                elif (raw_result - minimal) < 0 or (fp_result - minimal) < 0:
                    norm_results.append(0)
                else:
                    norm_results.append(max((raw_result - minimal) / (fp_result - minimal), 0))
    else:
        norm_results = raw_results
        # print('The input results have no FP precision, return original results.')
    assert range in [1, 100]
    if range == 100:
        norm_results = [i * 100 if i is not None else i for i in norm_results]
        # print('Result Normalization Succeeded.')
    elif range == 1 and norm_results == raw_results:
        norm_results = [i / 100. if i is not None else i for i in norm_results]
        # print('Result Normalization Succeeded.')
    return norm_results


if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    minimum = {
        'chid': 16.67,
        'winogrande': 50.,
        'race': 25.,
        'lambada': 0.,
        'rte': 50.,
        'piqa': 50.,
        'siqa': 33.33,
    }

    best_perfs = {}
    worst_perfs = {}
    for dataset_name in datasets:
        best_perfs[dataset_name] = -1
        worst_perfs[dataset_name] = 100
    for model_family_name, model_family_list in models_to_plot.items():
        for model_name in model_family_list:

            for dataset in datasets:
                results_module = import_module('assets.' + dataset + '_' + mode)
                dataset_results = getattr(results_module, model_name)
                dataset_results = normalize_results(dataset_results, range=1, minimal=minimum[dataset], \
                                                    w_fp=True)

                if normalize:
                    dataset_results = normalize_results(dataset_results, minimal=minimum[dataset])

                skip_flag = False
                for skip_combination in skip_combinations:
                    skip_dataset, skip_model = skip_combination.split('-')[0], skip_combination.split('-')[1]
                    if skip_dataset == dataset and (skip_model == model_name or skip_model in model_name):
                        skip_flag = True
                        print("The current dataset-model combination: {}-{} is meaningless, skip this result.".format(
                            dataset, model_name))
                        break

                # w4
                # import ipdb; ipdb.set_trace()
                if mode == 'w_only':
                    data_idx = {'fp16': 0, 'w8':1, 'w4': 2, 'w3': 3, 'w2': 4}[selected_bitwidth]
                elif mode == 'w_a':
                    data_idx = {'fp16': 0, 'w8a8': 1, 'w4a8': 2, 'wa4a': 3}[selected_bitwidth]
                if not skip_flag:
                    # skip the vacant result
                    if dataset_results[data_idx] is not None:
                        if best_perfs[dataset] < dataset_results[data_idx]:
                            best_perfs[dataset] = dataset_results[data_idx]
                            print('update best keeping ratio for {} dataset, model {}, ratio: {}.'.format(dataset, model_name, dataset_results[data_idx]))
                        # rule out the meaningless results
                        if worst_perfs[dataset] > dataset_results[data_idx] and dataset_results[data_idx] > 0:
                            worst_perfs[dataset] = dataset_results[data_idx]
                            print(
                                'update worst keeping ratio for {} dataset, model {}, ratio: {}.'.format(dataset, model_name,
                                                                                                        dataset_results[data_idx]))

    x_label = ['best keeping ratio', 'worst keeping ratio']
    stacked_perfs = [best_perfs, worst_perfs]
    print('best keeping ratio:{}'.format(best_perfs))
    print('worst keeping ratio:{}'.format(worst_perfs))
    results_2_plot = {}
    for i in range(len(x_label)):
        results_2_plot[x_label[i]]=[]
        results_2_plot[x_label[i]].append(stacked_perfs[i]['chid'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['winogrande'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['race'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['lambada'])
        # results_2_plot[x_label[i]].append(stacked_perfs[i]['rte'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['piqa'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['siqa'])
        results_2_plot[x_label[i]].append(stacked_perfs[i]['chid'])

    angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    # feature = ['CHID', 'Winogrande', 'RACE', 'Lambada', 'RTE', 'PIQA', 'SIQA', 'CHID']
    feature = ['CHID', 'Winogrande', 'RACE', 'Lambada', 'PIQA', 'SIQA', 'CHID']
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for k, v in results_2_plot.items():
        ax.plot(angles, v, 'o-', linewidth=2, label=k)
        ax.fill(angles, v, alpha=0.25)

    # plot 1.0 base line
    circle_angles = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    ax.plot(circle_angles, [1.0 for i in range(1000)], '--', linewidth=1, color='red', label='1.0 baseline')
    # ax.fill(circle_angles, [1.0 for i in range(1000)], alpha=0.25)

    ax.set_thetagrids(angles*180/np.pi, feature)
    plt.legend(loc = legend_loc)
    plt.savefig(save_path +  mode + '_' + selected_bitwidth + '.pdf')


