import os

import numpy as np
import matplotlib.pyplot as plt

from importlib import import_module


save_path = './results/radar_plot/'

normalize = False
provide_normalized_human_perf = True

modes = ['w_only', 'w_a', 'kv_cache']
'''
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

datasets=['chid', 'winogrande', 'race', 'lambada', 'piqa', 'siqa']
human_performances = {
    'chid': 87.1,
    'winogrande': 94.0,
    'race': 94.2,
    'lambada': None,
    'piqa': 95.0,
    'siqa': 86.9,
}

label_size = 20

# axis settings
axis_fontsize = 14

# legend settings
legend_loc = 'lower left'
legend_fontsize = 20


def normalize_results(raw_results, fp_idx=0, minimal=None, range=1, w_fp=True):
    # raw_results = result_pad(raw_results)
    has_nonzero_fp_result = raw_results[fp_idx] is not None and raw_results[fp_idx] != 0
    if has_nonzero_fp_result and w_fp:
        # do not consider the minimal value of the dataset
        fp_result = raw_results[fp_idx]
        if minimal is None:
            norm_results = [i / fp_result if i is not None else None for i in raw_results]
            print('Result Normalization Succeeded.')
        else:
            norm_results = [max((i - minimal) / (fp_result - minimal), 0) \
                                if i is not None else None for i in raw_results]
    else:
        norm_results = raw_results
        print('The input results have no FP precision, return original results.')
    assert range in [1, 100]
    if range == 100:
        norm_results = [i * 100 if i is not None else i for i in norm_results]
        print('Result Normalization Succeeded.')
    elif range == 1 and norm_results == raw_results:
        norm_results = [i / 100. if i is not None else i for i in norm_results]
        print('Result Normalization Succeeded.')
    return norm_results


if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for mode in modes:
        for model_family_name, model_family_list in models_to_plot.items():
            raw_fp_perf = {}
            for model_name in model_family_list:
                acc = {}
                minimum = {
                    'chid': 16.67,
                    'winogrande': 50.,
                    'race': 25.,
                    'lambada': 0.,
                    'rte': 50.,
                    'piqa': 50.,
                    'siqa': 33.33,
                }

                for dataset in datasets:
                    results_module = import_module('assets.' + dataset + '_' + mode)
                    dataset_results = getattr(results_module, model_name)
                    raw_fp_perf[dataset] = dataset_results[0]
                    dataset_results = normalize_results(dataset_results, range=1, minimal=minimum[dataset], \
                                                        w_fp=True)

                    if normalize:
                        dataset_results = normalize_results(dataset_results, minimal=minimum[dataset])
                    acc[dataset] = dataset_results

                normalized_human_perf = {}
                for dataset_name, human_perf in human_performances.items():
                    fp_perf = raw_fp_perf[dataset_name]
                    minimal = minimum[dataset_name]
                    if human_perf is not None and fp_perf is not None:
                        normalized_human_perf[dataset_name] = round(max((human_perf - minimal) / (fp_perf - minimal), 0), 2)
                    else:
                        normalized_human_perf[dataset_name] = None

                if mode == 'w_only':
                    x_label = ['FP16', 'W8', 'W4', 'W3', 'W2']
                elif mode == 'w_a':
                    x_label = ['FP16', 'W8A8', 'W4A8', 'W4A4']
                elif mode == 'kv_cache':
                    x_label = ['FP16', 'KV8', 'KV4', 'KV3', 'KV2']

                results_2_plot = {}
                for i in range(len(x_label)):
                    results_2_plot[x_label[i]]=[]
                    results_2_plot[x_label[i]].append(acc['chid'][i])
                    results_2_plot[x_label[i]].append(acc['winogrande'][i])
                    results_2_plot[x_label[i]].append(acc['race'][i])
                    results_2_plot[x_label[i]].append(acc['lambada'][i])
                    # results_2_plot[x_label[i]].append(acc['rte'][i])
                    results_2_plot[x_label[i]].append(acc['piqa'][i])
                    results_2_plot[x_label[i]].append(acc['siqa'][i])
                    results_2_plot[x_label[i]].append(acc['chid'][i])
                angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                feature = ['CHID', 'Winogrande', 'RACE', 'Lambada', 'PIQA', 'SIQA', 'CHID']
                if provide_normalized_human_perf:
                    feature = [dataset_name + '\n' + '(' + str(normalized_human_perf[dataset_name.lower()]) + \
                               ')' if dataset_name is not 'Lambada' else dataset_name for dataset_name in feature]
                fig = plt.figure()
                ax = fig.add_subplot(111, polar=True)

                for k, v in results_2_plot.items():
                    ax.plot(angles, v, 'o-', linewidth=2, label=k)
                    ax.fill(angles, v, alpha=0.25)
                ax.set_thetagrids(angles*180/np.pi, feature, fontsize=axis_fontsize)
                plt.legend(loc=legend_loc)
                plt.savefig(save_path + model_name + '_' + mode+'.pdf')


