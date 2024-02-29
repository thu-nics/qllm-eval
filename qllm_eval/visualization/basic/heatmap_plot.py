import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from importlib import import_module

from utils import *

save_name = None
save_path = './results/heatmap_plot/'

# whether use manually processed order or not
normalize = False

mode = 'w_only'
modes = ['w_only', 'w_a', 'kv_cache']
selected_models = 'llama2+falcon+bloomz+chatglm3+moe+opt'
# selected_models = 'llama2+bloomz+opt'
selected_bitwidth = 'w3'

label_size = 20


'''
# full list of the tested models.
models_to_plot = {
    'opt':      ['OPT_125M', 'OPT_1B3', 'OPT_2B7', 'OPT_6B7', 'OPT_13B', 'OPT_30B', 'OPT_66B'],
    'falcon':   ['Falcon_7B', 'Falcon_40B', 'Falcon_180B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_13B', 'LlaMA2_70B'],
    'bloom':    ['Bloom_560M', 'Bloom_1B1', 'Bloom_1B7', 'Bloom_3B', 'Bloom_7B1', 'Bloom_175B'],
    'bloomz':   ['Bloomz_560M', 'Bloomz_1B1', 'Bloomz_1B7', 'Bloomz_3B', 'Bloomz_7B1', 'Bloomz_175B'],
    'chatglm3': ['ChatGLM3_6B'],
}
'''
# selected models for easier display.
models_to_plot = {
    'opt':      ['OPT_6B7', 'OPT_13B'],
    'falcon':   ['Falcon_7B', 'Falcon_180B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_70B'],
    # 'bloom':    ['Bloom_3B', 'Bloom_7B1'],
    'bloomz':   ['Bloomz_3B', 'Bloomz_175B'],
    'chatglm3': ['ChatGLM3_6B'],
    'moe':      ['Mistral_7B', 'Mixtral_8x7B'],
}

# datasets=['chid', 'winogrande', 'race', 'lambada', 'rte', 'piqa', 'siqa']
datasets=['chid', 'winogrande', 'race', 'lambada', 'piqa', 'siqa']
datasets_ = ['CHID', 'Winogrande', 'Race', 'LAMBADA', 'PIQA', 'SIQA']

if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_name + '.pdf' if save_name is not None else \
                    'heatmap_' + selected_bitwidth + '.pdf'
    save_name = save_path + save_name


    # first gather the test results.
    model_perfs = {}
    selected_models = selected_models.split('+')
    for dataset in datasets:
        fp_perfs = []
        quant_perfs = []
        results_module = import_module('assets.' + dataset + '_' + mode)
        for selected_model_family in selected_models:
            for model in models_to_plot[selected_model_family]:
                perfs = getattr(results_module, model)
                fp_perfs.append(perfs[0])
                quant_idx = bitwidth_idx_mapping[mode][selected_bitwidth]
                quant_perfs.append(perfs[quant_idx])
            model_perfs.update({dataset: {'fp_perfs': fp_perfs, 'quant_perfs': quant_perfs}})

    spearman_corr = np.zeros((12, 12))
    # the order of the datasets is the same as <List> datasets. The FP results occupy the first 6 col & row, followed
    # by 6 quant results.
    for row in range(12):
        for col in range(12):

            row_dataset = datasets[row % 6]
            row_fp_flag = not(row // 6)
            row_perfs = np.array(model_perfs[row_dataset]['fp_perfs' if row_fp_flag else 'quant_perfs'])

            col_dataset = datasets[col % 6]
            col_fp_flag = not(col // 6)
            col_perfs = np.array(model_perfs[col_dataset]['fp_perfs' if col_fp_flag else 'quant_perfs'])

            sp_corr_value = stats.spearmanr(row_perfs, col_perfs).correlation
            spearman_corr[row][col] = sp_corr_value

    # plot the heatmap figure
    spearman_corr = pd.DataFrame(spearman_corr)

    x_labels = ["FP16 " + datasets_[i % 6] if i < 6 else selected_bitwidth.upper() + " " + datasets_[i % 6] for i in range(12)]
    y_labels = ["FP16 " + datasets_[i % 6] if i < 6 else selected_bitwidth.upper() + " " + datasets_[i % 6] for i in range(12)]

    plot = sns.heatmap(spearman_corr, cmap="YlGnBu", xticklabels=x_labels, yticklabels=y_labels, annot=True,
                       annot_kws={'fontsize': 6}, fmt='.2g')

    # plot.set(xlabel=x_labels, ylabel=y_labels)
    plot.xaxis.tick_top()
    plt.xticks(rotation=45, ha='left', rotation_mode='anchor')
    # plt.yticks(rotation=45)
    plt.subplots_adjust(left=0.22, right=0.89, top=0.77, bottom=0.13)
    # plt.xlabel('Dataset - Quantization Bitwidth', fontsize=label_size)
    # plt.ylabel('Dataset - Quantization Bitwidth', fontsize=label_size)
    plt.savefig(save_name)
    plt.show()







