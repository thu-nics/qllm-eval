import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from importlib import import_module


# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

####################################### Plot Settings #######################################
# saving path
save_path = './results/histrogram/'

# plot info
model_name = 'falcon_40b'
dataset_name = 'winogrande'
save_name = None # e.g. 'test_plot'

# plot mode
plot_mode = 'w_a'
assert plot_mode in ['w_only', 'w_a', 'kv_cache']
plot_candidates = {
    'w_only': ['fp16', 'w8', 'w4', 'w3', 'w2'],
    'w_a': ['fp16', 'w8a8', 'w4a8', 'w4a4'],
    'kv_cache': ['fp16', 'kv8', 'kv4', 'kv3', 'kv2'],
}[plot_mode]


# whether normalize the data with respect to its FP counterpart
data_normalization = False
# whether scale the accuracy to [0, 1.0].
accuracy_normalization = True

# label settings
x_label = 'Difference Between Two PPL Values'
y_label = 'Frequency'
label_size = 20

# axis settings
axis_fontsize = 20

# legend settings
legend_loc = 'upper right'
legend_fontsize = 20

# curve properties
curve_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
# line_style = ['-', '--', '-.', ':']
line_style = ['-', '-', '-', '-']
line_width = 2.0
# dot_style = ['.', '*', '^', ',']
dot_style = ['o', '^', 's', 'd', '+']
dot_size = 16

option_num = {
    'winogrande': 2,
}

##################################### Helper Functions ######################################
def result_pad(raw_results, fp_idx=0):
    raw_fp_result = raw_results[fp_idx]
    new_results = [x if x is not None else 0 for x in raw_results]
    new_results[fp_idx] = raw_fp_result
    return new_results

def normalize_results(raw_results, fp_idx=0, with_respect_to_fp=True):
    # raw_results = result_pad(raw_results)
    if raw_results[fp_idx] is not None and raw_results[fp_idx] != 0 and with_respect_to_fp:
        norm_results = [i / raw_results[fp_idx] * 100 if i is not None else None for i in raw_results]
        print('Result Normalization Succeeded.')
    elif raw_results[fp_idx] is not None and raw_results[fp_idx] != 0 and not with_respect_to_fp:
        norm_results = [i / 100. if i is not None else i for i in raw_results]
        print('Result Normalization Succeeded.')
    else:
        norm_results = raw_results
        print('The input results have no FP precision, return original results.')
    return norm_results

###################################### Plot Functions ######################################
if __name__ == '__main__':
    # create a folder to save the resulting plot.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # plot name
    save_name = save_name + '.pdf' if save_name is not None else \
                    model_name + '_' + plot_mode + '_' + dataset_name + '.pdf'
    save_name = save_path + save_name

    cleaned_results = {}
    for candidate in plot_candidates:
        result_file = './assets/hist_plots/falcon_40b/' + candidate + '/winogrande.json'
        # target_files = list(filter(lambda x: 'prediction' in x, glob(target_dir)))
        # for target_file in target_files:

        with open(result_file, 'r') as file:
            raw_results = json.load(file)
            cleaned_results_list = []
            for idx, results in raw_results.items():
                cleaned_result = {}
                for key in ['prediction', 'gold']:
                    cleaned_result.update({key: str(results[key])})
                for i in range(1, option_num[dataset_name] + 1):
                    label = 'label: ' + str(i)
                    cleaned_result.update({label: results[label]['PPL']})
                cleaned_results_list.append(cleaned_result)
            cleaned_results.update({candidate: cleaned_results_list})

    # init canvas
    fig, ax = plt.subplots(figsize=[8., 6.])

    # make the plot compact
    plt.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.12)

    # calculate the difference between two ppl values
    # Style 1: only plot the abs of the difference of the two ppl values.
    '''
    fp_ppl_difference = [abs(result['label: 1'] - result['label: 2']) for result in fp_cleaned_results]
    bins = np.linspace(0., 1., 100)

    plt.hist(fp_ppl_difference, bins, alpha=0.5, label='FP16')
    '''

    # Style 2: plot model_choice_ppl - gt_ppl (should be a positive value) and
    # gt_ppl - non_chosen_ppl (should be a negative value).
    # TODO: should the difference values be normalized by the base values?
    bins = np.linspace(-1., 1., 100)

    for bitwidth, results in cleaned_results.items():
        ppl_difference = []
        right_counter = 0
        label_1_counter = 0
        for result in results:
            gt_label = result['gold']
            wrong_label = '1' if gt_label == '2' else '2'
            ppl_difference.append(result['label: ' + wrong_label] - result['label: ' + gt_label])
            if result['prediction'] == result['gold']:
                right_counter += 1
            if result['prediction'] == '1':
                label_1_counter += 1
        accuracy = round(right_counter / 1267 * 100, 2)
        label_1_ratio = round(label_1_counter / 1267 * 100, 2)
        print(f"For bitwidth {bitwidth}, the {model_name} model made {right_counter} right predicitions out of 1267 "
              f"predictions (accuracy {accuracy}%), and there are {label_1_counter} \"1\" "
              f"predictions (ratio {label_1_ratio}%).")

        plt.hist(ppl_difference, bins, alpha=0.3, label=bitwidth.upper())

    plt.axvline(x=0., color='r', linestyle='--')

    # set legend location
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    # set axis label
    plt.xlabel(x_label, fontsize=label_size)
    plt.ylabel(y_label, fontsize=label_size)

    # Warning: please do not change the figure you are previewing
    # preview the figure
    plt.savefig(save_name)
    plt.show()













