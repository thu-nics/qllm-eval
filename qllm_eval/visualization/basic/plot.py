import os
import matplotlib.pyplot as plt
from importlib import import_module


# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

####################################### Plot Settings #######################################
# saving path
save_path = './results/family_dataset/'

# plot info
# dataset_name = 'lambada'
dataset_names = ['chid', 'winogrande', 'race', 'lambada', 'piqa', 'siqa']
save_name = None # e.g. 'test_plot'

# plot mode
# plot_mode = 'kv_cache'
plot_modes = ['w_only', 'w_a', 'kv_cache']
# assert plot_mode in ['w_only', 'w_a', 'kv_cache']

# whether normalize the data with respect to its FP counterpart
data_normalization = False
# whether scale the accuracy to [0, 1.0].
accuracy_normalization = True

# label settings
x_label = 'Precision'
y_label = 'Accuracy'
label_size = 20

# axis settings
axis_fontsize = 20

# legend settings
legend_loc = 'lower left'
legend_fontsize = 20

# curve properties
# curve_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
curve_colors = []
# line_style = ['-', '--', '-.', ':']
line_style = ['-', '-', '-', '-', '-', '-']
line_width = 2.0
# dot_style = ['.', '*', '^', ',']
dot_style = ['o', '^', 's', 'd', '+']
dot_size = 16

# selected representative instances
individuals_to_plot = {
    'opt':      ['OPT_2B7', 'OPT_66B'],
    'falcon':   ['Falcon_7B', 'Falcon_40B', 'Falcon_180B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_13B', 'LlaMA2_70B'],
    'bloom':    ['Bloom_3B', 'Bloom_175B'],
    'bloomz':   ['Bloomz_3B', 'Bloomz_175B'],
    'chatglm3': ['ChatGLM3_6B', 'ChatGLM3_6B_32K'],
    'moe': ['Mistral_7B', 'Mixtral_8x7B'],
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

    for dataset_name in dataset_names:
        for plot_mode in plot_modes:
            # plot name
            save_name = save_name + '.pdf' if save_name is not None else \
                            dataset_name + '_' + plot_mode + '.pdf'

            # plot the curves
            # model_families = ['opt', 'falcon', 'llama2', 'bloom', 'bloomz', 'chatglm3']
            model_families = ['llama2+chatglm3', 'falcon+moe']
            color_counter = 0
            for i, families in enumerate(model_families):
                # init canvas
                fig, ax = plt.subplots(figsize=[8., 6.])

                # make the plot compact
                plt.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.12)

                # set figure labels
                plt.xlabel(x_label, fontsize=label_size)
                plt.ylabel(y_label if not data_normalization else 'Normalized ' + y_label, fontsize=label_size)

                # set axes font size
                ax.tick_params(axis='x', labelsize=axis_fontsize)
                ax.tick_params(axis='y', labelsize=axis_fontsize)

                x_axis = {
                    'w_only': ['FP16', 'W8', 'W4', 'W3', 'W2'],
                    'w_a': ['FP16', 'W8A8', 'W4A8', 'W4A4'],
                    'kv_cache': ['FP16', 'KV8', 'KV4', 'KV3', 'KV2'],
                }[plot_mode]

                save_name = families + '_' + dataset_name + '_' + plot_mode + '.pdf'
                save_name = save_path + save_name
                families = families.split('+')
                for model_family in families:
                    individuals = individuals_to_plot[model_family]
                    for j, individual in enumerate(individuals):
                        if len(curve_colors) > 0:
                            curve_color = curve_colors[color_counter]  # specify the color you want to use
                            color_counter += 1
                        else:
                            curve_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]  # or use the default colors
                        results_module = import_module('assets.' + dataset_name + '_' + plot_mode)
                        individual_results = getattr(results_module, individual)
                        if accuracy_normalization:
                            individual_results = normalize_results(individual_results, with_respect_to_fp=False)
                        ax.plot(x_axis, individual_results, marker=dot_style[0], label=individual.replace('_', '-'), \
                                linestyle=line_style[j], markersize=dot_size, linewidth=line_width)

                # set legend location
                ax.legend(loc=legend_loc, fontsize=legend_fontsize)

                # Warning: please do not change the figure you are previewing
                # preview the figure
                plt.savefig(save_name)

                # plt.show()





