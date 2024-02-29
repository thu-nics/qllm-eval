import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from importlib import import_module


# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

####################################### Plot Settings #######################################
# saving path
save_path = './results/'

# plot info
family_name = 'llama2'
dataset_name = 'chid'
save_name = None # e.g. 'test_plot'

# plot mode
plot_mode = 'storage_overhead'

# whether normalize the data with respect to its FP counterpart
data_normalization = False
# whether scale the accuracy to [0, 1.0].
accuracy_normalization = True

# label settings
x_label = 'Parameter Scale'
y_label = 'Accuracy'
label_size = 20

# axis settings
axis_fontsize = 20

# legend settings
legend_loc = 'lower right'
legend_fontsize = 20

# curve properties
line_width = 2.0
curve_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
marker_style = ['o', '^', 's', 'd', '+']
marker_size = 100

# model family members
families_to_plot = {
    'opt':      ['OPT_125M', 'OPT_1B7', 'OPT_2B7', 'OPT_6B7', 'OPT_13B', 'OPT_30B', 'OPT_66B'],
    'falcon':   ['Falcon_7B', 'Falcon_40B', 'Falcon_180B'],
    'llama2':   ['LlaMA2_7B', 'LlaMA2_13B', 'LlaMA2_70B'],
    'bloom':    ['Bloom_560M', 'Bloom_1B7', 'Bloom_3B', 'Bloom_7B1', 'Bloom_175B'],
    'bloomz':   ['Bloomz_560M', 'Bloomz_1B7', 'Bloomz_3B', 'Bloomz_7B1', 'Bloomz_175B'],
    # 'chatglm3': ['ChatGLM3_6B'],
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
                    family_name + '_' + dataset_name + '_' + plot_mode + '.pdf'
    save_name = save_path + save_name

    # init canvas
    fig, ax = plt.subplots(figsize=[8., 6.])

    # make the plot compact
    plt.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.12)

    # set figure labels
    plt.xlabel(x_label, fontsize=label_size)
    plt.ylabel(y_label if not data_normalization else 'Normalized ' + y_label, fontsize=label_size)

    # set axes font size
    ax.tick_params(axis='x', labelsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=axis_fontsize)

    for i, member in enumerate(families_to_plot[family_name]):
        if len(curve_colors) > 0:
            curve_color = curve_colors[i]  # specify the color you want to use
        else:
            curve_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]  # or use the default colors
        model_size = float(member.split('_')[1].replace('B', '.')) if 'M' not in member.split('_')[1] else \
                        float(member.split('_')[1].replace('M', '.')) / 1000.
        bitwidth = [16, 8, 4, 3, 2]
        x_axis = [model_size * i for i in bitwidth[::-1]]

        results_module = import_module('assets.' + dataset_name + '_' + 'w_only')
        member_results = getattr(results_module, member)
        if accuracy_normalization:
            member_results = normalize_results(member_results, with_respect_to_fp=False)

        ax.plot(x_axis, member_results[::-1], label=member.replace('_', '-'), \
                linestyle='-', color=curve_color, linewidth=line_width)
        for x, y, marker in zip(x_axis, member_results[::-1], marker_style):
            ax.scatter(x, y, marker=marker, color=curve_color, s=marker_size)

    # set legend location
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    # set grid on
    plt.grid(linestyle = '--')

    # Warning: please do not change the figure you are previewing
    # preview the figure
    plt.savefig(save_name)
    plt.show()





