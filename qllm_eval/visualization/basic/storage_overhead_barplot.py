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
group_name = 'LlaMA2-1'
save_name = None # e.g. 'test_plot'

# plot mode
plot_mode = 'even_storage'

# whether normalize the data with respect to its FP counterpart
data_normalization = False
# whether scale the accuracy to [0, 1.0].
accuracy_normalization = True

# label settings
x_label = 'Dataset'
y_label = 'Accuracy'
label_size = 20

# axis settings
axis_fontsize = 20

# legend settings
legend_loc = 'upper left'
legend_fontsize = 20

# bar properties
bar_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
bar_width = 0.35

# selected representative instances
groups_to_plot = {
    'LlaMA2-1': ['7B_16', '13B_8', '70B_2'],
    'LlaMA2-2': ['7B_8', '13B_4'],
    'LlaMA2-3': ['7B_4', '13B_2'],
    'Falcon-1': ['7B_16', '40B_4', '40B_3', '40B_2'],
    'OPT-1': ['30B_16', '66B_8'],
    'OPT-2': ['13B_16', '30B_8', '66B_4', '66B_3'],
    'OPT-3': ['6B7_16', '13B_8', '30B_4', '30B_3', '66B_2'],
    'OPT-4': ['2B7_16', '6B7_8', '13B_4', '30B_2'],
    'OPT-5': ['1B3_16', '2B7_8', '6B7_4', '6B7_3', '13B_2'],
    'Bloom-1': ['3B_16', '7B1_8'],
    'Bloom-2': ['1B7_16', '3B_8', '7B1_4'],
    'Bloom-3': ['1B1_16', '1B7_8', '3B_4', '7B1_2'],
    'Bloom-4': ['560M_16', '1B1_8', '1B7_4', '3B_3', '3B_2'],
    'Bloomz-1': ['3B_16', '7B1_8'],
    'Bloomz-2': ['1B7_16', '3B_8', '7B1_4'],
    'Bloomz-3': ['1B1_16', '1B7_8', '3B_4', '7B1_2'],
    'Bloomz-4': ['560M_16', '1B1_8', '1B7_4', '3B_3', '3B_2'],
}

member_number_to_config_mapping = {
    2: {},
    3: {
        'bar_interval': 3,
        'internel_interval': 0.05,
        'bar_start_point': -0.4,
    },
    4: {},
    5: {},
}

bitwidth_idx_mappings = {16: 0, 8: 1, 4: 2, 3: 3, 2: 4}

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
                    group_name + '_' + plot_mode + '.pdf'
    save_name = save_path + save_name

    # init canvas
    fig, ax = plt.subplots(figsize=[16., 6.])

    # make the plot compact
    plt.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.12)

    # set figure labels
    plt.xlabel(x_label, fontsize=label_size)
    plt.ylabel(y_label if not data_normalization else 'Normalized ' + y_label, fontsize=label_size)

    # set axes font size
    ax.tick_params(axis='x', labelsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=axis_fontsize)

    x_axis = ['CHID', 'Winogrande', 'RACE', 'Lambada', 'RTE', 'PIQA', 'SIQA']

    # Construct the list containing the performance of each group member
    assert group_name in groups_to_plot.keys()
    family_name = group_name.split('-')[0]
    member_performances = []
    for member in groups_to_plot[group_name]:
        member_performance = []
        model_size, bitwidth = member.split('_')[0], member.split('_')[1]
        for dataset in x_axis:
            results_module = import_module('assets.' + dataset.lower() + '_' + 'w_only')
            individual_results = getattr(results_module, family_name + '_' + model_size)
            member_name = family_name + '-' + model_size + '-' + bitwidth + 'bit'
            member_performance.append(individual_results[bitwidth_idx_mappings[int(bitwidth)]])
        if accuracy_normalization:
            member_performance = normalize_results(member_performance, with_respect_to_fp=False)
        print(f'Performance of {family_name}-{model_size}-{bitwidth}bit: {member_performance}')
        member_performances.append({member_name: member_performance})

    # plot the bars
    member_num = len(member_performances)
    bar_interval = member_number_to_config_mapping[member_num]['bar_interval']
    internel_interval = member_number_to_config_mapping[member_num]['internel_interval']
    bar_start_point = member_number_to_config_mapping[member_num]['bar_start_point']
    x = [bar_start_point + i * bar_interval for i in range(len(x_axis))]
    x_major_locator = MultipleLocator(bar_interval)
    ax.xaxis.set_major_locator(x_major_locator)

    # plt.bar()
    # ax.set_xticklabels(x_axis)

    for idx, member in enumerate(member_performances):
        member_name = list(member.keys())[0]
        member_performance = member[member_name]
        bar_color = bar_colors[idx]
        # plt.bar(x_axis, member_performance, width=bar_width, color=bar_color, label=member_name)
        plt.bar([i + idx*(bar_width + internel_interval) for i in x], member_performance, width=bar_width, \
                color=bar_color, label=member_name)

    x_label = x_axis.insert(0, '')
    ax.set_xticklabels(x_axis)



    '''
    color_counter = 0
    for i, model_family in enumerate(model_families):
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
                    linestyle=line_style[j], color=curve_color, markersize=dot_size, linewidth=line_width)
    '''

    # set legend location
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    # Warning: please do not change the figure you are previewing
    # preview the figure
    plt.savefig(save_name)
    plt.show()





