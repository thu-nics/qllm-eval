import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./figures/')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--plot_mode', type=str, default='kv', choices=['w', 'wa', 'kv'])
parser.add_argument('--dataset_name', type=str, default='LongEval')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--x_label', type=str, default='Position', choices=['Position', "Length"])
parser.add_argument('--y_label', type=str, default='Accuracy')
parser.add_argument('--legend_loc', type=str, default='lower left')
args = parser.parse_args()


##################################### Helper Functions ######################################
def result_pad(raw_results, fp_idx=0):
    raw_fp_result = raw_results[fp_idx]
    new_results = [x if x is not None else 0 for x in raw_results]
    new_results[fp_idx] = raw_fp_result
    return new_results

def normalize_results(raw_results, fp_idx=0):
    # raw_results = result_pad(raw_results)
    if raw_results[fp_idx] is not None and raw_results[fp_idx] != 0:
        norm_results = [i / raw_results[fp_idx] * 100 if i is not None else None for i in raw_results]
        print('Result Normalization Succeeded.')
    else:
        norm_results = raw_results
        print('The input results have no FP precision, return original results.')
    return norm_results

MARKERS = ['o', '^', 's', 'd', '+']
COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
MODELS = ["Mixtral_8x7B", "Vicuna_7B"]
AWQ_MODELS = ["Vicuna_7B_AWQ"]


###################################### Plot Functions ######################################
if __name__ == '__main__':
    if args.plot_mode == 'kv':
        if args.x_label == "Position":
            import pos.kv_length_result as data
        else:
            import length.kv_length_result as data
    elif args.plot_mode == 'w':
        if args.x_label == "Position":
            import pos.w_length_result as data
        else:
            import length.w_length_result as data
    elif args.plot_mode == 'wa':
        if args.x_label == "Position":
            import pos.wa_length_result as data
        else:
            import length.wa_length_result as data

    # create a folder to save the resulting plot.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # plot name
    save_name = args.save_name + '.pdf' if args.save_name is not None else \
                    args.dataset_name + '_' + args.plot_mode + '_' + args.x_label + '.pdf'
    save_name = args.save_path + save_name

    # init canvas
    fig, ax = plt.subplots(figsize=[8., 6.])

    # make the plot compact
    plt.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.16)

    # set figure labels
    plt.xlabel(args.x_label, fontsize=25)
    plt.ylabel(args.y_label if not args.norm else 'Normalized ' + args.y_label, fontsize=25)
    plt.ylim(0,1.1)

    # set axes font size
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    x_axis = {
        'Position':   ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'Length':    ["4k", "6k", "9k", "11k", "13k", "16k"],
    }[args.x_label]

    # mode = {
    #     'w':   ['FP16', 'W8', 'W4', 'W3', 'W2'],
    #     'wa':    ['FP16', 'W8A8', 'W4A8', 'W4A4'],
    #     'kv': ['FP16', 'KV8', 'KV4', 'KV3', 'KV2'],
    # }[args.plot_mode]

    mode = {
        'w':   ['FP16', 'W8', 'W4', 'W3'],
        'wa':    ['FP16', 'W8A8', 'W4A8'],
        'kv': ['FP16', 'KV8', 'KV4', 'KV3'],
    }[args.plot_mode]

    for i, model_name in enumerate(MODELS):
        for j, _ in enumerate(mode):
            # ax.plot(x_axis, eval("data.{}".format(model_name))[j], marker=MARKERS[j], markersize=13, label=model_name + '-' + mode[j], linestyle="-", color=COLORS[i])
            ax.plot(x_axis, eval("data.{}".format(model_name))[j], marker=MARKERS[j], markersize=13, linestyle="-", color=COLORS[i])

    # if args.plot_mode == 'w' and args.x_label != 'Position':
    #     ax.plot(x_axis, data.Vicuna_7B_AWQ[3], marker=MARKERS[3], markersize=13, linestyle="--", color=COLORS[1])

    # set legend location
    # ax.legend(loc=args.legend_loc)
    legend_elements = []
    for i, _ in enumerate(mode):
        legend_elements.append(Line2D([0], [0], marker=MARKERS[i], color='w', label=mode[i], markerfacecolor='tab:brown', markersize=16))
    ax.legend(handles=legend_elements, fontsize=20)

    # for i, _ in enumerate(MODELS):
    #     legend_elements.append(Line2D([0], [0], color=COLORS[i], label=MODELS[i], linewidth=4))
    # ax.legend(handles=legend_elements, fontsize=20)

    # Warning: please do not change the figure you are previewing
    # preview the figure
    plt.savefig(save_name)
